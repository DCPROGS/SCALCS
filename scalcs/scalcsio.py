"""Reading MEC, SCN, SSD, ABF files and kinetic mechanisms from DCPROGS files.

The .mec reader is the one in this module, :func:`mec_get_list` and
:func:`mec_load`. They used to hand over to ``dcio.formats.mec`` whenever that
could be imported, but no such module exists in dcio, so they never did. The
hand-over was removed rather than left waiting: had such a module appeared,
every .mec file would have been read by it instead, silently, and without the
checks below.
"""

import json
import os
import struct
import time
from array import array
import numpy as np

from scalcs import mechanism

# ---------------------------------------------------------------------------
# Checking a .mec file before trusting it
#
# Without these the reader trusted every count it read, so a damaged file or
# a file that is not a .mec file at all failed with an IndexError deep in the
# reader, or - when a count came out in the billions - read empty bytes past
# the end of the file indefinitely. Each check says what is wrong instead.
# ---------------------------------------------------------------------------

# DC's programs give their files and records three-digit version numbers -
# 102 and 200 in every file seen. Anything else is not a .mec header.
_MEC_VERSIONS = range(100, 1000)

# Far more states than any mechanism these programs could hold; a count above
# it comes from bytes that are not a header.
_MAX_STATES = 1000


def _looks_like_text(head):
    printable = sum(32 <= byte < 127 or byte in (9, 10, 13) for byte in head)
    return len(head) > 0 and printable >= 0.95 * len(head)


def _mec_file_problem(data):
    """What makes `data`, the bytes of a whole file, unreadable as a .mec
    file - or None if nothing does.

    The file begins with its version, the number of records, two words the
    reader does not use, and then the byte at which each record starts.
    """
    if len(data) < 16:
        return 'it is not a .mec file: it is only %d bytes long' % len(data)
    version, nrecs = struct.unpack_from('<ii', data, 0)
    if version not in _MEC_VERSIONS or not 1 <= nrecs <= (len(data) - 16) // 4:
        head = data[:60]
        if _looks_like_text(head):
            text = ' '.join(head.decode('ascii', 'replace').split())
            return ('it is not a .mec file: it begins with the text %r - '
                    'perhaps a printout saved under a .mec name' % text[:50])
        return ('it is not a .mec file: its header gives version %d and %d '
                'records' % (version, nrecs))
    starts = struct.unpack_from('<%di' % nrecs, data, 16)
    outside = [start for start in starts if not 1 <= start <= len(data)]
    if outside:
        return ('its index puts a record at byte %s, but the file ends at '
                'byte %d' % (', '.join(map(str, outside)), len(data)))
    return None


def _record_header_problem(version, mecnum, k, kA, kB, kC, kD):
    """What is wrong with the header at the start of one record, or None.

    `k` is the number of states; kA to kD count the open, short-lived shut,
    long-lived shut and desensitised states, and must add up to it.
    """
    if version not in _MEC_VERSIONS or mecnum < 1 or not 1 <= k <= _MAX_STATES:
        return ('there is no valid record here: the header gives version %d, '
                'mechanism %d and %d states - the index of the file may be '
                'wrong' % (version, mecnum, k))
    if min(kA, kB, kC, kD) < 0 or kA + kB + kC + kD != k:
        return ('the header contradicts itself: it gives %d states, but %d '
                'open, %d short-lived shut, %d long-lived shut and %d '
                'desensitised, which make %d'
                % (k, kA, kB, kC, kD, kA + kB + kC + kD))
    return None


def _names_for_states(stored, k):
    """A name for each of the `k` states.

    The stored names are used only when there is one for every state. Files
    that store none (kstat 0) and independent-subunit models, which store the
    names of their subunits' states - 8 for a 16-state mechanism, say - give
    positional names to every state instead: S1, S2, ..., numbered as the
    file and its stored drawing number them. Putting subunit names on the
    first few states would load the mechanism labelled wrongly, which is
    worse than not naming it.
    """
    if len(stored) >= k:
        return list(stored[:k])
    return ['S{0:d}'.format(n) for n in range(1, k + 1)]


def _names_for_rates(stored, irate, jrate):
    """A name for each rate, given its from- and to-states.

    As with states, the stored names are used only when there is one for
    every rate. An independent-subunit model stores names for its basic rate
    constants only - npar of them, fewer than its rates - with the rest tied
    to those by constraints, and which rates the names belong to is not
    recorded plainly enough to rely on. Every rate is then named by its
    transition instead: q(13,9) is the rate from state 13 to state 9.
    """
    if len(stored) >= len(irate):
        return list(stored[:len(irate)])
    return ['q(%d,%d)' % (i, j) for i, j in zip(irate, jrate)]


def _rates_problem(irate, jrate, k):
    """None if every rate joins two of the `k` declared states."""
    used = list(irate) + list(jrate)
    bad = [state for state in used if not 1 <= state <= k]
    if bad:
        state = max(bad) if max(bad) > k else min(bad)
        return ('its rates connect state %d, but the header declares %d '
                'states' % (state, k))
    return None


# ---------------------------------------------------------------------------
# The mechanism drawing a .mec file carries
# ---------------------------------------------------------------------------

def _state_name(field, number):
    """The name in one 10-byte state-name field of a .mec file.

    The field is padded, but not always with the same byte: some writers pad
    with spaces and others with NULs, and older files leave it blank, never
    having stored the names. NUL counts as padding here too, and a field
    blank after that gets a positional name, ``S<number>``, so a mechanism
    whose names were never written still loads with distinguishable states.
    `number` is the state's number in the file - from 1, as the drawing
    stored in the file numbers it - so S4 is the state drawn as 4.

    Previously a blank field either raised IndexError (space padding, nothing
    for split() to return) or gave a name made of NULs (NUL padding, which
    split() does not strip), the second silently: every state of the
    mechanism got the same unprintable name.
    """
    tokens = field.decode('utf-8').replace('\x00', ' ').split()
    return tokens[0] if tokens else 'S{0:d}'.format(number)


def _layout_from_board(board, statenames):
    """Positions of the states on the "chess board" stored with a mechanism.

    The DOS programs drew a mechanism on a grid of two-character cells and
    saved the grid in the .mec file. A cell holds a state - its number,
    usually after a C or an O for shut or open: ``C9``, ``O1``, ``10``,
    ``03`` - or a piece of the line drawing, or nothing.

    Parameters
    ----------
    board : list of rows, each a list of two-byte cells
    statenames : list of str
        Names of the states in the file's own order, so that the state
        numbered *n* on the board is ``statenames[n - 1]``.

    Returns
    -------
    dict or None
        ``{name: (column, row)}`` in cell units, row 0 at the top and shifted
        so the left-most and top-most states are at 0. None unless the board
        places every state: many files store no drawing at all, and the cube
        mechanisms are drawn in an oblique projection that overwrites some of
        the state cells. Part of a layout is no use to anything that draws.
        None, too, when states share a name - some older files store no
        state names, and positions keyed by name cannot tell them apart.

    Only positions are read. The lines are deliberately not parsed: which
    states are connected is given by the rates, and boards in circulation
    draw their connections with corner pieces, long runs and that oblique
    projection, none of which can be read back reliably.
    A number with no corresponding state is skipped rather than raised on;
    a drawing is not worth failing to load a mechanism over.
    """
    placed = {}
    for row, cells in enumerate(board):
        for column, cell in enumerate(cells):
            text = bytes(cell).strip(b'\x00 ')
            lettered = text[:1].isalpha()
            if lettered:
                text = text[1:]
            if not text.isdigit():
                continue
            number = int(text)
            if lettered and number == 0:
                number = 10            # one digit after the letter: C0 is 10
            if 1 <= number <= len(statenames):
                placed.setdefault(statenames[number - 1], (column, row))

    if len(placed) != len(statenames) or not placed:
        return None
    left = min(column for column, _ in placed.values())
    top = min(row for _, row in placed.values())
    return {name: (column - left, row - top)
            for name, (column, row) in placed.items()}


# ---------------------------------------------------------------------------
# Modern JSON mechanism format (written by dcio.formats.modern.write_mechanism)
# ---------------------------------------------------------------------------

# State subsets in Q-matrix order (open first, then within-burst shut,
# between-burst shut, desensitised).
_STATE_CLASSES = ("A", "B", "C", "D")


def mechanism_from_dict(d):
    """Build a :class:`scalcs.mechanism.Mechanism` from a dcio mechanism dict.

    The dict is the structure produced by
    :func:`dcio.formats.modern.mechanism_to_dict` (schema ``mechanism/x``),
    covering both ``source_format='mec'`` (full Q-matrix mechanism, with a
    ``value`` per rate) and ``source_format='hjcfit_prt'`` (fitted printout,
    with ``initial``/``final`` per rate — the ``final`` value is used).

    Parameters
    ----------
    d : dict
        Parsed mechanism dictionary.

    Returns
    -------
    scalcs.mechanism.Mechanism

    Raises
    ------
    ValueError
        If *d* is not a dcio mechanism dictionary.
    """
    if not isinstance(d, dict) or not str(
        d.get("dcio_schema", "")
    ).startswith("mechanism/"):
        raise ValueError(
            "scalcsio.mechanism_from_dict: not a dcio mechanism dict "
            "(missing or wrong 'dcio_schema')"
        )

    states = d["states"]
    subsets = states.get("subsets", {})
    names = list(states.get("names", []))
    conductances = list(states.get("conductances_S", []))

    # Build the state list in A, B, C, D order so that the 1-based 'from'/'to'
    # indices in the rate table line up with the Q-matrix numbering.
    StateList = []
    j = 0
    for cls in _STATE_CLASSES:
        for _ in range(int(subsets.get(cls, 0))):
            name = names[j] if j < len(names) else f"{cls}{j + 1}"
            gamma = (
                conductances[j]
                if cls == "A" and j < len(conductances)
                else 0.0
            )
            StateList.append(mechanism.State(cls, name, gamma))
            j += 1

    RateList = []
    for r in d.get("rates", []):
        # 'value' for MEC mechanisms; fall back to 'final' fitted value (PRT).
        value = r.get("value", r.get("final"))
        eff = "c" if r.get("conc_dependent") else None
        RateList.append(
            mechanism.Rate(
                value,
                StateList[r["from"] - 1],
                StateList[r["to"] - 1],
                name=r.get("name", ""),
                eff=eff,
            )
        )

    CycleList = []
    for cyc in d.get("cycles", []):
        CycleList.append(mechanism.Cycle([names[idx - 1] for idx in cyc]))

    return mechanism.Mechanism(
        RateList, CycleList,
        mtitle=d.get("title", ""),
        rtitle=d.get("rate_set_title", ""),
    )


def mec_load_json(filename):
    """Load a mechanism from a modern dcio JSON file into a Mechanism.

    Parameters
    ----------
    filename : str or path-like
        Path to a ``.json`` file written by
        :func:`dcio.formats.modern.write_mechanism`.

    Returns
    -------
    scalcs.mechanism.Mechanism
    """
    with open(filename, encoding="utf-8") as fh:
        return mechanism_from_dict(json.load(fh))


def mec_get_list(mecfile):
    """
    Read list of mechanisms saved in mec file.

    Parameters
    ----------
    mecfile : filename

    Returns
    -------
    version : int
        Version of mec file.
    meclist : list; each element is another list containing
        jstart : int; start byte for mechanism in mefile.
        mecnum : int; mechanism sequence number in mecfile.
        mectitle : string
        ratetitle : string
    max_mecnum : int
        Number of different mechanisms in mec file.
    """

    with open(mecfile, 'rb') as whole:
        problem = _mec_file_problem(whole.read())
    if problem:
        raise ValueError('%s: %s' % (mecfile, problem))

    f = open(mecfile, 'rb')
    ints = array('i')

    ints.fromfile(f,1); version  = ints.pop()
    ints.fromfile(f,1); nrecs    = ints.pop()
    ints.fromfile(f,1); ints.pop()   # nextrec
    ints.fromfile(f,1); ints.pop()   # ireclast

    jstart = np.zeros(nrecs, 'int32')
    for i in range(nrecs):
        ints.fromfile(f, 1)
        jstart[i] = ints.pop()

    meclist = []
    max_mecnum = 0
    for i in range(nrecs):
        f.seek(jstart[i] - 1 + 4)
        ints.fromfile(f,1)
        mecnum = ints.pop()
        if mecnum > max_mecnum:
            max_mecnum = mecnum
        mectitle = f.read(74).decode("utf-8")
        ints.fromfile(f,5)
        ratetitle = f.read(74).decode("utf-8")
        meclist.append([jstart[i], mecnum, mectitle, ratetitle])

    f.close()
    return version, meclist, max_mecnum

def mec_choose_from_list(meclist, max_mecnum):
    """
    Choose mechanism from a list of mechanisms in file.

    Parameters
    ----------
    meclist : list; each element is another list containing:
        jstart : int; start byte for mechanism in mefile.
        mecnum : int; mechanism sequence number in mecfile.
        mectitle : string
        ratetitle : string
    max_mecnum : int
        Number of different mechanisms in mec file.

    Returns
    -------
    mecnum : int
        Sequence number of a mechanism to read.
    ratenum : int
        Sequence number of rate set to read.
    """

    # List all mechs and choose one.
    print (' Model #              title')
    ndisp = 0
    for i in range(1, (max_mecnum + 1)):
            present = False
            id = 0
            for j in range(len(meclist)):
                if i == meclist[j][1]:
                    present = True
                    id = j
            if present:
                print (i, meclist[id][2])
                ndisp += 1
                if ndisp % 20 == 0:
                    input('\n   Hit ENTER for more... \n')
    try:
        mecnum = int(input(
            "\nWhich mechanism would you like to read (1 to %d)? ... "
            %max_mecnum))
    except:
        print ("\nError: model number not entered!")
        mecnum = max_mecnum

    # List and choose rate constants.
    print (
        "\nFor model %d the following rate constants have been stored:"
        %mecnum)

    ndisp = 0
    for i in range(len(meclist)):
       if meclist[i][1] == mecnum:
           print ((i+1), meclist[i][3])
           ndisp += 1
           if ndisp % 20 == 0:
               input("\n   Hit ENTER for more... \n")
    try:
        ratenum = (int(input(
            "\nWhich rate set would you like to read?... ")) - 1)
    except:
        print ("Error: rate set number not entered!")

    if (ratenum < 0) or (ratenum > len(meclist)):
        print ("Error: not valid rate set number!")

    return mecnum, ratenum

def mec_load(mecfile, start):
    """
    Load chosen mec.

    Parameters
    ----------
    mecfile : filename
    start : int
        Start byte in mecfile for mechanism to read.

    Returns
    -------
    mec.Mechanism(RateList, StateList, ncyc) : instance of Mechanism class.

    Raises
    ------
    ValueError
        If there is no valid record at `start`, if the record contradicts
        itself, or if its stored drawing is larger than the file. The message
        names the file, the record and what is wrong.

    Notes
    -----
    States and rates take their stored names only when the file stores one
    for each; otherwise every state is named S1, S2, ... and every rate by
    its transition, q(i,j). See _names_for_states and _names_for_rates.

    Rate constants are taken exactly as stored. An independent-subunit model
    also stores constraints (ieq, jeq, ifq, jfq, efacq) tying its rates to
    its basic rate constants; they are read but not applied. Read as "rate =
    factor x rate", they hold in some records but contradict the stored
    rates, by factors of up to a million, in others, so their meaning is not
    certain enough to fill in rates from. A record that stores only its
    basic constants therefore loads with its other rates at zero.
    """

    doubles = array('d')
    floats = array ('f')
    ints = array('i')

    filesize = os.path.getsize(mecfile)
    where = '%s, record at byte %d' % (mecfile, start)
    f=open(mecfile, 'rb')	# open the .mec file as read only
    f.seek(start - 1);
    ints.fromfile(f, 1)
    version1 = ints.pop()
    ints.fromfile(f, 1)
    mecnum = ints.pop()
    mectitle = f.read(74);

    # Read number of states.
    ints.fromfile(f,1)
    k = ints.pop()
    ints.fromfile(f,1)
    kA = ints.pop()
    ints.fromfile(f,1)
    kB = ints.pop()
    ints.fromfile(f,1)
    kC = ints.pop()
    ints.fromfile(f,1)
    kD = ints.pop()

    problem = _record_header_problem(version1, mecnum, k, kA, kB, kC, kD)
    if problem:
        f.close()
        raise ValueError('%s: %s' % (where, problem))

    # In mec files version=103 all shut states are of type 'C'.
    # Check and leave just one in state 'C', others go as 'B'.
    if kB == 0:
        kB = kC - 1
        kC = 1

    ratetitle = f.read(74)

    # Read size of chess board to draw mechanism.
    ints.fromfile(f,1)
    ilast = ints.pop()
    ints.fromfile(f,1)
    jlast= ints.pop()

    # nrateq- number of non-zero rates in Q; = 2*ncon (always)
    ints.fromfile(f,1)
    nrateq = ints.pop()

    # Number of connections.
    ints.fromfile(f,1)
    ncon = ints.pop()

    # Number of concentration dependent rates
    ints.fromfile(f,1)
    ncdep = ints.pop()

    # Number of ligands
    ints.fromfile(f,1)
    nlig = ints.pop()

    # ? if char mechanism is presnt
    ints.fromfile(f,1)
    chardef = ints.pop()

    # ???
    ints.fromfile(f,1)
    boundef = ints.pop()

    # Number of cycles.
    ints.fromfile(f,1)
    ncyc = ints.pop()

    # Voltage.
    floats.fromfile(f,1)
    vref = floats.pop()
#    print 'vref=', vref

    # Number of voltage dependent rates.
    ints.fromfile(f,1)
    nvdep = ints.pop()

    # ???
    ints.fromfile(f,1)
    kmfast = ints.pop()

    # Independent subunit model.
    # False for all old models (npar=nrateq=2*ncon)
    # True when npar < nrateq=2*ncon. In this case must have nsetq>0
    ints.fromfile(f,1)
    indmod = ints.pop()

    # Number of basic rates constants.
    # Normally npar=nrateq and nsetq=0, but when indmod=T then npar<nrateq.
    ints.fromfile(f,1)
    npar = ints.pop()

    # ???
    ints.fromfile(f,1)
    nsetq = ints.pop()

    # ???
    ints.fromfile(f,1)
    kstat = ints.pop()

    # The "chess board" the DOS programs drew the mechanism on: a grid of
    # two-character cells, stored column by column. Kept, row by row, so the
    # positions of the states can be recovered once their names are known;
    # see _layout_from_board.
    #
    # Its size is checked against what is left of the file first. f.read()
    # past the end returns empty bytes rather than failing, so a size read
    # from damaged bytes - over a billion cells, in one file - made this loop
    # run, in effect, for ever.
    if ilast < 0 or jlast < 0 or f.tell() + 2 * ilast * jlast > filesize:
        f.close()
        raise ValueError(
            '%s: the stored drawing of the mechanism would be %d x %d cells, '
            'more than is left of the file' % (where, ilast, jlast))
    Jch = []
    for j in range(0, jlast):
        Ich = []
        for i in range(0, ilast):  # 500 is max
             charmod = f.read(2)#.decode("utf-8")
             Ich.append(charmod)
        Jch.append(Ich)
    board = []
    for i in range(0,ilast):
        IIch = []
        for j in range(0, jlast):
            IIch.append(Jch[j][i])
        board.append(IIch)

    # Read rate constants.
    irate = []
    for i in range(nrateq):
        ints.fromfile(f,1)
        irate.append(ints.pop())
    jrate = []
    for i in range(nrateq):
        ints.fromfile(f,1)
        jrate.append(ints.pop())
    problem = _rates_problem(irate, jrate, k)
    if problem:
        f.close()
        raise ValueError('%s: %s' % (where, problem))
    QT = np.zeros((k, k), 'float64')
    for i in range(nrateq):
        doubles.fromfile(f, 1)
        QT[irate[i]-1, jrate[i]-1] = doubles.pop()
    ratename = []
    for i in range(npar):
        ratename.append(f.read(10).decode("utf-8"))
        #print ratename[i], "QT[",irate[i],",",jrate[i],"]=", QT[irate[i]-1,jrate[i]-1]

    # Read ligand name and ligand molecules bound in each state.
    for j in range(0, nlig):
        ligname = f.read(20).decode("utf-8")
        #print "Number of ligand %s molecules bound to states:" %ligname
    nbound = np.zeros((nlig,k), 'int32')
    for i in range(nlig):
        for j in range(k):
            ints.fromfile(f, 1)
            nbound[i,j] = ints.pop()
        #print "to state",j+1,":",nbound[i,j]

    # Read concentration dependent rates.
    # from state
    ix = []
    for i in range(0, ncdep):
        ints.fromfile(f,1)
        ix.append(ints.pop())
    # to state
    jx = []
    for j in range(0, ncdep):
        ints.fromfile(f,1)
        jx.append(ints.pop())
        #if verbose: print "jx[",j,"]=",jx[j]
    # ligand bound in that particular transition
    il = []
    for i in range(0, ncdep):
        ints.fromfile(f,1)
        il.append(ints.pop())
        #if verbose: print "il[", i, "]=", il[i]

    # Read open state conductance.
    dgamma = []
    for j in range(0, kA):
        doubles.fromfile(f,1)
        dgamma.append(doubles.pop())
#    print 'dgamma=', dgamma

    # Get number of states in each cycle and connections.
    nsc = np.zeros(50, 'int32')
    for i in range(0, ncyc):
        ints.fromfile(f,1)
        nsc[i] = ints.pop()
    #print "nsc[", i, "]=", nsc[i]
    im = np.zeros((50, 100), 'int32')
    for i in range(0, ncyc):
        for j in range(0, nsc[i]):
            ints.fromfile(f,1)
            im[i, j] = ints.pop()
            #print "im[",i,",",j,"]=",im[i,j]
    jm = np.zeros((50,100), 'int32')
    for i in range(0, ncyc):
        for j in range(0, nsc[i]):
            ints.fromfile(f,1)
            jm[i,j] = ints.pop()
            #print "jm[",i,",",j,"]=",jm[i,j]

    # Read voltage dependent rates.
    # from state
    iv = []
    for i in range(0, nvdep):
        ints.fromfile(f,1)
        iv.append(ints.pop())
        #print "iv[",i,"]=",iv[i]
    # to state
    jv = []
    for j in range(0, nvdep):
        ints.fromfile(f,1)
        jv.append(ints.pop())
        #print "jv[", j,"]=",jv[j]

    hpar = []
    for i in range(0, nvdep):
        floats.fromfile(f,1)
        hpar.append(floats.pop())
        #print "hpar[",i,"]=",hpar[i]

    pstar = []
    for j in range(0, 4):
        floats.fromfile(f,1)
        pstar.append(floats.pop())
        #print "pstar[",j,"]=",pstar[j]

    kmcon = []
    for i in range(0, 9):
        ints.fromfile(f,1)
        kmcon.append(ints.pop())
        #print "kmcon[",i,"]=",kmcon[i]

    ieq = []
    for i in range(0, nsetq):
        ints.fromfile(f,1)
        ieq.append(ints.pop())
        #print "ieq[",i,"]=",ieq[i]

    jeq = []
    for j in range(0, nsetq):
        ints.fromfile(f,1)
        jeq.append(ints.pop())
        #print "jeq[", j, "]=", jeq[j]

    ifq = []
    for i in range(0,nsetq):
        ints.fromfile(f,1)
        ifq.append(ints.pop())
        #print "ifq[",i,"]=",ifq[i]

    jfq = []
    for j in range(0, nsetq):
        ints.fromfile(f,1)
        jfq.append(ints.pop())
        #print "jfq[",j,"]=",jfq[j]

    efacq = []
    for i in range(0, nsetq):
        floats.fromfile(f,1)
        efacq.append(floats.pop())
        #print "efacq[",i,"]=",efacq[i]

    statenames = []
    for i in range(0, kstat):
        statenames.append(_state_name(f.read(10), i + 1))
        #print "State name:", statename
    #print ("\n")

    # Some records go on with six more integers - nsub, kstat0, npar0, kcon,
    # npar1 and ncyc0 - which nothing here uses. Not every record has them:
    # in demo.mec each record ends right after its state names, so reading
    # them took 24 bytes of the next record, and ran off the end of the file
    # on the last one. They are no longer read.

    f.close()

    # one name per state and per rate; see _names_for_states and
    # _names_for_rates for when the stored ones cannot be used
    names = _names_for_states(statenames, k)
    ratename = _names_for_rates(ratename, irate, jrate)

    StateList = []
    j = 0
    for i in range(kA):
        StateList.append(mechanism.State('A', names[j], dgamma[j]))
        j += 1
    for i in range(kB):
        StateList.append(mechanism.State('B', names[j], 0))
        j += 1
    for i in range(kC):
        StateList.append(mechanism.State('C', names[j], 0))
        j += 1
    for i in range(kD):
        StateList.append(mechanism.State('D', names[j], 0))
        j += 1

    RateList = []
    for i in range(nrateq):
        cdep = False
        bound = None
        for j in range(ncdep):
            if ix[j] == irate[i] and jx[j] == jrate[i]:
                cdep = True
                bound = 'c'
        rate = QT[irate[i] - 1, jrate[i] - 1]
        # REMIS: please make sure the state indexing is correct
        RateList.append(mechanism.Rate(rate, StateList[irate[i]-1],
            StateList[jrate[i]-1], name=ratename[i], eff=bound))

    CycleList = []
    for i in range(ncyc):
#        mrconstrained = False
        CycleStates = []
        for j in range(nsc[i]):
            CycleStates.append(names[im[i, j]-1])
        CycleList.append(mechanism.Cycle(CycleStates))

    mec = mechanism.Mechanism(RateList, CycleList,
        mtitle=mectitle, rtitle=ratetitle)
    # names holds exactly one name per state, numbered as the drawing is
    mec.layout = _layout_from_board(board, names)
    return mec

def load_mec_from_prt(filename, verbose=False):    
    f = open(filename, 'r')
    linenum = 0
    
    
    ncyc, nsc, im2, jm2, mr = 0, [], [], [], []
    states, conductance, ligbound = [], [], []
    tres, tcrit, CHS = [], [], []
    rates, constrained = [], []
    k, kA, kB, kC, kD = 0, 0, 0, 0, 0

    f.readline()
    line = f.readline()
    if "Program HJCFIT Windows Version (Beta)" in line:
        version = 'win'
    else:
        version = 'dos'
    if verbose: print ('version=', version)

    while True:
        try:
            line = f.readline()
            #print 'line:', line
            if line == '':
                break
            line = line.strip("\r\n")
            linenum += 1
        except EOFError:
            print('Mecanism loading from PRT (or TXT) file finished.')
            
        if "HJCFIT: Fit of model to open-shut times with missed events" in line:
            if verbose: print ('This is possibly HJCFIT printout file.')
            
        if "No of states in each subset: kA, kB, kC, kD =" in line:
            kA = int(line.split()[-4].strip())
            kB = int(line.split()[-3].strip())
            kC = int(line.split()[-2].strip())
            kD = int(line.split()[-1].strip())
            k = kA + kB + kC + kD
            if verbose: print ("Number of states: kA, kB, kC, kD = {0:d}, {1:d}, {2:d}, {3:d}".
                format(kA, kB, kC, kD))

        if version == 'win' and "Number of open states =" in line and k == 0:
            kA = int(line.split()[-1].strip())
            if verbose:
                print("Number of open states = ", kA)
            line = f.readline()
            while (line != "\n") and ("conductance of state" in line):
                conductance.append(float(line.split()[-1].strip()))
                line = f.readline()
            k = len(conductance)
            kB = k - kA - 1
            kC = 1
            kD = 0
            if verbose: print ("Number of states: kA, kB, kC, kD = {0:d}, {1:d}, {2:d}, {3:d}".
                format(kA, kB, kC, kD))
            
        if "Number of ligands =" in line:
            nlig = int(line.split()[-1].strip())
            if verbose: print ("Number of ligands = {0:d}".format(nlig))
            f.readline() #'Concentration-dependent elements:'
            f.readline() #'  i   j     ligand #   Ligand name'
            line = f.readline()
            im, jm, lig = [], [], []
            while line != "\n":
                temp = line.split()
                im.append(int(temp[0]))
                jm.append(int(temp[1]))
                lig.append(temp[-1])
                line = f.readline()
            if verbose: print ("im=", im)
            if verbose: print ("jm=", jm)
            if verbose: print ("lig=", lig)
                
        
        if (version == 'dos') and ("Cycle #" in line) and (int(line[-1]) > ncyc):
            c1, c2 = [], []
            ncyc += 1
            line = f.readline()
            temp = line.split()
            c1.append(int(temp[1].strip(',')))
            c2.append(int(temp[2].strip(')')))
            mr.append([int(temp[1].strip(',')), int(temp[2].strip(')'))])
            line = f.readline()
            temp = line.split()
            while temp:
                c1.append(int(temp.pop(0).strip('-')))
                c2.append(int(temp.pop(0)))
            nsc.append(len(c1))
            im2.append(c1)
            jm2.append(c2)
            if verbose: print ("Cycle # ", ncyc)
            if verbose: print (nsc)
            if verbose: print (im2, jm2)

        if (version == 'win') and ("Microscopic reversibility" in line):
            line = f.readline()
            while "cycle=" in line:
                c1, c2 = [], []
                ncyc += 1
                temp = line.strip(';').split()
                nsc.append(len(temp[3].strip()))
                temp = temp[7:]
                for item in temp:
                    c1.append(int(item.strip(';')))
                im2.append(c1)
                line = f.readline()

        if (version == 'win') and ("by microscopic reversibility" in line):
            if len(mr) < ncyc:
                temp = []
                temp.append(int(line[3:5].strip()))
                temp.append(int(line[6:8].strip()))
                mr.append(temp)
            
#        if "state #    state name" in line: # in 'dos'
#            line = f.readline()
#            while line != "\n":
#                temp = line.split()
#                states.append(temp[1])
#                line = f.readline()
#            if len(states) == k:
#                if verbose: print states
#            else:
#                print "Warning: number of states does not correspond."
                
        if "Number of ligands bound" in line:
            line = f.readline()
            line = f.readline()     
            while len(states) < k:
                temp = line.split()
                if len(temp) == 3:
                    states.append(temp[1])
                    ligbound.append(int(temp[2]))
                line = f.readline()
            if verbose: print ('states: ', states)
            if verbose: print ('ligandsbound=', ligbound)
                
        if "Resolution for HJC calculations" in line:
            line = f.readline()
            while line != "\n":
                temp = line.split()
                if len(temp) == 4:
                    tres.append(float(temp[2]) * 1e-6)
                line = f.readline()
            if verbose: print ('resolution: ', tres)
            
        if "The following parameters are constrained:" in line:
            line = f.readline()
            while '-----' not in line:
                cnstr = []
                temp = line.split()
                r1 = int(temp[1]) - 1
                line = f.readline()
                temp = line.split()
                fac = float(temp[0])
                op = temp[1]
                r2 = int(temp[3]) - 1
                cnstr.append(r1)
                cnstr.append(op)
                cnstr.append(fac)
                cnstr.append(r2)
                line = f.readline()
                constrained.append(cnstr)
#            print constrained
            
        if (version == 'dos') and ("initial        final" in line):
            line = f.readline()
            while line != "\n":
                rate = []
                if 'q(' in line:
                    rate.append(int(line[8:10].strip()))
                    rate.append(int(line[11:13].strip()))
                    rate.append(line[18:29].strip())
                    rate.append(float(line[42:55].strip()))
                    if len(line) > 55:
                        rate.append(line[55:].strip().strip('(').strip(')'))
                rates.append(rate)
                line = f.readline()
            if constrained:
                for item in constrained:
                    if len(rates[item[0]]) > 4: # and rates[item[0, 4]] == 'constrained':
                        rates[item[0]].append(item[1])
                        rates[item[0]].append(item[2])
                        rates[item[0]].append(item[3])
            if verbose: print (rates)
            
        if 'Total number of rates' in line:
            numrates = int(line.split()[-1])
            numfixed = int(f.readline().split()[-1])
            numconstr = int(f.readline().split()[-1])
            nummr = int(f.readline().split()[-1])
            numec50 = int(f.readline().split()[-1])
            numfree = int(f.readline().split()[-1])
            if verbose: print ('\nTotal number of rates = {0:d}'.format(numrates) +
                '\nNumber that are fixed       = {0:d}'.format(numfixed) +
                '\nNumber that are constrained = {0:d}'.format(numconstr) +
                '\nNumber set by micro rev     = {0:d}'.format(nummr) +
                '\nNumber set by fixed EC50    = {0:d}'.format(numec50) +
                '\nNumber of free rates to be estimated = {0:d}'.format(numfree))

        if (version == 'win') and ("initial        final" in line):
            while len(rates) < numrates:
                rate = []
                line = f.readline()
                while line != "\n":
                    rate.append(int(line[8:10].strip()))
                    rate.append(int(line[11:13].strip()))
                    rate.append(line[18:29].strip())
                    rate.append(float(line[42:55].strip()))
                    rates.append(rate)
                    line = f.readline()
                    if line != "\n":
                        rates[-1].append(line.strip().strip('(').strip(')'))
                        line = f.readline()

    if verbose: print ('file contains {0:d} lines'.format(linenum))
    f.close()
    
    StateList = []
    j = 0
    for i in range(kA):
        StateList.append(mechanism.State('A', states[j], 50))
        j += 1
    for i in range(kB):
        StateList.append(mechanism.State('B', states[j], 0))
        j += 1
    for i in range(kC):
        StateList.append(mechanism.State('C', states[j], 0))
        j += 1
    for i in range(kD):
        StateList.append(mechanism.State('D', states[j], 0))
        j += 1
        
    RateList = []
    for i in range(numrates):
        bound = None
        mrc = False
        is_constr = False
        constr_func = None
        constr_args = None
        for j in range(len(im)):
            if im[j] == rates[i][0] and jm[j] == rates[i][1]:
                bound = 'c'
        for j in range(ncyc):
            if mr[j][0] == rates[i][0] and mr[j][0] == rates[i][1]:
                mrc = True
        if len(rates[i]) > 5 and rates[i][4] == 'constrained':
            is_constr = True
            if rates[i][5] == 'times':
                constr_func = mechanism.constrain_rate_multiple
                constr_args = [rates[i][7], rates[i][6]]
        
        rate = rates[i][3]
        RateList.append(mechanism.Rate(rate, StateList[rates[i][0]-1],
            StateList[rates[i][1]-1], name=rates[i][2], eff=bound, mr=mrc,
            is_constrained=is_constr, constrain_func=constr_func, constrain_args=constr_args))
            
    CycleList = []
    for i in range(ncyc):
#        mrconstrained = False
        CycleStates = []
        for j in range(nsc[i]):
            CycleStates.append(states[im2[i][j]-1])
        CycleList.append(mechanism.Cycle(CycleStates))

    return mechanism.Mechanism(RateList, CycleList)

#def set_load_from_prt(filename, verbose=False):    
#    f = open(filename, 'r')
#    linenum = 0
#    
#    scnfiles = []
#    conc, tres, tcrit, chsvec = [], [], [], []
#    
#    f.readline()
#    line = f.readline()
#    if "Program HJCFIT Windows Version (Beta)" in line:
#        version = 'win'
#    else:
#        version = 'dos'
#    if verbose: print 'version=', version
#
#    while True:
#        try:
#            line = f.readline()
#            #print 'line:', line
#            if line == '':
#                break
#            line = line.strip("\r\n")
#            linenum += 1
#        except EOFError:
#            print('Data set loading from PRT (or TXT) file finished.')
#            
#    return scnfiles, conc, tres, tcrit, chsvec

def mod_load(file):
    """
    Load mechanism from Channel Lab .mod file.

    Parameters
    ----------
    file : filename

    Returns
    -------
    mec.Mechanism(RateList, StateList, ncyc) : instance of Mechanism class.
    """

    # TODO: get cycles from mod file.
    f = open(file, 'r')	# open the .mec file as read only
    cl = f.readline().strip("\n")
    #print cl
    modtitle = f.readline().strip("\n")
    #print modtitle

    while True:
        try:
            line = f.readline()
            #print 'line:', line
            if line == '':
                break
            line = line.strip("\r\n")
        except EOFError:
            print('MOD reading finished.')

        if line == '[State1-20: labels]':
            statelabels = []
            for i in range(20):
                statelabels.append(f.readline().strip("\r\n"))
#            print statelabels

        if line == '[State1-20: onoff]':
            stateonoff = np.empty((20, 4))
            for i in range(20):
                onoff = f.readline().strip("\r\n")
                values = onoff.split(' ')
                stateonoff[i, 0] = int(values[0])
                stateonoff[i, 1] = int(values[1])
                stateonoff[i, 2] = float(values[2])
                stateonoff[i, 3] = float(values[3])
#            print stateonoff

        if line == '[Drug dependence labels]':
            druglabels = []
            for i in range(6):
                druglabels.append(f.readline().strip("\r\n"))
#            print druglabels

        if line == '[K1-31on]':
            kon = np.empty(31)
            for i in range(31):
                kon[i] = float(f.readline().strip("\r\n"))
#            print kon

        if line == '[K1-31off]':
            koff = np.empty(31)
            for i in range(31):
                koff[i] = float(f.readline().strip("\r\n"))
#            print koff

        if line == '[Kon concentration/voltage dependent]':
            konconc = np.empty((31, 7))
            for i in range(31):
                onconc = f.readline().strip("\r\n")
                values = onconc.split(' ')
                for j in range(7):
                    konconc[i, j] = int(values[j])
#            print konconc

        if line == '[Koff concentration/voltage dependent]':
            koffconc = np.empty((31, 7))
            for i in range(31):
                offconc = f.readline().strip("\r\n")
                values = offconc.split(' ')
                for j in range(7):
                    koffconc[i, j] = int(values[j])
#            print koffconc

        # [Rate constants for transitions between all states]
        # [Constraints for rate constants for transitions]
        # [Loop Constraints for rate constants for transitions]

        # [Kon steepness of voltage dependence]
        # [Kon activation range for voltage dependence]
        # [Koff steepness of voltage dependence]
        # [Koff activation range for voltage dependence]
        # [QMatrixMonteCarlo,#passes,#pts,#start,#channels,openstate,startstate,pApS,mVmM]
        # [extra storage space]
        # [adinterval,conductance,voltage,vrev,valence,kzero]
        # [Stimulus: Mode,AutoStep,ManualStep,Exp,Fc,pole,autofilt]
        # [Analysis: pksrch,pkave,basepts,basepos,startpos,risebeg,riseend]
        # [Analysis: expmodel,parse,numfitpts,numiter,numrestarts,tol,dual,startpts]
        # [Analysis: Fit limits hi,lo for vertex 1-72]
        # [Analysis: Fixed fit params for vertex 1-72]
        # [Analysis: Seeds for vertex 1-72, fitlims,autoseed,fixfree]
        # [FitWaveforms: start, stop positions for fit]
        # [Multi-stimulus file voltage / concentration parameters]
        # 20 [TState1-20: labels]

    f.close()

    StateList = []
    statesA = []
    statesB = []
    statesC = []
    for i in range(stateonoff.shape[0]):
        if stateonoff[i, 0] == 1:
            if stateonoff[i, 1] == 1:
                statesA.append(i)
            elif stateonoff[i, 3] == 1:
                statesC.append(i)
            else:
                statesB.append(i)
    newstates = []
    newstates.extend(statesA)
    newstates.extend(statesB)
    newstates.extend(statesC)
    k = len(newstates)

    for i in range(k):
            if stateonoff[newstates[i], 1] == 1:
                StateList.append(mechanism.State('A',
                    statelabels[newstates[i]], stateonoff[newstates[i], 2]))
            elif stateonoff[newstates[i], 3] == 1:
                StateList.append(mechanism.State('C',
                    statelabels[newstates[i]], 0))
            else:
                StateList.append(mechanism.State('B',
                    statelabels[newstates[i]], 0))

    RateList = []
    for i in range(k):
        li = (newstates[i]+1) // 5
        ci = (newstates[i]+1) % 5

        bound = None
        if newstates[i] != max(newstates):
            if (newstates[i] > 14) and (newstates[i] < 19):
                # States 16-19.
                ri = 9 * li + ci - 1
                if (kon[ri] != 0) or (koff[ri] != 0):
                    if konconc[ri1, 0] == 1:
                        bound = 'c'
                    else:
                        bound = None
                    RateList.append(mechanism.Rate(kon[ri1],
                        StateList[i],
                        StateList[newstates.index(newstates[i]+1)],
                        name='k'+str(i)+str(newstates.index(newstates[i]+1)),
                        eff=bound))
                    if koffconc[ri1, 0] == 1:
                        bound = 'c'
                    else:
                        bound = None
                    RateList.append(mechanism.Rate(koff[ri1],
                        StateList[newstates.index(newstates[i]+1)],
                        StateList[i],
                        name='k'+str(newstates.index(newstates[i]+1))+str(i),
                        eff=bound))

            elif ci == 0:
                # States 5, 10, 15
                ri = 9 * li + ci + 4 - 1
                if (kon[ri] != 0) or (koff[ri] != 0):
                    if konconc[ri, 0] == 1:
                        bound = 'c'
                    else:
                        bound = None
                    RateList.append(mechanism.Rate(kon[ri],
                        StateList[i],
                        StateList[newstates.index(newstates[i]+5)],
                        name='k'+str(i)+str(newstates.index(newstates[i]+5)),
                        eff=bound))
                    if koffconc[ri, 0] == 1:
                        bound = 'c'
                    else:
                        bound = None
                    RateList.append(mechanism.Rate(koff[ri],
                        StateList[newstates.index(newstates[i]+5)],
                        StateList[i],
                        name='k'+str(newstates.index(newstates[i]+5))+str(i),
                        eff=bound))

            else:
                # States 1-4, 6-9, 11-14
                ri1 = 9 * li + ci - 1
                ri2 = 9 * li + ci + 4 - 1

                if (kon[ri1] != 0) or (koff[ri1] != 0):
                    if konconc[ri1, 0] == 1:
                        bound = 'c'
                    else:
                        bound = None
                    RateList.append(mechanism.Rate(kon[ri1],
                        StateList[i],
                        StateList[newstates.index(newstates[i]+1)],
                        name='k'+str(i)+str(newstates.index(newstates[i]+1)),
                        eff=bound))
                    if koffconc[ri1, 0] == 1:
                        bound = 'c'
                    else:
                        bound = None
                    RateList.append(mechanism.Rate(koff[ri1],
                        StateList[newstates.index(newstates[i]+1)],
                        StateList[i],
                        name='k'+str(newstates.index(newstates[i]+1))+str(i),
                        eff=bound))

                if (((kon[ri2] != 0) or (koff[ri2] != 0)) and
                    ((newstates[i]+5) <= max(newstates))):
                    if konconc[ri2, 0] == 1:
                        bound = 'c'
                    else:
                        bound = None
                    RateList.append(mechanism.Rate(kon[ri2],
                        StateList[i],
                        StateList[newstates.index(newstates[i]+5)],
                        name='k'+str(i)+str(newstates.index(newstates[i]+5)),
                        eff=bound))
                    if koffconc[ri2, 0] == 1:
                        bound = 'c'
                    else:
                        bound = None
                    RateList.append(mechanism.Rate(koff[ri2],
                        StateList[newstates.index(newstates[i]+5)],
                        StateList[i],
                        name='k'+str(newstates.index(newstates[i]+5))+str(i),
                        eff=bound))

    return mechanism.Mechanism(RateList, mtitle=modtitle), modtitle

def mec_save_to_yaml(mec, fname):
    import yaml
    stream = open(fname, 'w')
    yaml.dump(mec, stream)
