"""Tests for the .mec reader's handling of damaged and unusual files.

A survey of one collection of 29 .mec files (374 records) found every way the
reader failed: a file that is not a .mec file at all; an index entry pointing at
no record, which made mec_load hang; records whose header contradicts itself or
whose rates use states the header does not declare; records that store no state
names, or only the names of the subunits of an independent-subunit model; and
six trailing fields, never used, that the reader read past the end of records
that do not have them.

The rules are tested on their own, with no file. The tests at the end load the
actual records, and need SCALCS_MEC_DIR to point at that collection.
"""
import contextlib
import glob
import io
import os
import struct

import numpy as np
import pytest

from scalcs import scalcsio


def mec_header(version=102, starts=(20000,), size=22000):
    """The first bytes of a .mec file, as mec_get_list reads them."""
    data = struct.pack('<4i', version, len(starts), 22000, starts[-1])
    data += struct.pack('<%di' % len(starts), *starts)
    return data + b'\x00' * (size - len(data))


class TestIsItAMecFile:

    def test_a_well_formed_header(self):
        assert scalcsio._mec_file_problem(mec_header()) is None

    def test_a_text_file_says_what_it_starts_with(self):
        """qHETGLY15.mec is an HJCFIT printout with a .mec extension. Read as
        binary, its first words gave version 1344277005 and 1919381362
        records."""
        text = (b'\r\n Program HJCFIT Windows Version\r\n\r\nCopyright '
                b'D. Colquhoun, I. Vais and A.G. Hawkes\r\n' + b' ' * 200)
        problem = scalcsio._mec_file_problem(text)
        assert 'not a .mec file' in problem
        assert 'Program HJCFIT Windows Version' in problem

    def test_a_record_that_would_start_beyond_the_end(self):
        problem = scalcsio._mec_file_problem(mec_header(starts=(20000, 90000)))
        assert '90000' in problem

    def test_an_impossible_number_of_records(self):
        data = struct.pack('<4i', 102, 1919381362, 0, 0) + b'\x00' * 100
        assert scalcsio._mec_file_problem(data) is not None


class TestRecordHeader:

    def test_a_well_formed_record(self):
        assert scalcsio._record_header_problem(102, 1, 10, 3, 6, 1, 0) is None

    def test_no_record_where_the_index_points(self):
        """Bgtvlt30.mec: the index says byte 20000, where the bytes give
        version 0, mechanism 0 and over a billion states. The reader went on
        to read a drawing of over a billion cells, one empty read at a time."""
        problem = scalcsio._record_header_problem(
            0, 0, 1852793632, 1952671086, 1936617321, 1952801312, 1852138871)
        assert 'no valid record' in problem

    def test_a_header_that_contradicts_itself(self):
        """qmechsDC1.mec @42458: 5 states, but 2 + 1 + 3 + 0 of the four
        kinds."""
        problem = scalcsio._record_header_problem(102, 23, 5, 2, 1, 3, 0)
        assert '5 states' in problem and '6' in problem


class TestStateNames:
    """A name is used only when the file stores one for every state."""

    def test_one_name_per_state(self):
        assert scalcsio._names_for_states(['AR*', 'AR', 'R'], 3) == \
            ['AR*', 'AR', 'R']

    def test_no_names_stored(self):
        """CKSIM3B.MEC stores none: kstat is 0."""
        assert scalcsio._names_for_states([], 3) == ['S1', 'S2', 'S3']

    def test_fewer_names_than_states_are_not_used_at_all(self):
        """ioana.mec, an independent-subunit model, has 16 states and stores
        8 names - the states of its two 4-state subunits, not states 1 to 8.
        Putting those names on the first 8 states would label the mechanism
        wrongly, so every state gets its positional name instead."""
        subunits = ['A2R*', 'A2R', 'AR', 'R', 'G2R*', 'G2R', 'GR', 'R']
        assert scalcsio._names_for_states(subunits, 16) == \
            ['S%d' % n for n in range(1, 17)]

    def test_more_names_than_states(self):
        assert scalcsio._names_for_states(['A', 'B', 'C', 'D'], 3) == \
            ['A', 'B', 'C']


class TestRateNames:
    """As with states: a stored name is used only if every rate has one."""

    def test_one_name_per_rate(self):
        assert scalcsio._names_for_rates(['beta1', 'alpha1'], [1, 2], [2, 1]) \
            == ['beta1', 'alpha1']

    def test_fewer_names_than_rates_name_every_rate_by_its_transition(self):
        """An independent-subunit model stores names for its basic rate
        constants only - 12 for ioana.mec's 48 rates - and which rates they
        belong to is not something to guess."""
        names = scalcsio._names_for_rates(['k+'], [1, 2, 2, 3], [2, 1, 3, 2])
        assert names == ['q(1,2)', 'q(2,1)', 'q(2,3)', 'q(3,2)']


class TestRatesUseDeclaredStates:

    def test_rates_within_the_states(self):
        assert scalcsio._rates_problem([1, 2, 2, 3], [2, 1, 3, 2], 3) is None

    def test_a_rate_to_a_state_the_header_does_not_declare(self):
        """qmechs.mec @22844 declares 3 states; its rates connect 6."""
        problem = scalcsio._rates_problem([1, 3, 1, 5, 2, 3, 3, 4, 4, 6],
                                          [3, 1, 5, 1, 3, 2, 4, 3, 6, 4], 3)
        assert '3 states' in problem and 'state 6' in problem


# ---------------------------------------------------------------------------
# The real files
# ---------------------------------------------------------------------------

MEC_DIR = os.environ.get('SCALCS_MEC_DIR', '')


def _path(name):
    path = os.path.join(MEC_DIR, name)
    if not os.path.isfile(path):
        pytest.skip('set SCALCS_MEC_DIR to a folder holding %s' % name)
    return path


def _load(name, start):
    with contextlib.redirect_stdout(io.StringIO()):
        return scalcsio.mec_load(_path(name), start)


def _records(path):
    with contextlib.redirect_stdout(io.StringIO()):
        _, entries, _ = scalcsio.mec_get_list(path)
    return sorted({entry[0] for entry in entries})


def _q_is_proper(mec):
    Q = np.asarray(mec.Q, dtype=float)
    return np.abs(Q.sum(axis=1)).max() < 1e-6 * np.abs(Q).max()


class TestRealFiles:

    def test_not_a_mec_file(self):
        with pytest.raises(ValueError, match='HJCFIT'):
            with contextlib.redirect_stdout(io.StringIO()):
                scalcsio.mec_get_list(_path('qHETGLY15.mec'))

    def test_an_index_pointing_at_no_record_is_an_error_not_a_hang(self):
        with pytest.raises(ValueError, match='20000'):
            _load('Bgtvlt30.mec', 20000)

    def test_a_drawing_larger_than_the_file_is_an_error_not_a_hang(self, tmp_path):
        """bgtv2a.mec with the drawing's height overwritten with a huge
        number: the header is otherwise sound, so only the size check on the
        drawing stands between the reader and a hang."""
        data = bytearray(open(_path('bgtv2a.mec'), 'rb').read())
        struct.pack_into('<i', data, 20000 - 1 + 176, 2000000000)
        damaged = tmp_path / 'damaged.mec'
        damaged.write_bytes(bytes(data))
        with pytest.raises(ValueError, match='drawing'):
            with contextlib.redirect_stdout(io.StringIO()):
                scalcsio.mec_load(str(damaged), 20000)

    def test_a_header_that_contradicts_itself(self):
        with pytest.raises(ValueError, match='5 states'):
            _load('qmechsDC1.mec', 42458)

    @pytest.mark.parametrize('name', ['qmechs.mec', 'qmechdem102.mec',
                                      'qmechdemw.mec', 'qmechsDC1.mec'])
    def test_rates_using_undeclared_states(self, name):
        with pytest.raises(ValueError, match='state 6'):
            _load(name, 22844)

    @pytest.mark.parametrize('start', [20000, 21681])
    def test_a_file_that_stores_no_state_names(self, start):
        mec = _load('CKSIM3B.MEC', start)
        assert sorted(mec.state_names) == ['S1', 'S2', 'S3']
        assert _q_is_proper(mec)

    @pytest.mark.parametrize('name, start, k', [('ioana.mec', 20000, 16),
                                                ('qmechsDC1.mec', 113150, 19)])
    def test_an_independent_subunit_model(self, name, start, k):
        mec = _load(name, start)
        assert sorted(mec.state_names) == sorted('S%d' % n for n in range(1, k + 1))
        assert _q_is_proper(mec)

    def test_the_last_record_of_a_file_whose_records_are_packed_tight(self):
        """In demo.mec each record ends exactly where the next begins, with no
        room for the six unused trailing fields the reader used to read: it
        read 24 bytes of the next record for the first four, and ran off the
        end of the file on the fifth."""
        path = _path('demo.mec')
        mecs = [_load('demo.mec', start) for start in _records(path)]
        assert len(mecs) == 5
        assert all(mec.k == 3 and _q_is_proper(mec) for mec in mecs)

    def test_every_record_loads_or_says_why_not(self):
        """Across the whole collection, a record either loads or raises
        ValueError saying what is wrong with it - never IndexError, EOFError,
        or a hang."""
        if not os.path.isdir(MEC_DIR):
            pytest.skip('set SCALCS_MEC_DIR to a folder of .mec files')
        loaded, refused = 0, []
        for path in sorted(glob.glob(os.path.join(MEC_DIR, '*'))):
            if os.path.basename(path) == 'qHETGLY15.mec':
                continue
            for start in _records(path):
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        scalcsio.mec_load(path, start)
                    loaded += 1
                except ValueError as exc:
                    refused.append((os.path.basename(path), start, str(exc)))
        assert loaded >= 360
        assert len(refused) <= 6, refused
