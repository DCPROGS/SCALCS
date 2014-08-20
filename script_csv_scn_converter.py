__author__="RLape"
__date__ ="$20-Aug-2014 13:55:03$"

import numpy as np

def convert_clampfit_to_scn(fname):
    """
    Convert
    """
    record = np.genfromtxt(fname, skip_header=1, delimiter=',')
    for i in range(len(record)):
        if np.isnan(record[i, 0]):
            record[i, 2] = 0
            record[i, 8] = record[i+1, 4] - record[i-1, 5]
    intervals = record[:, 8]
    amplitudes = record[:, 2].astype(int)
    flags = np.zeros((len(intervals)), dtype='b')
    to_filename = fname[:-3] + 'scn'
    scn_write(intervals, amplitudes, flags,
        filename=to_filename, type='Converted Clampfit ideal')
    return to_filename

def scn_write(intervals, amplitudes, flags, calfac=1.0, ffilt=-1.0, rms=0.0,
        treso=0.0, tresg=0.0, Emem=0.0,
        filename='new_saved.SCN', type='simulated'):
    """
    Write binary SCAN (DCprogs: http://www.ucl.ac.uk/Pharmacology/dcpr95.html)
    format file.

    Parameters
    ----------
    """

    # Preapare header.
    iscanver, ioffset = -103, 154
    nint, avamp = len(intervals), np.average(amplitudes)
    title = '{0: <70}'.format(type) # char*70
    t = time.asctime()
    expdate = t[8:10] + '-' + t[4:7] + '-' + t[20:24] # '00-ooo-0000' char*11
    tapeID = '{0: <24}'.format(type) # char*24
    ipatch = 0 # integer32

    # Write header.
    fout = open(filename, 'wb')
    fout.write(struct.pack('iii', iscanver, ioffset, nint))
    fout.write(title + expdate + tapeID)
    fout.write(struct.pack('ififff', ipatch, Emem, 0, avamp, rms, ffilt))
    fout.write(struct.pack('fff', calfac, treso, tresg))

    # Write data block.
    fout.write(struct.pack('f'*nint, *intervals))
    fout.write(struct.pack('h'*nint, *amplitudes))
    fout.write(struct.pack('b'*nint, *flags))
    fout.close()

if __name__ == "__main__":
    print 'Converting file: ', fname
    fscname = convert_clampfit_to_scn(fname)
    print 'Done!'



