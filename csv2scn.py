__author__="RLape"
__date__ ="$20-Aug-2014 13:55:03$"

import sys
import time
import struct
import numpy as np
import pandas as pd


def convert_clampfitCSV_to_scn(fname):
    """
    Convert Clampfit idealisation result saved as comma delimited .csv file.
    """
    record = np.genfromtxt(fname, skip_header=1, delimiter=',')  
    amplitudes = np.abs(record[:, 6] * record[:, 2] * 1000.0).astype(int)
    intervals = record[:, 8]
    flags = np.zeros((len(intervals)), dtype='b')
    to_filename = fname[:-3] + 'scn'
    scn_write(intervals, amplitudes, flags, calfac=0.001,
        filename=to_filename, type='Converted from Clampfit')
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

    if len(sys.argv) > 1:
        fname = sys.argv[1]
        print 'Converting file: ', fname
        fscname = convert_clampfitCSV_to_scn(fname)
        print 'Saved SCAN file: ', fscname
        print 'Done!'
    else:
        fname = "./dcpyps/samples/scn/single_channel_MK.xlsx"
        #data = pd.ExcelFile(fname)
        #record = data.parse(data.sheet_names[0], index_col=None, na_values=['NA'])
        

        

        print "No file was converted."
        print "Please, use this script this way: 'python csv2scn.py path_to_file.csv'"



