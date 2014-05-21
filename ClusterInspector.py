import numpy as np
from dcpyps import dcio
from dcpyps import dataset
from dcpyps.reports import ClusterReportHTML

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
    dcio.scn_write(intervals, amplitudes, flags,
        filename=to_filename, type='Converted Clampfit ideal')
    return to_filename
        
if __name__ == "__main__":

    fname = "C:/clusters/2013_11_05_0001.csv"
    fscname = convert_clampfit_to_scn(fname)

    tcrit = 0.1 # 100 ms
    tres = 0.0003 # 300 microsec
    minop = 50 # minimal number of openings

    rec = dataset.SCRecord([fscname], tres=0.0003, tcrit=0.1)
    rec.tcrit = 0.1
    for cluster in rec.bursts.all():
        print cluster

    ClusterReportHTML(fname, rec, runwin=20)
    