import os
import glob
import numpy as np
from pylab import figure, savefig, hist, scatter, xlim, plot
from pylab import title, xlabel, ylabel

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

    path = "C:/clusters/03mM"
    min_op = 10
    tres = 0.0003
    tcrit = 0.1
    run_win_av = 5


    os.chdir(path)
    print 'Checking directory', path
    filelist = glob.glob('*.csv')
    print 'Found {0:d} files:'.format(len(filelist))
    print filelist
    all_popen = []
    opav = []

    for fname in filelist:
        print '\n\nAnalysing file:', fname
        fscname = convert_clampfit_to_scn(fname)
        rec = dataset.SCRecord([fscname]) #, tres=0.0003, tcrit=0.1)
        rec.tres = tres
        rec.tcrit = tcrit
        for cluster in rec.bursts.all():
            print cluster
        ClusterReportHTML(fname, rec, runwin=run_win_av, minop=min_op)
        
        clusters = rec.bursts.get_long(min_op)
        all_popen.extend(clusters.get_popen_list())
        opav.extend(clusters.get_opening_length_mean_list())

        os.chdir(path)


    figure(figsize=(6, 4))
    hist(all_popen, bins=20)
    xlim([0, 1])
    file_popen = os.path.join(path + '/all_popen.png')
    title(file_popen)
    xlabel('Popen')
    savefig(file_popen, bbox_inches=0)

    ap = np.array(all_popen)
    print ('\n\n Average Popen for clusters of Popen>0.5: {0:.3f}'.
        format(np.average(ap[np.where(ap>0.5)])))


    p = np.linspace(0.0, 1.0, num=250, endpoint=False)
    o = p * tres / (1 - p)
    figure(figsize=(6, 4))
    scatter(np.array(all_popen), np.array(opav)*1000)
    plot(p, o*1000, 'r-')
    filepath = os.path.join(path + '/OPAVvPOPENS.png')
    title(filepath)
    ylabel('Average opening length, ms')
    xlabel('Popen')
    xlim([0, 1])
    savefig(filepath, bbox_inches=0)

#    fname = "C:/clusters/2013_11_05_0001.csv"
#    fscname = convert_clampfit_to_scn(fname)
#    tcrit = 0.1 # 100 ms
#    tres = 0.0003 # 300 microsec
#    minop = 50 # minimal number of openings
#    rec = dataset.SCRecord([fscname], tres=0.0003, tcrit=0.1)
#    rec.tcrit = 0.1
#    for cluster in rec.bursts.all():
#        print cluster
#    ClusterReportHTML(fname, rec, runwin=5)

