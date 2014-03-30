import os, time, sys, socket
import numpy as np
from pylab import figure, semilogx, savefig, hist, xlim, ylim, scatter
from pylab import plot, text, title, xlabel, ylabel
from dcpyps import scplotlib as scpl

class Cluster(object):
    """
    
    """
    
    def __init__(self):
        self.setup()
        
    def setup(self):
        self.intervals = []
        self.amplitudes = []
        
    def add_interval(self, interval, amplitude):
        self.intervals.append(interval)
        self.amplitudes.append(amplitude)
        
    def get_open_intervals(self):
        return self.intervals[1::2]
    
    def get_shut_intervals(self):
        return self.intervals[0::2]
    
    def get_openings_number(self):
        return len(self.get_open_intervals())
    
    def get_openings_average_length(self):
        return np.average(self.get_open_intervals())
    
    def get_shuttings_average_length(self):
        return np.average(self.get_shut_intervals())
    
    def get_total_open_time(self):
        return np.sum(self.get_open_intervals())
    
    def get_length(self):
        return np.sum(self.intervals)
        
    def get_popen(self):
        return self.get_total_open_time() / np.sum(self.intervals)
    
    def get_running_mean_popen(self, N):
        
        if len(self.intervals)-1 > 2*N:    
            openings = self.get_open_intervals()
            shuttings = self.get_shut_intervals()
            meanP = []
            for i in range(len(openings) - N):
                meanP.append(np.sum(openings[i: i+N]) / 
                    (np.sum(openings[i: i+N]) + np.sum(shuttings[i: i+N])))
            return meanP
        else:
            return self.get_popen()
                
    def __str__(self):
        ret_str = ('Cluster length = {0:.3f} ms; '.
            format(self.get_length() * 1000) +
            'number of openings = {0:d}; '.format(self.get_openings_number()) +
            'Popen = {0:.3f}'.format(self.get_popen()))
        return ret_str
###########################

class ClusterReportHTML():
    """
    """
    def __init__(self, filename, clusters):
        self.clusters = clusters
        self.setup(filename)
        
    def setup(self, filename):
        
        
        path, fname = os.path.split(filename)
        self.fname = fname.split('.')[0]
        os.chdir(os.path.expanduser("~"))
        if not os.path.isdir('clusters'):
            os.mkdir('clusters')
        os.chdir('clusters')
        if not os.path.isdir(self.fname):
            os.mkdir(self.fname)
        os.chdir(self.fname)
        self.dir = os.getcwd()
        htmlfile = os.path.join(self.dir, self.fname + '.html')
        self.f = open(htmlfile, 'w')
        
        str1 = "DC_PyPs: Cluster Inspector.<br>"
        str2 = ("Date and time of analysis: %4d/%02d/%02d %02d:%02d:%02d<br>"
            %time.localtime()[0:6])
        machine = socket.gethostname()
        system = sys.platform
        str3 = "Machine: %s; System: %s<br>" %(machine, system)
        str = ('<html>\n' + str1 + str2 + str3)
        self.f.write(str)
        
        all_openings, all_shuttings = np.array([]), np.array([])
        popens, shav = np.array([]), np.array([])
        opnum, opav, optot = np.array([]), np.array([]), np.array([])
        for cluster in self.clusters:
            all_openings = np.append(all_openings, cluster.get_open_intervals())
            all_shuttings = np.append(all_shuttings, cluster.get_shut_intervals())
            popens = np.append(popens, cluster.get_popen())
            opnum = np.append(opnum, cluster.get_openings_number())
            opav = np.append(opav, cluster.get_openings_average_length())
            optot = np.append(optot, cluster.get_total_open_time())
            shav = np.append(shav, cluster.get_shuttings_average_length())
            
        self.f.write ('<br>Record in ' + filename + ' contains {0:d} clusters '.
            format(len(self.clusters)) + 'with average Popen = {0:.3f}'.
            format(np.average(popens)))
        
        figure(figsize=(6, 4))
        hist(popens, bins=20)
        xlim([0, 1])
        file_popen = os.path.join(self.dir, self.fname + '_popen.png')
        title(file_popen)
        xlabel('Popen')
        savefig(file_popen, bbox_inches=0)
        self.f.write("<img src={0} width='450' height='300'>".format(file_popen))

        self.insert_pdf(all_openings*1000, 'open', 'Apparent open time, ms')
        self.f.write('<br> <br>')
        self.insert_pdf(all_shuttings*1000, 'shut', 'Apparent shut time, ms')
        self.insert_scatter(opnum, popens, 'OPNUMvPOPENS', 
            xtitle='Number of openings per cluster', 
            ytitle='Popen')
        self.f.write('<br> <br>')
        self.insert_scatter(opav*1000, popens, 'OPAVvPOPENS', 
            xtitle='Average opening length, ms', 
            ytitle='Popen')
        self.insert_scatter(optot*1000, popens, 'OPTOTvPOPENS', 
            xtitle='Total open time per cluster, ms', 
            ytitle='Popen')
        self.f.write('<br> <br>')
        self.insert_scatter(shav*1000, popens, 'SHAVvPOPENS', 
            xtitle='Average shutting length, ms', 
            ytitle='Popen')
        self.insert_scatter(opav*1000, shav*1000, 'OPAVvSHAV', 
            xtitle='Average opening length, ms', 
            ytitle='Average shutting length, ms')
        self.f.write('<br> <br>')
        
        N = 10 # number of openings
        for i in range(len(self.clusters)):
            prm = self.clusters[i].get_running_mean_popen(N)
            type = 'Cluster_{0:d}'.format(i+1)
            self.insert_stability_plot(prm, type, N, xtitle='Group number', 
                ytitle='Mean Popen')
        
        self.finalise()
        
    def insert_stability_plot(self, X, type, N, xtitle='', ytitle=''):
        figure(figsize=(6, 4))
        plot(X)
        if ytitle == 'Mean Popen': ylim([0, 1])
        filepath = os.path.join(self.dir, self.fname + '_' + type + '.png')
        title(filepath)
        text(0.5, 0.9,'Number of intervals in group = {0:d}'.format(2*N))
        xlabel(xtitle)
        ylabel(ytitle)
        savefig(filepath, bbox_inches=0)
        self.f.write("<img src={0} width='450' height='300'>".format(filepath))
            
    def insert_scatter(self, X, Y, type, xtitle='', ytitle=''):
        
        figure(figsize=(6, 4))
        scatter(X, Y)
        filepath = os.path.join(self.dir, self.fname + '_' + type + '.png')
        title(filepath)
        xlabel(xtitle)
        ylabel(ytitle)
        savefig(filepath, bbox_inches=0)
        self.f.write("<img src={0} width='450' height='300'>".format(filepath))
            
    def insert_pdf(self, X, type, xtitle=''):
        figure(figsize=(6, 4))
        x1, y1, dx = scpl.prepare_hist(X, 0.01)
        semilogx(x1, y1, 'k-')
        filepath = os.path.join(self.dir, self.fname + '_' + type + '.png')
        title(filepath)
        xlabel(xtitle)
        savefig(filepath, bbox_inches=0)
        self.f.write("<img src={0} width='450' height='300'>".format(filepath))

        
    def finalise(self):
        self.f.write('</html>')
        self.f.close()
        
########################

if __name__ == "__main__":

    fname = "./dcpyps/samples/scn/Cevents.csv"
    record = np.genfromtxt(fname, skip_header=1, delimiter=',')

    clusters = []
    cluster = Cluster()
    for i in range(len(record)):
        if not np.isnan(record[i, 0]) and i < len(record)-1:
            cluster.add_interval(record[i, 8] / 1000.0, record[i, 6] / 1000.0)
        else:
            print cluster
            clusters.append(cluster)
            cluster = Cluster()

    ClusterReportHTML(fname, clusters)