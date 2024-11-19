import math
import numpy as np
from numpy import linalg as nplin
import matplotlib.pyplot as plt
from matplotlib import cm

from samples import samples
from scalcs.pdfs import ExpPDF
from scalcs import qmatlib as qml
from scalcs.scalcslib import AsymptoticPDF, ExactPDFCalculator


class AdjacentPDF(AsymptoticPDF):
    """Calculates adjacent PDF components."""


    def __init__(self, mec, tres=0.0):
        super().__init__(mec, tres=tres)
        self.phiAr = self.phiA.reshape(1, self.kA)
        self.invQAA, self.invQFF = -nplin.inv(self.QAA), nplin.inv(self.QFF)
        self.Froots = -self.asymptotic_roots(open=False)
        self.Aroots = -self.asymptotic_roots(open=True)
        self.exact = ExactPDFCalculator(mec, tres)
        self.Feigs, self.FZ00, self.FZ10, self.FZ11 = self.exact.Zxx(open=False)
        self.Aeigs, self.AZ00, self.AZ10, self.AZ11 = self.exact.Zxx(open=True)


    def adjacent_open_to_shut_range_mean(self, u1, u2):
        """
        Calculate mean (ideal- no missed events) open times adjacent to a specified shut time range.

        Parameters
        ----------
        u1, u2 : floats
            Shut time range.
        """
        
        expQFFr = qml.expQ(self.QFF, u2) - qml.expQ(self.QFF, u1)
        col = self.QAF @ self.invQFF @ expQFFr @ self.QFA @ self.uA
        return (self.phiAr @ qml.powQ(self.invQAA, 2) @ col)[0, 0] / (self.phiAr @ self.invQAA @ col)[0, 0]

    def adjacent_open_to_shut_range_pdf_components(self, u1, u2):
        """
        Calculate time constants and areas for an ideal (no missed events)
        exponential probability density function of open times adjacent to a 
        specified shut time range.

        Parameters
        ----------
        u1, u2 : floats
            Shut time range.

        Returns
        -------
        eigs : ndarray, shape(k, 1)
            Eigenvalues.
        areas : ndarray, shape(k, 1)
            Component relative areas.
        """

        expQFFr = qml.expQ(self.QFF, u2) - qml.expQ(self.QFF, u1)
        col = self.QAF @ self.invQFF @ expQFFr @ self.QFA @ self.uA
        w = np.zeros(self.kA)
        eigs, A = qml.eigenvalues_and_spectral_matrices(-self.QAA)
        den = (self.phiAr @ self.invQAA @ col)[0, 0]
        #TODO: remove 'for'
        #w = np.array([(self.phiA @ A[i] @ col / den) for i in range(self.kA)])
        for i in range(self.kA):
            w[i] = (self.phiA @ A[i] @ col)[0] / den
        return eigs, w

    def HJC_adjacent_mean_open_to_shut_time_pdf(self, sht): #, tres): 
        """
        Calculate theoretical HJC (with missed events correction) mean open time
        given previous/next gap length (continuous function; CHS96 Eq.3.5). 

        Parameters
        ----------
        sht : array of floats
            Shut time interval.
        tres : float
            Time resolution.

        Returns
        -------
        mp : ndarray of floats
            Mean open time given previous gap length.
        mn : ndarray of floats
            Mean open time given next gap length.
        """
        
        FR = self.R(self.Froots, open=False)
        Q1 = self.dARSdS @ self.QAF @ self.expQFF
        col1 = Q1 @ self.uF
        row1 = self.HJCphiA @ Q1
        
        mp, mn = [], []
        for t in sht:
            eGFAt = qml.eGAF(t, self.tres, self.Feigs, self.FZ00, self.FZ10, self.FZ11, self.Froots,
                        FR, self.QFA, self.expQAA)
            denom = (self.HJCphiF @ eGFAt @ self.uA)[0]
            nom1 = (self.HJCphiF @ eGFAt @ col1)[0]
            nom2 = (row1 @ eGFAt @ self.uA)[0]
            mp.append(nom1 / denom)
            mn.append(nom2 / denom)
        
        return np.array(mp), np.array(mn)

    def HJC_dependency(self, top, tsh):
        """
        Calculate normalised joint distribution (CHS96, Eq. 3.22) of an open time
        and the following shut time as proposed by Magleby & Song 1992. 
        
        Parameters
        ----------
        top, tsh : array_like of floats
            Open and shut tims.
        tres : float
            Time resolution.

        Returns
        -------
        dependency : ndarray
        """
        
        FR = self.R(self.Froots, open=False)
        AR = self.R(self.Aroots, open=True)
        dependency = np.zeros((top.shape[0], tsh.shape[0]))
        
        for i in range(top.shape[0]):
            eGAFt = qml.eGAF(top[i], self.tres, self.Aeigs, self.AZ00, self.AZ10, self.AZ11, self.Aroots,
                    AR, self.QAF, self.expQFF)
            fo = (self.HJCphiA @ eGAFt @ self.uF)[0]
            
            for j in range(tsh.shape[0]):
                eGFAt = qml.eGAF(tsh[j], self.tres, self.Feigs, self.FZ00, self.FZ10, self.FZ11, self.Froots,
                    FR, self.QFA, self.expQAA)
                fs = (self.HJCphiF @ eGFAt @ self.uA)[0]
                fos = (self.HJCphiA @ eGAFt @ eGFAt @ self.uA)[0]
                dependency[i, j] = (fos - (fo * fs)) / (fo * fs)
        return dependency

    
class AdjacentPDFDisplay(AdjacentPDF):
    """ Prints adjacent PDF. """
    def __init__(self, mec, tres=0.0):
        super().__init__(mec, tres=tres)
        self.mec = mec

    def ideal_adjacent_dwells(self, t1, t2):
        
        adjacent_str = ('\nPDF of open times that precede shut times between {0:.3f} and {1:.3f} ms'.
                         format(t1 * 1000, t2 * 1000))
        e, a = self.adjacent_open_to_shut_range_pdf_components(t1, t2)
        adjacent_str += ExpPDF(1 / e, a / e).printout('\nOPEN TIMES ADJACENT TO SPECIFIED SHUT TIME RANGE')
        mean = self.adjacent_open_to_shut_range_mean(t1, t2)
        adjacent_str += ('Mean from direct calculation (ms) = {0:.6f}\n'.format(mean * 1000))
        return adjacent_str
    
    def calculate_adjacent_open_time_pdf(self, u1, u2, tmin=0.00001, tmax=1000, points=512):
        # Ideal pdf.
        eigs, w = self.ideal_open_time_pdf_components()
        tmax = (1 / eigs.max()) * 100
        t = np.logspace(math.log10(tmin), math.log10(tmax), points)
        fac = 1 / np.sum((w / eigs) * np.exp(-self.tres * eigs)) # Scale factor
        ipdf = t * ExpPDF(1 / eigs, w / eigs).calculate(t) * fac

        # Ajacent open time pdf
        eigs, w = self.adjacent_open_to_shut_range_pdf_components(u1, u2) 
    #    fac = 1 / np.sum((w / eigs) * np.exp(-tres * eigs)) # Scale factor
        apdf = t * ExpPDF(1 / eigs, w / eigs).calculate(t) * fac
        return t, ipdf, apdf


    def plot_adjacent_open_time_pdf(self, u1, u2, tmin=0.00001, tmax=1000, points=512):
        """Generate and display ideal pdf of all open times and ideal pdf of open times adjacent to specified shut
        time range.

        Parameters
        ----------
        tres : float
            Time resolution.
        u1, u2 : floats
            Shut time range.
        tmin, tmax : floats
            Time range for burst length ditribution.
        points : int
            Number of points per plot.
        """
                
        t, ipdf, apdf = self.calculate_adjacent_open_time_pdf(u1, u2, tmin=0.00001, tmax=1000, points=512)
        
        fig, ax = plt.subplots()
        ax.semilogx(t * 1000, ipdf, 'r--', label='Ideal open time PDF')
        ax.semilogx(t * 1000, apdf, 'b-', label='Adjacent open time PDF')

        # Apply square-root transformation to the y-axis
        #sqrt_transform = FuncTransform(lambda y: np.sqrt(y), lambda y: y**2)
        #ax.set_yscale('function', functions=(sqrt_transform.transform, sqrt_transform.inverted))

        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('PDF') # (sqrt scale)')
        ax.set_title('Adjacent opent time PDF')
        ax.legend()
        plt.show()

    def calculate_mean_open_next_to_shut(self):
        """
        Calculate plot of mean open time preceding/next-to shut time.

        Returns
        -------
        sht : ndarray of floats, shape (num of points,)
            Shut times.
        mp : ndarray of floats, shape (num of points,)
            Mean open time preceding shut time.
        mn : ndarray of floats, shape (num of points,)
            Mean open time next to shut time.
        """
        
        points=512
        tmax = (-1 / self.Froots.max()) * 5
        sht = np.logspace(math.log10(self.tres), math.log10(tmax), points)
        mp, mn = self.HJC_adjacent_mean_open_to_shut_time_pdf(sht)
        return sht, mp, mn

    def plot_mean_open_next_to_shut(self):
        sht, mp, mn = self.calculate_mean_open_next_to_shut()
        
        fig, ax = plt.subplots()
        ax.semilogx(sht * 1000, mp * 1000, 'r--', label='Mean open time preceding specified shut time')
        ax.semilogx(sht * 1000, mn * 1000, 'b--', label='Mean open time next to specified shut time')
        ax.set_xlabel('Shut time (ms)')
        ax.set_ylabel('Mean open time (ms)') # (sqrt scale)')
        ax.set_title('Mean open time adjacent to shut time')
        ax.legend()
        plt.show()

    def calculate_dependency(self, points=512):
        """
        Calculate 3D dependency plot.


        Returns
        -------
        top : ndarray of floats, shape (num of points,)
            Open times.
        tsh : ndarray of floats, shape (num of points,)
            Shut times.
        dependency : ndarray 
            Mean open time next to shut time.
        """
        
        tsmax = (-1 / self.Froots.max()) * 20
        tsh = np.logspace(math.log10(self.tres), math.log10(tsmax), points)
        tomax = (-1 / self.Aroots.max()) * 20
        top = np.logspace(math.log10(self.tres), math.log10(tomax), points)
        dependency = self.HJC_dependency(top, tsh)
        return np.log10(top*1000), np.log10(tsh*1000), dependency

    def plot_dependency(self):

        to, ts, d = self.calculate_dependency(points=128)
        fig = plt.figure()
        fig.suptitle('Dependency plot', fontsize=12)
        ax = fig.add_subplot(projection = '3d')
        to, ts = np.meshgrid(to, ts)
        ax.plot_surface(to, ts, d, rstride=1, cstride=1, cmap=cm.coolwarm,
            linewidth=0, antialiased=False)
        ax.set_zlim(-1.0, 1.0)
        plt.show()


if __name__ == '__main__':
    mec = samples.CH82()
    mec.set_eff('c', 0.0000001) 
    tres = 0.0001 # 10 us

    u1, u2 = 0.1e-3, 1e-3 # 1 ms, 10 ms
    pdf_adjacent = AdjacentPDFDisplay(mec, tres=tres)
    print(pdf_adjacent.ideal_adjacent_dwells(u1, u2))
    pdf_adjacent.plot_adjacent_open_time_pdf(u1, u2)
    pdf_adjacent.plot_mean_open_next_to_shut()
    pdf_adjacent.plot_dependency()

    
    
