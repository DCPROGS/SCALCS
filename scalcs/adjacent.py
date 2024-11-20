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
        Calculate mean open times adjacent to a specified shut time range.
        Parameters:
            u1, u2 : float
                Shut time range.
        Returns:
            float: Mean open time.
        """
        
        expQFFr = qml.expQ(self.QFF, u2) - qml.expQ(self.QFF, u1)
        col = self.QAF @ self.invQFF @ expQFFr @ self.QFA @ self.uA
        num = (self.phiAr @ qml.powQ(self.invQAA, 2) @ col)[0, 0]
        denom = (self.phiAr @ self.invQAA @ col)[0, 0]
        return num / denom

    def adjacent_open_to_shut_range_pdf_components(self, u1, u2):
        """
        Calculate eigenvalues and areas for an ideal exponential PDF open times adjacent to a 
        specified shut time range.
        Parameters:
            u1, u2 : float
                Shut time range.
        Returns:
            eigs : ndarray
                Eigenvalues.
            areas : ndarray
                Component relative areas.
        """

        expQFFr = qml.expQ(self.QFF, u2) - qml.expQ(self.QFF, u1)
        col = self.QAF @ self.invQFF @ expQFFr @ self.QFA @ self.uA
        eigs, A = qml.eigenvalues_and_spectral_matrices(-self.QAA)
        denom = (self.phiAr @ self.invQAA @ col)[0, 0]
        w = np.array([(self.phiA @ A[i] @ col)[0] / denom for i in range(self.kA)])
        return eigs, w

    def HJC_adjacent_mean_open_to_shut_time_pdf(self, sht): #, tres): 
        """
        Calculate theoretical HJC (with missed events correction) mean open time
        given previous/next gap length (continuous function; CHS96 Eq.3.5). 
        Parameters:
            sht : ndarray
                Shut time intervals.
        Returns:
            mp : ndarray
                Mean open time given the previous gap length.
            mn : ndarray
                Mean open time given the next gap length.
        """
        
        FR = self.R(self.Froots, open=False)
        Q1 = self.dARSdS @ self.QAF @ self.expQFF
        col1 = Q1 @ self.uF
        row1 = self.HJCphiA @ Q1
        
        mp, mn = [], []
        for t in sht:
            eGFAt = qml.eGAF(
                t, self.tres, self.Feigs, self.FZ00, self.FZ10, self.FZ11,
                self.Froots, FR, self.QFA, self.expQAA
            )
            denom = (self.HJCphiF @ eGFAt @ self.uA)[0]
            mp.append((self.HJCphiF @ eGFAt @ col1)[0] / denom)
            mn.append((row1 @ eGFAt @ self.uA)[0] / denom)

        return np.array(mp), np.array(mn)

    def HJC_dependency(self, top, tsh):
        """
        Calculate normalised joint distribution (CHS96, Eq. 3.22) of an open time
        and the following shut time as proposed by Magleby & Song 1992. 
        Parameters:
            top, tsh : ndarray
                Open and shut times.
        Returns:
            ndarray: Dependency values.
        """
        
        FR = self.R(self.Froots, open=False)
        AR = self.R(self.Aroots, open=True)
        dependency = np.zeros((len(top), len(tsh)))
        
        for i, t_open in enumerate(top):
            eGAFt = qml.eGAF(
                t_open, self.tres, self.Aeigs, self.AZ00, self.AZ10, self.AZ11,
                self.Aroots, AR, self.QAF, self.expQFF
            )
            fo = (self.HJCphiA @ eGAFt @ self.uF)[0]

            for j, t_shut in enumerate(tsh):
                eGFAt = qml.eGAF(
                    t_shut, self.tres, self.Feigs, self.FZ00, self.FZ10, self.FZ11,
                    self.Froots, FR, self.QFA, self.expQAA
                )
                fs = (self.HJCphiF @ eGFAt @ self.uA)[0]
                fos = (self.HJCphiA @ eGAFt @ eGFAt @ self.uA)[0]
                dependency[i, j] = (fos - (fo * fs)) / (fo * fs)

        return dependency

    
class AdjacentPDFDisplay(AdjacentPDF):
    """Displays and visualizes adjacent PDF components."""
    def __init__(self, mec, tres=0.0):
        super().__init__(mec, tres=tres)

    def ideal_adjacent_dwells(self, t1, t2):
        """
        Print PDF of open times that precede shut times within a range.
        Parameters:
            t1, t2 : float
                Shut time range (in seconds).
        Returns:
            str: Formatted output.
        """
        eigs, areas = self.adjacent_open_to_shut_range_pdf_components(t1, t2)
        mean = self.adjacent_open_to_shut_range_mean(t1, t2)
        result = [
            f"\nPDF of open times preceding shut times between {t1*1000:.3f} and {t2*1000:.3f} ms",
            ExpPDF(1 / eigs, areas / eigs).printout(
                "\nOPEN TIMES ADJACENT TO SPECIFIED SHUT TIME RANGE"
            ),
            f"Mean from direct calculation (ms): {mean * 1000:.6f}\n",
        ]
        return "\n".join(result)
    
    def calculate_adjacent_open_time_pdf(self, u1, u2, tmin=0.00001, tmax=1000, points=512):
        """
        Calculate ideal and adjacent open time PDFs.
        Parameters:
            u1, u2 : float
                Shut time range (in seconds).
            tmin, tmax : float
                Time range for the PDF.
            points : int
                Number of time points for calculation.
        Returns:
            t : ndarray
                Time points.
            ipdf, apdf : ndarray
                Ideal and adjacent open time PDFs.
        """
        eigs, weights = self.ideal_open_time_pdf_components()
        tmax = min(tmax, (1 / eigs.max()) * 100)
        t = np.logspace(math.log10(tmin), math.log10(tmax), points)
        fac = 1 / np.sum((weights / eigs) * np.exp(-self.tres * eigs))
        ipdf = t * ExpPDF(1 / eigs, weights / eigs).calculate(t) * fac

        eigs, weights = self.adjacent_open_to_shut_range_pdf_components(u1, u2)
        apdf = t * ExpPDF(1 / eigs, weights / eigs).calculate(t) * fac
        return t, ipdf, apdf

    #TODO: Move this function to pdfs.py module ExpPDF class
    def plot_pdf(self, t, ipdf, apdf, xlabel="Time (ms)", ylabel="PDF", title="Generic PDF"):
        """
        Plot PDF of ideal and adjacent open times.
        Parameters:
            t : ndarray
                Time points (in seconds).
            ipdf, apdf : ndarray
                Ideal and adjacent open time PDFs.
            xlabel, ylabel, title : str
                Labels and title for the plot.
        """
        fig, ax = plt.subplots()
        ax.semilogx(t * 1000, ipdf, "r--", label="Ideal open time PDF")
        ax.semilogx(t * 1000, apdf, "b-", label="Adjacent open time PDF")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        plt.show()

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
                
        t, ipdf, apdf = pdf_adjacent.calculate_adjacent_open_time_pdf(u1, u2)
        self.plot_pdf(t, ipdf, apdf, title="Adjacent Open Time PDF")

    def calculate_mean_open_next_to_shut(self, points=512):
        """
        Calculate mean open times preceding and next to a shut time.
        Parameters:
            points : int
                Number of points for calculation.
        Returns:
            sht : ndarray
                Shut times.
            mp, mn : ndarray
                Mean open times preceding and next to shut times.
        """
        
        tmax = (-1 / self.Froots.max()) * 5
        sht = np.logspace(math.log10(self.tres), math.log10(tmax), points)
        mp, mn = self.HJC_adjacent_mean_open_to_shut_time_pdf(sht)
        return sht, mp, mn

    def plot_mean_open_next_to_shut(self, points=512):
        """
        Plot mean open times preceding and next to a shut time.
        Parameters:
            points : int
                Number of time points for calculation.
        """

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
        Calculate dependency between open and shut times.
        Parameters:
            points : int
                Number of points for calculation.
        Returns:
            to, ts : ndarray
                Log-scaled open and shut times.
            dependency : ndarray
                Dependency matrix.
        """
        
        tsmax = (-1 / self.Froots.max()) * 20
        tomax = (-1 / self.Aroots.max()) * 20
        tsh = np.logspace(math.log10(self.tres), math.log10(tsmax), points)
        top = np.logspace(math.log10(self.tres), math.log10(tomax), points)
        dependency = self.HJC_dependency(top, tsh)
        return np.log10(top * 1000), np.log10(tsh * 1000), dependency

    def plot_dependency(self, points=128):
        """
        Plot 3D dependency of open and shut times.
        Parameters:
            points : int
                Number of points for each dimension.
        """

        to, ts, d = self.calculate_dependency(points)
        fig = plt.figure()
        fig.suptitle("Dependency Plot", fontsize=12)
        ax = fig.add_subplot(projection="3d")
        to, ts = np.meshgrid(to, ts)
        ax.plot_surface(to, ts, d, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax.set_zlim(-1.0, 1.0)
        ax.set_xlabel("Log10(Open Time (ms))")
        ax.set_ylabel("Log10(Shut Time (ms))")
        ax.set_zlabel("Dependency")
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

    
    
