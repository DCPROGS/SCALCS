import numpy as np
from numpy import linalg as nplin
from tabulate import tabulate

from samples import samples
from scalcs.pdfs import ExpPDF
from scalcs import qmatlib as qml
from scalcs.qmatlib import HJCMatrix


class AdjacentPDF(HJCMatrix):
    """ Calculates adjecent pdf components. """

    def __init__(self, mec, tres=0.0):
        super().__init__(mec.Q, kA=mec.kA, kB=mec.kB, kC=mec.kC, kD=mec.kD, tres=tres)
        self.uA = np.ones((self.kA))[:,np.newaxis]
        self.uF = np.ones((self.kF))[:,np.newaxis]
        self.phiAr = self.phiA.reshape(1, self.kA)
        self.invQAA, self.invQFF = -nplin.inv(self.QAA), nplin.inv(self.QFF)

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

    #TODO: recview next two functions
    def HJC_adjacent_mean_open_to_shut_time_pdf(self, shut_times):
        """
        Calculate theoretical HJC (with missed events correction) mean open time
        given previous/next gap length (continuous function; CHS96 Eq.3.5). 

        Parameters
        ----------
        shut_times : array of floats
            Shut time interval.

        Returns
        -------
        mp : ndarray of floats
            Mean open time given previous gap length.
        mn : ndarray of floats
            Mean open time given next gap length.
        """
        
        DARS = self.dARSdS()
        eigs, A = qml.eigenvalues_and_spectral_matrices(-self.Q)
        FZ00, FZ10, FZ11 = self.Zxx(open=False)
        Froots = asymptotic_roots(self.tres, open=False)
        FR = self.R(Froots, open=False)
        Q1 = DARS @ self.QAF @ self.expQFF
        col1 = Q1 @ self.uF
        row1 = self.HJCphiA @ Q1
        
        mp, mn = [], []
        for t in shut_times:
            eGFAt = qml.eGAF(t, self.tres, eigs, FZ00, FZ10, FZ11, Froots, FR, self.QFA, self.expQAA)
            denom = (self.HJCphiF @ eGFAt @ self.uA)[0]
            mp.append((self.HJCphiF @ eGFAt @ col1)[0] / denom)
            mn.append((row1 @ eGFAt @ self.uA)[0] / denom)
        return np.array(mp), np.array(mn)

    def HJC_dependency(self, top, tsh):
        """
        Calculate normalised joint distribution (CHS96, Eq. 3.22) of an open time
        and the following shut time as proposed by Magleby & Song 1992. 
        
        Parameters
        ----------
        top, tsh : array_like of floats
            Open and shut tims.

        Returns
        -------
        dependency : ndarray
        """
        
        eigs, A = qml.eigenvalues_and_spectral_matrices(-self.Q)
        FZ00, FZ10, FZ11 = self.Zxx(open=False)
        Froots = self.asymptotic_roots(open=False)
        FR = self.R(Froots, open=False) 
        AZ00, AZ10, AZ11 = self.Zxx(open=True)
        Aroots = self.asymptotic_roots(open=True)
        AR = self.R(Aroots, open=True)

        dependency = np.zeros((top.shape[0], tsh.shape[0]))
        for i in range(top.shape[0]):
            eGAFt = qml.eGAF(top[i], self.tres, eigs, AZ00, AZ10, AZ11, Aroots, AR, self.QAF, self.expQFF)
            fo = (self.HJCphiA @ eGAFt @ self.uF)[0]
            for j in range(tsh.shape[0]):
                eGFAt = qml.eGAF(tsh[j], self.tres, eigs, FZ00, FZ10, FZ11, Froots, FR, self.QFA, self.expQAA)
                fs = (self.HJCphiF @ eGFAt @ self.uA)[0]
                fos = (self.HJCphiA @ eGAFt @ eGFAt @ self.uA)[0]
                dependency[i, j] = (fos - (fo * fs)) / (fo * fs)
        return dependency
    
class AdjacentPDFPrints(AdjacentPDF):
    """ Prints adjacent PDF. """
    def __init__(self, mec, tres=0.0):
        super().__init__(mec, tres=tres)

    def ideal_adjacent_dwells(self, t1, t2):
        
        adjacent_str = ('\nPDF of open times that precede shut times between {0:.3f} and {1:.3f} ms'.
                         format(t1 * 1000, t2 * 1000))
        e, a = self.adjacent_open_to_shut_range_pdf_components(t1, t2)
        adjacent_str += ExpPDF(1 / e, a / e).printout('\nOPEN TIMES ADJACENT TO SPECIFIED SHUT TIME RANGE')
        #adjacent_str += expPDF_printout(e, a, 'OPEN TIMES ADJACENT TO SPECIFIED SHUT TIME RANGE')
        mean = self.adjacent_open_to_shut_range_mean(t1, t2) #     mec.QAA, mec.QAF, mec.QFF, mec.QFA, phiA)
        adjacent_str += ('Mean from direct calculation (ms) = {0:.6f}\n'.format(mean * 1000))
        return adjacent_str


if __name__ == '__main__':
    mec = samples.CH82()
    mec.set_eff('c', 0.0000001) 
    tres = 0.0001 # 10 us

    q_adjacent = AdjacentPDFPrints(mec, tres=tres)
    print(q_adjacent.ideal_adjacent_dwells(0.0001, 0.001))
