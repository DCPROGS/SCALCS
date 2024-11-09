import math
import numpy as np
import matplotlib.pyplot as plt

from samples import samples
from scalcs import qmatlib as qml
from scalcs import scalcslib as scl

class PopenCalculator(scl.AsymptoticPDF):
    """Calculates Popen for given mechanism."""

    def __init__(self, mec, tres=0.0):
        super().__init__(mec, tres=tres)
        self.mec = mec

    def _update_concentration(self, concentration):
        """
        Update the mechanism parameters.
        """
        self.mec.set_eff('c', concentration)
        super().__init__(self.mec, tres=self.tres)
        
    def popen_ideal(self, conc=1e-12):
        """
        Calculate Popen for a given concentration, adjusting for unresolved
        blockages if fast pore blocker is present.
        """
        self.mec.set_eff('c', conc)
        p = qml.pinf(self.mec.QGG)
        popen = np.sum(p[ : self.mec.kA]) / np.sum(p)
        #popen = scl.ideal_popen(self.mec) # if self.tres == 0 else scl.exact_popen(self.mec, self.tres)
        #return popen / (1 + conc / self.mec.fastKB) if self.mec.fastblock else popen
        return popen

    def popen_HJC(self, conc=1e-12, tres=1e-5):
        """
        Calculate HJC corrected Popen for a given concentration.
        """
        self.tres = tres
        self._update_concentration(conc)
        return self.apparentPopen


class PopenCurve:
    """
    A container class for calculating equilibrium open probability (Popen)
    and the dose-response curve parameters (EC50, Hill slope) for a given
    mechanism and concentration.
    """
    def __init__(self, mec=None, tres=0.0):
        """
        Initialize the PopenAnalysis class with mechanism and parameters.

        Parameters:
        mec : Mechanism
            The mechanism to be analyzed.
        """
        self.mec = mec
        self.calculator = PopenCalculator(mec, tres)

#        if mec is not None:
#            self._calculate_curve_parameters()

    def calculate_popen(self, c, tres=None):
        if tres is None:
            return self.calculator.popen_ideal(c)
        else:
            return self.calculator.popen_HJC(c, tres)

    def plot_popen_curve(self, cmin=1e-12, cmax=0.1, tres=None):
        """
        Plot the Popen curves for a given mechanism and time resolution.
                
        This function plots the equilibrium open probability (Popen) curves,
        including both the apparent (corrected for missed events) and ideal curves.
        """
        
        # Calculate Popen curve data
        crange = np.logspace(math.log10(cmin), math.log10(cmax), 1000)
        # Convert concentration to µM
        crange_um = crange * 1e6

        curve_ideal = np.array([self.calculate_popen(c) for c in crange])
        plt.semilogx(crange_um, curve_ideal, 'r--', label='Ideal')
        if tres is not None:
            curve_HJC = np.array([self.calculate_popen(c, tres) for c in crange])
            plt.semilogx(crange_um, curve_HJC, 'b-', label='Apparent (HJC)')
        
        # Label plot elements
        plt.ylabel('Popen')
        plt.xlabel('Concentration (µM)')
        plt.title('Popen curve')
        plt.legend()
        plt.grid(True)
        plt.show()


    def _get_maxPopen(self, curve, tolerance=1e-12, half_max_tolerance=0.05):
        """
        Analyze a curve to check for monotonicity, detect peak (if not monotonic), overall trend,
        and find half-maximal value.

        Parameters:
        arr (numpy.ndarray): Array containing curve data.
        tolerance (float): Tolerance to determine when the difference between points is negligible.
        half_max_tolerance (float): Tolerance for identifying the half-maximal point as a fraction of the peak value.

        Returns:
        result (dict): Dictionary containing curve properties:
            - "is_monotonic": Whether the curve is monotonic.
            - "peak_value": The peak value if there is a peak; otherwise, the max or min value.
            - "peak_index": The index of the peak.
            - "trend": 'ascending' or 'descending' based on overall direction.
            - "half_max_value": The value closest to half of the peak within tolerance.
            - "half_max_index": The index of the half-maximal value.
        """
        # Initial conditions
        is_monotonic = True
        peak_value = curve[0]
        peak_index = 0
        trend = "none"

        # Detect if array is generally ascending or descending
        initial_diff = curve[-1] - curve[0]
        trend = "ascending" if initial_diff > 0 else "descending"
        
        # Iterate through the array to check monotonicity and detect peak
        for i in range(1, len(curve)):
            diff = curve[i] - curve[i - 1]
            
            if np.any(abs(diff) < tolerance):  # Skip negligible differences
                continue
            
            if diff > 0:
                # Update peak for ascending part
                if curve[i] > peak_value:
                    peak_value = curve[i]
                    peak_index = i
                if trend == "descending":  # Not monotonic if it switches trend
                    is_monotonic = False
            elif diff < 0:
                # Update peak for descending part
                if curve[i] < peak_value:
                    peak_value = curve[i]
                    peak_index = i
                if trend == "ascending":  # Not monotonic if it switches trend
                    is_monotonic = False
        
        # If monotonic, get max or min value based on trend
        if is_monotonic:
            peak_value = curve[-1] if trend == "ascending" else curve[0]
            peak_index = len(curve) - 1 if trend == "ascending" else 0


        return {
            "is_monotonic": is_monotonic,
            "maxPopen": peak_value,
            "peak_index": peak_index,
            "trend": trend}

    def _get_hill_slope(self, popen_values, ec50, tres=None):
        """
        Calculates the Hill slope of the Popen curve around the EC50.

        Parameters
        ----------
        concentrations : ndarray
            Array of concentrations.
        popen_values : ndarray
            Array of Popen values corresponding to each concentration.
        ec50 : float
            The EC50 concentration value.
        tolerance : float
            The range around EC50 to use for slope calculation, as a fraction of EC50 (e.g., 0.05 for ±5%).

        Returns
        -------
        hill_slope : float
            Calculated Hill slope around EC50.
        """

        P0 = min(popen_values)
        Pmax = max(popen_values)
        n = 64
        dc = (math.log10(ec50 * 1.1) - math.log10(ec50 * 0.9)) / (n - 1)
        c = np.zeros(n)
        y = np.zeros(n)
        for i in range(n):
            c[i] = (ec50 * 0.9) * pow(10, i * dc)
            y[i] = self.calculate_popen(c[i], tres)

        # Find two points around EC50.
        i50 = 0
        s1, s2 = 0, 0
        i = 0
        while i50 ==0 and i < n-1:
            if (c[i] <= ec50) and (c[i+1] >= ec50):
                i50 = i
                y1 = math.log10(math.fabs((y[i] - P0) / (Pmax - y[i])))
                y2 = math.log10(math.fabs((y[i+1] - P0) / (Pmax - y[i+1])))
                s1 = (y2 - y1) / (math.log10(c[i+1]) - math.log10(c[i]))
                y3 = math.log10(math.fabs((y[i+1] - P0) / (Pmax - y[i+1])))
                y4 = math.log10(math.fabs((y[i+2] - P0) / (Pmax - y[i+2])))
                s2 = (y4 - y3) / (math.log10(c[i+2]) - math.log10(c[i+1]))
            i += 1

        # Interpolate linearly for Hill slope at EC50
        b = (s2 - s1) / (c[i50+1] - c[i50])
        nH = s1 + b * (ec50 - c[i50])
        return nH   
    
    def _get_EC50(self, tres=None):
        """
        Estimate numerically the equilibrium EC50 for a specified mechanism.
        If monotonic this is unambiguous. If not monotonic then returned is
        a concentration for 50% of  the peak response to the left of the peak.

        Parameters
        ----------
        tres : float
            Time resolution (dead time).

        Returns
        -------
        EC50 : float
            Concentration at which Popen is 50% of its maximal value.
        """

        c1 = 0
        c2 = self.Cmax
        epsy = 0.001    # accuracy in Popen
        perr = 2 * epsy
        epsc = 0.1e-9    # accuracy in concentration 0.1 nM
        nstepmax = int(math.log10(math.fabs(c1 - c2) / epsc) / math.log10(2) + 0.5)
        nstep = 0
        while math.fabs(perr) > epsy and nstep <= nstepmax:
            conc = (c1 + c2) / 2
            perr = math.fabs((self.calculate_popen(conc, tres) - self.P0) / (self.Pmax - self.P0)) - 0.5
            if perr < 0:
                c1 = conc
            elif perr > 0:
                c2 = conc
            nstep += 1
        return conc

    def analyse_curve(self, tres=None):
        """ """
        start = int(math.log10(0.1e-12))
        end = int(math.log10(0.1))
        crange = np.logspace(start, end, (end-start)*100)
        curve = np.array([self.calculate_popen(c, tres) for c in crange])

        self.P0 = min(curve)
        self.Pmax = max(curve)

        result = self._get_maxPopen(curve)
        self.Cmax = crange[result['peak_index']]

        result['EC50'] = self._get_EC50(tres)
        nH = self._get_hill_slope(curve, result['EC50'])
        result['nH'] = nH
        return result
        
    def printout(self, tres=None):
        """
        """
        out = ('\n******************' )
        out += ('\nIdeal Popen curve:')
        result1 = self.analyse_curve()
        out += self.print_pars(result1)
        if tres is not None:
            out += ('\n\n****************')
            out += ('\nHJC Popen curve:')
            result2 = self.analyse_curve(tres)
            out += self.print_pars(result2)

#        if mec.fastblock:
#            out += ('\nThis Popen curve was corrected for fast block ' + 
#                'with KB = {0:.5g} mM.'.format(mec.fastKB * 1000))
        #out += ('\nHJC Popen curve:\n' + self.print_pars())
        
        return out

    def print_pars(self, result):
        sections = [
            (f"\nIs Monotonic: {result['is_monotonic']}"),
            (f"\nTrend: {result['trend']}"),
            (f"\nmaximal Popen = {result['maxPopen']}"),
            (f"\nEC50 = {result['EC50'] * 1e6} uM"),
            (f"\nHill slope (nH) = {result['nH']}")]
        return ''.join(sections)

    ############################   REVIEW   ##################################


def Popen(mec, tres=0.0, conc=0.0, eff='c'):
    """
    Calculate equilibrium open probability (Popen) and correct for
    unresolved blockages in case of presence of fast pore blocker.

    Parameters
    ----------
    mec : dcpyps.Mechanism
        The mechanism to be analysed.
    tres : float
        Time resolution (dead time).
    conc : float
        Concentration.

    Returns
    -------
    Popen : float
        Open probability value at a given concentration.
    """
    mec.set_eff(eff, conc)
    #q_dwells = SCDwells(mec.Q, mec.kA, mec.kB, mec.kC, mec.kD, tres=tres)
    #q_dwells.tres = tres
    if tres == 0:
        p = qml.pinf(mec.QGG)
        popen = np.sum(p[:mec.kA]) / np.sum(p)
        #popen = q_dwells.Popen()
    else:
        #popen = q_dwells.apparent_mean_open_time / (q_dwells.apparent_mean_open_time + q_dwells.apparent_mean_shut_time)
        hmopen, hmshut = scl.exact_mean_open_shut_time(mec, tres)
        popen = (hmopen / (hmopen + hmshut))
    if mec.fastblock:
        popen = popen / (1 + conc / mec.fastKB)
    return popen

def calculate_Popen_plot(mec, tres):
    """
    Calculate Popen curve parameters and data for Popen curve plot.

    Parameters
    ----------
    mec : instance of type Mechanism
    tres : float
        Time resolution (dead time).

    Returns
    -------
    c : ndarray of floats, shape (num of points,)
        Concentration in Moles.
    pe : ndarray of floats, shape (num of points,)
        Open probability corrected for missed events.
    pi : ndarray of floats, shape (num of points,)
        Ideal open probability.
    """

    iEC50 = EC50(mec, 0)
    eEC50 = EC50(mec, tres)
    pmax, cx = maxPopen(mec, 0)
    nH = Hill_slope(mec, 0)

    # Plot ideal and corrected Popen curves.
    cmin = iEC50 / 20
    cmax = iEC50 * 500
    log_start = int(np.log10(cmin)) - 1
    log_end = int(np.log10(cmax)) - 1
    points = 512

    c = np.logspace(log_start, log_end, points)
    pe = np.zeros(points)
    pi = np.zeros(points)
    H = np.zeros(points)
    for i in range(points):
        pe[i] = Popen(mec, tres, c[i])
        pi[i] = Popen(mec, 0, c[i])
        H[i] = pmax / (math.pow((iEC50 / c[i]), nH) + 1) # Hill equation

    return c, pe, pi#,  H


def Popen0(mec, tres, eff='c'):
    """
    Find Popen at concentration = 0.

    Parameters
    ----------
    mec : instance of type Mechanism
    tres : float
        Time resolution (dead time).

    Returns
    -------
    P0 : float
        Open probability in absence of effector.
    """
    popen = Popen(mec, 0, conc=0) 
    return popen if popen < 1e-10 else Popen(mec, tres, conc=0)

def maxPopen(mec, tres, eff='c'):
    """
    Estimate numerically maximum equilibrium open probability.
    In case Popen curve goes through a maximum, the peak open
    probability is returned.

    Parameters
    ----------
    mec : instance of type Mechanism
    tres : float
        Time resolution (dead time).

    Returns
    -------
    maxPopen : float
        Maximum equilibrium open probability.
    conc : float
        Concentration at which Popen curve reaches maximal value.
    """

    flat, monot = False, True
    conc = 1e-9    # start at 1 nM
    poplast = Popen(mec, tres, conc)

    niter = 0
    while (not flat and conc < 100 and monot):
        conc *= math.sqrt(10)
        popen = Popen(mec, tres, conc)
        if decline(mec, tres) and (math.fabs(popen) < 1e-12):
            flat = math.fabs(poplast) < 1e-12
        else:
            rel = (popen - poplast) / popen
            if niter > 1 and popen > 1e-5:
                if (rel * rellast) < -1e-10: # goes through min/max
                    monot = False
                flat = ((math.fabs(rel) < 1e-5) and
                    (math.fabs(rellast) < 1e-5))
            if conc < 0.01:    # do not leave before 10 mM ?
                flat = False
            rellast = rel
        poplast = popen
        niter += 1

    if not monot:    # find maxPopen and cmax more accurately
        c1, c2 = conc / math.sqrt(10), conc # conc before and after max
        epsc, epsy =  c1 / 1000, 1e-4 # accuracy in concentration and Popen
        perr = 2 * epsy
        fac = 1.01
        maxnstep  = int(math.log10(math.fabs(c1 - c2) / epsc) / math.log10(2) + 0.5)
        nstep = 0
        while nstep <= maxnstep and math.fabs(perr) > 0:
            conc = 0.5 * (c1 + c2)
            conc1 = conc / fac
            P1 = Popen(mec, tres, conc)
            conc1 = conc * fac
            P2 = Popen(mec, tres, conc)
            perr = P2 - P1
            if perr < 0:
                c1 = conc1
            else:
                c2 = conc1

    return Popen(mec, tres, conc), conc

def decline(mec, tres, eff='c'):
    """
    Find whether open probability curve increases or decreases
    with ligand concentration. Popen may decrease if ligand is inhibitor.

    Parameters
    ----------
    mec : instance of type Mechanism
    tres : float
        Time resolution (dead time).

    Returns
    -------
    decline : bool
        True if Popen curve dectreases with concentration.
    """
    return (Popen(mec, tres, conc=1) < Popen0(mec, tres))

def EC50(mec, tres, eff='c'):
    """
    Estimate numerically the equilibrium EC50 for a specified mechanism.
    If monotonic this is unambiguous. If not monotonic then returned is
    a concentration for 50% of  the peak response to the left of the peak.

    Parameters
    ----------
    mec : instance of type Mechanism
    tres : float
        Time resolution (dead time).

    Returns
    -------
    EC50 : float
        Concentration at which Popen is 50% of its maximal value.
    """

    P0 = Popen0(mec, tres)
    maxP, c2 = maxPopen(mec, tres)
    c1 = 0
    epsy = 0.001    # accuracy in Popen
    perr = 2 * epsy
    epsc = 0.1e-9    # accuracy in concentration 0.1 nM
    nstepmax = int(math.log10(math.fabs(c1 - c2) / epsc) / math.log10(2) + 0.5)
    nstep = 0
    while math.fabs(perr) > epsy and nstep <= nstepmax:
        conc = (c1 + c2) / 2
        perr = math.fabs((Popen(mec, tres, conc) - P0) / (maxP - P0)) - 0.5
        if perr < 0:
            c1 = conc
        elif perr > 0:
            c2 = conc
        nstep += 1
    return conc

def Hill_slope(mec, tres, eff='c'):
    """
    Calculate Hill slope, nH, at EC50 of a calculated Popen curve.
    This is Python implementation of DCPROGS HJC_HILL.FOR subroutine.

    Parameters
    ----------
    mec : instance of type Mechanism
    tres : float
        Time resolution (dead time).

    Returns
    -------
    nH : float
        Hill slope.
    """

    P0 = Popen0(mec, tres)
    Pmax, cmax = maxPopen(mec, tres)
    if decline(mec, tres):
        P0, Pmax = Pmax, P0
    ec50 = EC50(mec, tres)
    # Calculate Popen curve
    n = 64
    dc = (math.log10(ec50 * 1.1) - math.log10(ec50 * 0.9)) / (n - 1)
    c = np.zeros(n)
    y = np.zeros(n)
    for i in range(n):
        c[i] = (ec50 * 0.9) * pow(10, i * dc)
        y[i] = Popen(mec, tres, c[i])

    # Find two points around EC50.
    i50 = 0
    s1, s2 = 0, 0
    i = 0
    while i50 ==0 and i < n-1:
        if (c[i] <= ec50) and (c[i+1] >= ec50):
            i50 = i
            y1 = math.log10(math.fabs((y[i] - P0) / (Pmax - y[i])))
            y2 = math.log10(math.fabs((y[i+1] - P0) / (Pmax - y[i+1])))
            s1 = (y2 - y1) / (math.log10(c[i+1]) - math.log10(c[i]))
            y3 = math.log10(math.fabs((y[i+1] - P0) / (Pmax - y[i+1])))
            y4 = math.log10(math.fabs((y[i+2] - P0) / (Pmax - y[i+2])))
            s2 = (y4 - y3) / (math.log10(c[i+2]) - math.log10(c[i+1]))
        i += 1

    # Interpolate linearly for Hill slope at EC50
    b = (s2 - s1) / (c[i50+1] - c[i50])
    nH = s1 + b * (ec50 - c[i50])
    return nH



def printout(mec, tres):
    """
    """
    out = ('\n*******************************************\nPopen CURVE\n' )
    if mec.fastblock:
        out += ('\nThis Popen curve was corrected for fast block ' + 
            'with KB = {0:.5g} mM.'.format(mec.fastKB * 1000))
    out += ('\nHJC Popen curve:\n' + print_pars(mec, tres))
    out += ('\nIdeal Popen curve:\n' + print_pars(mec, 0))
    return out

def print_pars(mec, tres):
    emaxPopen, conc = maxPopen(mec, tres)
    return ('maxPopen = {0:.5g}; '.format(emaxPopen) + 
           ' EC50 = {0:.5g} microM; '.format(EC50(mec, tres) * 1000000) + 
           ' nH = {0:.5g}'.format(Hill_slope(mec, tres)))



def plot_popen_curve(mec, tres):
    """
    Plot the Popen curves for a given mechanism and time resolution.
    
    Parameters:
    mec - Mechanism object (contains the mechanism parameters)
    tres - Time resolution (dead time)
    
    This function plots the equilibrium open probability (Popen) curves,
    including both the apparent (corrected for missed events) and ideal curves.
    """
    
    # Calculate Popen curve data
    c, pe, pi = calculate_Popen_plot(mec, tres)
    
    # Convert concentration to µM
    c_um = c * 1e6
    
    # Plot apparent and ideal Popen curves
    plt.semilogx(c_um, pe, 'b-', label='Apparent (Corrected)')
    plt.semilogx(c_um, pi, 'r--', label='Ideal')
    
    # Label plot elements
    plt.ylabel('Popen')
    plt.xlabel('Concentration (µM)')
    plt.title('Apparent and Ideal Popen Curves')
    plt.legend()
    plt.grid(True)
    
    plt.show()
    


if __name__ == '__main__':
    c = 0.0000001 # 0.1 uM
    mec = samples.CH82()
    mec.set_eff('c', c)
    tres = 0.0001 # 100 us

    #print(printout(mec, tres))
    #plot_popen_curve(mec, tres)

    popen_analysis = PopenCurve(mec)
    popen_analysis.plot_popen_curve(cmin=0.1e-6, cmax=0.1e-3, tres=tres)
    print(popen_analysis.printout(tres=tres))