#! /usr/bin/env python
"""
Example of realistic concentration jump calculation.
"""

import matplotlib.pyplot as plt
from dcpyps import samples
from dcpyps import cjumps

if __name__ == "__main__":

    mec = samples.CH82()
    mec.printout()

    # Here one can tweak the parameters of the jump.
    step_size = 8e-6 # The sample step. All time parameters in seconds
    pulse_centre = 10e-3
    rise_time = 250e-6 # 10-90% rise time for error functions
    pulse_width = 10e-3
    record_length = 50e-3
    peak_conc = 10e-6    # in molar
    baseline_conc = 0.0
    cjargs = (peak_conc, baseline_conc, pulse_centre, pulse_width,
                rise_time, rise_time)
    
    print ('\nCalculating jump with {0:.6f} microsec rise...'.
        format(rise_time/1e-6))
    cjumps.printout(mec, peak_conc, pulse_width)
    t, c, Popen, P  = cjumps.solve_jump(mec, record_length, step_size,
        cjumps.pulse_erf, cjargs)
    maxP = max(Popen)
    maxC = max(c)
    c1 = (c / maxC) * 0.2 * maxP + 1.02 * maxP

    plt.plot(t * 1000, Popen,'b-', t * 1000, c1, 'g-')
    plt.ylabel('Open probability')
    plt.xlabel('Time, ms')
    plt.title('Concentration jump')
    plt.show()

    print ('\ndone!')
