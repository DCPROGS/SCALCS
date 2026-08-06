#! /usr/bin/env python
"""Example of realistic concentration jump calculation.

Demonstrates the ErfPulse (realistic concentration jump) and SquarePulse
APIs in scalcs.cjumps.
"""

import matplotlib.pyplot as plt
from scalcs.samples import samples
from scalcs import cjumps

if __name__ == "__main__":

    mec = samples.CH82()
    mec.printout()

    # --- Realistic concentration jump (erf profile) ---
    pulse = cjumps.ErfPulse(
        cmax=10e-6,       # 10 µM peak concentration
        width=10e-3,      # 10 ms pulse width
        cb=0.0,           # zero background
        centre=10e-3,     # pulse centred at 10 ms
        rise=250e-6,      # 10-90% rise time
        decay=250e-6,     # 90-10% decay time
    )

    result = cjumps.solve(mec, pulse, reclen=50e-3, step=8e-6)
    t, c, Popen, P = result          # backward-compatible 4-tuple unpacking

    maxP = Popen.max()
    maxC = c.max()
    c_scaled = (c / maxC) * 0.2 * maxP + 1.02 * maxP   # overlay on Popen axis

    plt.figure()
    plt.plot(t * 1000, Popen, 'b-', label='Popen')
    plt.plot(t * 1000, c_scaled, 'g-', label='Concentration (scaled)')
    plt.ylabel('Open probability')
    plt.xlabel('Time (ms)')
    plt.title('Realistic concentration jump (erf profile)')
    plt.legend()

    # --- Analytical printout for a square pulse ---
    square = cjumps.SquarePulse(cmax=10e-6, width=10e-3)
    print(cjumps.printout(mec, square))

    plt.show()
    print('\ndone!')
