"""
DC_PyPs
=======

A collection of functions for Q matrix manipulations.

Notes
-----
DC_PyPs project are pure Python implementations of Q-Matrix formalisms
for ion channel research. To learn more about kinetic analysis of ion
channels see the references below.

References
----------
CH82: Colquhoun D, Hawkes AG (1982)
On the stochastic properties of bursts of single ion channel openings
and of clusters of bursts. Phil Trans R Soc Lond B 300, 1-59.

HJC92: Hawkes AG, Jalali A, Colquhoun D (1992)
Asymptotic distributions of apparent open times and shut times in a
single channel record allowing for the omission of brief events.
Phil Trans R Soc Lond B 337, 383-404.

CH95a: Colquhoun D, Hawkes AG (1995a)
The principles of the stochastic interpretation of ion channel mechanisms.
In: Single-channel recording. 2nd ed. (Eds: Sakmann B, Neher E)
Plenum Press, New York, pp. 397-482.

CH95b: Colquhoun D, Hawkes AG (1995b)
A Q-Matrix Cookbook.
In: Single-channel recording. 2nd ed. (Eds: Sakmann B, Neher E)
Plenum Press, New York, pp. 589-633.

CHS96: Colquhoun D, Hawkes AG, Srodzinski K (1996)
Joint distributions of apparent open and shut times of single-ion channels
and maximum likelihood fitting of mechanisms.
Phil Trans R Soc Lond A 354, 2555-2590.
"""

import bisectHJC
import dcio
import dcpypsrc
from mechanism import * # this should really be in the namespace
                        # so that dcpyps.Mechanism is something like
                        # numpy.array
import pdfs
import popen
# import qmatgui
import qmatlib
import rcj
import scalcslib
import scburst
import scplotlib
import version

import samples
