# DC\_PyPs #

## Introduction ##
The goal of DC\_PyPs is to provide a collection of tools for scientific research on ion channels. The package is derived from the DCPROGS suite (http://www.ucl.ac.uk/Pharmacology/dcpr95.html) and will consist of a Python port and wrap of Fortran code with ~30 years usage at University College London. The rationale is to preserve and cultivate these tools for future research applications. The included programs are used to process raw data, to plot and fit dwell time distributions, to perform statistical tests on data and to fit kinetic mechanisms to random time series.

## Installation ##
[Download](http://code.google.com/p/dc-pyps/downloads/list) the source distribution, unpack, change to the source directory and run (as administrator):
```
$ python setup.py install
```

## Getting started ##
The sample from [CH82](CH82.md) that is implemented in dcpyps/samples/samples.py illustrates how to set up the rates and states of a gating mechanism. The demo (demo.py) shows how to compute open probabilities and burst properties for this mechanism.

## References ##
CH82: Colquhoun D, Hawkes AG (1982) On the stochastic properties of bursts of single ion channel openings and of clusters of bursts. Phil Trans R Soc Lond B 300, 1-59.

HJC92: Hawkes AG, Jalali A, Colquhoun D (1992) Asymptotic distributions of apparent open times and shut times in a single channel record allowing for the omission of brief events. Phil Trans R Soc Lond B 337, 383-404.

CH95a: Colquhoun D, Hawkes AG (1995a) The principles of the stochastic interpretation of ion channel mechanisms. In: Single-channel recording. 2nd ed. (Eds: Sakmann B, Neher E) Plenum Press, New York, pp. 397-482.

CH95b: Colquhoun D, Hawkes AG (1995b) A Q-Matrix Cookbook. In: Single-channel recording. 2nd ed. (Eds: Sakmann B, Neher E) Plenum Press, New York, pp. 589-633.

CHS96: Colquhoun D, Hawkes AG, Srodzinski K (1996) Joint distributions of apparent open and shut times of single-ion channels and maximum likelihood fitting of mechanisms. Phil Trans R Soc Lond A 354, 2555-2590.