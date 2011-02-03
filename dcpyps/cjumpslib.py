"""Library of routines for calculating responses to concentration jumps."""

__author__="Andrew Plested"
__date__ ="$Dec 20, 2010 12:07:36 PM$"

import numpy as np

import qmatlib as qml

def erf(x):
    """
    Accurate, fast approximation of the error function from
    http://www.johndcook.com/blog/2009/01/19/stand-alone-error-function-erf/

    arguments -- x - a scalar
    returns -- a scalar
    """

    # constants
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911

    # Save the sign of x.
    sign = 1
    if x < 0:
        sign = -1
    x = abs(x)

    # A & S 7.1.26
    t = 1.0 / (1.0 + p * x)
    y = (1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t *
        np.exp(-x * x))

    return sign*y

def erf_pulse(pulse_width, pulse_conc, rise_t, pulse_centre, t_step):
    """
    Construct a "top-hat" pulse with rise and fall from error function.

    arguments --    pulse_width - approximate FWHM in microseconds
                    pulse_conc - ligand concentration during jump in Molar
                    rise_t - desired 10 - 90% rise time (can use 0. for square pulse)
                    centre - position of pulse in simulation trace
                    time_step - sampling interval in microseconds

    returns -- dictionary of microsecond time point,concentration pairs
    """

    erf_rise = 1.8  #10-90% of erf(x)
    step_factor = float(rise_t) / ( erf_rise * t_step )

    z = {}

    # lock to sampling intervals
    left_fwhm  = int((pulse_centre - (float(pulse_width) / 2)) / t_step)*t_step
    right_fwhm = int((pulse_centre + (float(pulse_width) / 2)) / t_step)*t_step

    # fill in with true top hat at pulse_conc - will overwrite flanks with erf
    for r in range(left_fwhm, right_fwhm, t_step):

        z [r] = pulse_conc

    # Return a perfect top hat if rise time is 0
    # or shorter than sampling interval
    if rise_t < t_step :
        return z

    #reduce number of points but scale with rise_t
    pts = int(float(rise_t) / 3) 
    for s in range(-pts, pts + 1): #symmetrical

        # scale x into correct intervals
        x = float(s) / step_factor
        # y-shift erf to 0 to 1; scale by concentration
        y = (erf(x) + 1.) / 2. * pulse_conc          

        # flatten differences under 10 pM during constant-concentration sections
        if y < 10e-12:
            y = 0.

        if y > pulse_conc - 10e-12:
            y = pulse_conc

        #inflexion points centred on left_fwhm, right_fwhm
        t_rise = left_fwhm + x * step_factor * t_step
        t_fall = right_fwhm + x * step_factor * t_step

        # truncate rise and fall if pulse is too thin to reach max.
        if t_rise <= pulse_centre:
            z[t_rise] = y

        if t_fall >= pulse_centre:
            z[t_fall] = pulse_conc-y

    return z

def rcj_single(mec, parameters):

    jump = make_jump(parameters)
    relax = rcj_calc (jump, mec, parameters)

    return jump, relax

def make_family():
    '''
    testing - make a family of pulses to examine properties
    alternate main()
    no arguments
    returns nothing
    '''

    center = 10000
    width = 2500. #desired 10-90% in microseconds
    family = []
    peak_conc = 30e-3
    step_size = 8
    max_length =0
    for rise_time in range(50,500,50):

        profile = erf_pulse(width,peak_conc,rise_time,center,step_size)
        family.append(profile)

        if len(profile) > max_length:
            max_length = len (profile)

    rcj_fileops.family_write(family,max_length)

def make_jump(pa):
    '''
    argument --
    pa : the parameter dictionary
    returns --
    ju : dictionary of sample point - concentration pairs
    '''

    #Make blank concentration jump profile as dictionary
    #again, t in microseconds
    ju = {}

    pw = pa['pulse_width']
    pc = pa['peak_conc']
    rt = pa['rise_time']
    pce= pa['pulse_centre']
    ss = pa['step_size']
    rl = pa['record_length']

    for t in range(0,rl,ss):
        ju [float(t)] = 0.     # fill dictionary with blanks

    profile = erf_pulse(pw, pc, rt, pce, ss )

    ju.update(profile)        # overwrite blank dictionary keys with pulse values

    return ju

def rcj_calc (jux, mec, paras, eff='c'):
    """
    arguments --
    jux         : concentration jump profile dictionary
    mec :
    mech_rates  : dictionary of rate name - constant pairs
    paras       : dictionary of parameters that defines pulse
    returns - relaxation and jump
    """

    rlx = {}
    dt = paras['step_size'] * 1.e-6  # Convert from microseconds to seconds.
    firststep = True

    # Sort the dictionary to get time elements in order.
    for t_point in sorted(jux):
        # Extract concentration.
        c = jux [t_point]
        mec.set_eff(eff, c)

        if not firststep:
            # Not the first step, so use P from last step to calculate
            # new occupancy. Note that P is not used again in the calculation,
            # so can be updated safely.
            w = coefficient_calc(mec, P)

            #loop over states to get occupancy of each
            for s in range(mec.k):
                # r is a running total over contributions of all components
                r = 0
                for ju, k in zip(w[:, s], mec.eigenvals):
                    r += ju * np.exp(k * dt)
                P[s] = r
        else:
            # First iteration, just calculate Pinf at initial concentration.
            P = qml.pinf(mec.Q)
            firststep = False

        # Must copy P, otherwise every value in dictionary = P (entire
        # output has last value!)
        rlx [t_point] = P.copy()
    return rlx

def coefficient_calc(mec, p_occup):
    """
    Calculate weighted components for relaxation for each state p * An
    """

    w = np.zeros([mec.k, mec.k])
    for n in range (mec.k):
        w[n, :] = np.dot(p_occup, mec.A[n, :, :])
    return w

def compose_rcj_out (cjump,relax_dict,o_states,offset=1.2,output_option=2):
    '''
    write output to text file as lines of t,j,p0,p1,p2....pn,pOpen with option to omit
    arguments --
        cjump           : dictionary of agonist profile values against time
        relax_dict      : dictionary of state occupancies against time
        o_states        : list of open states for P-open calculation
        offset          : float by which to offset the jump from the response
        output_option   : By default, 2: give occupancies and P_O; 1 : P_O Only; 0: Occupancies only
    returns --
        lines           : A list of strings

    '''
    lines = []

    for time_key in sorted(relax_dict):
        line = str(time_key)+'\t'+str(cjump[time_key]+offset)       #positive offset = 1.2 for jump
        isochrone = relax_dict[time_key]

        if output_option != 1:
            #Full occupancy
            for elem in isochrone:
                line += '\t'+str(elem)

        if output_option != 0:
            #Open Probability
            Popen = 0.
            for o in range(o_states):
                Popen = Popen + isochrone[o]

            line += '\t'+str(Popen)

        lines.append(line)

    return lines

def convert_to_arrays(cjump, relax_dict, kA):
    """

    """

    t = np.zeros(len(cjump))
    cj = np.zeros(len(cjump))
    rl = np.zeros(len(cjump))
    index = 0

    for time_key in sorted(relax_dict):
        t[index] = time_key
        cj[index] = cjump[time_key] * 1.01

        #Open Probability
        for o in range(kA):
            rl[index] += relax_dict[time_key][o]
        index +=1

    return t, cj, rl