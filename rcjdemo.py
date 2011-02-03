#! /usr/bin/python
"""
Example of realistic concentration jump calculation.
"""

import matplotlib.pyplot as plt
from dcpyps import samples
from dcpyps import cjumpslib as cjl

if __name__ == "__main__":

    mec = samples.CH82()

    # Here one can tweak the parameters of the jump.
    jump_params = {
    'step_size'     : 8 , # The sample step. All time params in microseconds.
    'pulse_centre'  : 10000 ,
    'rise_time'     : [250] , # list of 10-90% rise times for error functions
    'pulse_width'   : 2500 ,
    'record_length' : 50000,
    'peak_conc'     : 30e-3          # in molar
    }

    for rise in jump_params['rise_time']:

        print 'Calculating jump with %s microsec rise...' %(rise)

        P_copy = jump_params.copy()
        P_copy['rise_time'] = rise

        jump, relax = cjl.rcj_single(mec, P_copy)
        
        # rlx contains Popen trace only
        t, cjmp, rlx = cjl.convert_to_arrays(jump, relax, mec.kA)

        plt.subplot(211)
        plt.plot(t * 0.001, cjmp * 1000, 'b-')
        plt.ylabel('Concentration, mM')
        plt.xlabel('Time, ms')
        plt.title('Concentration jump')

        plt.subplot(212)
        plt.plot(t * 0.001, rlx,'r-')
        plt.ylabel('Popen')
        plt.xlabel('Time, ms')

        plt.subplots_adjust(left=None, bottom=0.1, right=None, top=None,
            wspace=0.4, hspace=0.5)
        plt.show()



#        jump_data_out = cjl.compose_rcj_out(jump, relax, mec.kA, 1.2, 1)
        #1.2 : offset for c-jump
        #1   : returns jump and receptor P-O traces only
#        f=open('jump250.xls', 'w')
#        for l in jump_data_out:
#            f.write(l+'\n')
#        f.close()

    print ('done!')


