import numpy as np
import qmat
import qmatpy
import time
import sys

def init_matrix_colqu(c_agonist, debug=False):
    """Returns the example matrix given by Colquhoun & Hawkes,
    chapter 18, p. 465, eq. 127, for a given agonist concentration.
    Example as in chapter 20, p. 593, eq. 4"""

    # Initialize rates
    beta_1 = 15.0
    beta_2 = 15000.0
    alpha_1 = 3000.0
    alpha_2 = 500.0
    k_m1 = 2000.0
    k_m2 = 2000.0
    k_p1 = 5.0e07 * c_agonist
    k_star_p2 = 5.0e08 * c_agonist
    k_p2 = 5.0e08 * c_agonist
    k_star_m2 = 1.0/3.0

    # Set up Q matrix
    Q = np.array([[             0, k_star_p2,       0,  alpha_1,    0],
                  [ 2.0*k_star_m2,         0, alpha_2,        0,    0],
                  [             0,    beta_2,       0, 2.0*k_m2,    0],
                  [        beta_1,         0,    k_p2,        0, k_m1],
                  [             0,         0,       0, 2.0*k_p1,    0]])
    Q *= 1e-3 # convert to 1/ms
    qmat.init_matrix(Q) # Update diagonal elements

    if debug: print "Initialised Colquhoun-Hawkes matrix:\n", Q

    return Q


if __name__ == "__main__":

    np.set_printoptions(precision=3)

    # solution as in Ch. 20, p. 597, eq. 18
    p_inf1_ref = np.array([
            2.483e-5, 1.862e-3, 6.207e-5, 4.965e-3, 9.931e-1])
    p_inf1_hjc = np.array([
            2.483e-5, 1.862e-3, 6.207e-5, 4.965e-3, 9.931e-1])
    lambda_ref = np.array([
            0.000e0, 1.018e-1, 2.022e0, 3.094e0, 1.941e1])
    lambda_hjc = np.array([
            0.000e0, 1.018e-1, 2.022e0, 3.094e0, 1.941e1])
    Q_100 = init_matrix_colqu(100.0e-09)
    Q_10 = init_matrix_colqu(10.0e-09)
    t = np.arange(0,1000.0,0.01)

    # C++
    time0 = time.time()

    p_inf0_cpp = qmat.p_inf(Q_10)
    p_inf1_cpp = qmat.p_inf(Q_100)
    lambda_cpp, A_cpp = qmat.mat_solve(Q_100)
    y_cpp = qmat.p(t, p_inf0_cpp, p_inf1_cpp, lambda_cpp, A_cpp)

    print "\nC++ implementation took", 
    print np.round((time.time()-time0)*1e3, 3), "ms"

    # Pure Python
    time0 = time.time()

    p_inf0_py = qmatpy.p_inf(Q_10)
    p_inf1_py = qmatpy.p_inf(Q_100)
    lambda_py, A_py = qmatpy.mat_solve(Q_100)
    y_py = qmatpy.p(t, p_inf0_py, p_inf1_py, lambda_py, A_py)

    print "Python implementation took",
    print np.round((time.time()-time0)*1e3, 3), "ms"

    # Results
    sys.stdout.write("\np_inf (blue book) = %s\n" % np.array_str(p_inf1_ref, precision=3))
    sys.stdout.write("p_inf (hjcfit) =    %s\n" % np.array_str(p_inf1_hjc, precision=3))
    sys.stdout.write("p_inf (C++) =       %s\n" % np.array_str(p_inf1_cpp, precision=3))
    sys.stdout.write("p_inf (Python) =    %s\n\n" % np.array_str(p_inf1_py, precision=3))

    sys.stdout.write("lambda (blue book) = %s\n" % np.array_str(lambda_ref, precision=3))
    sys.stdout.write("lambda (hjcfit) =    %s\n"    % np.array_str(lambda_hjc, precision=3))
    sys.stdout.write("lambda (C++) =       %s\n" % np.array_str(lambda_cpp, precision=3))
    sys.stdout.write("lambda (Python) =    %s\n\n" % np.array_str(lambda_py, precision=3))
    
    try:
        import matplotlib.pyplot as plt
        plt.plot(t,y_cpp[0],t,y_py[0])
        plt.show()
    except ImportError:
        print "Matplotlib is required for plotting"
