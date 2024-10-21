from samples import samples
from scalcs.qmatprint import QMatrixPrints, SCBurstPrints
from scalcs.qmatprint import CorrelationPrints, TCritPrints, ExactPDFPrints, AsymptoticPDFPrints, AdjacentPDFPrints


print('\n\tNUMERICAL EXAMPLE CH82')
# Load Colquhoun & Hawkes 1982 numerical example
c = 0.0000001 # 0.1 uM
mec = samples.CH82()
mec.set_eff('c', c)
#print(mec)

# Create an instances of the QMatrix, QOccupancies and QTransitions classes
q_matrix = QMatrixPrints(mec.Q, mec.kA, mec.kB, mec.kC, mec.kD)
print(q_matrix.print_Q)
print(q_matrix.print_pinf) # print equilibrium state occupancies
print(q_matrix.print_Popen)
print(q_matrix.print_state_lifetimes)
print(q_matrix.print_transition_matrices)
print(q_matrix.print_subset_probabilities)
print(q_matrix.print_initial_vectors)
print(q_matrix.print_DC_table)
print(q_matrix.print_initial_vectors_for_openings_shuttings)
print(q_matrix.print_open_time_pdf)
print(q_matrix.print_shut_time_pdf)

# Calculating burst parameters
q_burst = SCBurstPrints(mec.Q, mec.kA, mec.kB, mec.kC, mec.kD)
print(q_burst.print_all)

# Calculating correlations
q_corrs = CorrelationPrints(mec.Q, mec.kA, mec.kB, mec.kC, mec.kD)
#print(q_corrs.print_all)

q_asymp = AsymptoticPDFPrints(mec.Q, mec.kA, mec.kB, mec.kC, mec.kD, tres=0.0001)
print(q_asymp.print_all)

#q_dwells = HJCDwellsPrints(mec.Q, mec.kA, mec.kB, mec.kC, mec.kD)
#q_dwells.tres = 0.0001 # 10 us
#print(q_dwells.print_all)
#print(q_dwells.print_adjacent_dwells(q_dwells.tres, q_dwells.tres * 2))
#print(q_dwells.print_adjacent_dwells(q_dwells.tres, q_dwells.tres * 2))

q_exact = ExactPDFPrints(mec.Q, mec.kA, mec.kB, mec.kC, mec.kD, tres=0.0001)
#q_exact.tres = 0.0001 # 10 us
print(q_exact.open_time_pdf)
print(q_exact.shut_time_pdf)

tcrits = TCritPrints(mec)
print(tcrits.print_all)

q_adjacent = AdjacentPDFPrints(mec.Q, mec.kA, mec.kB, mec.kC, mec.kD, tres=0.0001)
print(q_adjacent.ideal_adjacent_dwells(0.0001, 0.001))
