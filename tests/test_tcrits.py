import unittest
import numpy as np
from scipy import optimize as so

from scalcs.pdfs import TCrits
from scalcs.qmatlib import QMatrix
from samples import samples

class TestTCrits(unittest.TestCase):
    def setUp(self):

        # Setting up test data
        mec = samples.CH82()
        mec.set_eff('c', 0.1e-6) # 0.1 uM
        q_dwells = QMatrix(mec)
        e, w = q_dwells.ideal_shut_time_pdf_components()
        self.tcrits_obj = TCrits(1 / e, w / e)
        
        # Expected critical times and misclassified values
        self.expected_tcrits = np.array([[0.0002331703859231374, 0.0015961890924042492],  # DC criterion
                                         [0.0002783170202703173, 0.001991173682237133],  # C&N criterion
                                         [0.0003941418085441515, 0.002676264472148571]]) # Jackson criterion

        self.expected_misclassified = {
            "DC": {
                "1_2": {"pf": 0.011878926921784511, "ps": 0.011878926921784511, "enf": 0.008667901717923407, "ens": 0.0032110252470535067},
                "2_3": {"pf": 0.0004211382166996774, "ps": 0.0004211382166996774, "enf": 0.0003108228748879419, "ens": 0.00011031534197327652},
            },
            "CN": {
                "1_2": {"pf": 0.005035169014601427, "ps": 0.013592030070958043, "enf": 0.0036740987160767836, "ens": 0.0036740987160767836},
                "2_3": {"pf": 0.00018644458249004374, "ps": 0.0005253234970906684, "enf": 0.00013760622721676122, "ens": 0.00013760622721676122},
            },
            "Jackson": {
                "1_2": {"pf": 0.0005567767704375295, "ps": 0.017326661238954176, "enf": 0.00040627291983123923, "ens": 0.004683617156951672},
                "2_3": {"pf": 4.5369367285873026e-05, "ps": 0.0007060044809844568, "enf": 3.348505695387487e-05, "ens": 0.00018493483267978327},
            }
        }

    def test_tcrits(self):
        for i in range(self.tcrits_obj.num_components):
            for crit_id in range(3):
                tcrit_computed = self.tcrits_obj.tcrits[crit_id, i]
                tcrit_expected = self.expected_tcrits[crit_id, i]
                self.assertAlmostEqual(tcrit_computed, tcrit_expected, places=6)

    def test_misclassified(self):
        for i in range(self.tcrits_obj.num_components):
            for crit_id, criterion_name in enumerate(['DC', 'CN', 'Jackson']):
                enf, ens, pf, ps = self.tcrits_obj.misclassified_results[crit_id, i]
                expected = self.expected_misclassified[criterion_name][f"{i+1}_{i+2}"]
                self.assertAlmostEqual(pf, expected["pf"], places=6)
                self.assertAlmostEqual(ps, expected["ps"], places=6)
                self.assertAlmostEqual(enf, expected["enf"], places=6) 
                self.assertAlmostEqual(ens, expected["ens"], places=6)

if __name__ == '__main__':
    unittest.main()
