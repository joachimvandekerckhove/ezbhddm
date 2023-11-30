import unittest
import numpy as np
from parameter_set import Hddm_Parameter_Set
from prior import Hddm_Prior

class TestPrior(unittest.TestCase):

    def test_init(self):
        self.assertIsInstance(Hddm_Prior(), Hddm_Prior)

class TestParameter(unittest.TestCase):

    def test_init(self):
        self.assertIsInstance(Hddm_Parameter_Set(), Hddm_Parameter_Set)

    def test_random(self):
        self.assertIsInstance(Hddm_Parameter_Set.random(Hddm_Prior(), 3), Hddm_Parameter_Set)

    def test_eq(self):
        self.assertTrue(Hddm_Parameter_Set(bound_mean = 1, bound_sdev = 1, bound = np.array([1, 1, 1]),
                                           drift_mean = 1, drift_sdev = 1, drift = np.array([1, 1, 1]),
                                           nondt_mean = 1, nondt_sdev = 1, nondt = np.array([1, 1, 1]),
                                           betaweight = 0) == \
                        Hddm_Parameter_Set(bound_mean = 1, bound_sdev = 1, bound = np.array([1, 1, 1]),
                                           drift_mean = 1, drift_sdev = 1, drift = np.array([1, 1, 1]),
                                           nondt_mean = 1, nondt_sdev = 1, nondt = np.array([1, 1, 1]),
                                           betaweight = 0))

    def test_ineq(self):
        self.assertFalse(Hddm_Parameter_Set(bound_mean = 1, bound_sdev = 1, bound = np.array([1, 1, 1]),
                                            drift_mean = 1, drift_sdev = 1, drift = np.array([1, 1, 1]),
                                            nondt_mean = 1, nondt_sdev = 1, nondt = np.array([1, 1, 1]),
                                            betaweight = 0) == \
                         Hddm_Parameter_Set(bound_mean = 1, bound_sdev = 1, bound = np.array([1, 1, 1]),
                                            drift_mean = 1, drift_sdev = 1, drift = np.array([1, 1, 1]),
                                            nondt_mean = 1, nondt_sdev = 1, nondt = np.array([1, 1, 1]),
                                            betaweight = 1))

    def test_sub(self):
        self.assertEqual(Hddm_Parameter_Set(bound_mean = 3, bound_sdev = 3, bound = np.array([3, 3, 3]),
                                            drift_mean = 3, drift_sdev = 3, drift = np.array([3, 3, 3]),
                                            nondt_mean = 3, nondt_sdev = 3, nondt = np.array([3, 3, 3]),
                                            betaweight = 3)
                       - Hddm_Parameter_Set(bound_mean = 2, bound_sdev = 2, bound = np.array([2, 2, 2]),
                                            drift_mean = 2, drift_sdev = 2, drift = np.array([2, 2, 2]),
                                            nondt_mean = 2, nondt_sdev = 2, nondt = np.array([2, 2, 2]),
                                            betaweight = 2),
                         Hddm_Parameter_Set(bound_mean = 1, bound_sdev = 1, bound = np.array([1, 1, 1]),
                                            drift_mean = 1, drift_sdev = 1, drift = np.array([1, 1, 1]),
                                            nondt_mean = 1, nondt_sdev = 1, nondt = np.array([1, 1, 1]),
                                            betaweight = 1))


if __name__ == '__main__':
    unittest.main()


print("Done")
