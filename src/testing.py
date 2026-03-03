import unittest
import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .data_loader import DataPreprocessor
from .models.ensemble import UnificationLayer

warnings.filterwarnings("ignore")


class TestIDSComponents(unittest.TestCase):
    def test_preprocessor_clean_logic(self):
        print("\n[Test] Preprocessor Data Cleaning")

        df = pd.DataFrame(
            {
                "Feature1": [1.0, np.inf, np.nan, -np.inf],
                "IPV4_SRC_ADDR": ["1.1.1.1", "2.2.2.2", "3.3.3.3", "4.4.4.4"],
            }
        )

        mp = DataPreprocessor("dummy_path.csv")
        cleaned = mp.clean_data(df)

        self.assertNotIn("IPV4_SRC_ADDR", cleaned.columns)
        vals = cleaned["Feature1"].values
        self.assertEqual(vals[1], 0.0)
        self.assertEqual(vals[2], 0.0)
        self.assertEqual(vals[3], 0.0)
        print(" -> PASSED")

    def test_unification_scaling(self):
        print("\n[Test] Unification Layer Scaling")
        unifier = UnificationLayer()

        train_scores = np.array([0.0, 50.0, 100.0])
        unifier.register_stats("m1", train_scores)

        unified_mid = unifier.unify("m1", np.array([50.0]))
        self.assertAlmostEqual(unified_mid[0], 0.5, places=4)

        unified_high = unifier.unify("m1", np.array([200.0]))
        self.assertAlmostEqual(unified_high[0], 1.0, places=4)

        unified_low = unifier.unify("m1", np.array([-10.0]))
        self.assertAlmostEqual(unified_low[0], 0.0, places=4)
        print(" -> PASSED")

    def test_rank_ensemble_math(self):
        print("\n[Test] Rank Ensemble Math")

        n_samples = 3
        base_scores = np.array([[0.1, 0.2], [0.9, 0.8], [0.5, 0.6]])

        n_models = 2
        ranks = np.zeros_like(base_scores)
        for i in range(n_models):
            ranks[:, i] = rankdata(base_scores[:, i], method="min")

        avg_rank = np.mean(ranks, axis=1)
        final = (avg_rank - 1) / (n_samples - 1)

        self.assertAlmostEqual(final[0], 0.0)
        self.assertAlmostEqual(final[1], 1.0)
        self.assertAlmostEqual(final[2], 0.5)
        print(" -> PASSED")


if __name__ == "__main__":
    unittest.main()
