import unittest
import numpy as np
import pandas as pd
import torch
import warnings
from preprocessing import DataPreprocessor
from ensemble import UnificationLayer, BaseEnsemble

# Filter unnecessary warnings for tests
warnings.filterwarnings("ignore")

class TestIDSComponents(unittest.TestCase):

    def test_preprocessor_clean_logic(self):
        """Test if DataPreprocessor.clean_data handles NaNs and Infinity correctly."""
        print("\n[Test] Preprocessor Data Cleaning")
        
        # Create dummy dataframe with dirty data
        df = pd.DataFrame({
            'Feature1': [1.0, np.inf, np.nan, -np.inf],
            'IPV4_SRC_ADDR': ['1.1.1.1', '2.2.2.2', '3.3.3.3', '4.4.4.4']
        })
        
        # Instantiate without file load
        mp = DataPreprocessor("dummy_path.csv")
        
        cleaned = mp.clean_data(df)
        
        # 1. Check IP column removal
        self.assertNotIn('IPV4_SRC_ADDR', cleaned.columns, "IP Column should be dropped")
        
        # 2. Check Inf/NaN handling
        # Should replace Inf with NaN then fill with 0
        vals = cleaned['Feature1'].values
        self.assertEqual(vals[1], 0.0, "Inf should be 0")
        self.assertEqual(vals[2], 0.0, "NaN should be 0")
        self.assertEqual(vals[3], 0.0, "-Inf should be 0")
        print(" -> PASSED")

    def test_unification_scaling(self):
        """Test MinMax scaling logic in UnificationLayer."""
        print("\n[Test] Unification Layer Scaling")
        unifier = UnificationLayer()
        
        # Train stats
        train_scores = np.array([0.0, 50.0, 100.0])
        unifier.register_stats('m1', train_scores)
        
        # Test Case 1: Middle value
        unified_mid = unifier.unify('m1', np.array([50.0]))
        self.assertAlmostEqual(unified_mid[0], 0.5, places=4)
        
        # Test Case 2: Out of bounds (Calibration clipping)
        unified_high = unifier.unify('m1', np.array([200.0]))
        self.assertAlmostEqual(unified_high[0], 1.0, places=4)
        
        unified_low = unifier.unify('m1', np.array([-10.0]))
        self.assertAlmostEqual(unified_low[0], 0.0, places=4)
        print(" -> PASSED")

    def test_rank_ensemble_math(self):
        """
        Manually verify Rank Averaging logic without loading full models.
        Logic: Average Rank / (N-1)
        """
        print("\n[Test] Rank Ensemble Math")
        from scipy.stats import rankdata
        
        # Synthetic Base Scores from 2 Models for 3 Samples
        # Model A: [0.1, 0.9, 0.5] -> Rank: [0, 2, 1] relative order
        # Model B: [0.2, 0.8, 0.6] -> Rank: [0, 2, 1] relative order
        # Since we use rankdata, let's trace exact values
        
        n_samples = 3
        # Scores:
        # Sample 0: Low, Low
        # Sample 1: High, High
        # Sample 2: Mid, Mid
        base_scores = np.array([
            [0.1, 0.2], 
            [0.9, 0.8], 
            [0.5, 0.6]
        ])
        
        # Expected Ranks (method='min'):
        # Col 0: 0.1->1, 0.5->2, 0.9->3
        # Col 1: 0.2->1, 0.6->2, 0.8->3
        
        # Avg Ranks:
        # Row 0: (1+1)/2 = 1
        # Row 1: (3+3)/2 = 3
        # Row 2: (2+2)/2 = 2
        
        # Normalized (rank-1)/(N-1) -> (rank-1)/2:
        # Row 0: 0.0
        # Row 1: 1.0
        # Row 2: 0.5
        
        n_models = 2
        ranks = np.zeros_like(base_scores)
        for i in range(n_models):
            ranks[:, i] = rankdata(base_scores[:, i], method='min')
            
        avg_rank = np.mean(ranks, axis=1)
        final = (avg_rank - 1) / (n_samples - 1)
        
        self.assertAlmostEqual(final[0], 0.0)
        self.assertAlmostEqual(final[1], 1.0)
        self.assertAlmostEqual(final[2], 0.5)
        print(" -> PASSED")

if __name__ == '__main__':
    unittest.main()
