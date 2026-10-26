import unittest

import numpy as np
import pandas as pd

from src.model_a_infer import ModelAInfer


class ModelAFeatureParityTest(unittest.TestCase):
    def test_build_features_adds_river_season_and_squeeze_fields(self):
        infer = ModelAInfer.__new__(ModelAInfer)
        infer._enrich = lambda legs_df, vessel_features_override=None, context_override=None: legs_df.copy()
        def _apply_pattern_features(d):
            d = d.copy()
            for col in ["pat_avg_h", "pat_med_h", "pat_std_h", "pat_count", "pat_speed_nm_h", "pat_lock_cnt", "pat_lock_delay_h"]:
                d[col] = np.nan
            return d

        def _apply_knn_features(d):
            d = d.copy()
            for col in ["knn_pred_h", "knn_med_h", "knn_std_h", "knn_mean_d", "knn_n_tight"]:
                d[col] = np.nan
            return d

        infer._apply_pattern_features = _apply_pattern_features
        infer._apply_knn_features = _apply_knn_features
        infer._apply_encoding_priors = lambda d: d

        legs = pd.DataFrame(
            {
                "START_PORT_AREA": ["Houston", "St. Louis, MO"],
                "START_PORT_LAT": [29.76, 38.63],
                "START_PORT_LON": [-95.37, -90.20],
                "END_PORT_AREA": ["Mobile", "Memphis"],
                "END_PORT_LAT": [30.69, 35.15],
                "END_PORT_LON": [-88.04, -90.05],
                "TIME_CREATED": [pd.Timestamp("2026-08-10T12:00:00Z"), pd.Timestamp("2026-01-10T12:00:00Z")],
                "ESTIMATED_DISTANCE": [100.0, 200.0],
                "ESTIMATED_SPEED": [10.0, 8.0],
                "LOCKS_HOURS_INCLUDED": [2.0, 3.0],
                "NUM_LOCKS": [2, 4],
                "SERVICE_TYPE": [2, 1],
                "route_min_width": [50.0, 100.0],
                "route_min_length": [200.0, 400.0],
                "route_dur_h": [12.0, 30.0],
                "TUG_BREADTH": [25.0, 50.0],
                "TUG_LENGTH": [100.0, 300.0],
                "EDITED_ROUTE": [False, True],
                "TIME_IN_START_PORT": [None, None],
                "TIME_IN_END_PORT": [None, None],
                "IDLE_FUEL_PER_DAY": [1.0, 1.0],
                "FUEL_BURN_PER_DAY": [2.0, 2.0],
                "TOTAL_DEADWEIGHT": [10.0, 20.0],
                "TOW_SIZE": [1.0, 2.0],
                "TOTAL_LOADED_TONS": [100.0, 200.0],
                "CAPACITY_WEIGHT": [200.0, 400.0],
                "TOTAL_LOADED_BARRELS": [300.0, 400.0],
                "CAPACITY_BARRELS": [600.0, 800.0],
                "START_PORT_ID": [1, 2],
                "END_PORT_ID": [3, 4],
            }
        )

        feat = infer.build_features(legs)

        self.assertEqual(feat.loc[0, "RIVER_SYSTEM"], "GulfCoast")
        self.assertEqual(feat.loc[1, "RIVER_SYSTEM"], "UpperMiss")
        self.assertEqual(feat.loc[0, "season4"], "summer")
        self.assertEqual(feat.loc[1, "season4"], "winter")
        self.assertEqual(feat.loc[0, "season_river"], "summer_GulfCoast")
        self.assertEqual(feat.loc[1, "season_river"], "winter_UpperMiss")
        self.assertEqual(int(feat.loc[0, "hurricane_gulf"]), 1)
        self.assertEqual(int(feat.loc[1, "ice_season"]), 1)
        self.assertEqual(int(feat.loc[1, "spring_flood"]), 0)
        self.assertTrue(np.isclose(feat.loc[0, "squeeze_beam"], 0.5))
        self.assertTrue(np.isclose(feat.loc[0, "squeeze_len"], 0.5))
        self.assertTrue(np.isclose(feat.loc[1, "squeeze_len"], 0.75))


if __name__ == "__main__":
    unittest.main()
