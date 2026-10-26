import importlib
import os
import unittest


class ModelADefaultsTest(unittest.TestCase):
    def test_safe_model_a_defaults(self):
        old_target = os.environ.pop("MODEL_A_TARGET_MODE", None)
        old_priors = os.environ.pop("MODEL_A_USE_ENCODING_PRIORS", None)
        try:
            from src import config

            importlib.reload(config)
            self.assertEqual(config.MODEL_A_TARGET_MODE, "naive_residual")
            self.assertFalse(config.MODEL_A_USE_ENCODING_PRIORS)
        finally:
            if old_target is not None:
                os.environ["MODEL_A_TARGET_MODE"] = old_target
            if old_priors is not None:
                os.environ["MODEL_A_USE_ENCODING_PRIORS"] = old_priors
            importlib.reload(config)


if __name__ == "__main__":
    unittest.main()
