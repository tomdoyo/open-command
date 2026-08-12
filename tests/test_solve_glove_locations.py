import unittest

import pandas as pd

from src.solve_glove_locations import clean_teleports


class CleanTeleportsTest(unittest.TestCase):
    def test_unsorted_rows_match_chronological_input(self):
        detections = pd.DataFrame({
            "frame_idx": [0, 1, 2, 3, 4, 5],
            "x_in": [0.0, 1.0, 2.0, 7.0, 4.0, 5.0],
            "z_in": [0.0] * 6,
        })
        shuffled = detections.iloc[[0, 1, 2, 4, 5, 3]]

        expected = clean_teleports(detections, fps=60)["frame_idx"].tolist()
        actual = clean_teleports(shuffled, fps=60)["frame_idx"].tolist()

        self.assertEqual(expected, [0, 1, 2, 4, 5])
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
