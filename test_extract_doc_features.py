import unittest
import os
from extract_doc_features import extract_doc_features


class TestExtractDocFeatures(unittest.TestCase):
    INPUTS_FOLDER = "inputs"
    VALIDATION_FOLDER = "validation"

    def test_empty(self):
        with os.scandir(self.INPUTS_FOLDER) as entries:
            for entry in entries:
                with self.subTest():
                    print(entry)
                    self.assertTrue(True)
        # for p1, p2 in param_list:
        #     with self.subTest():
        #         self.assertEqual(p1, p2)
        # self.assertRaises(TypeError, Polynomial)


if __name__ == '__main__':
    TestExtractDocFeatures.INPUTS_FOLDER = os.getenv("INPUTS_FOLDER", default=TestExtractDocFeatures.INPUTS_FOLDER)
    TestExtractDocFeatures.VALIDATION_FOLDER = os.getenv("VALIDATION_FOLDER", default=TestExtractDocFeatures.VALIDATION_FOLDER)
    unittest.main()
