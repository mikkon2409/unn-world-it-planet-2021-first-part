import unittest
import os
import json
import Levenshtein

from extract_doc_features import extract_doc_features


class TestExtractDocFeatures(unittest.TestCase):
    INPUTS_FOLDER = "inputs"
    VALIDATION_FOLDER = "validation"

    def test_empty(self):
        with os.scandir(self.INPUTS_FOLDER) as entries:
            for entry in entries:
                with self.subTest(entry.name):
                    root = os.path.splitext(entry.name)[0]

                    with open(os.path.join(self.VALIDATION_FOLDER, f"{root}.json")) as json_file:
                        data = json.load(json_file)
                        print(data)

                    output = extract_doc_features(os.path.join(self.INPUTS_FOLDER, entry.name))
                    self.assertEqual(data["red_areas_count"], output["red_areas_count"])
                    self.assertEqual(data["blue_areas_count"], output["blue_areas_count"])
                    self.assertEqual(data["table_cells_count"], output["table_cells_count"])
                    val_title, out_title = data["text_main_title"], output["text_main_title"]
                    val_text, out_text = data["text_block"], output["text_block"]

                    def text_evaluation(first, second):
                        distance = Levenshtein.distance(first, second)
                        norm_distance = distance / max(len(val_title), len(out_title))
                        return norm_distance

                    self.assertEqual(text_evaluation(val_title, out_title), 0.0)
                    self.assertEqual(text_evaluation(val_text, out_text), 0.0)


if __name__ == '__main__':
    TestExtractDocFeatures.INPUTS_FOLDER = os.getenv("INPUTS_FOLDER",
                                                     default=TestExtractDocFeatures.INPUTS_FOLDER)
    TestExtractDocFeatures.VALIDATION_FOLDER = os.getenv("VALIDATION_FOLDER",
                                                         default=TestExtractDocFeatures.VALIDATION_FOLDER)
    unittest.main()
