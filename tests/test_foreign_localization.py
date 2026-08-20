import unittest

from foreign_localization import (
    parse_frame_index,
    parse_normalized_point,
)


class ForeignLocalizationTest(unittest.TestCase):
    def test_parse_point(self):
        self.assertEqual(
            parse_normalized_point("target <point>412, 287</point>"), (412, 287)
        )
        self.assertEqual(parse_normalized_point("target (412,287)"), (412, 287))
        self.assertIsNone(parse_normalized_point("Normal"))
        self.assertIsNone(parse_normalized_point("<point>1200, 10</point>"))
        self.assertEqual(
            parse_normalized_point("not Normal: <point>412, 287</point>"), (412, 287)
        )

    def test_parse_video_frame_index(self):
        self.assertEqual(parse_frame_index("frame 2 <point>400,300</point>"), 1)
        self.assertEqual(parse_frame_index("第3帧 <point>400,300</point>"), 2)
        self.assertEqual(parse_frame_index('{"frame_num": 1}'), 0)

    def test_frame_index_defaults_to_last_frame(self):
        self.assertEqual(parse_frame_index("<point>400,300</point>"), 3)
        self.assertEqual(parse_frame_index("frame 9 <point>400,300</point>"), 3)


if __name__ == "__main__":
    unittest.main()
