import unittest

import self_speculation


class PackageSmokeTest(unittest.TestCase):
    def test_package_is_importable(self) -> None:
        self.assertEqual(self_speculation.__version__, "0.1.0")


if __name__ == "__main__":
    unittest.main()
