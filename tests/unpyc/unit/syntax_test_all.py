import os
import unittest

from test_utilities import make_tests_from_folder


class DecompileTestAll(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.maxDiff = 10000


test_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'test_cases'
)
make_tests_from_folder(test_dir, DecompileTestAll)
