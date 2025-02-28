import unittest
from gradescope_utils.autograder_utils.decorators import weight
import os

class TestSubmission(unittest.TestCase):
    @weight(0)
    def test_ipynb_file_submitted(self):
        """Check if exactly one .ipynb file is submitted in any nested directory"""
        submission_dir = '/autograder/submission'
        ipynb_files = []

        # Walk through all directories and subdirectories.
        for root, dirs, files in os.walk(submission_dir):
            for file in files:
                if file.endswith('.ipynb'):
                    ipynb_files.append(os.path.join(root, file))

        # Check that there is exactly one .ipynb file.
        self.assertEqual(len(ipynb_files), 1, f"Expected 1 .ipynb file, found {len(ipynb_files)}: {ipynb_files}")
