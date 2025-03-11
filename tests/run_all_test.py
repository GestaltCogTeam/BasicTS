import os
import unittest

# find all test files
test_loader = unittest.defaultTestLoader
dir_of_this_script = os.path.dirname(os.path.realpath(__file__))
top_dir = os.path.dirname(dir_of_this_script)
test_suite = test_loader.discover(start_dir=dir_of_this_script, pattern='test_*.py', top_level_dir=top_dir)

# run all tests
test_runner = unittest.TextTestRunner()
test_runner.run(test_suite)
