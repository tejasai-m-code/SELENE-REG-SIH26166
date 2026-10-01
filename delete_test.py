import re
with open('tests/test_phase7_evidence.py', 'r') as f:
    text = f.read()

# I will just write a python script that replaces the whole file with a correctly formatted one
import os
os.system('del tests\test_phase7_evidence.py')
