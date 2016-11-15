import tensorflow as tf
import numpy as np
import os

import data_utils

test_filename = "pnp-test.txt"

file_path = os.path.join(os.path.abspath(os.path.curdir), "data", test_filename)
f = open(file_path, 'r', encoding = "ISO-8859-1")
data = list(f.readlines())
f.close()
data = [s.strip().split() for s in data]

proper_nouns_strings = [" ".join(d[1:]) for d in data]

print(proper_nouns_strings)

f = open("output.txt", 'w', encoding = "ISO-8859-1")
for noun in proper_nouns_strings:
    f.write("Example: "+noun+" guess="+" drug "+" gold= confidence="+str(10)+"\n")
f.close()
