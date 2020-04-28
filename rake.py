# Code requires rake-nltk to be installed in the directory.

import rake
import operator

sample_file = open("Pre-processed Text file already containing keywords.txt", 'r')
text = sample_file.read()
keywords = rake_object.run(text)
print "Keywords:", keywords