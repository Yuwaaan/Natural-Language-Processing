#Here are some libraries you're likely to use. You might want/need others as well.
import re
import sys
from random import random
from math import log
from collections import defaultdict


tri_counts=defaultdict(int) #counts of all trigrams in input
bi_counts=defaultdict(int)
chartypes=defaultdict(int)


def preprocess_line(line):
    text = line.lower()
    comp = re.compile('[^A-Z^a-z^0-9^ ^#^.]')
    text = comp.sub('', text)
    digcov = re.compile('[0-9]')
    text = digcov.sub('0', text)
    return text



#here we make sure the user provides a training filename when
#calling this program, otherwise exit with a usage error.
if len(sys.argv) != 2:
    print("Usage: ", sys.argv[0], "<training_file>")
    sys.exit(1)

infile = sys.argv[1] #get input argument: the training file
# python ./assignment1.py training.en



#This bit of code gives an example of how you might extract trigram counts
#from a file, line by line. If you plan to use or modify this code,
#please ensure you understand what it is actually doing, especially at the
#beginning and end of each line. Depending on how you write the rest of
#your program, you may need to modify this code.
with open(infile) as f:
    for line in f:
        line = preprocess_line(line)
        for c in line:
            if c not in chartypes:
                chartypes[c] = 1

        for j in range(len(line)-(3)):
            trigram = line[j:j+3]
            tri_counts[trigram] += 1

        for j in range(len(line)-(2)):
            bigram = line[j:j + 2]
            bi_counts[bigram] += 1

# chartypes to chardict
chardict=dict()
i = 0
for k in chartypes:
    chardict[i] = k
    i += 1

# add 0 counts
for i in range(len(chardict)):
    for j in range(len(chardict)):
        for k in range(len(chardict)):
            string = chardict[i]+chardict[j]+chardict[k]
            if string not in tri_counts:
                tri_counts[string] = 0
print (tri_counts)

#Some example code that prints out the counts. For small input files
#the counts are easy to look at but for larger files you can redirect
#to an output file (see Lab 1).
# print("Trigram counts in ", infile , ", sorted alphabetically:")
# for trigram in sorted(tri_counts.keys()):   # 按字母排序
#     print(trigram, ": ", tri_counts[trigram])
# print("Trigram counts in ", infile , ", sorted numerically:")
# for tri_count in sorted(tri_counts.items(), key=lambda x:x[1], reverse = True):  # 按次数排序
#     print(tri_count[0], ": ", str(tri_count[1]))



def count_to_prob(tri_counts):
    # tot_counts = sum(counts.values())
    # for k in counts:
    #     counts[k] = counts[k] / tot_counts
    # probs = dict([(k, counts[k]/tot_counts) for k in counts.keys()])
    alpha = 0.01
    chartype = len(chartypes)
    tri_probs = defaultdict(int)
    for trigram in tri_counts:
        tri_key = list(trigram)
        tri_key.pop()
        bigram = "".join(tri_key)
        probs = (tri_counts[trigram] + alpha) / (bi_counts[bigram] + alpha * chartype)
        tri_probs[trigram] = probs
    return tri_probs

tri_probs = count_to_prob(tri_counts)
# print(tri_probs)

# with open('demo_tri.txt', 'w', encoding='utf-8') as new_file:
#     for tri_prob in sorted(tri_probs.items(), key=lambda x: x[1], reverse=True):  # 按次数排序
#         new_file.write(tri_prob[0] + ": " + str(tri_prob[1]) + "\n")

with open('train_tri.txt', 'w', encoding='utf-8') as new_file:
    for trigram in sorted(tri_probs.keys()):  # 按字母排序
        new_file.write(trigram + ": " + str(tri_probs[trigram]) + "\n")