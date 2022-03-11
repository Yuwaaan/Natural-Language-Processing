import re
import sys
from random import random
from math import log
from collections import defaultdict


#provide a training filename
if len(sys.argv) != 2:
    print("Usage: ", sys.argv[0], "<training_file>")
    sys.exit(1)
infile = sys.argv[1] #get input argument: the training file


def preprocess_line(line):
    """
    Preprocess a line
    :param line: an original line in the training set.
    :return: Preprocessed line.
    """
    text = line.lower()  # lowercase
    comp = re.compile('[^a-z^0-9^ ^#^.]')
    text = comp.sub('', text)   # remove unnecessary characters
    digcov = re.compile('[0-9]')
    text = digcov.sub('0', text)   # convert all digits to '0'
    text = "##" + text + "#"  # add'##'at the beginning and '#' at the end of a line
    return text


tri_counts=defaultdict(int) #counts of all trigrams in input
bi_counts=defaultdict(int)  #counts of all bigrams in input
chartypes=defaultdict(int)  #counts of all characters in input

#read file
with open(infile) as f:
    for line in f:
        text= preprocess_line(line)
        #get all characters in training set
        for c in text:
            if c not in chartypes:
                chartypes[c] = 1
        #get all trigrams in training set
        for j in range(len(text)-(3)):
            trigram = text[j:j+3]
            tri_counts[trigram] += 1
        #get all bigrams in training set
        for j in range(len(text)-(2)):
            bigram = text[j:j+2]
            bi_counts[bigram] += 1
f.close()


# put all the characters in one list
charlist=list()
for k in chartypes:
    charlist.append(k)
#All characters are arranged arbitrarily to form trigrams.
#There are 30 * 30 * 30 = 27,000 trigrams in total, and
#all trigrams probabilities are initialized to 0
for i in charlist:
    for j in charlist:
        for k in charlist:
            string = i + j + k
            if string not in tri_counts:
                tri_counts[string] = 0



def count_to_prob(tri_counts, ext):
    """
    Use add-alpha smoothing to estimate the probability

    :param tri_counts: A dictionary, key is each trigram,and
           value is the number of trigrams in the training set.
    :param ext: The suffix of the infile, that is, the language
            of the training set
    :return: A dictionary, key is each trigram, and value is
    the estimated probability obtained by trigram using the
    add alpha smoothing method.
    """
    if(ext=="en"):  #infile is an English training set
        alpha = 0.08
    elif(ext=="de"):  #infile is an German training set
        alpha = 0.10
    elif(ext=="es"):  #infile is an Spanish training set
        alpha = 0.09
    else:
        alpha = 0.09
    chartype = len(chartypes)    #calculate character type
    # construct probability estimation dictionary
    tri_probs = defaultdict(int)
    for trigram in tri_counts:
        tri_key = list(trigram)
        tri_key.pop()
        bigram = "".join(tri_key)
        #calculate using the add alpha method
        probs = (tri_counts[trigram] + alpha) / (bi_counts[bigram] + alpha * chartype)
        # store probability in dictionary
        tri_probs[trigram] = probs
    return tri_probs

#Calculate probability
tri_probs = count_to_prob(tri_counts, infile.split(".")[-1])



#write outefile
outfile = "model-" + infile
with open(outfile, 'w', encoding='utf-8') as new_file:
    for trigram in sorted(tri_probs.keys()):  # trigrams sort by characters
        new_file.write(trigram + '\t' + str(tri_probs[trigram]) + '\n')