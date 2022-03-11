import sys
import re


#here we make sure the user provides a test filename when
#calling this program, otherwise exit with a usage error.
if len(sys.argv) != 3:
    print("Usage: ", sys.argv[0], "<test_file>", "<model_file>")
    sys.exit(1)
test_file = sys.argv[1] #get input argument: the test file
model_file = sys.argv[2] #get input argument: the model file

# read model
model = {}
with open(model_file) as f:
    for line in f:
        con = line[:2]
        if con not in model:
            model[con] = {}
        model[con][line[2]] = float(line[4:])

N = 0

def preprocess_line(line):
    text = line.lower()
    comp = re.compile('[^A-Z^a-z^0-9^ ^#^.]')
    text = comp.sub('', text)
    digcov = re.compile('[0-9]')
    text = digcov.sub('0', text)
    return text

pp = 1
lines = []

with open(test_file) as f:
    for line in f:
        pro_line = preprocess_line(line)
        newline = "##" + pro_line + "#"
        N += len(newline)-2
        lines.append(newline)
        print(newline, "N=", N)
f.close()

for text in lines:
    #print(text)
    for j in range(len(text)-2):
        tri = text[j:j+3]
        tri_prob = model[tri[:2]][tri[2]]
        if tri_prob == 0:
            sys.exit(1)
        pp *= tri_prob**(-1/N)
        #print("tri: ", tri, "tri_prob: ", tri_prob, ". pp: ", pp)
        if pp == 0:
            print("Something wrong. pp=0")
            print("line: ", line)
            print("tri: ", tri)
            sys.exit(1)

print("final pp: ", pp)
