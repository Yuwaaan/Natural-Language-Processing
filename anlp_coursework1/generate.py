import sys
#import json
import random

#here we make sure the user provides a model filename when
#calling this program, otherwise exit with a usage error.
if len(sys.argv) != 2:
    print("Usage: ", sys.argv[0], "<model_file>")
    sys.exit(1)
infile = sys.argv[1] #get input argument: the model file

model = {}
with open(infile) as f:
    for line in f:
        con = line[:2]
        if con not in model:
            model[con] = {}
        # print(line[2])
        model[con][line[2]] = float(line[4:])
        # print(model[con])

f.close()

def generate_from_LM(model):
    #print(json.dumps(model, indent=4, sort_keys=True))
    # for i, (k, v) in enumerate(model.items()):
    #     if i in range(0, 10000):
    #         print(k, v)

    output_str = "##"
    
    # print(current)

    i = 0
    while i < 298:
        last_two = output_str[-2:]
        # print("iteration ", i, "output str: " + output_str)
        # print("last two: " + last_two)
        
        #current = dict(sorted(model[last_two].items(), key=lambda data: data[1]))

        # if no trigrams in the given model start with the two characters
        if last_two not in model:
            output_str += "#"
            i += 1
        # if some trigrams start with the two characters
        else:
            current = model[last_two]
            #首先随机生成一个0，1之间的随机数，作为概率点。将每个字符的概率平铺在0-1区间内，x在哪个字符的概率范围内就生成哪个字符
            cum_prob = random.uniform(0, 1)
            # print("cum_prob=", cum_prob)
            # print(current)
            for c in current:   # ‘y', 'z', ...
                if cum_prob < current[c]:
                    output_str = output_str + c
                    i += 1
                    # print("c: ", c)
                    break
                else:
                    cum_prob -= current[c]
                    # print("cum_prob=", cum_prob)
        # print()

    print(output_str)

generate_from_LM(model)
