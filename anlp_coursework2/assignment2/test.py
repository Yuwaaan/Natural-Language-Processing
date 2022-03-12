import pickle
import pprint


# file=open(".\data\en_ewt-ud-embeds-first_token.pkl","rb")
# data=pickle.load(file)
# pprint.pprint(data)
# file.close()

import json
# file_path = ".\data\en_ewt-ud-train.json"
# with open(file_path,'r')as f:
#     data, =json.load(f)
#     print(data)

tokens_list = []
tags_list = []

for line in open('.\data\en_ewt-ud-test.json','r'):
    tokens_list.append(json.loads(line)['tokens'])
    tags_list.append(json.loads(line)['tags'])
res = []
for i in range(len(tokens_list)):
    for j in range(len(tokens_list[i])):
        # print(tokens_list[i][j] + ':' + tags_list[i][j])
        res.append(tokens_list[i][j])
print(len(res)) # 25098

