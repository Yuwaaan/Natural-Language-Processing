#!/usr/bin/env python
# coding: utf-8

import numpy as np
from collections import Counter
from keras.preprocessing.sequence import pad_sequences
import pickle
import platform
import os

import json
import numpy as np
import sys
from keras.models import Sequential, load_model
from keras import models
from keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from keras.preprocessing.sequence import pad_sequences
from keras_contrib.layers import CRF
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from pprint import pprint


from keras.layers import *
from keras.models import Model
import matplotlib.pyplot as plt



os.environ["CUDA_VISIBLE_DEVICES"] = "3"



def _parse_data(fh):
    split_text = '\n'

    string = fh.read().decode('utf-8')
    string = string.replace('\r', '')
#     print (string.strip().split(split_text + split_text))
    data = [[row.split() for row in sample.split(split_text)] for
            sample in
            string.strip().split(split_text + split_text)]
    
    fh.close()
    return data


def _process_data(data, vocab, chunk_tags, maxlen=None, onehot=False):
    print (chunk_tags,11111)
    if maxlen is None:
        maxlen = max(len(s) for s in data)
        print (maxlen)
    word2idx = dict((w, i+1) for i, w in enumerate(vocab))
    x = [[word2idx.get(w[0].lower(), 1) for w in s if len(w)==2] for s in data]
    print (chunk_tags)
    y_chunk = [[chunk_tags.index(w[1]) for w in s if len(w)==2] for s in data]


    y_chunk = pad_sequences(y_chunk, maxlen, value=-1)

    if onehot:
        y_chunk = numpy.eye(len(chunk_tags), dtype='float32')[y_chunk]
    else:
        y_chunk = numpy.expand_dims(y_chunk, 2)
    return x, y_chunk,word2idx


train = _parse_data(open('train_cq(1).data', 'rb'))
# val= _parse_data(open('data1/dev_data.data', 'rb'))
# train=train+val

fr = open('namecq.txt', 'r', encoding = 'utf-8')
dic = []
for line in fr.readlines():
    line = line.strip('\n')
    b = line.split(' ')
    dic.append(b)
dic = dict(dic)
labels = dic.keys()


chunk_tags = ['O']
for label in labels:
    chunk_tags.append('B-'+label)
    chunk_tags.append('I-'+label)
ner2label=dict(zip(chunk_tags,range(len(chunk_tags))))
word_counts = Counter(row[0].lower() for sample in train for row in sample)
vocab = [w for w, f in iter(word_counts.items()) if f >= 2]
X,y,word2idx=_process_data(train,vocab, chunk_tags,maxlen=80)
Y = [[ner2label[w[1]] for w in s if len(w)==2] for s in train]
count=0
x_new=[]
y_new=[]

for x,y in zip(X,Y):
    
    if set(y)==set([2]):
        print (111)
        count+=1
    else:
        x_new.append(x)
        y_new.append(y)
np.save("x.npy",x_new)
np.save("y.npy",y_new)



# train = _parse_data(open('trainlstm.txt', 'rb'))

X, y, word2idx = _process_data(train,vocab, chunk_tags,maxlen=80)







voc_size = len(word2idx.keys())
max_len = 32
tag_size =len(chunk_tags)
epoches = 50


word2num =word2idx

ner2label=dict(zip(chunk_tags,range(len(chunk_tags))))
label2ner = {
    w: k for k, w in ner2label.items()
}

def gen_datasets():

    X = np.load('x.npy')
    Y = np.load('y.npy')
    X_, Y_ = shuffle(X, Y)
    X_ = pad_sequences(X_, maxlen=max_len, value=0)
    Y_ = pad_sequences(Y_, maxlen=max_len, value=0)
    Y_ = np.expand_dims(Y_, 2)
    X_train, X_test, Y_train, Y_test = train_test_split(X_, Y_, test_size=0.2, random_state=0)
    return X_train, X_test, Y_train, Y_test



def train():
    inputs_sentence = Input(shape=(max_len,))
    em=Embedding(voc_size+1, 128, mask_zero=True)(inputs_sentence)
    blstm=Bidirectional(LSTM(64, return_sequences=True))(em)
    dr=Dropout(rate=0.5)(blstm)
    den=Dense(tag_size)(dr)
    crf = CRF(tag_size, sparse_target=True)
    c=crf(den)
    
    
    model = Model(inputs=inputs_sentence, outputs=c)
    model.summary()
    model.compile('adam', loss=crf.loss_function, metrics=[crf.accuracy])
    X_train, X_test, Y_train, Y_test = gen_datasets()
    # 可视化
    tb = TensorBoard(log_dir='./tb_logs/0914', histogram_freq=0, write_graph=True, write_images=False,
                     embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    cp = ModelCheckpoint("das.hdf5", monitor='val_crf_viterbi_accuracy', verbose=0,
                         save_best_only=False, save_weights_only=False, mode='auto', period=1)
    his=model.fit(X_train, Y_train, batch_size=100, epochs=epoches,
              validation_data=[X_test, Y_test], callbacks=[tb, cp])


    score = model.evaluate(X_test, Y_test, batch_size=100)

    model.save('keras_crf')
    return his




def create_custom_objects():

    instanceHolder = {"instance": None}

    class ClassWrapper(CRF):
        def __init__(self, *args, **kwargs):
            instanceHolder["instance"] = self
            super(ClassWrapper, self).__init__(*args, **kwargs)

    def loss(*args):
        method = getattr(instanceHolder["instance"], "loss_function")
        return method(*args)

    def accuracy(*args):
        method = getattr(instanceHolder["instance"], "accuracy")
        return method(*args)

    return {"ClassWrapper": ClassWrapper ,"CRF": ClassWrapper, "crf_loss": loss, "crf_viterbi_accuracy":accuracy}


if __name__ == '__main__':
    his=train()



num2word = {
    w: k for k, w in word2num.items()
}



model = load_model('das.hdf5', custom_objects=create_custom_objects())
X_train, X_test, Y_train, Y_test = gen_datasets()




def get_entiy(words, y_pred):

    fr = open('namecq.txt', 'r', encoding='utf-8')
    dic = []
    for line in fr.readlines():
        line = line.strip('\n')
        b = line.split(' ')
        dic.append(b)


    ner_dict = dict(dic)
    for k in ner_dict.keys():
        ner_dict[k] = ''
    ner_dict['O'] = ''

    ner_list = dict(dic)
    for k in ner_list.keys():
        ner_list[k] = []
    ner_list['O'] = []


    for i in range(len(y_pred)):

        ner = label2ner[y_pred[i]].split('-')[-1]
        ner_dict[ner] += words[i]
        for n,s in ner_dict.items():
            if n != ner and s:
                ner_list[n].append(s)
                ner_dict[n] = ''

    ner_list.pop('O')
    tmplist=[]

    for i in ner_list.keys():
        tmp=[j+'-'+i for j in ner_list[i]]
        tmplist.extend(tmp)
    return tmplist


#sub预测出的实体 ans真实实体
sub=[]
ans=[]

y_pre=np.argmax(model.predict(X_test), axis=-1)
print (y_pre)

for ind,i in enumerate(y_pre):
    words = "".join([num2word[l] for l in X_test[ind][X_test[ind] > 0]])
    y_pred = i[X_test[ind] > 0]
    sub.append(get_entiy(words, y_pred))

    y_pred = Y_test[ind][X_test[ind] > 0]
    y_pred = np.squeeze(y_pred)
    ans.append(get_entiy(words, y_pred))

res_ans=[]
for i in ans:
    res_ans.append(' '.join(i))

res_sub=[]
for i in sub:
    res_sub.append(' '.join(i))

def micro_f1(sub_lines, ans_lines, split=' '):

    correct=[]
    total_sub = 0
    total_ans = 0

    for sub_line, ans_line in zip(sub_lines, ans_lines):
        sub_line = set(sub_line.split(split))
        ans_line = set(ans_line.split(split))

        c = sum(1 for i in sub_line if i in ans_line) if sub_line != {''} else 0

        total_sub += len(sub_line) if sub_line != {''} else 0
        total_ans += len(ans_line) if ans_line != {''} else 0
        correct.append(c)

    p = np.sum(correct) / total_sub if total_sub != 0 else 0
    r = np.sum(correct) / total_ans if total_ans != 0 else 0

    f1 = 2*p*r /(p+r) if (p+r) != 0 else 0

    print ('total sub:', total_sub)
    print ('total ans:', total_ans)
    print ('correct:', np.sum(correct), correct)
    print ('precision:', p)
    print ('recall', r)
    print ('f1', f1)

    return f1,p,r

f1,p,r = micro_f1(res_ans, res_sub, split=' ')


plt.figure()
plt.title('Loss')
plt.plot(his.history['loss'], label='train') #绘制训练集损失
plt.plot(his.history['val_loss'], label='val') #绘制验证集损失
plt.legend(["train", "val"])
plt.show()



plt.figure()
plt.title('acc')
plt.plot(his.history['crf_viterbi_accuracy'], label='train') #绘制训练集损失
plt.plot(his.history['val_crf_viterbi_accuracy'], label='val') #绘制验证集损失
plt.legend(["train", "val"])
plt.show()


def sen2entitys(words):
    x =[word2idx.get(w.lower(), 1) for w in words]
    X_ = pad_sequences(np.array([x]), maxlen=max_len, value=0)
    y_pre=np.argmax(model.predict(X_), axis=-1)
    return [{"word":i.split("-")[0],"type":dic[i.split("-")[1]]} for i in get_entiy(words,y_pre[X_ >0])]

sen2entitys(words)




