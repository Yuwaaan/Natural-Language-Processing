#!/usr/bin/env python
# coding: utf-8


txt = [i.strip() for i in open("helpme.txt", encoding='utf-8').readlines()]

target = [i.split(" ")[1:] for i in txt]




train_data = [i.split(" ")[0][9:] for i in txt]
train_data = train_data[:387] + train_data[388:]
target = target[:387] + target[388:]




labelall = []
for ind, i in enumerate(target):
    for j in i:
        if j == "":
            print(ind)
        labelall.append(j)




label = list(set(labelall))




wuindex = label.index("无")








from tqdm import tqdm

new_target = []
traindata = []
for i in range(len(target)):

    tmp = [0] * len(label)
    for j in target[i]:
        tmp[label.index(j)] = 1
    if j != "无":
        new_target.append(tmp)
        traindata.append(train_data[i])
target = new_target









import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
mode = 0
maxlen = 24  # 文本最大长度截取到24
learning_rate = 5e-5
min_learning_rate = 1e-5
config_path = 'C:/Users/zhangyiwen07/Downloads/chinese_rbt3_L-3_H-768_A-12/bert_config_rbt3.json'
checkpoint_path = 'C:/Users/zhangyiwen07/Downloads/chinese_rbt3_L-3_H-768_A-12/bert_model.ckpt'
dict_path = 'C:/Users/zhangyiwen07/Downloads/chinese_rbt3_L-3_H-768_A-12/vocab.txt'
modelname = "bert_duo.hdf5"




import json
import numpy as np
from keras.utils import to_categorical
from random import choice
from tqdm import tqdm
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from keras.layers import Dense, Activation, Multiply, Add, Lambda
import re, os
import codecs
import jieba.posseg as pseg
import jiagu
import jieba
import keras
import keras.backend as K

from keras.callbacks import *
import os

token_dict = {}

with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)


class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')  # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]')  # 剩余的字符是[UNK]
        return R


# 输出是 ['[CLS]', u'今', u'天', u'天', u'气', u'不', u'错', '[SEP]']
tokenizer = OurTokenizer(token_dict)




# 数据生成器和模型部分
from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam


# 让每条文本的长度相同，用0填充
def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])





from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(traindata, target, test_size=0.1, random_state=678)




X1, X2, Y = [], [], []
for i in X_train:
    #     d = self.data[i]
    text = i[:maxlen]
    x1, x2 = tokenizer.encode(first=text)
    X1.append(x1)
    X2.append(x2)
X1 = seq_padding(X1)
X2 = seq_padding(X2)
X_train = [X1, X2]




X1, X2, Y = [], [], []
for i in X_test:
    #     d = self.data[i]
    text = str(i)[:maxlen]
    x1, x2 = tokenizer.encode(first=text)
    X1.append(x1)
    X2.append(x2)

X1 = seq_padding(X1)
X2 = seq_padding(X2)
X_test = [X1, X2]





def build_bert(nclass):
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)  # 加载预训练模型

    for l in bert_model.layers:
        l.trainable = True

    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))

    x = bert_model([x1_in, x2_in])
    x = Lambda(lambda x: x[:, 0])(x)  # 取出[CLS]对应的向量用来做分类
    x = Dropout(0.5)(x)
    p = Dense(nclass, activation='sigmoid')(x)

    model = Model([x1_in, x2_in], p)
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(1e-4),  # 用足够小的学习率
                  metrics=['accuracy'])
    print(model.summary())
    return model


model = build_bert(len(label))
early_stopping = EarlyStopping(monitor='val_acc', patience=3)  # 早停法，防止过拟合
plateau = ReduceLROnPlateau(monitor="val_acc", verbose=1, mode='max', factor=0.5, patience=2)  # 当评价指标不在提升时，减少学习率
checkpoint = ModelCheckpoint(modelname, monitor='val_acc', verbose=2, save_best_only=True, mode='max',
                             save_weights_only=True)  # 保存最好的模型

# 模型训练
model.fit(X_train, np.array(y_train),
          epochs=20, batch_size=64,
          callbacks=[early_stopping, plateau, checkpoint], validation_data=(X_test, np.array(y_test)))

# In[268]:


# 模型训练
model.fit(X_train, np.array(y_train),
          epochs=20, batch_size=64,
          callbacks=[early_stopping, plateau, checkpoint], validation_data=(X_test, np.array(y_test)))




model.load_weights(modelname)




testpre = model.predict(X_test)





def getres(testpre):
    pre = []

    for i in range(len(testpre)):
        tmp = []
        for j in np.where(np.array(testpre[i]) > 0.4)[0]:
            #             print (j)
            tmp.append(label[j])

        pre.append(tmp)
    return pre





pre = getres(testpre)
true = getres(np.array(y_test))




count = 0
allcount = 0
for i in range(len(pre)):
    if pre[i] == true[i]:
        count += 1
    allcount += 1
print("如果标签全部对应acc为:", count / allcount)





def micro_f1(sub_lines, ans_lines, split=' '):
    correct = []
    total_sub = 0
    total_ans = 0
    for sub_line, ans_line in zip(sub_lines, ans_lines):
        sub_line = sub_line
        ans_line = ans_line
        c = sum(1 for i in sub_line if i in ans_line) if sub_line != {''} else 0
        total_sub += len(sub_line) if sub_line != {''} else 0
        total_ans += len(ans_line) if ans_line != {''} else 0
        correct.append(c)
    p = np.sum(correct) / total_sub if total_sub != 0 else 0
    r = np.sum(correct) / total_ans if total_ans != 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) != 0 else 0
    print('total sub:', total_sub)
    print('total ans:', total_ans)
    print('correct: ', np.sum(correct), correct)
    print('precision: ', p)
    print('recall: ', r)
    print("f1:", f1)
    return f1, p, r


f1, p, r = micro_f1(pre, true, split=' ')




pre




true




true = getres(np.array(y_test))

