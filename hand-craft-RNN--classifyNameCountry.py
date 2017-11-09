part1:读入以及准备数据;
==
import os
from io import open
import string
#from compiler.ast import flatten
#将数据读入到一个字典里;
fileList=os.listdir('/home/ly/pytorch_exercise/data/data for classify names')
for i in range(len(fileList)):
    fileList[i]=fileList[i].split('.')[0]
filepath='/home/ly/pytorch_exercise/data/data for classify names/'
dic={}

#不同语言之间字符编码格式转换;
import unicodedata
all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1 # Plus EOS marker
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

for files in fileList:
    dic[files]=open(filepath+files+'.txt').read().strip().split('\n')
for files in fileList:
    for i in range(len(dic[files])):
        dic[files][i]=unicodeToAscii(dic[files][i])
#现在数据都存储在dic字典中，编码格式也都转化过了，下一步将其转化为向量;
tensorDic={}
def wordToVec(s):
    temp=[]
    for i in range(len(s)):
        temp.append([0 for j in range(56)])
    for i in range(len(s)):
        for j in range(56):
            if all_letters[j]==s[i]:
                temp[i][j]=1
                break
    return temp

import torch
def wordToTensor(s):
    temp=torch.zeros(len(s),1,n_letters)
    for i in range(len(s)):
        for j in range(56):
            if all_letters[j]==s[i]:
                temp[i][0][j]=1
                break
    return temp
def categoryToTensor(n):
    temp=torch.zeros(18)
    temp[n-1]=1
    return temp

import random
from torch.autograd import Variable
def randomChoice():
    #返回随即抽取的数据和label；
    n=random.randint(0,17)
    target=dic.keys()[n]
    #targetTensor=categoryToTensor(n)
    targetTensor=torch.Tensor([n]).long()
    m=random.randint(0,len(dic[dic.keys()[n]])-1)
    trainData=dic[dic.keys()[n]][m]
    trainDataTensor=wordToTensor(dic[dic.keys()[n]][m])
    return target,trainData,Variable(targetTensor),Variable(trainDataTensor);
    #训练的时候每次抽一个放进去;
    
part2:hand craft a simple RNN;
==
import torch.nn as nn
from torch.autograd import Variable
#喂的时候是名字中的一个字符作为一个时刻喂进去的;
class RNN(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(RNN,self).__init__()
        self.hidden_size=hidden_size
        self.i2h=nn.Linear(input_size+hidden_size,hidden_size)
        self.i2o=nn.Linear(input_size+hidden_size,output_size)
        self.softmax=nn.LogSoftmax()
    def forward(self,input,hidden):     #input_size和input的输入维度是不同的,input_size应该是一维的,input应该是多维的吧？？
        combined=torch.cat((input,hidden),1)
        hidden=self.i2h(combined)
        output=self.i2o(combined)
        output=self.softmax(output)
        return output,hidden
    def initHidden(self):
        return Variable(torch.zeros(1,self.hidden_size))   
        #这里应该是1*1*self.hidden_size还是1*self.hidden_size？？设置成1*1*self.hidden_size在torch.cat()过程中会出错!
n_hidden=128
n_categories=len(dic.keys())
rnn=RNN(n_letters,n_hidden,n_categories)

criterion = nn.NLLLoss()

part3:组织训练以及结果展示;
==
learn_rate=0.005
def train(categoryTensor,wordTensor):
    hidden=rnn.initHidden()
    rnn.zero_grad()
    for i in range(wordTensor.size()[0]):
        output,hidden=rnn(wordTensor[i],hidden)
    #output=output.view(18)
    #categoryTensor=categoryTensor.long()
    loss=criterion(output,categoryTensor)
    loss.backward()
    for p in rnn.parameters():
        p.data.add_(-learn_rate,p.grad.data)
    return output,loss.data[0]

import time
import math
def categoryFromOutput(output):
    top_n, top_i = output.data.topk(1) # Tensor out of Variable with .data
    category_i = top_i[0][0]
    return dic.keys()[category_i], category_i
def timeSince(beginTime):
    now=time.time()
    s=now-beginTime
    m=math.floor(s/60)
    s=s-m*60
    return '%dm %ds' % (m,s)
beginTime=time.time()
n_iters=100000
print_every=1000
plot_every=1000
current_loss = 0
all_losses = []
for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomChoice()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss
    # Print iter number, loss, name and guess
    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f / %s %s' % (iter, iter / n_iters * 100, timeSince(beginTime), loss,  guess, correct))

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0
