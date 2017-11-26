from final import *
import numpy as np
import random
import sys
import csv


train_file = sys.argv[1]
test_file = sys.argv[2]
def Accuracy(y1,y2):
    temp = 0;
    if(len(y1)!=len(y2)):
        return -1
    else:
        for i in range(len(y1)):
            if(y1[i]==y2[i]):
                temp = temp +1
        temp2 = len(y1)
        return (temp*1.0)/temp2



y_train = []
y_test = []
x_train = []
x_test = []
cnt_train = 0
cnt_test = 0
# with open(train_file,'r') as f:
#     read_words = f.read()
#     read_words = read_words.split('\n')
#     read_words = read_words[:-1]
#     for t in read_words:
#         words = t.split(',')
#         x_train = x_train + [words[:-1]]
#         y_train = y_train + [int(words[-1])]
#
# with open(test_file,'r') as f:
#     read_words = f.readlines()
#     read_words = read_words[:-1]
#     for t in read_words:
#         words = t.split(',')
#         x_test = x_test + [words[:-1]]
#         y_test = y_test + [int(words[-1])]


with open(train_file,'r') as f:
    reader = csv.reader(f)
    for row in reader:
        cnt_train = cnt_train + 1
        y_train = y_train + [[int(row[0])]]
        b = (row[1:28*28+1])
        x_train = x_train + [b]

with open(test_file,'r') as f:
    reader = csv.reader(f)
    for row in reader:
        y_test = y_test + [int(row[0])]
        cnt_test = cnt_test + 1
        x_test = x_test + [row[1:28*28+1]]


x_train = x_train
y_train = y_train
x_train = np.array(x_train,dtype = float);
x_test = np.array(x_test,dtype = float)

print len(x_train)

temp = NN(len(x_train[0]),10,10,0.3,'tanh','euclidean');
temp.run(x_train,y_train,2)
y_predict = temp.predict(x_test)
ans = y_predict
print ans
print y_test
acc = Accuracy(y_test,ans);
print acc
