import numpy as np
import random
import math

random.seed(0)

def rand_val(range_val):
    val = (2*range_val)*random.random() - range_val
    return val

def activation_function(val,type):
    ret_val = 0
    if type=='sigmoid':
        ret_val = 1 + math.exp(-1*val)
        ret_val = 1/ret_val
    elif type=='tanh':
        ret_val = math.tanh(val)
    return ret_val

def diff_activation_function(val,type):
    ret_val = 0
    if type=='sigmoid':
        ret_val1 = 1 + math.exp(-1*val)
        ret_val1 = 1/ret_val1
        ret_val = ret_val1*(1-ret_val1)
    elif type=='tanh':
        ret_val1 = math.tanh(val)
        ret_val = 1-(ret_val1*ret_val1)

    return ret_val

def cost_func(y,y1,type):
    y = np.array(y)
    y1 = np.array(y1)
    if type == 'euclidean':
        y2 = np.square((np.subtract(y,y1)))
        y2 = np.sum(y2)
        y2=y2/2;
        y3 = np.subtract(y1,y)
    return y2,y3

class NN:
    def __init__(self,num_in,num_out,num_hid,lmda,activate,func):
        self.num_inputs = num_in;
        self.num_outputs = num_out
        self.num_hidden = num_hid
        self.lmda = lmda
        self.activation = activate
        self.loss_func = func
        self.n = []
        self.n = self.n + [num_in];
        self.n = self.n + [num_hid];
        self.n = self.n + [num_out];
        self.y = []
        self.eta = []
        self.delta = []
        for i in range(len(self.n)):
            self.y = self.y + [[0.5 for x in range(self.n[i])]]
            self.eta = self.eta + [[1 for x in range(self.n[i])]]
            self.delta = self.delta + [[0 for x in range(self.n[i])]]
        self.w = []
        for i in range(len(self.n)-1):
            temp = [[rand_val(0.5) for x in range(self.n[i+1])] for z in range(self.n[i])]
            temp = np.array(temp)
            self.w = self.w + [temp]

        self.y = np.asarray(self.y)
        self.eta = np.asarray(self.eta)
        self.delta = np.asarray(self.delta)

    def forward_propagate(self,data):
        if len(data)!=len(self.y[0]):
            return -1
        for i in range(len(self.y[0])):
            self.eta[0][i] = data[i]
            self.y[0][i] = activation_function(data[i],self.activation)
        for i in range(len(self.y)-1):
            if i == 0:
                continue
            for j in range(len(self.y[i])):
                temp = self.w[i-1]
                self.eta[i][j] = np.dot(self.y[i-1],self.w[i-1][:,j])
                self.y[i][j]= activation_function(np.dot(self.y[i-1],self.w[i-1][:,j]),self.activation)

        i=len(self.y)-1;
        for j in range(len(self.y[i])):
            temp = self.w[i-1]
            self.eta[i][j] = np.dot(self.y[i-1],self.w[i-1][:,j])
            self.y[i][j]=self.eta[i][j];
        return 0


    def back_propagate(self,labels):
        if len(labels)!=len(self.y[(len(self.y)-1)]):
            return -1,0
        error,ydiff = cost_func(self.y[len(self.y)-1],labels,self.loss_func)
        # print self.y[2][0]
        # print ydiff;
        # print error;
        for i in range(len(self.delta)-1,-1,-1):
            if i==len(self.delta)-1:
                for j in range(len(self.delta[i])):
                    self.delta[i][j] = ydiff[j];
            else:
                for j in range(len(self.delta[i])):
                    self.delta[i][j] = np.dot(self.delta[i+1],self.w[i][j])*diff_activation_function(self.eta[i][j],self.activation)

        for i in range(len(self.w)):
            for j in range(len(self.w[i])):
                for k in range(len(self.w[i][j])):
                    self.w[i][j][k] = self.w[i][j][k]+self.lmda*(self.delta[i+1][k]*self.y[i][j])

        return 0,error

    def run(self,input_data,label_data,iterations):
        if len(input_data)!=len(label_data):
            return -1;
        for i in range(iterations):
            error = 0
            for j in range(len(input_data)):
                print self.y
                flag = self.forward_propagate(input_data[j])
                if(flag==-1):
                    return -1
                flag,e1 = self.back_propagate(label_data[j])
                if(flag == -1):
                    return -1
                else:
                    error = error + e1
        return 0

    def predict(self,test_data):
        ret = []
        for i in range(len(test_data)):
            flag = self.forward_propagate(test_data[i])
            if(flag==-1):
                return -1
            ret = ret + self.y[len(self.y)-1]

        return ret

temp = NN(2,1,3,0.1,'tanh','euclidean')
input_data = [[1,1],[1,0],[0,1],[0,0]];
label_data = [[1],[0],[0],[0]];
test_data = [[1,1],[1,0],[0,1],[0,0]];
temp.run(input_data,label_data,100000)
y_predict = temp.predict(test_data)
print(y_predict)
