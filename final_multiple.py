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
    elif type=='none':
        ret_val = val
    elif type == 'relu':
        if(val>0):
            ret_val= val
        else:
            ret_val = 0


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
    elif type=='none':
        ret_val = val
    elif type=='relu':
        if(val>0):
            ret_val = 1
        else:
            ret_val = 0

    return ret_val

def cost_func(y,y1,type):
    y = np.array(y)
    y1 = np.array(y1)
    if type == 'euclidean':
        y2 = np.square((np.subtract(y,y1)))
        y2 = 0.5*np.sum(y2)
        y3 = np.subtract(y1,y)
        # y2 = math.sqrt(y2)

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
        self.n = self.n + num_hid;
        self.n = self.n + [num_out];
        self.y = []
        self.eta = []
        self.delta = []
        self.bias = []
        for i in range(len(self.n)):
            if i == 0:
                self.bias = self.bias + [[0 for x in range(self.n[i])]]
            else:
                self.bias = self.bias + [[0.5 for x in range(self.n[i])]]


        for i in range(len(self.n)):
            self.y = self.y + [[0.5 for x in range(self.n[i])]]
            self.eta = self.eta + [[1 for x in range(self.n[i])]]
            self.delta = self.delta + [[0 for x in range(self.n[i])]]

        self.w = []

        for i in range(len(self.n)-1):
            temp = [[rand_val(0.5) for x in range(self.n[i+1])] for z in range(self.n[i])]
            temp = np.array(temp)
            self.w = self.w + [temp]

        self.y = np.array(self.y)
        # self.w = np.asarray(self.w)
        self.eta = np.array(self.eta)
        self.delta = np.array(self.delta)

    def forward_propagate(self,data):
        if len(data)!=len(self.y[0]):
            return -1
        for i in range(len(self.y[0])):
            self.eta[0][i] = data[i]
            self.y[0][i] = activation_function(data[i],self.activation)
        for i in range(len(self.y)):
            if i == 0:
                continue

            for j in range(len(self.y[i])):

                temp = self.w[i-1]

                self.eta[i][j] = self.bias[i][j] + np.dot(self.y[i-1],self.w[i-1][:,j])
                if(j==len(self.y[i])-1):
                    self.y[i][j]= activation_function(self.eta[i][j],'sigmoid')
                else:
                    self.y[i][j] = activation_function(self.eta[i][j],self.activation)
        return 0


    def back_propagate(self,labels):
        if len(labels)!=len(self.y[(len(self.y)-1)]):
            return -1,0
        error,ydiff = cost_func(self.y[len(self.y)-1],labels,self.loss_func)

        # print ydiff
        for i in range(len(self.delta)-1,-1,-1):
            if i==len(self.delta)-1:
                for j in range(len(self.delta[i])):
                    self.delta[i][j] = ydiff[j]*diff_activation_function(self.eta[i][j],'sigmoid')

                    # print self.delta[i][j]
            else:
                for j in range(len(self.delta[i])):
                    self.delta[i][j] = np.dot(self.delta[i+1],self.w[i][j])*diff_activation_function(self.eta[i][j],self.activation)
                    # print self.delta[i][j]

        for i in range(len(self.bias)):
            for j in range(len(self.bias[i])):
                self.bias[i][j] = self.bias[i][j]+ self.lmda*self.delta[i][j]


        for i in range(len(self.w)):
            for j in range(len(self.w[i])):
                for k in range(len(self.w[i][j])):
                    self.w[i][j][k] = self.w[i][j][k]+self.lmda*(self.delta[i+1][k]*self.y[i][j])

        return 0,error

    def run(self,input_data,label_data,iterations):
        if len(input_data)!=len(label_data):
            return -1;
        for i in range(iterations):
            print "iteration No:",i+1
            error = 0
            count_data = 0
            for j in range(len(input_data)):
                count_data = count_data+1;
                flag = self.forward_propagate(input_data[j])
                # print self.w
                if(flag==-1):
                    return -1
                temp = np.zeros(self.num_outputs)
                temp[label_data[j]]=1
                flag,e1 = self.back_propagate(temp)
                if(flag == -1):
                    return -1
                else:
                    error = error + e1
            # print "W: "
            # print self.w
            print "Error:", error/count_data
                # print self.y[len(self.y)-1],e1

            # print error

        return 0

    def predict(self,test_data):
        ret = []
        for i in range(len(test_data)):
            flag = self.forward_propagate(test_data[i])
            if(flag==-1):
                return -1
            # print self.y[len(self.y)-1]
            temp = self.y[len(self.y)-1]
            lab = np.argmax(temp)
            # print lab
            ret = ret + [lab]
        # print ret
        return ret

# temp = NN(2,2,3,0.5,'tanh','euclidean')
# input_data = [[1,1],[1,0],[0,1]]
# label_data = [[0,1],[1,0],[1,0]]
# test_data = [[1,0],[1,1]]
# temp.run(input_data,label_data,1000)
# y_predict = temp.predict(test_data)
# print(y_predict)
