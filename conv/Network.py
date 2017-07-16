#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Sigmoid激活函数类
from FullConnectedLayer import *
import numpy as np
from conv import *
from activator import *
from pool import *
import sys
# 神经网络类
class Network(object):
    def __init__(self):
        '''
        构造函数
        '''
        self.layers = []
        self.convLayer = ConvLayer(28, 28, 
                 1, 6, 
                 6, 1, 
                 1, 2, ReluActivator(),
                 0.001)
        self.layers.append(self.convLayer)

        self.poolLayer = MaxPoolingLayer(13, 13, 
                 1, 4, 
                 4, 1)
        self.layers.append(self.poolLayer)

        self.fcLayer1 = FullConnectedLayer(
                100, 300, 0.001,
                SigmoidActivator())
        self.layers.append(self.fcLayer1)

        self.fcLayer2 = FullConnectedLayer(
                300, 10, 0.001,
                SigmoidActivator())
        self.layers.append(self.fcLayer2)
            
    def predict(self, sample):
        '''
        使用神经网络实现预测
        sample: 输入样本
        '''
        output = sample
        self.convLayer.forward(output)
        output = self.convLayer.output

        self.poolLayer.forward(output)
        output = self.poolLayer.output
        output = output.reshape((output.size, 1))

        self.fcLayer1.forward(output)
        output = self.fcLayer1.output

        self.fcLayer2.forward(output)
        output = self.fcLayer2.output

        return output


    def train(self, labels, data_set, epoch):
        '''
        训练函数
        labels: 样本标签
        data_set: 输入样本
        epoch: 训练轮数
        '''
        for i in range(epoch):
            for d in range(len(data_set)):
                temp = np.array(labels[d])
                l = temp.reshape(len(temp), -1)
                temp = np.array(data_set[d])
                d = temp.reshape((1, 28, 28))
                self.train_one_sample(l, d)

    def train_one_sample(self, label, sample):
        self.predict(sample)
        self.calc_gradient(label)
        self.update_weight()

    def calc_gradient(self, label):
        delta = self.layers[-1].activator.backward(
            self.layers[-1].output
        ) * (label - self.layers[-1].output)

        self.fcLayer2.backward(delta)
        delta = self.fcLayer2.delta

        self.fcLayer1.backward(delta)
        delta = self.fcLayer1.delta
        delta = delta.reshape((1, 10, 10))

        self.poolLayer.backward(delta)
        delta = self.poolLayer.delta

        self.convLayer.backward(delta)

    def update_weight(self):
        for layer in self.layers:
            layer.update()