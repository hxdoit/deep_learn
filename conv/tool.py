# -*- coding:utf-8 -*-
import numpy as np
class IdentityActivator(object):
    def forward(self, weighted_input):
        #return weighted_input
        return weighted_input
    def backward(self, output):
        return 1
# 对numpy数组进行element wise操作
def element_wise_op(array, op):
    for i in np.nditer(array,
                       op_flags=['readwrite']):
        i[...] = op(i)
def conv(input_array, 
         kernel_array,
         output_array, 
         stride, bias):
    '''
    计算卷积，自动适配输入为2D和3D的情况
    '''
    channel_number = input_array.ndim
    output_width = output_array.shape[1]
    output_height = output_array.shape[0]
    kernel_width = kernel_array.shape[-1]
    kernel_height = kernel_array.shape[-2]
    for i in range(output_height):
        for j in range(output_width):
            output_array[i][j] = (    
                get_patch(input_array, i, j, kernel_width, 
                    kernel_height, stride) * kernel_array
                ).sum() + bias
# 为数组增加Zero padding
def padding(input_array, zp):
    '''
    为数组增加Zero padding，自动适配输入为2D和3D的情况
    '''
    if zp == 0:
        return input_array
    else:
        if input_array.ndim == 3:
            input_width = input_array.shape[2]
            input_height = input_array.shape[1]
            input_depth = input_array.shape[0]
            padded_array = np.zeros((
                input_depth, 
                input_height + 2 * zp,
                input_width + 2 * zp))
            padded_array[:,
                zp : zp + input_height,
                zp : zp + input_width] = input_array
            return padded_array
        elif input_array.ndim == 2:
            input_width = input_array.shape[1]
            input_height = input_array.shape[0]
            padded_array = np.zeros((
                input_height + 2 * zp,
                input_width + 2 * zp))
            padded_array[zp : zp + input_height,
                zp : zp + input_width] = input_array
            return padded_array
def get_patch(input_array, i, j, kernel_width, 
                    kernel_height, stride):
    i*=stride
    j*=stride
    max_height = i + kernel_height
    max_width = j + kernel_width
    if input_array.ndim == 3:
        max_z = input_array.shape[0] + 1
        return input_array[0:max_z, i:max_height, j:max_width]
    else:
        return input_array[i:max_height, j:max_width]




