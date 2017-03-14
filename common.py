#!/bin/env python
#encoding=utf8

import os,sys


##
# @file common.py
# @brief  scikit-learn练习的通用工具函数
# @author  xinxianwei01@163.com
# @version 1.0.0.0
# @date 2017-03-14


def array_list_to_list(input_list):
    res_list = []
    for arr in input_list:
        for a in arr:
            res_list.append(a)
    return res_list



