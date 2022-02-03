#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 17:20:47 2019

@author: prapti
"""
from matplotlib import pyplot as plt

kernel = [[0,1,0],
          [1,-4,1],
          [0,1,0]]
plt.imshow(kernel, cmap='gray')