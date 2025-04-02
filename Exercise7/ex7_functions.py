# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 21:18:35 2025

@author: adm_fs
"""

import numpy as np
import matplotlib.pyplot as plt

pi = 3.14

def make_spider(values, names, color, title):
    
    N = len(values)
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    values = list(values)
    values += values [:1]
     
    plt.rc('figure', figsize=(7, 7))
 
    ax = plt.subplot(1,1,1, polar=True)
 
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
 
   
    plt.xticks(angles[:-1], names, color='black', size=12)
    ax.tick_params(axis='x', rotation=0)
    
    ax.set_rlabel_position(0)
    plt.yticks([-1,0,1], ["-1","0","1"], color="black", size=10)
    plt.ylim(-1.5,1.5)
 
    ax.plot(angles, values, color = color, linewidth=1, linestyle='solid')
    ax.fill(angles, values, color = color, alpha = 0.3)
 
  
    plt.title(title, fontsize=20, x = 0.5, y = 1.1)
 
