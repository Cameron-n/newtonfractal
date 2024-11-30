# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 18:00:57 2024

@author: camer
"""
import numpy as np
from sympy import lambdify
import matplotlib.pyplot as plt

def newton_numpy():
    x, y = 2*[np.linspace(-1,1,num=1001)]
    X, Y = np.meshgrid(x,y)
    Z = X + Y*1j
    
    l = lambdify("z", "z - (z**3-1)/(3*z**2)")
    
    colors = {
        -0.5-0.866025j:[1,0,0],
        -0.5+0.866025j:[0,1,0],
        1+0j:[0,0,1],
        }
    
    cool_array = np.zeros([1001,1001,3])
    fig, ax = plt.subplots(1,1)
    im = ax.imshow(cool_array)
    plt.axis('off')
    for _ in range(20):
        Z = l(Z)
        
        Z = np.round(Z,decimals=6)
        
        for i in range(1001):
            for j in range(1001):
                try:
                    cool_array[i][j] = colors[Z[i][j]]
                except:
                    pass 
        plt.imshow(cool_array)
        im.set_data(cool_array)
        fig.canvas.draw_idle()
        plt.pause(0.001)

def newton_method(newton, z, tol, max_ite):

    iterations = 0
    z_previous = z
    while iterations <= max_ite:
        try:
            z = newton(z)
        except (ZeroDivisionError, OverflowError):
            return 0
        dif = z - z_previous
        if abs(dif) < tol:
            root = round(z.real, 6) + round(z.imag, 6)*1j
            return root
        z_previous = z
        iterations += 1
                    
    return 0
        
def newton_no_numpy():
    colors = {
        -0.5-0.866025j:[1,0,0],
        -0.5+0.866025j:[0,1,0],
        1+0j:[0,0,1],
        }
    cool_array = np.zeros([1001,1001,3])
    newton = lambdify("z", "z - (z**3-1)/(3*z**2)")
    for i in range(1001):
        for j in range(1001):
            z = -1 + j*2/1000 + (-1 + i*2/1000)*1j
            root = newton_method(newton, z, 10**-6, 20)
            if root != 0:
                cool_array[i][j] = [rgb for rgb in colors[root]]
