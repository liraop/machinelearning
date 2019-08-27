#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 22:43:27 2019

@author: liraop
"""

from numpy import *


def compute_errors(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[0, 1]
        totalError += (y - (m * x + b)) **2
    return totalError/float(len(points))

def step_gradient(current_b , current_m, points, learningRate):
    #gradient descent
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/N) * (y - ((current_m * x) + current_b))
        m_gradient += -(2/N) * x * (y - ((current_m * x) + current_b))
    new_b = current_b - (learningRate * b_gradient)
    new_m = current_m - (learningRate * m_gradient)
    return [new_b, new_m]
 
def gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations):
    b = initial_b
    m = initial_m
    
    for i in range(num_iterations):
        b, m = step_gradient(b , m, array(points), learning_rate)
    return [b, m]

def run():
    points = genfromtxt('./data/income.csv', delimiter=',')
    #hyperparameters 
    learning_rate = 0.00001
    #y = mx + b (slope formula)
    initial_b = 0
    initial_m = 0
    num_iterations = 100
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print(b, m)    

if __name__ == '__main__':
   run() 