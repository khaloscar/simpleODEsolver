# My own ODE solver
# This will be a project to implement some different numerical methods for solving ODE's
# The implemented methods will be able to handle vectors
#
# The following features should be added:
#       - RK1, RK2, RK4, RK45 (handling vectors)
#               * Each method should have descriptions
#               * Maybe add error handling...
#               * Should handle vectors
#               * Should pass func. args through
#       - Should be able to plot the numerical stability region in complex space
#       - Maybe provide a method to find the order of precision NO...
#       - Could set input type to np.array or list, so that it can handle both types by user input (first assume input is np.array..)
#       - Other features ??
#
#
#
# Author notes:
#       - Do it with classes??

import numpy as np
import matplotlib.pyplot as plt

## Numerical Methods

def trapz(fun, tspan, u0, stepsize, *args):
    # Trapezoidal method
    N = int((tspan[-1] - tspan[0])/stepsize)
    tt = np.linspace(tspan[0], tspan[1], N+1)
    yy = np.array([u0])

    for i in range(N):
        tn = i*stepsize
        w1 = fun(tn, yy[-1], *args)
        w2 = fun(tn + stepsize, yy[-1] + stepsize*w1, *args)
        ynext = yy[-1] + w1 -w2
        yy = np.append(yy, [ynext], axis=0)
    return tt, yy

# Leap frog
def leap_frog(fun, tspan, u0, stepsize, *args):
    # Leap frog method

    N = int((tspan[-1] - tspan[0])/stepsize)
    tt = np.linspace(tspan[0], tspan[1], N+1)
    yy = np.array([u0])

    for i in range(N):
        tn = i*stepsize
        w1 = fun(tn, yy[-1], *args)
        ynext = yy[-1] + 2*stepsize*w1
        yy = np.append(yy, [ynext], axis=0)
    return tt, yy

def RK1(fun, tspan, u0, stepsize, *args):
    # Aka Eulers forward method

    N = int((tspan[-1] - tspan[0])/stepsize)
    tt = np.linspace(tspan[0], tspan[-1], N+1)
    yy = np.array([u0])
    tn = 0
    for i in range(N):
        tn = i*stepsize
        ynext = yy[-1] + stepsize*fun(tn, yy[-1], *args)
        yy = np.append(yy, [ynext], axis=0)
    return tt, yy

def euler_bw(fun, tspan, u0, stepsize, *args):
    # Euler backwards... For whatever reason
    # Not sure if implemented correctly....

    N = int((tspan[-1] - tspan[0])/stepsize)
    tt = np.linspace(tspan[0], tspan[1], N+1)
    yy = np.array([u0])

    for i in range(N):
        tn = i*stepsize
        yy = np.append(yy, [yy[-1] - stepsize*fun(tn, yy[-1], *args)], axis=0)

    return tt, yy

def RK2(fun, tspan, u0, stepsize, *args):
    # Aka Heuns
    N = int((tspan[-1] - tspan[0])/stepsize)
    tt = np.linspace(tspan[0], tspan[-1], N+1)
    yy = np.array([u0])
    tn = 0
    for i in range(N):
        tn = i*stepsize
        F1 = fun(tn, yy[-1], *args)
        F2 = fun(tn + stepsize, yy[-1] + stepsize*F1, *args)
        ynext = yy[-1] + stepsize/2 * (F1 + F2)
        yy = np.append(yy, [ynext], axis=0)
    return tt, yy

def RK4(fun, tspan, u0, stepsize, *args):
    N = int((tspan[-1] - tspan[0])/stepsize)
    tt = np.linspace(tspan[0], tspan[-1], N+1)
    yy = np.array([u0])
    tn = 0
    for i in range(N):
        tn = i*stepsize
        F1 = fun(tn, yy[-1], *args)
        F2 = fun(tn + stepsize/2, yy[-1] + stepsize*F1/2, *args)
        F3 = fun(tn + stepsize/2, yy[-1] + stepsize*F2/2, *args)
        F4 = fun(tn + stepsize, yy[-1] + stepsize*F2, *args)
        ynext = yy[-1] + stepsize/6 * (F1 + 2*F2 + 2*F3 + F4)
        yy = np.append(yy, [ynext], axis=0)
    return tt, yy    


# A function tester tests the implemented method against
# A well known ODE, so that we know that the method
# actually works
# Testing function

def testfun(t, y):
    return -y

# The function for testing the implemented methods
def tester(method = RK4):
    tspan = (0,4)
    u0 = 1
    k = 0.01
    yVal, xVal = method(testfun, tspan, u0, k)
    plt.plot(yVal, xVal)
    plt.title(f'Testing method {method}')
    plt.show()


if __name__ == "__main__":
    tester(euler_bw)