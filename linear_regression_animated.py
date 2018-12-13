# It is the animated implementation of Linear Regression

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import random

# Learning rate
l_rate = 0.019

random.seed(7)
# Data points
x = np.arange(1,10)
y = np.array(random.sample(x, len(x)))

# Iniatializing parameter
th_0 = 20
th_1 = -2

# Number of training examples
m = len(x)
# Hypothesis function
h = lambda x: th_0 + x*th_1  

fig = plt.figure()
ax = plt.axes(xlim=(min(x)-1, max(x)+1), ylim=(min(y)-1, max(y)+1))
line, = ax.plot(x, y, lw=1)
plt.plot(x,y,'r.')

def init():
    line.set_data(x, y)
    return line,

def animate(i):
    global th_0, th_1
    # Compute partial derivative of cost function
    temp =  h(x) - y
    dJ_th_0 = 1/float(m) * np.sum(temp)
    dJ_th_1 = 1/float(m) * np.sum(temp*x)

    # Updating parameter
    th_0 -= l_rate * dJ_th_0
    th_1 -= l_rate * dJ_th_1

    line.set_data(x, h(x))
    return line,

anim = animation.FuncAnimation(fig, animate, init_func=init,
                    frames=200, interval=20, blit=True)
plt.show()
print th_0, th_1
