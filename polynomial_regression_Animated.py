# It is the animated implementation of Polynomial Regression

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import random

# Learning rate
l_rate = 0.023

random.seed(3)

# Data points
x_t = np.arange(1,10)
x = np.array([[1, i, i**2/float(10), i**3/float(10**2),] for i in range(1, 10)], dtype=float)
y = np.array([random.sample(x_t, len(x))], dtype=float).T

x_p = []
y_p = []

# Iniatializing parameter
parm = np.array([[0 for _ in range(len(x[0]))]], dtype=float).T

# Number of training examples
m = len(x)

fig = plt.figure()
ax = plt.axes(xlim=(min(x_t)-1, max(x_t)+1), ylim=(min(min(y))-1, max(max(y))+1))
line, = ax.plot(x_t, (y.T)[0], lw=1)
plt.plot(x_t, y, 'r.')

##################### Normal Equation ##################
parm_n = np.dot(np.linalg.pinv(np.dot(x.T, x)), np.dot(x.T, y))
plt.plot(x_t, np.dot(x, parm_n), 'b--', lw=1)
J_n = 1/float(2*m) * np.sum(np.square( np.dot(x, parm_n) - y))
##################### Normal Equation ##################

def init():
    line.set_data(x, y)
    return line,

def animate(i):
    global parm, J
    # Compute partial derivative of cost function
    dJ_th = 1/float(m) * np.dot((np.dot(x, parm)-y).T , x)

    # Cost function
    J = 1/float(2*m) * np.sum(np.square( np.dot(x, parm) - y))

    # Collect the value of cost function at each iteration
    y_p.append(J)
    x_p.append(len(x_p)+1)

    # Updating parameter
    parm -= l_rate*dJ_th.T
    line.set_data(x_t, np.dot(x, parm))
    return line,

anim = animation.FuncAnimation(fig, animate, init_func=init,
                    frames=200, interval=10, blit=True)


plt.show()
plt.plot(x_p, y_p, 'r')
plt.title('Cost Function')
plt.show()
print parm, '\n'
print J
print parm_n
print J_n