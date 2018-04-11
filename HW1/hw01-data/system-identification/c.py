# Load mat file
import numpy as np
import scipy.io

mdict = scipy.io.loadmat("train.mat")

# Assemble xu matrix
x = mdict["x"]   # position of a car
v = mdict["xd"]  # velocity of the car
xprev = mdict["xp"]   # position of the car ahead
vprev = mdict["xdp"]  # velocity of the car ahead

acc = mdict["xdd"]  # acceleration of the car

a, b, c, d, e = 0, 0, 0, 0, 0

# Your code to compute a, b, c, d

print("Fitted dynamical system:")
print("xdd_i = {:.3f} x_i + {:.3f} xd_i + {:.3f} x_i-1 + {:.3f} xd_i-1 + {:.3f}".format(a, b, c, d, e))
