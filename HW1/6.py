
# coding: utf-8

# In[2]:


##  6 System Identifcation by ordinary least squares regression, part(a)
from scipy.io import loadmat
import numpy as np
from numpy.linalg import inv


# In[79]:


a = loadmat('hw01-data/system-identification/a.mat')
x = np.transpose(a['x'])
u = np.transpose(a['u'])

X = np.hstack((x[:29, :], u[:29, :]))
y =  x[1:30, :]

XT = np.transpose(X)
w = np.dot(inv(XT @ X) @ XT, y)

print(w)


# In[83]:


##  6 System Identifcation by ordinary least squares regression, part(b)

b = loadmat('hw01-data/system-identification/b.mat')
u = b['u']
x = b['x']
u = u.reshape(u.shape[0:2])
x = x.reshape(x.shape[0:2])
n = u.shape[0]

Y = x[1:n, :].transpose()
X = np.vstack((x[0:n-1, :].transpose(), u[0:n-1, :].transpose()))
XT = X.transpose()
W = Y @ XT @ inv(X @ XT)
A = W[:, 0:3]
B = W[:, 3:6]
print(A)
print('\n')
print(B)


# In[23]:


##  6 System Identifcation by ordinary least squares regression, part(c)

train = loadmat('hw01-data/system-identification/train.mat')
x = train['x'].transpose()
xd = train['xd'].transpose()
xdd = train['xdd'].transpose()
xp = train['xp'].transpose()
xdp = train['xdp'].transpose()
n = x.shape[0]

X = np.hstack((x, xd, xp, xdp, np.ones((n, 1))))
y = xdd
XT = X.transpose()
w = np.dot(inv(XT @ X) @ XT, y)
print(w)

