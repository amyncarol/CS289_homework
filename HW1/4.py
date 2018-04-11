
# coding: utf-8

# In[58]:


import numpy as np
from scipy.linalg import eig
from scipy.linalg import svd

def compute_svd(a):
    U, s, Vh = svd(a, full_matrices=False)
    print(a)
    print(U)
    print(s)
    print(Vh)

A = np.array([[2, -4], [-1, -1]])
B = np.array([[3, 1], [1, 3]])
C = np.dot(A, B)
D = np.dot(B, A)
c = np.array([[3,1],[1, 3],[2,-4],[-1,-1]])


# In[59]:


w, vl, vr = eig(C,  left=True)
print(C)
print(w)
print(vr)
print(vl)


# In[60]:


w, vl, vr = eig(D,  left=True)
print(D)
print(w)
print(vr)
print(vl)


# In[61]:


compute_svd(A)


# In[62]:


compute_svd(np.dot(A,A))


# In[63]:


compute_svd(np.dot(A,B))


# In[64]:


compute_svd(np.dot(B,A))


# In[65]:


compute_svd(c)

