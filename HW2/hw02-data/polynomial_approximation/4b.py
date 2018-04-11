
# coding: utf-8

# In[29]:


import numpy as np
import matplotlib.pyplot as plt


t1 = lambda x : 1+x
t2 = lambda x: 1+x + 1.0/2*x**2
t3 = lambda x : 1+x +1.0/2*x**2+1.0/6*x**3
t4 = lambda x : 1+x +1.0/2*x**2+1.0/6*x**3+1.0/24*x**4

x = np.linspace(-4, 3, 100)
plt.plot(x, np.exp(x), label = 'True')
plt.plot(x, t1(x), label = '1st')
plt.plot(x, t2(x), label = '2nd')
plt.plot(x, t3(x), label = '3rd')
plt.plot(x, t4(x), label = '4th')
plt.legend()
plt.savefig('1.pdf', dpi=100, format='pdf')
plt.show()


# In[30]:


x = np.linspace(-20, 8, 100)
plt.plot(x, np.exp(x), label = 'True')
plt.plot(x, t1(x), label = '1st')
plt.plot(x, t2(x), label = '2nd')
plt.plot(x, t3(x), label = '3rd')
plt.plot(x, t4(x), label = '4th')
plt.legend()
plt.savefig('2.pdf', dpi=100, format='pdf')
plt.show()

