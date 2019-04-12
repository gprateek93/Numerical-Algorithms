#!/usr/bin/env python
# coding: utf-8

# In[64]:


import matplotlib.image as im
import matplotlib.pyplot as plt
import scipy.linalg as sc
import numpy as np


# In[160]:


p = im.imread('/Users/prateekgupta/Downloads/kodim01.png');


# In[184]:


def split(img, n):
    a = img.shape[0];
    b = img.shape[1];
    c = img.shape[2];
    m = 3*n*n;
    l = (a*b*c)//m
    C = img.reshape(l,m);
    return C;
C = split(p,16);


# In[172]:


def join(C,n,w,h):
    img = C.reshape(h,w,3);
    return img;
img = join(C,16,p.shape[1],p.shape[0]);


# In[179]:


def compress(C,r):
    u,e,v = sc.svd(C); 
    u_hat = u[:,0:r];
    e_hat = np.zeros((r,r),dtype = float);
    for i in range(0,r):
        e_hat[i][i] = e[i];
    v_hat = v[0:r,:];
    A = u_hat @ e_hat;
    B = v_hat;
    s = 0;
    p= 0;
    for i in range(0,e.shape[0]):
        if(i<r):
            s+=e[i]**2;
        else:
            s+=e[i]**2;
            p+=e[i]**2;
    error = p/s;
    return (A,B,error);
A,B,error = compress(C,15);


# In[178]:


def relError(A,B):
    diff = A-B;
    C = sc.norm(diff)/sc.norm(A);
    return C**2;
D = relError(p,join(A@B,16,p.shape[1],p.shape[0]));


# In[ ]:





# In[ ]:





# In[ ]:




