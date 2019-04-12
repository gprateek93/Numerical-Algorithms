import makeNetwork as mk
import numpy as np
import matplotlib.pyplot as plt

def applyA(network,v):
    m,nodes = network
    current = np.zeros(m)
    for node in nodes:
        flow = (v[node[0]] - v[node[1]])/node[2]
        current[node[0]] += flow
        current[node[1]] -= flow
    current[0] += v[0]
    current[m-1] += v[m-1]
    return current

def getB(network):
    m,nodes = network
    b = np.zeros(m)
    b[0] = -1
    return b

def cg(Afun, b, tolerance):
    residual_error = []
    x = np.random.random(b.shape[0])
    residual = b - Afun(x)
    search_dir = residual
    residual_entry = np.linalg.norm(residual)/np.linalg.norm(b)
    residual_error.append(residual_entry)
    while True:
        if residual_entry <= tolerance:
            break
        alpha = np.dot(residual,residual) / np.dot(search_dir,Afun(search_dir))
        x += alpha * search_dir
        residual_new = residual - alpha * Afun(search_dir)
        beta = np.dot(residual_new,residual_new) / np.dot(residual,residual)
        search_dir = residual_new + beta * search_dir
        residual = residual_new
        residual_entry = np.linalg.norm(residual)/np.linalg.norm(b)
        residual_error.append(residual_entry)
    return x,residual_error

def getDiag(network):
    m,nodes = network
    res = np.zeros(m)
    res[0] = res[m-1] = 1
    for node in nodes:
        res[node[0]]+= 1/(node[2])
        res[node[1]]+= 1/(node[2])
    return res

def pcg(Afun, b, d , tolerance):
    P = np.linalg.inv(np.diag(d))
    return cg(lambda v: P@Afun(v),P@b, tolerance)

#plot random 1:
network = mk.makeNetwork('random1',1000)
x, error = cg(lambda v: applyA(network, v),-getB(network), 10**-6)
error = np.log(error)
plt.plot(error)
plt.xlabel('No. of Iterations')
plt.ylabel('Log residual error')
plt.savefig('Plot1.png')
plt.show()

#plot random 2:
network = mk.makeNetwork('random2', 1000)
x2,error2 = pcg(lambda v: applyA(network, v), -getB(network), getDiag(network), 10**-6)
x1,error1 = cg(lambda v: applyA(network, v),-getB(network), 10**-6)
fig = plt.figure()
cg, = plt.plot(np.log(error1),label = 'Conjugate Gradient')
pcg,= plt.plot(np.log(error2),label = 'Preconditioned Conjugate Gradient')
plt.legend(handles = [cg,pcg])
plt.xlabel('No. of Iterations')
plt.ylabel('Log residual error')
plt.savefig('Plot2.png')
plt.show()
