import fcontour
import numpy as np
import matplotlib.pyplot as plt

#part a
class Conic:
    def __init__(self,a,b,c,p,q,r):
        param = [a,b,c,p,q,r]
        self.param = param

    def evaluate(self,x):
        quad = self.param[0] * (x[0]**2) + self.param[1] * x[0] * x[1] + self.param[2] * (x[1] ** 2)
        lin  = self.param[3] * x[0] + self.param[4] * x[1]
        constant = self.param[5]
        exp = quad + lin + constant
        return exp

    def jacobian(self,x):
        quad_1 = 2 * self.param[0] * x[0]
        lin_1 =  self.param[1] * x[1]
        con_1 = self.param[3]
        grad_1 = quad_1+lin_1+con_1
        quad_2 = 2 * self.param[2] * x[1]
        lin_2 =  self.param[1] * x[0]
        con_2 = self.param[4]
        grad_2 = quad_2+lin_2+con_2
        return [grad_1, grad_2]

    def linearize(self,x):
        return lambda y: self.evaluate(x) + self.jacobian(x) @ (y - x)

def newtonStep(conic1,conic2,x):
    X = np.array([conic1.evaluate(x), conic2.evaluate(x)])
    J = np.array([conic1.jacobian(x), conic2.jacobian(x)])
    x_n = x - np.linalg.inv(J) @ X
    return x_n

conic_1 = Conic(1, 0, 1, -2, 0, 0)
conic_2 = Conic(1, 0, 1, -6, 0, 8)
x = (3, 4)
i = 0
while True:
    x_n = newtonStep(conic_1, conic_2, x)
    X = [x[0],x_n[0]]
    Y = [x[1],x_n[1]]
    plt.plot(X,Y,'o')
    plt.plot(X,Y,'r--')
    fcontour.fcontour(conic_1.evaluate, [-5, 5], [-5, 5])
    fcontour.fcontour(conic_2.evaluate, [-5, 5], [-5, 5])
    fcontour.fcontour(conic_1.linearize(x), [-5, 5], [-5, 5])
    fcontour.fcontour(conic_2.linearize(x), [-5, 5], [-5, 5])
    name = 'img_plot' + str(i)
    plt.savefig(name)
    plt.show()
    if x_n[0] == x[0] and x_n[1] == x[1]:
        break
    x = x_n
    i+=1
