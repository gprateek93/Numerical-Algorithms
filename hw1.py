def data1():
    X = [10000,10000.1,10000.2];
    return X;



print(data1());




def var1(X):
    sm = 0;
    sq = 0;
    n = len(X)
    for i in range(0,n):
        sm+=X[i];
        sq+=X[i]*X[i];
    ans = sq/n - (sm/n)**2;
    return ans;




print(var1(data1()));




def var2(X):
    sm = 0;
    n = len(X);
    for i in range(0,n):
        sm+=X[i];
    mean = sm/n;
    sq = 0;
    for j in range(0,n):
        sq+=(X[j]-mean)**2;
    return sq/n;




print(var2(data1()));




import random;
def data2():
    n = 100000;
    X = [];
    for i in range(0,n):
        X.append(random.uniform(1,10));
    return X;




dat = data2();
print(dat);




def var3(X):
    N = 10;
    sm = 0;
    if(len(X)<=N):
        for i in range(0,len(X)):
            sm+=X[i];

    else:
        m = len(X)//2;
        sm = var3(X[0:m]) + var3(X[m:len(X)]);
    return sm;




print(var3(dat));




def test(X):
    sm = 0;
    n = len(X);
    for i in range(0,n):
        sm+=X[i];
    return sm;




print(test(dat));




import numpy as np
P = np.array(dat);
print(P);
sm = P.sum();
print(sm);
