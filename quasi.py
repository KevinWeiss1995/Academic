#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math
import time
import matplotlib.pyplot as plt
import itertools


# In[2]:


def rosen(x):
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

def eval_f(func, x, y):
    z = np.ndarray([np.size(x),np.size(y)])
    for i in range(np.size(x)):
        for j in range(np.size(y)):
            z[i,j] = func([x[i],y[j]])

def eval_f(func, x, y):
    z = np.ndarray([np.size(x),np.size(y)])
    for i in range(np.size(x)):
        for j in range(np.size(y)):
            z[i,j] = func([x[i],y[j]])
    return z

def eval_gradient(func, x, h):
    fx = func(x) # initialization
    grad = np.zeros(x.shape)
    # implement derivative for every index
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        # evaluate function value at x+h
        ix = it.multi_index
        old_value = x[ix]
        x[ix] = old_value + h 
        fxh = func(x) 
        x[ix] = old_value 

        # calculate partial derivative
        grad[ix] = (fxh - fx) / h 
        it.iternext() 

    return grad

def gen_P(dim,index):
    P = np.identity(dim)
    if len(index)!=0:
        list = np.array(range(dim))
        p = np.sort(index)
        rem = np.delete(list,p)
        newlist = np.append(p,rem)
        P = P.take(newlist, axis=0)
    
    return P

def backtracking_line(func, x, df, gamma, itmax):
    for i in range(itmax):
        fx1 = func(x - gamma*df) 
        fx2 = func(x) - 0.75*gamma*(np.linalg.norm(df))**2
        if fx1 <= fx2: break
        else: gamma = 0.75*gamma
    return gamma

def eval_hessian(func, x, h, p=[]):
    # Initialize
    if len(p)==0: p=np.array(range(x.shape[0])) 
    hess = np.zeros((p.shape[0],p.shape[0]))
    it = np.nditer(p, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        if len(p)==len(x): ix = it.multi_index; ii = it.multi_index
        else: ix = it.value; ii = it.multi_index
        
        # derivative at first point (left)
        x1 = x; x1[ix] = x[ix]
        df1 = eval_gradient(func, x1, h)
        # derivative at second point (right)
        x2 = x; x2[ix] = x[ix] + h
        df2 = eval_gradient(func, x2, h)
        # differentiate between the two derivatives
        d2f = (df2 - df1)/h       
        hess[ii] = d2f[p]
        it.iternext()
        
    return hess

def chk_singul(hf,alpha0):
    alpha = alpha0
    while True:
        hf = hf + alpha*np.identity(len(hf))
        if np.linalg.det(hf) == 0:
            alpha = alpha*1.1
        elif np.linalg.det(hf) > 0:
            break
    return hf, alpha


# In[3]:

def PH_newton_method(func,x0,back_tr,gamma,p0=[],h=1e-7,tol=1e-4,itmax=10000,level=0):
    t0 = time.time()
    if level >= 1: print('- Newton Method ------------------------------')
    ierr = 0
    gamma0 =gamma; alpha =0.01
    x, xk = x0, x0
    p = p0
    q0 = (np.ones((1,x.shape[0]))-x0)/np.linalg.norm(np.ones((1,x.shape[0]))-x0)
    qk = q0*np.ones((1,x.shape[0]))
    P = gen_P(x.shape[0],p)
            
    for k in range(itmax+1):
        if len(p)==0:
            df = eval_gradient(func,x,h)
            hf_raw = eval_hessian(func,x,h,p)
            hf =  hf_raw + alpha*np.identity(hf_raw.shape[0])
            q = np.linalg.solve(hf, df)
        else:
            df_raw = eval_gradient(func,x,h)
            df = np.matmul(P,df_raw)
            df_h = df[:len(p)]; df_t = df[len(p):]
            hf_raw = eval_hessian(func,x,h,p)
            hf_h =  hf_raw + alpha*np.identity(hf_raw.shape[0])
            q_h = np.linalg.solve(hf_h, df_h)
            # Need update adaptively
            sigma = np.linalg.norm(df_h)/np.linalg.norm(q_h)
            qs = np.concatenate((sigma*q_h,df_t))
            q = np.matmul(P.T,qs)
    
        # backtracking line search
        if back_tr == True: gamma = backtracking_line(func,x,df,gamma,itmax)
            
        diff = np.linalg.norm(gamma*q);
        x = x - gamma*q
        xk = np.vstack((xk,x))
        qk = np.vstack((qk,q/np.linalg.norm(q)))
        
        if level>=2:
            print("%d:" %k,end=" "); print(diff); print(x)
        
        if  diff < tol:
            ierr = 1; etime = time.time()-t0
            if level >= 1:
                print('  Optimization terminated successfully.')
                print('    step=%g, tol=%g, itmax=%d' %(h,tol,itmax))
                print('    Backtracking line search = %s' %back_tr)
                if len(p0)==0:
                    print('    Hessian approximation = False')
                else:
                    print('    Hessian approximation: ', p0)
                print('    Current function value = %g' %func(x))
                print('    Newton method converged in iter = %d' %k)
                print('    Elapsed time : %4f\n' %(etime))
                print('    Error = %g' %diff)
                print('  Optimized solution is ',x)
                print('-----------------------------------------------\n')
            break
    if k >= itmax:
        etime = time.time()-t0
        if level >= 1:
            print('    Newton method: did not converged, iter = %d' %k)
            print('    Elapsed time : %4f' %(etime))
            print('    Error = %g' %diff)
    return x, xk, qk, k, etime, ierr

level = 1
func = rosen
dim = 8
x0 = 0.9*np.ones(dim)
p0 = np.array([0,1,2])
q0 = (np.ones((1,x0.shape[0]))-x0)/np.linalg.norm(np.ones((1,x0.shape[0]))-x0)
sol = 1.0*np.ones(dim)
x, xk, qk, k, etime, ierr = PH_newton_method(func,x0,back_tr=False,gamma=0.001,p0=p0,level=level)
qav = np.sum(qk, axis=0)/len(qk)
diffx = np.linalg.norm(sol-x)
diffq = np.linalg.norm(qav-q0)
data = str(diffq)+str(diffx)
print("diffq", "diffx", data)