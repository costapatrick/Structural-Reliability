# -*- coding: utf-8 -*-
"""
Created on Wed May 13 18:52:04 2020
FORM-HLRF algorithm Haldar and Mahadevan,
Probability, Reliability and Statistical Methods in Engineering Design
Page 207.
@author: MVREAL
"""
import numpy as np
from scipy import optimize
from scipy.stats import norm
import pandas as pd

#
# Limit state function: g(x)
# This function is used to evaluate the value of g(x)
# and the derivatives of g(x)
#
# Step 0 - Example 8.1 - Haldar & Mahadevan, p. 228
#
def g(x):
    g=x[0]*x[1]-1140
    return g
#
# GRAM-SCHMIDT transformation
#
def gramschmidt(A, n):
    rk=np.zeros(n)
    rj=np.zeros(n)
    rk0=np.zeros(n)
    #
    R=np.zeros((n,n))
    R[n-1,:]=A[n-1,:].copy()
    for k in range(n-2,-1,-1):
        rk0=A[k,:].copy()
        rk0projection=np.zeros(n)
        for j in range(n-1,k,-1):
            rj=R[j,:].copy()
            projection=(rj.dot(rk0))/(rj.dot(rj))
            rk0projection=rk0projection+projection*rj
        rk=rk0-rk0projection
        R[k,:]=rk.copy()
    for i in range(n):
        R[i,:]=R[i,:]/np.linalg.norm(R[i,:])
#
    return R
#
#
# Function to calculate the second order derivative: d2g/dxidxj
#
def second_order_derivative(x,i,j):
    epsilon=1.e-4 # tolerance for the increments
    h1=epsilon    # increments: h1 and h2, when i is not equal to j
    h2=epsilon    # different increments can be adopted
    h=epsilon     # increment h
    a=x[i]        # reference value for x[i]
    b=x[j]        # reference value for x[j]
#
# Code: gmn where m and n are equal to:
# Index 0 = no increment is applied to the variables i and j
# Index 1 = a decrement equal to -h is applied to the variable i (or j)
# Index 2 = an incremente equal to +h is applied to the variable i (or j)
#
    if i==j:
        x0=np.copy(x)
        x0[i]=a-h
        g10=g(x0)
        x0[i]=a
        g00=g(x0)
        x0[i]=a+h
        g20=g(x0)
        d2g=(g10-2.*g00+g20)/h**2 # second order derivative: d2g/dxi2
    else:
        x0=np.copy(x)
        x0[i]=a+h1
        x0[j]=b+h2
        g22=g(x0)
        x0[i]=a+h1
        x0[j]=b-h2
        g21=g(x0)
        x0[i]=a-h1
        x0[j]=b+h2
        g12=g(x0)
        x0[i]=a-h1
        x0[j]=b-h2
        g11=g(x0)
        d2g=(g22-g21-g12+g11)/(4.*h1*h2) # second order derivative: d2g/dxidxj
    #
    return d2g

#
# Penalty function m(y) for FORM-iHLRF algorithm
#
def mfunc(normy,g,c):
    my=1./2.*normy**2+c*np.abs(g)
    return my
#
# Equivalent normal distribution parameters
# xval = value of the variable x (scalar)
# xpar1,xpar2,xpar3,xpar4 = parameters of the original pdf (scalars)
# namedist = name of the x probability distribution ('string')
#
def normeqv(xval,xpar1,xpar2,xpar3,xpar4,namedist):
#
# Normal distribution
#
    if namedist.lower() in ['norm','normal','gauss']:
        mux=xpar1
        sigmax=xpar2
        muxneq=mux
        sigmaxneq=sigmax
#
# Uniform or constant distribution  
#      
    elif namedist.lower() in ['uniform','uniforme','const']:
        a=xpar1
        b=xpar2
        c=(b-a)
        pdfx=1./c
        cdfx=(xval-a)/c
        zval=norm.ppf(cdfx)
        sigmaxneq=(norm.pdf(zval))/pdfx
        muxneq=xval-zval*sigmaxneq
#
# Lognormal distribution       
#
    elif namedist.lower() in ['lognormal','lognorm','log']:
        mux=xpar1
        sigmax=xpar2
        zetax=np.sqrt(np.log(1.+(sigmax/mux)**2))
        lambdax=np.log(mux)-0.50*zetax**2
        sigmaxneq=zetax*xval
        muxneq=xval*(1.-np.log(xval)+lambdax)
#
# Gumbel distribution
#
    elif namedist.lower() in ['gumbel','extvalue1','evt1max']:
        mux=xpar1
        sigmax=xpar2
        alphan=(np.pi/np.sqrt(6.00))/(sigmax)
        un=mux-np.euler_gamma/alphan
        cdfx=np.exp(-np.exp(-alphan*(xval-un)))
        pdfx=alphan*np.exp(-alphan*(xval-un))*cdfx
        zval=norm.ppf(cdfx)
        sigmaxneq=norm.pdf(zval)/pdfx
        muxneq=xval-zval*sigmaxneq
#
    return muxneq,sigmaxneq                        
#
#
# Data input
#
# Number of variables of the problem
n=int(2)
# Equivalent normal mean and standard deviation of the variables
muxneqk=np.zeros(n)
sigmaxneqk=np.zeros(n)
#
# Original mean and standard deviation of the variables x
# Example 8.1 - Haldar & Mahadevan, pag. 228
namevar=np.array(['fy','Z'])
mux0=np.array([38.00,54.00])
sigmax0=np.array([3.80,2.70])
# Names of the probability density functions of the variables x
dist=['lognormal','lognormal']
#
#
#   Algorithm FORM-iHLRF: Beck, 2019, pag. 101.
#
#
# Step 1 - Determination of equivalent correlation coefficients and
#          Jacobian matrices Jxz and Jzx
#
# Uncorrelated variates in this problem.
#
# Step 2 - Initialize de xk value with mux0
#
# Initialization of the variable yk1
# Jacobian matrices of x==>y and y==>x transformations
Imatrix=np.eye(n)
D=sigmax0*Imatrix
Jyx=np.linalg.inv(D)
Jxy=np.copy(D)
yk1=np.zeros(n)
xk1=mux0+Jxy.dot(yk1)
#
# Error tolerance for yk and g(x)
epsilon=1e-6
delta=1e-6*np.abs(g(xk1))
# Initial values for errors and iteration counters
erro1=1000.00
erro2=1000.00
kiter=0
# Value of dx increment for the evaluation of the derivatives
eps=1.e-8
#

while (erro1>epsilon or erro2>delta) and kiter<100:
#
    kiter+=1
    xk=np.copy(xk1)
#
# Calculation of the equivalent normal distribution parameters for xk
#
    for i in range(n):
        xval=xk[i]
        mux=mux0[i]
        sigmax=sigmax0[i]
        namedist=dist[i]
        muxneqk[i],sigmaxneqk[i]=normeqv(xval,mux,sigmax,0,0,namedist)
#
# Step 3 - Update of the Jacobian matrices Jyx and Jxy
#
    Dneq=sigmaxneqk*Imatrix
    Jyx=np.linalg.inv(Dneq)
    Jxy=np.copy(Dneq)
#
#  Step 4 - Transformation from xk to yk
#
    yk=Jyx.dot(xk-muxneqk)
    normyk=np.linalg.norm(yk)
    beta=np.linalg.norm(yk)

#
#  Step 5 - Evaluation of g(xk)
#
    gxk=g(xk)

#
# Step 6 - Evaluation of the gradients of g(x) in relation to yk
#
#
# a. Calculation of the partial derivatives of g(x) in relation to xk
#
    gradxk=optimize.approx_fprime(xk, g,eps)
#
# b. Calculation of the partial derivatives of g(x) in relation to yk
#
    gradyk=np.transpose(Jxy).dot(gradxk)
    normgradyk=np.linalg.norm(gradyk)
#
# c. Calculation of the direction cosines for xk
#
# Direction cosines
    alpha=gradyk/normgradyk

#
# Step 7. Vector yk updating to yk+1 by HLRF algorithm
#
    dk=((np.dot(gradyk,yk)-gxk)/normgradyk**2)*gradyk-yk
    lambdak=1.00
    yk1=yk+lambdak*dk
#
# Parameters of iHLRF method
#
    gamma=2.0
    a=0.1
    b=0.5
#
    gyk=gxk
    normyk=np.linalg.norm(yk)
    normyk1=np.linalg.norm(yk1)
    c1=normyk/normgradyk
#
    if erro2>delta:
        c2=0.5*normyk1**2/np.abs(gyk)
        ck=gamma*np.max([c1,c2])
    else:
        ck=gamma*c1
#
    k=-1
    f1=1.00
    f2=0.00
    while f1>f2 and k<10:
        k+=1
        lambdak=b**k
        yk1=yk+lambdak*dk
        xk1=muxneqk+Jxy.dot(yk1)
        gyk1=g(xk1)
        normyk1=np.linalg.norm(yk1)
        f1=mfunc(normyk1,gyk1,ck)-mfunc(normyk,gyk,ck)
        gradm=yk+ck*gradyk*np.sign(gyk)
        normgradm=np.linalg.norm(gradm)
        f2=a*lambdak*np.dot(gradm,dk)
#        f2=-a*lambdak*normgradm**2 # Beck pg. 85: It does not work!!
#        res=np.array([k,ck,lambdak,gxk,gyk1,f1,f2])
#        print(res)
#
    yk1=yk+lambdak*dk
    
#

#
# Step 8. Transformation from yk+1 to xk+1
#
    xk1=muxneqk+Jxy.dot(yk1)

#
# Step 9. Convergence test for yk and g(x)
#
    prod=normgradyk*normyk
# Evaluation of the error in the yk1 vector
    if np.abs(prod)>eps:
        erro1=1.-np.abs(np.dot(gradyk,yk)/(normgradyk*normyk))
    else:
        erro1=1000.00
# Evaluation of the error in the limit state function g(x)
    erro2=np.abs(gxk)
# Printing of the updated values
    print('\nIteration number = {0:d} g(x) ={1:0.5e} erro1 ={2:0.5e} Beta ={3:0.4f}'
          .format(kiter,gxk,erro1,beta))
    datadict={'xvar':namevar,'prob_dist':dist,'mux':muxneqk,'sigmax':sigmaxneqk,
          'xk':xk,'yk':yk,'alpha':alpha}
    data=pd.DataFrame(datadict)
    print(data)
#
# Formulation of Second Order Reliability Method - SORM
#
print('\nSORM results:')
#
# Failure probability calculation
#
pfform=norm.cdf(-beta)
#
# Calculation of the Hessian Matrix
# 
bmatrix=np.zeros((n,n))
dmatrix=np.zeros((n,n))
amatrix=np.eye(n)
hmatrix=np.zeros((n,n))

np.set_printoptions(precision=4)

#
# Calculation of the Hessian matrix D: d2g/dyidyj
#
for i in range(n):
    for j in range(n):
        dmatrix[i,j]=second_order_derivative(xk,i,j)*sigmaxneqk[i]*sigmaxneqk[j]
        
print('\nHessian matrix:') 
print(dmatrix)  
 
#
# Calculation of the matrix B 
#
bmatrix=1./normgradyk*dmatrix
print('\nNorm of the gradient of g(y) =',normgradyk)
print('\nB matrix:') 
print(bmatrix)

#
# Calculation of the orthogonal matrix H
#
amatrix[n-1,:]=alpha.copy()
#
hmatrix=gramschmidt(amatrix,n)

print('\nH matrix:') 

print(hmatrix)   

#
# Calculation of the curvature matrix K
#
kmatrix=hmatrix.dot(bmatrix.dot(hmatrix.T))
print('\nK = curvatures matrix:') 
print(kmatrix) 

#
# Calculation of the failure probability using SORM Breitung equation
#
factor=1.00
for i in range(n-1):
    factor=factor*1./np.sqrt(1.00+beta*kmatrix[i,i])
pfsorm=pfform*factor
betasorm=-norm.ppf(pfsorm)
#
# Print the result 
# 
print('\npfFORM =',pfform)
print('\nfactor =',factor)
print('\npfSORM =',pfsorm)
print('\nBetaSORM =',betasorm)

