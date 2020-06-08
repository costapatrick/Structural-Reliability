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
# Step 0 - Example 7-7- Haldar & Mahadevan, 2000, pag. 215
#
def gfunc1(x):
    g=x[0]*x[1]-1140.00
    return g
#
# Penalty function m(y) for FORM-iHLRF algorithm
#
def mfunc(normy,g,c):
    my=1./2.*normy**2+c*np.abs(g)
    return my
#

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
# Example  - Haldar & Mahadevan, 2000, pag. 218
namevar=np.array(['fy','Z'])
mux0=np.array([38.00,54.00])
sigmax0=np.array([3.80,2.70])
# Names of the probability density functions of the variables x
dist=['lognormal','normal']
#
#
#
#   Algorithm FORM-HLRF: Beck, 2019, pag. 101.
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
delta=1e-6*np.abs(gfunc1(xk1))
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
    gxk=gfunc1(xk)

#
# Step 6 - Evaluation of the gradients of g(x) in relation to yk
#
#
# a. Calculation of the partial derivatives of g(x) in relation to xk
#
    gradxk=optimize.approx_fprime(xk, gfunc1,eps)
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
        gyk1=gfunc1(xk1)
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

