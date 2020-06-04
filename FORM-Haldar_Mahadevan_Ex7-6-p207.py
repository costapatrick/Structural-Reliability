# -*- coding: utf-8 -*-
"""
Created on Wed May 13 18:52:04 2020
FORM algorithm Haldar and Mahadevan,
Probability, Reliability and Statistical Methods in Engineering Design
Page 207.
@author: MVREAL
"""
import numpy as np
import scipy.optimize
from scipy import optimize
from scipy.stats import norm

#
# Limit state function: g(x)
# This function is used to evaluate the value of g(x)
# and the derivatives of g(x)
#
# Step 1 - Example 7.3 - Haldar & Mahadevan pag. 208 
#
def gfunc1(x):
    g=x[0]*x[1]-1140.00
    return g
#
# Limit state function: g(beta)
# This function is used to evaluate the value of beta
# on the condition that g(beta)=0.00
#
# Step 1 - Example 7.6 - Haldar & Mahadevan pag. 207 
#
def gfunc2(beta,muxneqk,sigmaxneqk,alpha):
    g=(muxneqk[0]-alpha[0]*beta*sigmaxneqk[0])*(muxneqk[1]-alpha[1]*beta*sigmaxneqk[1])-1140.00
    return g
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
# Variables values at iteration k
xk=np.zeros(n)
yk=np.zeros(n)
# Variable y value at iteration k+1
yk1=np.zeros(n)
# Gradients of variables xk and yk
gradxk=np.zeros(n)
gradyk=np.zeros(n)
# Direction cosines of the variables
oldalpha=np.zeros(n)
alpha=np.zeros(n)
#
# Original mean and standard deviation of the variables x
# Example 7.6 - Haldar & Mahadevan pag. 207
#
# Step 2 - Initial value for beta for the Newton-Raphson method
#
beta=3.00
#
# Step 3 - Initialize de xk value with mux0
#
# Initialization of xk and muxneqk with mux0
mux0=np.array([38.00,54.00])
sigmax0=np.array([3.80,2.70])
# Names of the probability density functions of the variables x
dist=['lognormal','normal']
#
xk=mux0.copy()
muxneqk=mux0.copy()
# Jacobian matrices of x==>y and y==>x transformations
Imatrix=np.eye(n)
D=sigmax0*Imatrix
Jyx=np.linalg.inv(D)
Jxy=D.copy()
# Initialization of the variable yk1
yk1=Jyx.dot(xk-mux0)
# Error tolerance for yk and g(x)
epsilon=1e-3
delta=1e-3*gfunc1(xk)
# Initial values for errors and iteration counters
erro1=1000.00
erro2=1000.00
kiter=0
# Value of dx increment for the evaluation of the derivatives
eps=1.e-8
#
#
#   Algorithm FORM: Haldar & Mahadevan, pag. 207
#
while (erro1>epsilon or erro2>delta) and kiter<10:
#
    kiter+=1
    erro3=1000.00
    miter=0
# Iteration until direction cosines converge
    while (erro3>0.005):
        miter+=1
        yk=yk1.copy()
        xk=muxneqk+Jxy.dot(yk)

#
# Step 4 - Calculation of the equivalent normal distribution parameters for xk
#
        for i in range(n):
            xval=xk[i]
            mux=mux0[i]
            sigmax=sigmax0[i]
            namedist=dist[i]
            muxneqk[i],sigmaxneqk[i]=normeqv(xval,mux,sigmax,0,0,namedist)
#
# Step 5 - Calculation of the partial derivatives of g(x) in relation to yk
#
    #
    # Update of the Jacobian matrices
    #
        Dneq=sigmaxneqk*Imatrix
        Jyx=np.linalg.inv(Dneq)
        Jxy=Dneq.copy()
    #
    # Evaluation of the gradients of g(x)
    #
        gradxk=optimize.approx_fprime(xk, gfunc1,eps)
        gradyk=Jxy.dot(gradxk)
        normgradyk=np.linalg.norm(gradyk)
#
# Step 6 - Calculation of the direction cosines for xk
#
    # Direction cosines
        oldalpha=alpha.copy()
        alpha=gradyk/normgradyk
        diffalpha=np.abs(alpha-oldalpha)
#
# Step 7 - Update the values of yk
#
    # Update of yk values
        yk1=-alpha*beta
    # Convergence test: the max difference between alpha and oldalpha must be less than 0.005
        erro3=diffalpha.max()
#        print(miter,xk,muxneqk,sigmaxneqk,gradxk,alpha,erro3)
#
# Step 8 - Evaluation of beta value that satisfies the condition g(x)=0
#    
    betai=beta
    beta=scipy.optimize.newton(gfunc2,betai,args=(muxneqk,sigmaxneqk,alpha))
    normyk=np.linalg.norm(yk)
    gxk=gfunc1(xk)

#
    prod=normgradyk*normyk
# Evaluation of the error in the yk vector
    if np.abs(prod)>eps:
        erro1=1.-np.abs(np.dot(gradyk,yk)/(normgradyk*normyk))
    else:
        erro1=1000.00
# Evaluation of the error in the limit state function g(x)
    erro2=np.abs(gxk)
# Printing of the updated values
    print('k        yk                alphak          Beta    erro1        g(x)           xk')
    np.set_printoptions(formatter={'float': '{: 0.4f}'.format}) 
    print(kiter,yk1,alpha,'{0:10.4f} {1:10.4e} {2:10.4f}'.format(beta,erro1,erro2),xk)

# Vector yk updating
    yk1=-beta*alpha

