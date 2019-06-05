# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 16:23:54 2019
Reinforced concrete beam cross-section reliability analysis
Monte Carlo Simulation Method
@author: mauroreal@furg.br
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
# Number of Monte Carlo Simulations
N=1000000;
# Concrete compressive strength: normal distribution
fcm=48; # Units: MPa
Vfc=0.10
sfc=Vfc*fcm
fc=np.random.normal(fcm,sfc,N)
# Histogram for fc
hx, hy, _ = plt.hist(fc, bins=50, density=True,color="lightblue")
plt.title('Generate random numbers \n from a normal distribution for fc')
plt.text(30, 0.07, r'$\mu=48,\ \sigma=4.8$')
plt.grid()
plt.show()
# Reinforcement steel yielding strength: lognormal distribution
fym=550 # Units: MPa
Vfy=0.08
zetafy=np.log(1+Vfy**2)
lambdafy=np.log(fym)-0.50*zetafy**2;
fy=np.random.lognormal(lambdafy,Vfy,N);
# Histogram for fy
hx, hy, _ = plt.hist(fy, bins=50, density=True,color="darkblue")
plt.title('Generate random numbers \n from a lognormal distribution fy')
plt.text(400, .008, r'$\mu=550,\ \sigma=44$')
plt.grid()
plt.show()
# Dead load: normal distribution
gk=20 # Units: kN/m
gm=gk
Vg=0.10
sg=Vg*gm
g=np.random.normal(gm,sg,N);
# Histogram for g
hx, hy, _ = plt.hist(g, bins=50, density=True,color="darkgreen")
plt.title('Generate random numbers \n from a normal distribution for g')
plt.text(12.5, 0.175, r'$\mu=20,\ \sigma=2$')
plt.grid()
plt.show()
# Live load q: Gumbel distribution for Maxima
qk=30 # Units: kN/m
qm=qk
Vq=0.30
sq=Vq*qm
alfaq=np.sqrt((np.pi**2)/(6.*(sq)**2));
uq=qm-0.577216/alfaq;
q=np.random.gumbel(uq,1./alfaq,N);
# Histogram for q
hx, hy, _ = plt.hist(q, bins=50, density=True,color="red")
plt.title('Generate random numbers \n from a Gumbel distribution for q')
plt.text(80, 0.04, r'$\mu=30,\ \sigma=9$')
plt.grid()
plt.show()
# Beam geometric parameters
b=0.30;         # beam width (m)
d=0.90;         # beam effective depth
h=1.000;        # beam height (m)
As=0.002512;    # reinforcement cross-sectional area (m2)
L=10;           # beam span (m)
# Ultimate limit state function g(x)=MR - MS = 0
MR=1000.*As*fy*(d-0.5*(As*fy/(0.85*fc*b))); # Internal resistant bending moment
# The factor 1000. converts from MNm to kNm
MS=(g+q)*(L**2)/8; # External loading bendig moment (kNm)
G=MR-MS; # Limit state function
meanG=np.mean(G)    # mean value of G
stdG=np.std(G)      # standard deviation of G
# Histogram for G
hx, hy, _ = plt.hist(G, bins=50, density=True,color="orange")
plt.title('Reinforce concrete beam limit state function')
plt.grid()
plt.show()
# Evaluation of the number of failures
index=np.where(G>=0,0,1);
Nf=sum(index);
# Evaluation of the reliability index Beta as the ratio between the
# the mean value and the standard deviation of  g(x)
Beta1=np.mean(G)/np.std(G)
mu=0;
sigma=1;
Pf1=norm.cdf(-Beta1)
print("\nBeta1 = {0:0.4f} and Pf1 = {1:0.4e}".format(Beta1,Pf1))
# Evaluation of the reliability index Beta as the inverse of the normal 
# distribution cumulative density function of Pf (failure probability)
Pf2=Nf/N
Beta2=-norm.ppf(Pf2,mu,sigma)
print("\nBeta2 = {0:0.4f} and Pf2 = {1:0.4e}".format(Beta2,Pf2))



