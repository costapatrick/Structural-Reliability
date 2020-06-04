"""
Running example 7.3 from Haldar & Mahadevan 2000 with an Implicit limit state function
"""

import paransys

# Call ParAnsys
form = paransys.FORM()

# Console log On
form.Info(True)

# Create random variables
form.CreateVar('y', 'lognormal', 38.00, cv=0.10)
form.CreateVar('z', 'normal', 54, cv=0.050)

# A clean Implicit limit state function
def myFunc(y, z):
    return y * z - 1140

# But it's a function, so the parameters doesn't need to have the same name of random variables and we can have some fun, like this. 
#   Of course, here I'm trying keep the result, but you can see the idea... 
#   It's possible to connect complex functions here or create a function that do the connection (calling a lot of functions inside it)
def CrazyFunction(x1, x2):
    res = x1 * x2 - 1140

    return res


# Create limit state
#   You can hange the function name at userf=myFunc/CrazyFunction, but it's always called as userf() inside the LS
#       Please, see that myFunc() uses x,y,m and CrazyFunction() uses x1,x2,x3, but inside userf() is used the random variables names
form.SetLimState('userf(y,z)', userf=CrazyFunction)

# Run
values = form.Run(dh=0.01, meth='iHLRF')
