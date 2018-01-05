
# coding: utf-8

# In[1]:

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import time
from MLmodel.RidgeRegression.RidgeRegression import RidgeRegression


# In[4]:

class PolynomialRegression(RidgeRegression):
    
    def get_polynomial_basis(self, max_degree):
        polynomials=[]
        for deg in range(0,max_degree+1):
            deg_polys=self.get_polynomials_of_a_degree(deg)
            for poly in deg_polys:
                polynomials.append(poly)
        return polynomials
    
    def __init__(self, max_degree=1,regular_coef=0.0):
        polynomial_basis=self.get_polynomial_basis(max_degree)
        
        RidgeRegression.__init__(self,polynomial_basis,regular_coef)
        return  
    
    def get_polynomials_of_a_degree(self, deg):
        polynomials=[]
        for k in range(0,deg+1):
            ph=self.make_a_polynomial(deg-k,k)
            polynomials.append(ph)
        return polynomials

    def make_a_polynomial(self, a,b):
    #     def phi (x,y):
    #         return x**a*y**b;
    #     return phi
        return lambda x,y: x**a*y**b

