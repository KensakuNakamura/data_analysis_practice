
# coding: utf-8

# In[10]:

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import time

from MLmodel.MLmodel import MLmodel


# In[ ]:




# # 呼び出し方
# 
# ridge = RidgeRegression(basis=phi, coef=0.0)
# 
# ridge = RidgeRegression(basis_type="poly", max_degree=3, coef=0.01)
# 
# 
# ridge_fit=ridge(X,Y)
# 
# ridge_fit_predict=ridge_fit(X_0)

# In[ ]:




# In[7]:

class RidgeRegression(MLmodel):
    
    def __init__(self, basis=None, regular_coef=0.0):
        self.basis = basis
        self.regular_coef = regular_coef
        self.num_func = len(basis)
    
    def fit(self, X_train=None, t_train=None):
        MLmodel.fit(self,X_train,t_train)
        PHI = np.zeros((self.num_data, self.num_func))

        #関数Φ（行：サンプル数,列：基底関数の個数）の各要素に値を入れる
        for i,ph in enumerate(self.basis):
            PHI[:,i] = np.reshape(ph(self.X_train[0],self.X_train[1]),self.num_data)
        
        PHI_trans=np.transpose(PHI)
        self.w = np.linalg.inv(PHI_trans.dot(PHI)).dot(np.dot(PHI_trans, self.t_train_array))
        
        return
    
    def predict(self, X_test):
        MLmodel.predict(self,X_test)
        PHI = np.zeros((self.num_data, self.num_func))

        #関数Φ（行：サンプル数,列：基底関数の個数）の各要素に値を入れる
        for i,ph in enumerate(self.basis):
            PHI[:,i] = np.reshape(ph(self.X_test[0],self.X_test[1]),self.num_data)

            self.t_test_array=PHI.dot(self.w)
            self.t_test=self.t_test_array.reshape(self.X_test[0].shape)

        return        


# In[8]:

class PolynomialRegression(RidgeRegression):
    def __init__(self,max_degree=1,regular_coef=0.0):
        #polynomial_basis。
        #polynomial_basis=...
        polynomial_basis=""
        
        RidgeRegression.__init__(self,polynomial_basis,regular_coef)
        return  


# In[ ]:



