
# coding: utf-8

# In[20]:

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import time


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




# In[21]:

class MLmodel:
    
    def __init__(self):
        return
    
    def set_data(self,X_train=None,t_train=None):

        if type(X_train)!=type(None) :
            X_train_old=self.X_train if hasattr(self,'X_train') else None
            self.X_train=X_train
            self.num_data=self.X_train[0].size

        if type(t_train)!=type(None) :
            t_train_old=self.t_train if hasattr(self,'t_train') else None
            self.t_train=t_train
            self.t_train_array=np.reshape(t_train,np.size(t_train))
        
        if self.X_train[0].size!=t_train.size:
            self.X_train=X_train_old
            self.t_train=t_train_old

            assert  False ,"Numbers of data of X and t must be the same."

        return
    
    def fit(self, X_train=None, t_train=None):

        if type(X_train)!=type(None) or type(t_train)!=type(None) :
            self.set_data(X_train,t_train)
        return
    
    def predict(self, X_test):
        self.X_test=X_test
        self.t_test=None
        
        return
    
    def plot(self,training_data=True,test_data=False):
                
        fig = plt.figure()
        ax = Axes3D(fig)
        
        if training_data:
            ax.plot_wireframe(self.X_train[0],self.X_train[1],self.t_train) #<---ここでplot
            ax.scatter(self.X_train[0],self.X_train[1],self.t_train) #<---ここでplot

        if test_data:
            
            assert type(self.X_test)!=type(None), "X_test none"
            assert type(self.t_test)!=type(None), "t_test none"

            
            ax.scatter(self.X_test[0],self.X_test[1],self.t_test,color='green') #<---ここでplot
            ax.plot_wireframe(self.X_test[0],self.X_test[1],self.t_test,color='green') #<---ここでplot

        return


# In[22]:

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


# In[23]:

class PolynomialRegression(RidgeRegression):
    def __init__(self,max_degree=1,regular_coef=0.0):
        #polynomial_basis。
        #polynomial_basis=...
        polynomial_basis=""
        
        RidgeRegression.__init__(self,polynomial_basis,regular_coef)
        return  


# In[ ]:


def test():
	print ("test")
