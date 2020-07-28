from numpy.linalg import inv # for inverse of matrix
import numpy as np


#Simulation in Python


def simulateLinearRegressionData(m, n):
    """
  Parameters
  ----------
  m - number of samples
  n - number of features
  
  
  Return
  ---------
    matrix X (m*n-1)
    vector y(m)
    vector beta(n)
    
  note: for convenience 1 has been added to the last column of X
    """
    np.random.seed(1)
    X = np.zeros((m,n-1))
    beta = np.zeros(n)
    for i in range(n-1):
      mu = np.random.randint(0, 50)
      sigma = np.random.normal(3, 0.1)
      X[:,i] = np.random.normal(mu, sigma, size=m)
      beta[i] =  np.random.normal(mu, sigma)
      
    X = np.hstack((X, np.ones((m,1))))
    beta[-1] = np.random.normal()
    beta =  np.reshape(beta, (n, -1))
    y = np.dot(X, beta)
    
    return X, y, beta
    
    
#closed-form in python
def estimateBetaLR(X, y):
  """
  function for closed-form estimation of beta
  """
  return np.dot(inv(np.dot(X.T, X)), 
                  np.dot(X.T, y)
                  )
                  
                  
#Gradient Descent in python



def calc_rss(_beta, x, y):
  """
  function to calculate RSS 
  """
  return float((y-x.dot(_beta)).T.dot(y-x.dot(_beta)))
  
  
def oracle(_beta, x, y):
  """
  Oracle computes and returns the gradient of RSS 
  """
  return -2 * np.dot(x.T, (y-np.dot(x,_beta)))
  
  
def gd(x, y, maxit, step_size):
  """
  Input
  -------
  x - matrix m * n
  y - vector of target values
  maxit - number of iterations
  step_size - step-size t
  
  Return
  -------
  beta - the gradient descent estimate of beta
  rss_history - the rss computed at each iteration of gd
  """
  np.random.seed(2)
  beta = np.random.normal(size=(x.shape[1], 1)) #initial guess
  rss_history = []
  for i in range(maxit):
    rss_history.append(calc_rss(beta, x, y))
    step_direction = -oracle(beta, x, y)
    beta = beta + step_size*step_direction
    
  return beta, rss_history
 
# Assessment
def calc_tss(y):
  return np.dot((y-np.mean(y)).T, 
                ((y-np.mean(y)))
                )
  
def rse_r2(beta,x, y):
  rss = calc_rss(beta,x, y )
  rse = np.sqrt(rss/(len(y)-x.shape[1]-1))
  tss = calc_tss(y)
  r2 = 1-(rss/tss)
  return rse, r2

# Diagnostics
def plot_residuals(x, y, beta):
  y_hat = x.dot(beta)
  e = y - y_hat
  if x.shape[1] > 1:
    plt.scatter(y_hat, e)
    plt.show()
  else:
    plt.scatter(x[:,0], e)
    
    
# using libraries
from sklearn.linear_model import LinearRegression
lm = LinearRegression().fit(X[:,:-1], y)
lm.coef_
