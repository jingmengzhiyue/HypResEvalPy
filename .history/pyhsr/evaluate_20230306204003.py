import numpy as np

def rmse(X1, X2):
    diff = X1 - X2
    squared_diff = np.square(diff)
    mse = np.mean(squared_diff)
    rmse = np.sqrt(mse)
    return rmse