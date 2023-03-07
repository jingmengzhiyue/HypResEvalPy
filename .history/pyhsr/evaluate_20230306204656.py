import numpy as np

def rmse(X1, X2):
    diff = X1 - X2
    squared_diff = np.square(diff)
    mse = np.mean(squared_diff)
    Recult_rmse = np.sqrt(mse)
    return Recult_rmse

def cc(GT, RC, mask=None):
    _, _, bands = RC.shape
    out = np.zeros(bands)
    for i in range(bands):
        tar_tmp = RC[:, :, i]
        ref_tmp = GT[:, :, i]
        if mask is None:
            cc = np.corrcoef(tar_tmp.ravel(), ref_tmp.ravel())[0, 1]
        else:
            mask_indices = np.nonzero(mask)
            cc = np.corrcoef(tar_tmp[mask_indices], ref_tmp[mask_indices])[0, 1]
        out[i] = cc
    Recult_CC = np.mean(out)
    Recult_CC = np.mean(Recult_CC)
    return Recult_CC

def Index_DD(GT, RC, mode):
    if mode == 1:
        Recult_DD = np.linalg.norm(GT.ravel() - RC.ravel(), ord=1) / GT.size
    elif mode == 2:
        rows, cols, bands = RC.shape
        Recult_DD = 1 / (bands * rows * cols) * np.linalg.norm(RC.reshape(-1, 1) - GT.reshape(-1, 1), ord=1)
    return Recult_DD

import numpy as np

def Index_ERGAS(GT, RC, downsampling_scale):
    m, n, k = GT.shape
    mm, nn, kk = RC.shape
    m = min(m, mm)
    n = min(n, nn)
    k = min(k, kk)
    imagery1 = GT[0:m, 0:n, 0:k]
    imagery2 = RC[0:m, 0:n, 0:k]

    ergas = 0
    for i in range(k):
        mse = np.mean((imagery1[:, :, i] - imagery2[:, :, i])**2)
        rmse = np.sqrt(mse)
        ergas += (rmse / np.mean(imagery1[:, :, i]))**2

    Result_ERGAS = 100 * np.sqrt(ergas / k) / downsampling_scale
    return Result_ERGAS
