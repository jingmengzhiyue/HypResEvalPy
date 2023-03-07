import numpy as np
from scipy import signal, ndimage

def RMSE(X1, X2):
    diff = X1 - X2
    squared_diff = np.square(diff)
    mse = np.mean(squared_diff)
    Recult_rmse = np.sqrt(mse)
    return Recult_rmse

def CC(GT, RC, mask=None):
    _, _, bands = RC.shape

    out = np.zeros(bands)
    for i in range(bands):
        tar_tmp = RC[:, :, i]
        ref_tmp = GT[:, :, i]

        if mask is not None:
            mask_ = np.nonzero(mask)
            tar_tmp = tar_tmp[mask_]
            ref_tmp = ref_tmp[mask_]

        cc = np.corrcoef(tar_tmp.flatten(), ref_tmp.flatten())
        out[i] = cc[0, 1]

    Recult_CC = np.mean(out)
    return Recult_CC

def DD(GT, RC):

    Recult_DD = np.linalg.norm(GT.ravel() - RC.ravel(), ord=1) / GT.size

    # rows, cols, bands = RC.shape
    # Recult_DD = 1 / (bands * rows * cols) * np.linalg.norm(RC.reshape(-1, 1) - GT.reshape(-1, 1), ord=1)
    return Recult_DD



def ERGAS(ref_img, RC, downsampling_scale):
    m, n, k = ref_img.shape
    mm, nn, kk = RC.shape
    m = min(m, mm)
    n = min(n, nn)
    k = min(k, kk)
    imagery1 = ref_img[0:m, 0:n, 0:k]
    imagery2 = RC[0:m, 0:n, 0:k]

    ergas = 0
    for i in range(k):
        mse = np.mean((imagery1[:, :, i] - imagery2[:, :, i])**2)
        rmse = np.sqrt(mse)
        ergas += (rmse / np.mean(imagery1[:, :, i]))**2

    Result_ERGAS = 100 * np.sqrt(ergas / k) / downsampling_scale
    return Result_ERGAS



def PSNR(X_1, X_2):
    mse = np.mean((X_1 - X_2) ** 2)
    max_val = np.max(X_1)
    psnr = 20 * np.log10(max_val / np.sqrt(mse))
    return psnr

def RSNR(GT, RC, mask=None):
    tar = GT
    ref = RC
    _, _, bands = ref.shape

    if mask is None:
        ref = np.reshape(ref, (-1, bands))
        tar = np.reshape(tar, (-1, bands))

        msr = np.linalg.norm(ref - tar, 'fro') ** 2  # RSNR
        max2 = np.linalg.norm(ref, 'fro') ** 2  # RSNR
        rsnrall = 10 * np.log10(max2 / msr)  # RSNR

        out = {}
        out['all'] = rsnrall
        Result_RSNR = out['all']

    else:
        ref = np.reshape(ref, (-1, bands))
        tar = np.reshape(tar, (-1, bands))
        mask = mask != 0

        msr = np.mean((ref[mask, :] - tar[mask, :]) ** 2, axis=0)
        max2 = np.max(ref, axis=0) ** 2

        psnrall = 10 * np.log10(max2 / msr)
        out = {}
        out['all'] = psnrall
        out['ave'] = np.mean(psnrall)
        Result_RSNR = out['all']

    return Result_RSNR

def SAM(imagery1, imagery2):
    tmp = (np.sum(imagery1*imagery2, axis=2) + np.finfo(float).eps) \
        / (np.sqrt(np.sum(imagery1**2, axis=2)) + np.finfo(float).eps) \
        / (np.sqrt(np.sum(imagery2**2, axis=2)) + np.finfo(float).eps)
    sam = np.mean(np.real(np.arccos(tmp)))
    return sam


def SSIM(x, y, k1=0.01, k2=0.03, L=255):
    c1 = (k1 * L)**2
    c2 = (k2 * L)**2
    mu_x = np.mean(x, axis=(1, 2), keepdims=True)
    mu_y = np.mean(y, axis=(1, 2), keepdims=True)
    sigma_x = np.std(x, axis=(1, 2), keepdims=True)
    sigma_y = np.std(y, axis=(1, 2), keepdims=True)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y), axis=(1, 2), keepdims=True)
    ssim = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2) / \
           ((mu_x**2 + mu_y**2 + c1) * (sigma_x**2 + sigma_y**2 + c2))
    return np.mean(ssim)



def UIQI(tensor1, tensor2):
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    tensor1_sq = tensor1 * tensor1
    tensor2_sq = tensor2 * tensor2
    tensor1_tensor2 = tensor1 * tensor2

    tensor1_mean = np.mean(tensor1)
    tensor2_mean = np.mean(tensor2)

    tensor1_sq_mean = np.mean(tensor1_sq)
    tensor2_sq_mean = np.mean(tensor2_sq)
    tensor1_tensor2_mean = np.mean(tensor1_tensor2)

    numerator = 4 * tensor1_tensor2_mean * tensor1_mean * tensor2_mean
    denominator = (tensor1_sq_mean + tensor2_sq_mean) * (tensor1_mean ** 2 + tensor2_mean ** 2)

    uiqi = numerator / (denominator + c1 + c2)

    return uiqi

def fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()