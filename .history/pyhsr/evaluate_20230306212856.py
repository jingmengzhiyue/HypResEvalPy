import numpy as np

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
        if mask is None:
            cc = np.corrcoef(tar_tmp.ravel(), ref_tmp.ravel())[0, 1]
        else:
            mask_indices = np.nonzero(mask)
            cc = np.corrcoef(tar_tmp[mask_indices], ref_tmp[mask_indices])[0, 1]
        out[i] = cc
    Recult_CC = np.mean(out)
    Recult_CC = np.mean(Recult_CC)
    return Recult_CC

def DD(GT, RC):

    Recult_DD = np.linalg.norm(GT.ravel() - RC.ravel(), ord=1) / GT.size

    # rows, cols, bands = RC.shape
    # Recult_DD = 1 / (bands * rows * cols) * np.linalg.norm(RC.reshape(-1, 1) - GT.reshape(-1, 1), ord=1)
    return Recult_DD



def ERGAS(GT, RC, downsampling_scale):
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


def ssim(img1, img2, k1=0.01, k2=0.03, win_size=11):
    c1 = (k1 * 255) ** 2
    c2 = (k2 * 255) ** 2
    window = np.ones((win_size, win_size)) / (win_size ** 2)
    mu1 = np.zeros_like(img1)
    mu2 = np.zeros_like(img2)
    sigma1 = np.zeros_like(img1)
    sigma2 = np.zeros_like(img2)
    sigma12 = np.zeros_like(img1)

    # Compute means
    for i in range(img1.shape[2]):
        mu1[:, :, i] = np.convolve(img1[:, :, i], window, mode='valid')
        mu2[:, :, i] = np.convolve(img2[:, :, i], window, mode='valid')

    # Compute variances and covariances
    for i in range(img1.shape[2]):
        sigma1[:, :, i] = np.convolve(np.square(img1[:, :, i] - mu1[:, :, i]), window, mode='valid')
        sigma2[:, :, i] = np.convolve(np.square(img2[:, :, i] - mu2[:, :, i]), window, mode='valid')
        sigma12[:, :, i] = np.convolve((img1[:, :, i] - mu1[:, :, i]) * (img2[:, :, i] - mu2[:, :, i]), window, mode='valid')

    # Compute SSIM
    ssim_map = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 + sigma2 + c2))
    mssim = np.mean(ssim_map)
    return mssim


def uiqi(tensor1, tensor2):
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