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


def SSIM(img1, img2, cs_map=False):
    """Return the Structural Similarity Map corresponding to input images img1 
    and img2 (images are assumed to be uint8)
    
    This function attempts to mimic precisely the functionality of ssim.m a 
    MATLAB provided by the author's of SSIM
    https://ece.uwaterloo.ca/~z70wang/research/ssim/ssim_index.m
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    L = 255 #bitdepth of image
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = signal.fftconvolve(window, img1, mode='valid')
    mu2 = signal.fftconvolve(window, img2, mode='valid')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = signal.fftconvolve(window, img1*img1, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(window, img2*img2, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(window, img1*img2, mode='valid') - mu1_mu2
    if cs_map:
        return (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2)), 
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        return ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2))

def msssim(img1, img2):
    """This function implements Multi-Scale Structural Similarity (MSSSIM) Image 
    Quality Assessment according to Z. Wang's "Multi-scale structural similarity 
    for image quality assessment" Invited Paper, IEEE Asilomar Conference on 
    Signals, Systems and Computers, Nov. 2003 
    
    Author's MATLAB implementation:-
    http://www.cns.nyu.edu/~lcv/ssim/msssim.zip
    """
    level = 5
    weight = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    downsample_filter = np.ones((2, 2))/4.0
    im1 = img1.astype(np.float64)
    im2 = img2.astype(np.float64)
    mssim = np.array([])
    mcs = np.array([])
    for l in range(level):
        ssim_map, cs_map = SSIM(im1, im2, cs_map=True)
        mssim = np.append(mssim, ssim_map.mean())
        mcs = np.append(mcs, cs_map.mean())
        filtered_im1 = ndimage.filters.convolve(im1, downsample_filter, 
                                                mode='reflect')
        filtered_im2 = ndimage.filters.convolve(im2, downsample_filter, 
                                                mode='reflect')
        im1 = filtered_im1[::2, ::2]
        im2 = filtered_im2[::2, ::2]
    return (np.prod(mcs[0:level-1]**weight[0:level-1])*
                    (mssim[level-1]**weight[level-1]))



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