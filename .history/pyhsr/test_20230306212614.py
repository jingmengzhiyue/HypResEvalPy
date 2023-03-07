import evaluate
import numpy as np


X1 = 1.2*np.ones(27,).reshape((3,3,3))
X2 = np.ones(27,)
rmsr = evaluate.RMSE(X1,X2)

CC = evaluate.CC(X1,X2)

DD = evaluate.DD(X1,X2)


ERGAS = evaluate.ERGAS(X1, X2, 4)
PSNR = evaluate.PSNR(X1, X2)
RSNR = evaluate.RSNR(X1, X2)
SAM = evaluate.SAM(X1, X2)
print("f")