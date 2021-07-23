"""
Created on June 08, 2021
@author: Zhongjie Yu
@author: Mingye Zhu
"""
import numpy as np


def calc_rmse(y_pred, y):
    """Calculate RMSE for MOMoGP.
    Details in Sec. 5.2 Equation (15)
    """
    RMSE = 0
    y_d = y.shape[1]
    for k in range(y_d):
        SE = (y_pred[:,0,k] - y[:,k]) ** 2
        RMSE_k = np.sqrt(SE.sum()/ len(y))

        RMSE += RMSE_k

    return RMSE/y_d


def calc_mae(y_pred, y):
    """Calculate MAE for MOMoGP.
    Details in Sec. 5.2 Equation (16)
    """
    MAE = 0
    y_d = y.shape[1]
    for k in range(y_d):
        SE = (y_pred[:,0,k] - y[:,k]) ** 2
        MAE_k = np.sqrt(SE).sum()/ len(y)

        MAE += MAE_k

    return MAE/y_d


def calc_nlpd(y_pred, y_cov, y):
    """Calculate NLPD for MOMoGP.
    Details in Sec. 5.2 Equation (17)
    """
    NLPD = 0
    y_d = y.shape[1]
    count = 0
    for k in range(y_pred.shape[0]):
        sigma = np.sqrt(np.abs(np.linalg.det(y_cov[k,:,:])))
        if sigma == 0:
            count+=1
            continue
        d1 = (y[k, :].reshape((1, 1, y_d)) - y_pred[k, :, :]).reshape((1, y_d))
        a = 1/(np.power((2*np.pi),y_d/2)*sigma)
        ni =np.linalg.pinv(y_cov[k, :, :])
        b = a * np.exp(np.dot(np.dot(-1 / 2 * d1, ni), d1.T))
        if b > 0.0000000001:  
            NLPD_k = -np.log(b)
        else:
            NLPD_k = 0

        NLPD += NLPD_k

    return NLPD/(y_pred.shape[0])







