"""
Created on June 08, 2021
@author: Zhongjie Yu
@author: Mingye Zhu
"""
import numpy as np
import pandas as pd
from MOMoGPstructure import query, build_MOMoGP
from MOMoGP import structure
import random
import torch
from utils import calc_rmse, calc_mae, calc_nlpd
from dataloader import load_data
import argparse

random.seed(23)
np.random.seed(23)
torch.manual_seed(23)
torch.cuda.manual_seed(23)


def train_MOMoGP(args):
    """Train MOMoGP with given args.
    
    Parameters
    ----------
    args
        Argument list.
    """
    # load data
    print("Train MOMoGP on", args.data)
    x_train, x_test, y_train, y_test, args=load_data(args)
    D = x_train.shape[1] # input dimensions
    P = y_train.shape[1] # output dimensions

    # hyperparameter settings
    lr = args.lr
    rerun = args.rerun
    epoch = args.epoch
    Kpx = args.Kpx
    M = args.M
    RMSEE=[]
    MAEE=[]
    NLPDD=[]
    scores = np.zeros((rerun,3))

    # args for structure learning
    opts = {
        'min_samples': 0,
        'X': x_train,
        'Y': y_train,
        'qd': Kpx-1,
        'max_depth': 100,
        'max_samples': M,
        'log': True,
        'jump': True,
        'reduce_branching': True
    }

    # run #rerun times
    for k in range(rerun):
        
        # built the root structure
        root_region, gps_ = build_MOMoGP(**opts)
        root, gps = structure(root_region,scope = [i for i in range(P)], gp_types=['matern1.5_ard'])

        # train GP experts with their own hyperparameters
        outer_LMM = np.zeros((len(gps),epoch))
        for i, gp in enumerate(gps):
            idx = query(x_train, gp.mins, gp.maxs)
            gp.x = x_train[idx]
            y_scope = y_train[:,gp.scope]
            gp.y = y_scope[idx]
            print(f"Training GP {i + 1}/{len(gps)} ({len(idx)})")
            outer_LMM[i,:]= gp.init(cuda=args.cuda, lr=lr, steps=epoch, iter=i)
        root.update()

        # on test data
        mu, cov= root.forward(x_test[:,:], y_d=P)

        # evaluate RMSE, MAE and NLPD
        RMSE = calc_rmse(mu, y_test)
        MAE = calc_mae(mu, y_test)
        NLPD = calc_nlpd(mu, cov, y_test)
        print(RMSE, MAE, NLPD)

        RMSEE.append(RMSE)
        MAEE.append(MAE)
        NLPDD.append(NLPD)
        scores[k,0] = RMSE
        scores[k,1] = MAE
        scores[k,2] = NLPD

    # print ecaluation
    # uncomment if you want to save the evaluation
    #np.savetxt('MOMoGP_scores_parkinsons.csv', scores, delimiter=',')
    print(f"MOMoGP  RMSE: {RMSEE}")
    print(f"MOMoGP  MAE: {MAEE}")
    print(f"MOMoGP  NLPD: {NLPDD}")
    print(f"MOMoGP  RMSE mean: {np.mean(np.array(RMSEE))} std:{np.std(np.array(RMSEE))}")
    print(f"MOMoGP  MAE mean: {np.mean(np.array(MAEE))} std:{np.std(np.array(MAEE))}")
    print(f"MOMoGP  NLPD mean: {np.mean(np.array(NLPDD))} std:{np.std(np.array(NLPDD))}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Args go here
    parser.add_argument('--data', type=str, default='parkinsons',
                        help='Select dataset')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning Rate in optimization')
    parser.add_argument('--M', type=int, default=None,
                        help='Threshold of observations in the subspace')
    parser.add_argument('--Kpx', type=int, default=None,
                        help='Number of splits in the covariate space')
    parser.add_argument('--epoch', type=int, default=None,
                        help='Number of epochs in GP leaf optimization')
    parser.add_argument('--cuda', dest='cuda', action='store_true')
    parser.add_argument('--cpu', dest='cuda', action='store_false')
    parser.set_defaults(cuda=True)
    parser.add_argument('--rerun', type=int, default=1,
                        help='Rerun MOMoGP')
    # set args
    args, unparsed = parser.parse_known_args()

    train_MOMoGP(args)




















