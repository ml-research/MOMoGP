"""
Created on June 08, 2021
@author: Zhongjie Yu
@author: Mingye Zhu
"""
import numpy as np
import pandas as pd
import random


random.seed(23)
np.random.seed(23)

def load_data(args):
    """Load data from the given dataset name.

    Parameters
    ----------
    args
        Argument list.

    Returns
    -------
    x_train
        Input training data after normalisation.
    x_test
        Input test data after normalisation.
    y_train
        Output training data after normalisation.
    y_test
        Output test data after normalisation.
    args
        Argument list assigned with default hyperparameters.
    """
    if args.data=="parkinsons":
        x_train = pd.read_csv('./data/Parkinsons/x_train.csv')
        x_test = pd.read_csv('./data/Parkinsons/x_test.csv')
        y_train = pd.read_csv('./data/Parkinsons/y_train.csv')
        y_test = pd.read_csv('./data/Parkinsons/y_test.csv')

        # normalize
        mu_x,std_x = x_train.mean().to_numpy(), x_train.std().to_numpy()
        mu_y,std_y = y_train.mean().to_numpy(), y_train.std().to_numpy()
        x_train = (x_train - mu_x)/std_x
        x_test = (x_test - mu_x)/std_x
        y_train = (y_train - mu_y)/std_y
        y_test = (y_test - mu_y)/std_y
        x_train = x_train.iloc[:,:].values
        x_test = x_test.iloc[:,:].values
        y_train = y_train.iloc[:,:].values
        y_test = y_test.iloc[:,:].values

        # set default haperparameters
        if args.lr is None:
            args.lr =0.1
        if args.M is None:
            args.M = 550 
        if args.Kpx is None:
            args.Kpx = 2
        if args.epoch is None:
            args.epoch = 200

    elif args.data=="scm20d":
        from scipy.io import arff
        d_input = 61
        dataarff = arff.loadarff('./data/scm20d/scm20d.arff')
        data = pd.DataFrame(dataarff[0])
        train = data.sample(frac=0.8, random_state=58)
        test = data.drop(train.index)
        x_train, y_train = train.iloc[:, :d_input].values, train.iloc[:, d_input:].values
        x_test, y_test = test.iloc[:, :d_input].values, test.iloc[:, d_input:].values

        # normalise data
        mu_x,std_x = np.mean(x_train, axis=0), np.std(x_train, axis=0)
        mu_y,std_y = np.mean(y_train, axis=0), np.std(y_train, axis=0)
        x_train = (x_train - mu_x)/std_x
        x_test = (x_test - mu_x)/std_x
        y_train = (y_train - mu_y)/std_y
        y_test = (y_test - mu_y)/std_y

        # Apply PCA
        from sklearn.decomposition import PCA
        pca = PCA(n_components=30)
        pca.fit(x_train)
        x_train = pca.transform(x_train)
        x_test = pca.transform(x_test)
        
        # set default haperparameters
        if args.lr is None:
            args.lr =0.1
        if args.M is None:
            args.M = 1100 
        if args.Kpx is None:
            args.Kpx = 3
        if args.epoch is None:
            args.epoch = 150

    elif args.data=="wind":
        d_input = 8
        data = pd.read_csv('./data/wind/windturbine.csv')
        data = pd.DataFrame(data).dropna()
        train = data.sample(frac=0.8, random_state=58)
        test = data.drop(train.index)
        x_train, y_train = train.iloc[:, :d_input].values, train.iloc[:, d_input:].values
        x_test, y_test = test.iloc[:, :d_input].values, test.iloc[:, d_input:].values

        # normalise data
        mu_x,std_x = np.mean(x_train,axis=0), np.std(x_train,axis=0)
        mu_y,std_y = np.mean(y_train,axis=0), np.std(y_train,axis=0)
        x_train = (x_train - mu_x)/std_x
        x_test = (x_test - mu_x)/std_x
        y_train = (y_train - mu_y)/std_y
        y_test = (y_test - mu_y)/std_y
        
        # set default haperparameters
        if args.lr is None:
            args.lr =0.1
        if args.M is None:
            args.M = 550 
        if args.Kpx is None:
            args.Kpx = 2
        if args.epoch is None:
            args.epoch = 250

    elif args.data=="energy":
        d_input = 32
        data = pd.read_csv('./data/energy/Adelaide_Data.csv')
        data = pd.DataFrame(data).dropna()
        train = data.sample(frac=0.8, random_state=58)
        test = data.drop(train.index)
        x_train, y_train = train.iloc[:, :d_input].values, train.iloc[:, d_input:].values
        x_test, y_test = test.iloc[:, :d_input].values, test.iloc[:, d_input:].values

        # normalise data
        mu_x,std_x = np.mean(x_train,axis=0), np.std(x_train,axis=0)
        mu_y,std_y = np.mean(y_train,axis=0), np.std(y_train,axis=0)
        x_train = (x_train - mu_x)/std_x
        x_test = (x_test - mu_x)/std_x
        y_train = (y_train - mu_y)/std_y
        y_test = (y_test - mu_y)/std_y
        
        # set default haperparameters
        if args.lr is None:
            args.lr =0.1
        if args.M is None:
            args.M = 5500 
        if args.Kpx is None:
            args.Kpx = 4
        if args.epoch is None:
            args.epoch = 150

    elif args.data=="usflight":
        x_train = pd.read_csv('./data/usflight/x_train.csv')
        x_test = pd.read_csv('./data/usflight/x_test.csv')
        y_train = pd.read_csv('./data/usflight/y_train.csv')
        y_test = pd.read_csv('./data/usflight/y_test.csv')

        # normalise data
        mu_x,std_x = x_train.mean().to_numpy(), x_train.std().to_numpy()
        mu_y,std_y = y_train.mean().to_numpy(), y_train.std().to_numpy()
        x_train = (x_train - mu_x)/std_x
        x_test = (x_test - mu_x)/std_x
        y_train = (y_train - mu_y)/std_y
        y_test = (y_test - mu_y)/std_y
        x_train = x_train.iloc[:,:].values
        x_test = x_test.iloc[:,:].values
        y_train = y_train.iloc[:,:].values
        y_test = y_test.iloc[:,:].values
        
        # set default haperparameters
        if args.lr is None:
            args.lr =0.02
        if args.M is None:
            args.M = 5500
        if args.Kpx is None:
            args.Kpx = 10
        if args.epoch is None:
            args.epoch = 150

    else:
        raise Exception("Incorrect dataset, can only be the following:\n pakinsons\n scm20d\n wind\n energy\n usflight\n")

    print(args.data, "loaded")
    print('training data shape, x:', x_train.shape, 'y:', y_train.shape)
    print('test data shape, x:', x_test.shape, 'y:', y_test.shape)

    return x_train, x_test, y_train, y_test, args
