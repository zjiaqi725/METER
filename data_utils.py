import pandas as pd
import numpy as np
import torch
import scipy.io
from sklearn.utils import shuffle
from utils import shingle

def load_dataset(args):
    
    nfile = None
    lfile = None
    if args.dataset == 'NSL':
        nfile = 'datasets/nsl.txt'
        lfile = 'datasets/nsllabel.txt'
    elif args.dataset == 'KDD':
        nfile = 'datasets/kdd.txt'
        lfile = 'datasets/kddlabel.txt'
    elif args.dataset == 'AT':
        data = pd.read_csv('datasets/ambient_temperature_system_failure.csv')
        numeric = data["value"].values#.reshape(-1, 1)
        labels = data["label"].values
        X = shingle(numeric, 10) # shape (windowsize, len-win+1)
        numeric = torch.FloatTensor(np.transpose(X))
        t1, _ = np.shape(numeric) 
        labels=labels[:t1]
    elif args.dataset == 'CPU':
        data = pd.read_csv('datasets/cpu_utilization_asg_misconfiguration.csv')
        numeric = data["value"].values
        labels = data["label"].values
        X = shingle(numeric, 10)
        numeric = torch.FloatTensor(np.transpose(X))
        t1, _ = np.shape(numeric) 
        labels=labels[:t1]
    elif args.dataset == 'MT':
        data = pd.read_csv('datasets/machine_temperature_system_failure.csv')
        numeric = data["value"].values
        labels = data["label"].values
        X = shingle(numeric, 10) 
        numeric = torch.FloatTensor(np.transpose(X))
        t1, _ = np.shape(numeric) 
        labels=labels[:t1]
    elif args.dataset == 'NYC':
        data = pd.read_csv('datasets/nyc_taxi.csv')
        numeric = data["value"].values
        labels = data["label"].values
        X = shingle(numeric, 10) 
        numeric = torch.FloatTensor(np.transpose(X))
        t1, _ = np.shape(numeric) 
        labels=labels[:t1]
    elif args.dataset in ["MNIST_AbrRec", "MNIST_GrdRec", "F_MNIST_AbrRec",
                          "F_MNIST_GrdRec", "GAS", "RIALTO", "INSECTS_Abr", "INSECTS_Incr",
                          "INSECTS_IncrGrd", "INSECTS_IncrRecr"]:
        data = pd.read_csv("datasets/"+args.dataset+".csv", dtype=np.float64, header=None)
        data_label = data.pop(data.columns[-1])
        numeric = torch.FloatTensor(data.values)
        labels = data_label.values.reshape(-1)   
    else:
        df = scipy.io.loadmat('datasets/'+args.dataset+".mat")
        numeric = torch.FloatTensor(df['X'])
        labels = (df['y']).astype(float).reshape(-1)
            
    device = torch.device(args.dev if torch.cuda.is_available() else "cpu")
    
    if args.dataset in ['KDD', 'NSL']:
        numeric = torch.FloatTensor(np.loadtxt(nfile, delimiter = ','))
        labels = np.loadtxt(lfile, delimiter=',')
    if args.dataset == 'KDD':
        labels = 1 - labels
    
    return numeric, labels
        

    
    
            
