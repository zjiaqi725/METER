# -*- coding: utf-8 -*-
"""
@Project: METERCoder20230310
@File:    METER
@Time:    2023/03/10 11:30
@Description: The METER for Unsupervised Anomaly Detection 
in a streaming/online manner with the problem of concept drift
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils
import torch.nn.init as init
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn import metrics
import scipy.spatial as sp
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import argparse
import scipy.io
import warnings
import math
import os
import h5py
import pickle
from METER import METER
from data_utils import load_dataset
from utils import get_result

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
torch.set_num_threads(1)

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='NSL')
parser.add_argument("--mode", default='static')
parser.add_argument("--emb_rate", type=float, help="rate of embedding dim/input dim", default=2)
parser.add_argument("--train_rate", type=float, help="rate of training set", default=0.1)
parser.add_argument("--epochs", type=int, help="number of epochs", default=1000)
parser.add_argument("--lr", type=float, help="learning rate", default=1e-2)
parser.add_argument("--dev", help="device", default="cuda")  
parser.add_argument("--seed", type=int, help="random seed", default=0)
#DSD
parser.add_argument("--encoder_layer", type=int, default=10)
parser.add_argument("--decoder_layer", type=int, default=10)
parser.add_argument("--hyperenc_layer", type=int, default=100)
parser.add_argument("--hyperdec_layer", type=int, default=100)
parser.add_argument("--hlr", type=float, help="learning rate of hypernetwork", default=1e-2)
#IEC
parser.add_argument("--thres_rate", type=float, help="threshold rate for the pseudo labels from SCD", default=0.05)
parser.add_argument("--uncertainty_threshold", type=float, help="threshold of the concept uncertainty", default=0.1)
parser.add_argument("--uncertainty_avg_threshold", type=float, help="threshold of the offline updating strategy", default=0.1)
parser.add_argument("--alpha", type=float, help="smoothing factor", default=0.5)
args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
device = torch.device(args.dev if torch.cuda.is_available() else "cpu")


numeric, labels = load_dataset(args)
N = int(args.train_rate*len(numeric))
emb_rate = args.emb_rate
alpha = args.alpha
params = {
          'train_rate': N, 'batch_size':1, 'lr':args.lr
         }
model = METER(args, numeric[0].shape[0],params,alpha,N=N, lr=args.lr, hlr=args.hlr, device=device, emb_rate=emb_rate,
                       encoder_layer=args.encoder_layer, decoder_layer=args.decoder_layer, 
                       hyperenc_layer=args.hyperenc_layer, hyperdec_layer=args.hyperdec_layer).to(device)
def weights_init(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight.data)  
model.apply(weights_init) 


batch_size = params['batch_size']
print("------------------METER------------------") 
print('dataset:',args.dataset, 'rate of training set:', args.train_rate, 'lr:', args.lr, 'epochs:', args.epochs,"random seed:", args.seed)
print('in_dim:',numeric[0].shape[0], 'emb_rate:', args.emb_rate, 'embedding dim:', int(numeric[0].shape[0]*args.emb_rate))
print('dataset size:',numeric.shape, 'anomaly rate:', sum(labels)/len(labels))

labels_infer = np.delete(labels, np.where(labels==0)[0][:N])
numeric_infer = torch.FloatTensor(np.delete(numeric.numpy(), np.where(labels==0)[0][:N],0))
# print('###',numeric.shape,numeric_infer.shape)
data_loader = DataLoader(numeric_infer, batch_size=batch_size)
init_data = numeric[labels == 0][:N].to(device)   
model.init_data = init_data
print('init_data:', init_data.shape)

torch.set_grad_enabled(True)
if args.mode == 'static':
    static_encoder_weight, static_decoder_weight = model.train_autoencoder(Variable(init_data).to(device), 
                                                                            epochs=args.epochs, batch_size=trans_batch_size, 
                                                                            mode=args.mode)
elif args.mode == 'dynamic':
    encoder_weight,decoder_weight = model.train_autoencoder(Variable(init_data).to(device), epochs=args.epochs, 
                                                            batch_size=trans_batch_size, mode=args.mode)
elif args.mode == 'hybrid':
    static_encoder_weight, static_decoder_weight = model.train_autoencoder(Variable(init_data).to(device), 
                                                                            epochs=args.epochs, mode='static')
    encoder_weight_hybrid, decoder_weight_hybrid = model.train_autoencoder(Variable(init_data).to(device), epochs=2*args.epochs,  mode='hybrid',
                            static_encoder_weight=static_encoder_weight, static_decoder_weight=static_decoder_weight) 
elif args.mode == 'hybrid+edl':
    static_encoder_weight, static_decoder_weight = model.train_autoencoder(Variable(init_data).to(device), 
                                                                            epochs=args.epochs, 
                                                                            mode='static')
    encoder_weight_hybrid, decoder_weight_hybrid = model.train_autoencoder(Variable(init_data).to(device), epochs=2*args.epochs,
                                                              mode='hybrid+edl',
                            static_encoder_weight=static_encoder_weight, static_decoder_weight=static_decoder_weight,
                            thres_rate = args.thres_rate)
else:
    raise Exception('wrong mode setting')
        
torch.set_grad_enabled(False)
if args.mode == 'hybrid+edl':
    data_loader_initial = DataLoader(init_data, batch_size=1)
    uncertainty_list = []
    for data in data_loader_initial:
        _, _, _, _, _, distr_uncertainty,_= model(Variable(data).to(device),static_encoder_weight, static_decoder_weight)
        uncertainty_list.append(distr_uncertainty)
    uncertainty_avg_threshold = max(uncertainty_list)  
    #torch.mean(torch.stack(uncertainty_list))  max(uncertainty_list)  
    print('uncertainty_avg_threshold:', uncertainty_avg_threshold, 
          'len_uncertainty_list:',  len(uncertainty_list),  #uncertainty_list ,#
          'uncertainty_avg:', torch.mean(torch.stack(uncertainty_list)))

err = []
pr = []
mod = []
data_un = []
# distr_un = []
use_dynamic = []
uncertainty_list = []
distr_uncertainty_avg = 0
start = time.time()
update_time = 0
count_overuncert = 0
max_count_uncert = 10
window_size = 50
for i, data in enumerate(data_loader):
    output, count_dy, expected_p,fake_label, data_uncertainty, distr_uncertainty,use_dy = model(data.to(device),
                                                                          static_encoder_weight, static_decoder_weight)
    if expected_p != None:
        # distr_uncertainty_avg = args.alpha*distr_uncertainty + (1-args.alpha)*distr_uncertainty_avg
        uncertainty_list.append(distr_uncertainty)        
        if i>=window_size:
            count_overuncert = sum([1 for d in uncertainty_list[i-window_size:i] if d > uncertainty_avg_threshold])
        # cumu_uncert = cumu_uncert + distr_uncertainty_avg/uncertainty_avg_threshold
            if count_overuncert>max_count_uncert:
            # if cumu_uncert>cumu_uncert_threshold:
                print(f'-----------------model update:{update_time+1} from {i}------------------')
                torch.set_grad_enabled(True)
                current_data = numeric_infer[i-window_size:i]  
                static_encoder_weight, static_decoder_weight = model.train_autoencoder(Variable(current_data).to(device), 
                                                                                    epochs=200, batch_size=trans_batch_size, 
                                                                                    mode='static')
                encoder_weight_hybrid, decoder_weight_hybrid = model.train_autoencoder(Variable(current_data).to(device), 
                                                                      epochs=500, batch_size=trans_batch_size, 
                                                                      mode='hybrid+edl',
                                    static_encoder_weight=static_encoder_weight, static_decoder_weight=static_decoder_weight,
                                    thres_rate = args.thres_rate)
                update_time=update_time+1
                uncertainty_avg_threshold = args.alpha*uncertainty_avg_threshold + (1-args.alpha)*torch.mean(torch.stack(uncertainty_list))
                torch.set_grad_enabled(False)
                output, count_dy, expected_p, fake_label, data_uncertainty, distr_uncertainty,use_dy = model(data.to(device))
               
        err.append(output)
        pr.append(expected_p)
        mod.append(1-fake_label)
        data_un.append(data_uncertainty)
        use_dynamic.append(use_dy)
    else:
        err.append(output)
        
end = time.time()
print('infer_time:', end-start)
print('count_dynamic:', count_dy, 'dynamic_routing_rate:', count_dy/len(numeric_infer) )
scores = np.array([i.cpu() for i in err])
fake_label = np.array([i.cpu() for i in mod])
data_uncertainty = np.array([i.cpu() for i in data_un])

metrics_dict, y_pred = get_result(args, labels_infer, scores)

# metrics_dict['dataset shape'] = numeric.shape
log = list(metrics_dict.items())

obj = labels_infer , scores,  data_uncertainty
with open(r"METER_log_"+ args.dataset + ".txt", "wb") as f:
    pickle.dump(obj, f)
    

