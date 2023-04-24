# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 04:03:43 2023

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils
import torch.nn.init as init
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn import metrics
import scipy.spatial as sp
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import argparse
import scipy.io
from hypnettorch.mnets import MLP
from hypnettorch.hnets import HMLP
from sklearn.metrics import roc_auc_score,accuracy_score,f1_score,auc,precision_recall_curve
import warnings
import matplotlib.pyplot as plt
import math
from edl_pytorch import Dirichlet, evidential_classification



class METER(nn.Module):
    def __init__(self, args, in_dim, params, alpha, N,  lr, hlr, device, emb_rate=2,
                 encoder_layer=10, decoder_layer=10, hyperenc_layer=100, hyperdec_layer=100):
        super(METER, self).__init__()
        self.params = params
        self.in_dim = in_dim
        self.out_dim = int(in_dim*emb_rate)
        self.init_data = torch.randn(N, self.in_dim).to(device)
        self.init_data.requires_grad = False
        self.encoder_layer = encoder_layer
        self.decoder_layer = decoder_layer
        self.hyperenc_layer = hyperenc_layer
        self.hyperdec_layer = hyperdec_layer
        '''Static'''
        self.encoder_static = MLP(self.in_dim, self.out_dim, hidden_layers=[self.encoder_layer,self.encoder_layer], 
                                  activation_fn=torch.nn.LeakyReLU(),
                                  no_weights=False)    
        self.decoder_static = MLP(self.out_dim, self.in_dim, hidden_layers=[self.decoder_layer,self.decoder_layer], 
                                  activation_fn=torch.nn.LeakyReLU(),
                                  no_weights=False)
        '''dynamic'''
        self.encoder = MLP(self.in_dim, self.out_dim, hidden_layers=[self.encoder_layer,self.encoder_layer], 
                           activation_fn=torch.nn.LeakyReLU(),
                           no_weights=True)    
        self.decoder = MLP(self.out_dim, self.in_dim, hidden_layers=[self.decoder_layer,self.decoder_layer], 
                           activation_fn=torch.nn.LeakyReLU(),
                           no_weights=True)
        self.hyperen = HMLP(target_shapes=self.encoder.param_shapes,cond_in_size=in_dim,
                            layers=(self.hyperenc_layer,self.hyperenc_layer) )
        self.hyperde = HMLP(target_shapes=self.decoder.param_shapes,cond_in_size=in_dim,
                            layers=(self.hyperdec_layer,self.hyperdec_layer) )

        self.edl_model = nn.Sequential(nn.Linear(self.in_dim, self.in_dim*2),  # two input dim
                                       nn.ReLU(),
                                       Dirichlet(self.in_dim*2, 2),  # two output classes
                                       )
        self.alpha = alpha
        self.clock = 0
        self.last_update = -1
        #self.hyperen.internal_params,self.hyperde.internal_params
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)     
        self.optimizer_d = torch.optim.Adam(self.parameters(), lr=hlr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.96)
        self.scheduler_d = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_d, gamma=0.96)
        self.loss_fn = nn.MSELoss()
        self.loss_la = nn.MSELoss(reduce=False)
        self.count = 0
        self.device = device
        self.args = args
        
    
    def train_autoencoder(self, data, epochs,  mode='dynamic',
                          static_encoder_weight=None,static_decoder_weight=None,thres_rate=0.05,lamb=0.001):
        self.mean, self.std = self.init_data.mean(0), self.init_data.std(0)
        new = (data - self.mean) / self.std
        new[:, self.std == 0] = 0
        new = Variable(new)
        loss_list = np.zeros((epochs))

        for epoch in range(epochs):
            losses_recon = []
            losses_edl = []
            self.optimizer.zero_grad()
            self.optimizer_d.zero_grad()
            train_data_loader = DataLoader(new, batch_size=int(new.shape[0]*0.3))
            for data in train_data_loader:
                if mode == 'dynamic':
                    encoder_weight = self.hyperen.forward(cond_input=data.mean(0).view(1, -1)) # Generate the weights 
                    decoder_weight = self.hyperde.forward(cond_input=data.mean(0).view(1, -1)) # Generate the weights 
                    z,_ =self.encoder.forward(data,weights=encoder_weight)
                    output,_ = self.decoder.forward(z,weights=decoder_weight)
                    
                elif mode == 'static':
                    z,static_encoder_weight = self.encoder_static.forward(data.to(self.device))
                    output,static_decoder_weight = self.decoder_static.forward(z)
                    
                elif mode == 'hybrid':
                    encoder_weight = self.hyperen.forward(cond_input=data.mean(0).view(1, -1)) # Generate the weights 
                    decoder_weight = self.hyperde.forward(cond_input=data.mean(0).view(1, -1)) # Generate the weights 
                    encoder_weight_hybrid = [] 
                    for j in range(len(encoder_weight)):
                        encoder_weight_hybrid.append(encoder_weight[j]+[i for i in static_encoder_weight][j])
                    decoder_weight_hybrid = [] 
                    for j in range(len(decoder_weight)):
                        decoder_weight_hybrid.append(decoder_weight[j]+[i for i in static_decoder_weight][j])  
    
                    z,_ =self.encoder.forward(data.to(self.device),
                                            weights=encoder_weight_hybrid)
                    output,_ = self.decoder.forward(z,weights=decoder_weight_hybrid)                    

                elif mode == 'hybrid+edl':
                    encoder_weight = self.hyperen.forward(cond_input=data.mean(0).view(1, -1)) # Generate the weights 
                    decoder_weight = self.hyperde.forward(cond_input=data.mean(0).view(1, -1)) # Generate the weights 
                    encoder_weight_hybrid = [] 
                    for j in range(len(encoder_weight)):
                        encoder_weight_hybrid.append(encoder_weight[j]+[i for i in static_encoder_weight][j])
                    decoder_weight_hybrid = [] 
                    for j in range(len(decoder_weight)):
                        decoder_weight_hybrid.append(decoder_weight[j]+[i for i in static_decoder_weight][j])  
                    z,_ =self.encoder.forward(data.to(self.device),
                                            weights=encoder_weight_hybrid)
                    output,_ = self.decoder.forward(z,weights=decoder_weight_hybrid)
                    
                else:
                    raise Exception('wrong mode setting')
                loss_recon = self.loss_fn(output, data) 
                if mode == 'hybrid+edl':
                #edl detection
                    static_z,_ = self.encoder.forward(new.to(self.device), weights=static_encoder_weight)
                    static_output,_ = self.decoder.forward(static_z,weights=static_decoder_weight)
                    total_static_loss = self.loss_la(static_output, new).mean(-1) 
                    thres = total_static_loss.reshape(-1,1).sort(0,True)[0][int(len(total_static_loss.reshape(-1,1))*thres_rate)]
                    # print('output.shape:', output.shape)
                    fake_label = torch.zeros((output.shape[0],output.shape[1],1))  #,output.shape[2]
                    fake_label = torch.from_numpy(np.array(np.where(total_static_loss>thres, 1, 0),dtype=np.int64))
                    
                    pred_dirchlet = self.edl_model(static_output) # new
                    loss_edl = evidential_classification(pred_dirchlet, fake_label, lamb=lamb) # regularization coefficient 
                
                    loss = loss_recon + loss_edl
                    losses_edl.append(loss_edl.cpu().detach().numpy())
                else:
                    loss = loss_recon 
                losses_recon.append(loss_recon.cpu().detach().numpy())
                loss.backward()   #retain_graph=True
                if mode == 'static':
                    self.scheduler.step()
                else:
                    self.scheduler_d.step() 

            loss_list[epoch] = loss

          
        if mode == 'dynamic':
            return encoder_weight, decoder_weight
        elif mode == 'static':
            return static_encoder_weight, static_decoder_weight
        elif mode in ['hybrid' , 'hybrid+edl']:
            return encoder_weight_hybrid, decoder_weight_hybrid
            

    def forward(self, x, static_encoder_weight=None, static_decoder_weight=None):
        new = (x - self.mean) / self.std
        new[:, self.std == 0] = 0
        encoder_weight = self.hyperen.forward(cond_input=new.view(1, -1)) 
        decoder_weight = self.hyperde.forward(cond_input=new.view(1, -1)) 
        expected_p = None
        data_uncertainty = None 
        distributional_uncertainty = None
        use_dynamic = 0
        self.count = 0
        if self.args.mode == 'dynamic':
            z_emb,_ = self.encoder.forward(new,weights=encoder_weight)
            output,_ = self.decoder.forward(z_emb,weights=decoder_weight)
        elif self.args.mode == 'static':
            z_emb,_ = self.encoder_static.forward(new)
            output,_ = self.decoder_static.forward(z_emb)
        elif self.args.mode == 'hybrid':
            encoder_weight_hybrid = [] 
            for j in range(len(encoder_weight)):
                encoder_weight_hybrid.append(encoder_weight[j]+[i for i in static_encoder_weight][j])
            decoder_weight_hybrid = [] 
            for j in range(len(decoder_weight)):
                decoder_weight_hybrid.append(decoder_weight[j]+[i for i in static_decoder_weight][j]) 
            z_emb,_ = self.encoder.forward(new,weights=encoder_weight_hybrid)
            output,_ = self.decoder.forward(z_emb,weights=decoder_weight_hybrid)
        elif self.args.mode == 'hybrid+edl':
            encoder_weight_hybrid = [] 
            for j in range(len(encoder_weight)):
                encoder_weight_hybrid.append(encoder_weight[j]+[i for i in static_encoder_weight][j])
            decoder_weight_hybrid = [] 
            for j in range(len(decoder_weight)):
                decoder_weight_hybrid.append(decoder_weight[j]+[i for i in static_decoder_weight][j]) 
            #static
            z_emb,_ = self.encoder_static.forward(new)
            output,_ = self.decoder_static.forward(z_emb)
            
            pred_sd = self.edl_model(output)
            # print('pred_sd', pred_sd, output, z_emb)
            total_pred_sd = pred_sd.sum(-1, keepdims=True)
            # print('total_pred_sd', total_pred_sd.shape)
            expected_p = pred_sd / total_pred_sd
            eps = 1e-7
            # distributional uncertainty of each sample
            point_entropy = - torch.sum(expected_p * torch.log(expected_p + eps), dim=1)
            data_uncertainty = torch.sum((pred_sd / total_pred_sd) * 
                                         (torch.digamma(total_pred_sd + 1) - torch.digamma(pred_sd + 1)), dim=1)
            # print('data_uncertainty', data_uncertainty)
            distributional_uncertainty = point_entropy - data_uncertainty
            # print(distributional_uncertainty)
            distributional_uncertainty_threshold = self.args.uncertainty_threshold
            # print('fake_label:', expected_p[0][0].round())
            
            if expected_p[0][0].round() == 0 or distributional_uncertainty>distributional_uncertainty_threshold:   #dynamic
                # print('dynamic')
                use_dynamic = 1
                self.count = self.count + 1
                z_emb,_ = self.encoder.forward(new,weights=encoder_weight_hybrid)
                output,_ = self.decoder.forward(z_emb,weights=decoder_weight_hybrid)
        else:
            raise Exception('wrong mode setting')

        loss_values = torch.norm(new - output,  p=1) 
        score = loss_values 
        if expected_p == None:
            return score ,self.count, expected_p, None, data_uncertainty , distributional_uncertainty,use_dynamic
        else:
            return score ,self.count, expected_p, expected_p[0][0].round(), data_uncertainty , distributional_uncertainty,use_dynamic
