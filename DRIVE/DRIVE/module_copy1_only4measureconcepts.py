import pytorch_lightning as pl
import torch
import sys
sys.path.append("/DRIVEcodeV1/DCG_Core")
from dataloader import *
from dataloader_comma import *
from dataloader_nuscenes import * 
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torch.nn as nn 
from utils import pad_collate
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import copy
from PGD import PGD_input, PGD_layer
import time
import re
from losses import topk_overlap_loss_batch


class LaneModule(pl.LightningModule):
    '''Pytorch lightning module to train angle, distance or multitask procedures'''

    def __init__(self, model,teachmodel, bs, multitask="angle", dataset="comma", topkbatchloss=topk_overlap_loss_batch,time_horizon=1, ground_truth="desired", intervention=False, dataset_path=None,img_noise =None, dataset_fraction=1.0):
        super(LaneModule, self).__init__()
        self.dataset_fraction = dataset_fraction
        self.model = model
        self.dataset = dataset
        self.ground_truth = ground_truth
        self.intervention = intervention
        self.dataset_path = dataset_path
        self.num_workers = 8
        self.multitask = multitask
        self.bs = bs
        self.time_horizon = time_horizon
        self.img_noise =img_noise
        self.loss = self.mse_loss
        self.bce_loss = nn.BCELoss()
        self.topk_loss_batch = topkbatchloss
        self.pgd_img = None
        self.pgd_conc = None

        self.teachmodel = teachmodel
        self.freeze_teachmodel()
   
    def freeze_teachmodel(self):
  
        for param in self.teachmodel.parameters():
            param.requires_grad = False

        # self.pgd_img =PGD_input(self.model,loss_function=self.calculate_loss,bs=self.bs)
        # self.pgd_conc=PGD_layer(self.model,loss_function=self.calculate_loss,bs=self.bs)
    def forward(self, x, angle, distance, vego):
        return self.model(x, angle, distance, vego)


    def on_load_checkpoint(self, checkpoint):
    
        state_dict = checkpoint['state_dict']
        model_state_dict = {re.sub('^model\\.', '', k): v for k, v in state_dict.items() if k.startswith('model.')}
        self.model.load_state_dict(model_state_dict)
        teachmodel_state_dict = {re.sub('^model\\.', '',k): v for k, v in state_dict.items() if k.startswith('model.')}
        self.teachmodel.load_state_dict(teachmodel_state_dict)
        
        self.teachmodel.eval()

        self.to('cuda')
        
        return self
    
    def on_save_checkpoint(self, checkpoint):
       
        model_state_dict = self.model.state_dict()
        teachmodel_state_dict = self.teachmodel.state_dict()

        checkpoint['state_dict'] = {
            **{'model.' + k: v for k, v in model_state_dict.items()},
            **{'teachmodel.' + k: v for k, v in teachmodel_state_dict.items()},
        }


        return checkpoint

    def create_pgd_objects(self, model, bs, lf):
        pgd_img = PGD_input(model=self.model, loss_function=lf, bs=bs)
        pgd_conc = PGD_layer(model=self.model, loss_function=lf, bs=bs)
        return pgd_img, pgd_conc

    #perform a PGD attack per epoch
    def on_train_epoch_start(self):
         self.pgd_img, self.pgd_conc = self.create_pgd_objects(self.model, self.bs, self.calculate_loss)

    def calculate_diff(self,model1, model2):
        state_dict_1 = model1.state_dict()
        state_dict_2 = model2.state_dict()
        diff_dict = {}
        assert set(state_dict_1.keys()) == set(state_dict_2.keys()), "Keys do not match between the two models."
        for key in state_dict_1.keys():
            diff_tensor = state_dict_1[key] - state_dict_2[key]
            diff_dict[key] = diff_tensor

        return diff_dict


    def mse_loss(self, input, target, mask, reduction="mean"):
        input = input.float()
        target = target.float()

        out = (input[~mask]-target[~mask])**2
        return out.mean() if reduction == "mean" else out 

    def calculate_loss(self, logits, angle, distance):
        sm = nn.Softmax(dim=1)
        if self.multitask == "multitask":
            logits_angle, logits_dist, param_angle, param_dist = logits
            # import pdb; pdb.set_trace()
            mask = distance.squeeze() == 0.0
            if not self.intervention:
                loss_angle = torch.sqrt(self.loss(logits_angle.squeeze(), angle.squeeze(), mask))
            else: 

                angle, distance = distance, angle
                mask = distance.squeeze() == 0.0
                loss_angle = self.bce_loss(sm(logits_angle.float()).squeeze()[~mask], angle.float().squeeze()[~mask])
            loss_distance = torch.sqrt(self.loss(logits_dist.squeeze(), distance.squeeze(), mask))
            if loss_angle.isnan() or loss_distance.isnan():
                print("ERROR,loss_angle.isnan() or loss_distance.isnan()")
            loss = loss_angle, loss_distance
            self.log_dict({"train_loss_angle": loss_angle}, on_epoch=True, batch_size=self.bs)
            self.log_dict({"train_loss_distance": loss_distance}, on_epoch=True, batch_size=self.bs)
            return loss_angle, loss_distance, param_angle, param_dist
        else:
            mask = distance.squeeze() == 0.0
            loss = torch.sqrt(self.loss(logits.squeeze(), angle.squeeze(), mask))
            return loss
    

    def training_step(self, batch, batch_idx):
        self.model.train()

        _, image_array, vego, angle, distance, m_lens, i_lens, s_lens, a_lens, d_lens = batch
        time0 =time.time()
        adv_images = self.pgd_img.perturb(batch)

        time1 =time.time()
        self.teachmodel.eval()
        self.model.train()
        self.teachmodel.eval()
        logits, attns,probs = self.model(image_array, angle, distance, vego)
        logits_ptb,attns_ptb,probs_ptb = self.model(adv_images,angle, distance, vego)
        logits_tc, attns_tc,probs_tc = self.teachmodel(image_array, angle, distance, vego)
        time2 =time.time()
        loss1 = self.calculate_loss(logits, angle, distance)#this model-gt
        loss4 = torch.mean(self.topk_loss_batch(probs,probs_ptb))
        loss5 = torch.mean(self.topk_loss_batch(probs_tc,probs))

        if self.multitask == "multitask":
  
            loss2 = self.calculate_loss(logits_ptb,logits[0],logits[1])
            loss3 = self.calculate_loss(logits,logits_tc[0],logits_tc[1])
            loss_angle1, loss_dist1, param_angle1, param_dist1 = loss1
            loss_angle2, loss_dist2, param_angle2, param_dist2 = loss2
            loss_angle3, loss_dist3, param_angle3, param_dist3 = loss3
            param_angle, param_dist = 0.3, 0.7
            loss_1m = (param_angle * loss_angle1) + (param_dist * loss_dist1)
            loss_2m = (param_angle * loss_angle2) + (param_dist * loss_dist2)
            loss_3m = (param_angle * loss_angle3) + (param_dist * loss_dist3)
            self.log_dict({"loss_1m": loss_1m}, on_epoch=True, batch_size=self.bs)
            self.log_dict({"loss_dist1": loss_dist1}, on_epoch=True, batch_size=self.bs)
            self.log_dict({"loss_angle1": loss_angle1}, on_epoch=True, batch_size=self.bs)
            self.log_dict({"loss_dist2": loss_dist2}, on_epoch=True, batch_size=self.bs)
            self.log_dict({"loss_angle2": loss_angle2}, on_epoch=True, batch_size=self.bs)
            
           
            # loss = loss_1m+0.01*loss_2m+0.01*loss_3m+1*torch.mean(loss4)+300*torch.mean(loss5)#sum losses with certain weights;ablation study by modify its   composition 
            # loss = loss_1m+0.01*loss_2m+0.01*loss_3m#ablation
            loss = loss_1m+0.3*torch.mean(loss4)+300*torch.mean(loss5)
        else:
            loss2 = self.calculate_loss(logits_ptb,logits,distance)#perturbed inputs-clean input
            loss3 = self.calculate_loss(logits,logits_tc,distance)#this model-teach model
            loss = loss1+0.07*loss2+0.07*loss3+300*torch.mean(loss4)+7000*torch.mean(loss5)#abaltion 
            # loss =loss1+0.05*loss2+0.05*loss3
            # loss =loss1+1*torch.mean(loss4)+300*torch.mean(loss5)#A D E loss
            self.log_dict({"loss1": loss1}, on_epoch=True, batch_size=self.bs)
            self.log_dict({"loss2": loss2}, on_epoch=True, batch_size=self.bs)
            self.log_dict({"loss3": loss3}, on_epoch=True, batch_size=self.bs)
        time3 =time.time()
        
        # print("pgd cost time=",time1-time0)
        # print("model forward time=",time2-time1)
        # print("loss calculaiton time =",time3-time2)

        self.log_dict({"loss4": loss4}, on_epoch=True, batch_size=self.bs)
        self.log_dict({"loss5": loss5}, on_epoch=True, batch_size=self.bs)
        self.log_dict({"train_loss": loss}, on_epoch=True, batch_size=self.bs)
        return loss
    

    def predict_step(self, batch, batch_idx):
        _, image_array, vego, angle, distance, m_lens, i_lens, s_lens, a_lens, d_lens = batch
        # noise = torch.randn(image_array)*0.08
        # noise_image_array = image_array+noise
        if self.time_horizon > 1:
            logits_all = []
            logits_all_noise=[]

            for i in range(self.time_horizon, vego.shape[1], self.time_horizon):
                for j in range(self.time_horizon):
                    input_ids_img, input_ids_vego, input_ids_angle, input_ids_distance = image_array[:,0:i+j, :, :, :], vego[:,0:i+j], angle[:,0:i+j], distance[:,0:i+j]
                    # input_ids_img_n=noise_image_array[:,0:i+j, :, :, :]
                    if self.multitask == "angle" and len(logits_all) > 0:
                        angle[:,i+j] = torch.tensor(logits_all)[-1]
                    if self.multitask == "distance" and len(logits_all) > 0:
                        distance[:,i+j] = torch.tensor(logits_all)[-1]
                    if self.multitask == "multitask":
                        logits, attns,_ = self.model(input_ids_img, input_ids_angle, input_ids_distance, input_ids_vego)
                        # logits_noise, attns,_ = self.model(input_ids_img_n, input_ids_angle, input_ids_distance, input_ids_vego)

                        logits = logits[0][:, -1], logits[1][:, -1]
                        # logits_noise = logits_noise[0][:, -1], logits[1][:, -1]
                    else:
                        logits, attns,_ = self.model(input_ids_img, input_ids_angle, input_ids_distance, input_ids_vego)[:, -1]
                        # logits_noise, attns,_ = self.model(input_ids_img_n, input_ids_angle, input_ids_distance, input_ids_vego)[:, -1]

                    logits_all.append(logits)
                    # logits_all_noise.append(logits_noise)
            return torch.tensor(logits_all), angle[:,self.time_horizon:], distance[:,self.time_horizon:]

        
        # logits_noise, attns,concepts = self.model(noise_image_array, angle, distance, vego)
        logits, attns,concepts = self.model(image_array, angle, distance, vego)
        return logits, angle, distance,concepts


    def validation_step(self, batch, batch_idx):
        _, image_array, vego, angle, distance, m_lens, i_lens, s_lens, a_lens, d_lens = batch
        logits, attns, _ = self.model(image_array, angle, distance, vego)
        loss = self.calculate_loss(logits, angle, distance)
        if self.multitask == "multitask":
            loss_angle, loss_dist, param_angle, param_dist = loss
            param_angle, param_dist = 0.3, 0.7
            loss = (param_angle * loss_angle) + (param_dist * loss_dist)
            self.log_dict({"val_loss_dist": loss_dist}, on_epoch=True, batch_size=self.bs)
            self.log_dict({"val_loss_angle": loss_angle}, on_epoch=True, batch_size=self.bs)
        self.log_dict({"val_loss": loss}, on_epoch=True, batch_size=self.bs)
        
        return loss

    def test_step(self, batch, batch_idx):
        _, image_array, vego, angle, distance, m_lens, i_lens, s_lens, a_lens, d_lens = batch
        if self.time_horizon > 1:
            logits_all = []
            for i in range(self.time_horizon,vego.shape[1], self.time_horizon):
                for j in range(self.time_horizon)+1:
                    input_ids_img, input_ids_vego, input_ids_angle, input_ids_distance = image_array[:,0:i+j, :, :, :], vego[:,0:i+j], angle[:,0:i+j], distance[:,0:i+j]
                    if self.multitask == "angle":
                        angle[:,i+j] = logits[:,-1]
                    if self.multitask == "distance":
                        distance[:,i+j] = input_ids_distance[:,-1]
                    logits, attns = self(input_ids_img, input_ids_angle, input_ids_distance, input_ids_vego)[:, -1]
                    logits_all.append(logits)
            loss = self.calculate_loss(torch.tensor(logits_all), angle[:,self.time_horizon:], distance[:,self.time_horizon:])
            self.log_dict({"test_loss": loss}, on_epoch=True, batch_size=self.bs)
            return loss
    
        _, image_array, vego, angle, distance, m_lens, i_lens, s_lens, a_lens, d_lens = batch
        logits, attns = self(image_array, angle, distance, vego)
        loss = self.calculate_loss(logits, angle, distance)
        if self.multitask == "multitask":
            loss_angle, loss_dist, param_angle, param_dist = loss
            param_angle, param_dist = 0.3, 0.7
            loss = (param_angle * loss_angle) + (param_dist * loss_dist)
            self.log_dict({"test_loss_dist": loss_dist}, on_epoch=True, batch_size=self.bs)
            self.log_dict({"test_loss_angle": loss_angle}, on_epoch=True, batch_size=self.bs)
        self.log_dict({"test_loss": loss}, on_epoch=True, batch_size=self.bs)
        return loss

    def training_epoch_end(self, outputs):
        losses = torch.mean(torch.stack([x['loss'] for x in outputs]))
        self.log_dict({"train_loss_accumulated": losses }, batch_size=self.bs)

    def validation_epoch_end(self, outputs):
        losses = torch.mean(torch.stack([x for x in outputs]))
        self.log_dict({"val_loss_accumulated": losses }, batch_size=self.bs)

    def test_epoch_end(self, outputs):
        losses = torch.mean(torch.stack([x for x in outputs]))
        self.log_dict({"test_loss_accumulated": losses }, batch_size=self.bs)

    def train_dataloader(self):
        return self.get_dataloader(dataset_type="train")

    def val_dataloader(self):
        return self.get_dataloader(dataset_type="val")

    def test_dataloader(self):
        return self.get_dataloader(dataset_type="test")

    def predict_dataloader(self):
        return self.get_dataloader(dataset_type="test")

    def configure_optimizers(self):
        # g_opt = torch.optim.Adam(self.model.parameters(), lr=1e-5, weight_decay=1e-5)
        g_opt = torch.optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-5)

        return g_opt

    def get_dataloader(self, dataset_type):
        if self.dataset == "once":
            ds = ONCEDataset(dataset_type=dataset_type, multitask=self.multitask) 
        elif self.dataset == "comma":
            ds = CommaDataset(dataset_type=dataset_type,img_noise=self.img_noise, multitask=self.multitask if not self.intervention else "intervention", ground_truth=self.ground_truth, dataset_path=self.dataset_path, dataset_fraction=self.dataset_fraction)
        elif self.dataset == 'nuscenes':
            ds = NUScenesDataset(dataset_type=dataset_type, multitask=self.multitask if not self.intervention else "intervention", ground_truth=self.ground_truth, max_len=20, dataset_path=self.dataset_path, dataset_fraction=self.dataset_fraction)
        return DataLoader(ds, batch_size=self.bs, num_workers=self.num_workers, collate_fn=pad_collate)
        