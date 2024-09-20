import torch
import torch.optim as optim
import pytorch_lightning as pl
import sys
sys.path.append("/DRIVEcodeV1/DCG_Core")
from main import *
from module_copy1 import *
from model_copy1 import *
from model_copy2 import *
from pytorch_lightning.callbacks.progress import TQDMProgressBar
import yaml
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import argparse
from pathlib import Path
import pandas as pd
import os

#conduct model parameter attack after model training
#Please replace the following directory path with your own path.

parser = get_arg_parser()
args = parser.parse_args()

model = VTN(multitask=args.task, backbone=args.backbone, concept_features=args.concept_features, device=f"cuda:0", train_concepts=args.train_concepts)
teachmodel = TVTN(multitask=args.task, backbone=args.backbone, concept_features=args.concept_features, device=f"cuda:0", train_concepts=args.train_concepts)
save_path ="/hpc2hdd/home/tianlangxue/XAI4AD/concept_gridlock/my_save_path"
checkpoint_path = '/hpc2hdd/home/tianlangxue/XAI4AD/comma2k19data/ckpts_final/ckpts_final_comma_angle_none_True_1/lightning_logs/version_119_40epoch_5loss/checkpoints/epoch=34-step=4200.ckpt'
cleanmodule = LaneModule.load_from_checkpoint(
    checkpoint_path=checkpoint_path,
    model=model, 
    teachmodel=teachmodel,
    multitask=args.task,
    dataset="comma",
    dataset_path="/hpc2hdd/home/tianlangxue/XAI4AD/comma2k19data",
    bs=2,  
    ground_truth="desired",
)



model.eval()

def unfreeze_all_parameters(module):
    for param in module.model.parameters():
        param.requires_grad = True

unfreeze_all_parameters(cleanmodule)

def add_noise_to_parameters(module, noise_scale=0.01):
    for param in module.parameters():
        param.data += torch.randn_like(param) * noise_scale
    return module


noisemodel = add_noise_to_parameters(cleanmodule)
new_checkpoint_path = '/hpc2hdd/home/tianlangxue/XAI4AD/comma2k19data/ckpts_final/ckpts_final_comma_angle_none_True_1/lightning_logs/version_119_40epoch_5loss/checkpoints/epoch=34-step=4200_GNV1.ckpt'

original_checkpoint = torch.load(checkpoint_path)

new_checkpoint = {
    'pytorch-lightning_version': original_checkpoint.get('pytorch-lightning_version', pl.__version__),
    'hyper_parameters': vars(args),  
    'state_dict': noisemodel.state_dict(),
}

for key in original_checkpoint:
    if key not in new_checkpoint and key not in ['state_dict', 'hyper_parameters']:
        new_checkpoint[key] = original_checkpoint[key]

torch.save(new_checkpoint, new_checkpoint_path)
print(f'New checkpoint with noise saved to {new_checkpoint_path}')

trainer = pl.Trainer(
    fast_dev_run=args.dev_run,
    gpus=1,
    accelerator='gpu',
    devices=[args.gpu_num] if torch.cuda.is_available() else None,
    max_epochs=args.max_epochs,
)

predictions = trainer.predict(noisemodel, ckpt_path=new_checkpoint_path)
for pred in predictions:
    if args.task != "multitask":
        predictions, preds_1, preds_2 = pred[0], pred[1], pred[2]
        save_preds(predictions, preds_1, f"{args.dataset}_{args.task}_{args.backbone}_{args.concept_features}_{args.n_scenarios}", save_path)
    else:
        preds, angle, dist = pred[0], pred[1], pred[2]
        preds_angle, preds_dist = preds[0], preds[1]
        save_preds(preds_angle, angle, f"angle_multi_{args.dataset}_{args.task}_{args.backbone}_{args.concept_features}", save_path)
        save_preds(preds_dist, dist, f"dist_multi_{args.dataset}_{args.task}_{args.backbone}_{args.concept_features}", save_path)