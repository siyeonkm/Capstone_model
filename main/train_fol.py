import sys
import os
import numpy as np
import time

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils import data
from torchsummaryX import summary

from lib.utils.train_val_utils import train_fol_ego, val_fol_ego
from lib.models.rnn_ed import FolRNNED, EgoRNNED
from lib.utils.fol_dataloader import HEVIDataset
from config.config import *

from tensorboardX import SummaryWriter

# device 설정
print("Cuda available: ", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load args
args = parse_args()

# 모델 초기화
fol_model = FolRNNED(args).to(device)
all_params = fol_model.parameters()

if args.with_ego:
    print("Initializing pre-trained ego motion predictor...")
    ego_pred_model = EgoRNNED(args).to(device)
    ego_pred_model.load_state_dict(torch.load(args.best_ego_pred_model))
    print("Pre-trained ego_motion predictor done!")
    all_params = list(ego_pred_model.parameters()) + list(fol_model.parameters())
optimizer = optim.RMSprop(all_params, lr=args.lr)

# Train, validate 데이터셋 초기화
print("Initializing train and val datasets...")
dataloader_params ={
        "batch_size": args.batch_size,
        "shuffle": args.shuffle,
        "num_workers": args.num_workers
}

val_set = HEVIDataset(args, 'val')
print("Number of validation samples:", val_set.__len__())
val_gen = data.DataLoader(val_set, **dataloader_params)

# model summary 출력
if args.with_ego:
    summary(ego_pred_model,
            torch.zeros(1, args.segment_len, 3).to(device))

summary(fol_model,
        torch.zeros(1, args.segment_len, 4).to(device),
        torch.zeros(1, args.segment_len, 50).to(device),
        torch.zeros(1, args.segment_len, args.pred_timesteps, 3).to(device))

writer = SummaryWriter('summary/fol_ego/exp-1')


### Train
all_val_loss = []
min_loss = 1e6
best_fol_model = None
best_ego_model = None
for epoch in range(1, args.nb_fol_epoch+1):
    train_set = HEVIDataset(args, 'train')
    train_gen = data.DataLoader(train_set, **dataloader_params)
    print("Number of training samples:", train_set.__len__())

    start = time.time()
    # train
    train_loss, train_fol_loss, train_ego_pred_loss = train_fol_ego(epoch,
                                                                args,
                                                                fol_model,
                                                                ego_pred_model,
                                                                optimizer,
                                                                train_gen,
                                                                verbose=True)
    writer.add_scalar('data/train_loss', train_loss, epoch)
    writer.add_scalar('data/train_fol_loss', train_fol_loss, epoch)
    writer.add_scalar('data/train_ego_pred_loss', train_ego_pred_loss, epoch)

    # val
    val_loss, val_fol_loss, val_ego_pred_loss = val_fol_ego(epoch,
                                                    args,
                                                    fol_model,
                                                    ego_pred_model,
                                                    val_gen,
                                                    verbose=True)
    writer.add_scalar('data/val_loss', val_loss, epoch)
    writer.add_scalar('data/val_fol_loss', val_fol_loss, epoch)
    writer.add_scalar('data/val_ego_pred_loss', val_ego_pred_loss, epoch)

    all_val_loss.append(val_loss)

    # print time
    elipse = time.time() - start
    print("Elipse: ", elipse)

    # loss 줄어들때마다 checkpoint 저장
    if val_loss < min_loss:
        try:
            os.remove(best_fol_model)
            os.remove(best_ego_model)
        except:
            pass

        min_loss = val_loss
        saved_fol_model_name = 'fol_epoch_' + str(format(epoch,'03')) + '_loss_%.4f'%val_fol_loss + '.pt'
        saved_ego_pred_model_name = 'ego_pred_epoch_' + str(format(epoch,'03')) + '_loss_%.4f'%val_ego_pred_loss + '.pt'

        print("Saving checkpoints: " + saved_fol_model_name + ' and ' + saved_ego_pred_model_name)
        if not os.path.isdir(args.checkpoint_dir):
            os.mkdir(args.checkpoint_dir)

        torch.save(fol_model.state_dict(), os.path.join(args.checkpoint_dir, saved_fol_model_name))
        torch.save(ego_pred_model.state_dict(), os.path.join(args.checkpoint_dir, saved_ego_pred_model_name))

        best_fol_model = os.path.join(args.checkpoint_dir, saved_fol_model_name)
        best_ego_model = os.path.join(args.checkpoint_dir, saved_ego_pred_model_name)
