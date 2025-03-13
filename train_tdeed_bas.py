#!/usr/bin/env python3
"""
File containing the main training script to train T-DEED for SN-BAS challenge 2025.
"""

# Standard imports
import argparse
import os
import time
import torch
import torch.nn as nn
import numpy as np
import random
from torch.utils.data import DataLoader
import wandb
import sys
from tqdm import tqdm

# Local imports
from util.io import load_json, store_json, load_text
from dataset.datasets import get_datasets
from model.model import TDEEDModel, update_labels_2heads
from model.modules import step
from torch.optim.lr_scheduler import ChainedScheduler, LinearLR, CosineAnnealingLR
from util.eval import mAPevaluate, mAPevaluateTest
from dataset.frame import ActionSpotVideoDataset


import torch.nn.functional as F
# Constants
EVAL_SPLITS = ['test', 'challenge']
STRIDE = 1
STRIDE_SN = 12
STRIDE_SNB = 2

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='SoccerNetBall_baseline_liujian')
    parser.add_argument('-ag', '--acc_grad_iter', type=int, default=1, help='Use gradient accumulation')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training from')
    return parser.parse_args()

def update_args(args, config):
    args.frame_dir = config['frame_dir']
    args.save_dir = config['save_dir'] + '/' + args.model
    args.store_dir = os.path.join(config['save_dir'], 'StoreClips-dense', config['dataset'])
    args.store_mode = config['store_mode']
    args.batch_size = config['batch_size']
    args.clip_len = config['clip_len']
    args.crop_dim = config['crop_dim']
    args.dataset = config['dataset']
    args.event_team = config['event_team']
    args.radi_displacement = config['radi_displacement']
    args.epoch_num_frames = config['epoch_num_frames']
    args.feature_arch = config['feature_arch']
    args.learning_rate = config['learning_rate']
    args.mixup = config['mixup']
    args.modality = config['modality']
    args.num_classes = config['num_classes']
    args.num_epochs = config['num_epochs']
    args.warm_up_epochs = config['warm_up_epochs']
    args.start_val_epoch = config['start_val_epoch']
    args.temporal_arch = config['temporal_arch']
    args.n_layers = config['n_layers']
    args.sgp_ks = config['sgp_ks']
    args.sgp_r = config['sgp_r']
    args.only_test = config['only_test']
    args.criterion = config['criterion']
    args.num_workers = config['num_workers']
    if 'joint_train' in config:
        args.joint_train = config['joint_train']
        args.joint_train['store_dir'] = os.path.join(args.save_dir, 'StoreClips', args.joint_train['dataset'])
    else:
        args.joint_train = None
    return args

def get_lr_scheduler(args, optimizer, num_steps_per_epoch):
    cosine_epochs = args.num_epochs - args.warm_up_epochs
    print('Using Linear Warmup ({}) + Cosine Annealing LR ({})'.format(args.warm_up_epochs, cosine_epochs))
    return args.num_epochs, ChainedScheduler([
        LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=args.warm_up_epochs * num_steps_per_epoch),
        CosineAnnealingLR(optimizer, num_steps_per_epoch * cosine_epochs)])

def save_checkpoint(model, optimizer, scaler, epoch_idx, args, is_best=False):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch_idx,
        'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
        'args': vars(args),
    }
    latest_path = os.path.join(args.save_dir, 'checkpoint_latest.pt')
    torch.save(checkpoint, latest_path)
    if is_best:
        best_path = os.path.join(args.save_dir, 'checkpoint_best.pt')
        torch.save(checkpoint, best_path)
    print(f"Saved checkpoint to {latest_path}{' and ' + best_path if is_best else ''}")

def load_checkpoint(model, optimizer, scaler, device, checkpoint_path):
    """加载检查点并恢复训练状态"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scaler is not None and checkpoint['scaler_state_dict'] is not None:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Loaded checkpoint from {checkpoint_path}, resuming from epoch {start_epoch}")
    return start_epoch, model, optimizer


# OridianlPredD = nn.Sigmoid()

# def displace_predict(predD):
#     displacement = OridianlPredD(predD)
#     displacement = (torch.sum(displacement>0.5, dim=2))
#     return displacement-4
    
def modify_tensor( input_tensor):
    # 获取输入张量的形状
    batch_size, seq_len, num_classes = input_tensor.shape
    # 找到每行第一个 1 的索引
    first_one_indices = torch.argmax(input_tensor, dim=-1)
    # 创建列索引矩阵，形状为 (1, 1, num_classes)
    column_indices = torch.arange(num_classes, device=input_tensor.device).unsqueeze(0).unsqueeze(0)
    # 扩展 first_one_indices 以进行广播比较，形状变为 (batch_size, seq_len, 1)
    expanded_first_one_indices = first_one_indices.unsqueeze(-1)
    # 创建掩码，判断每个位置的列索引是否小于等于该行第一个 1 的索引
    mask = column_indices <= expanded_first_one_indices
    # 将掩码转换为与输入张量相同的数据类型
    result = mask.to(input_tensor.dtype)
    return result



def epoch(model, loader, device, optimizer=None, scaler=None, lr_scheduler=None,
          acc_grad_iter=1, fg_weight=5, loss_weights=[1, 1, 2], num_classes=None, args=None):
    if optimizer is None:
        inference = True
        model.eval()
    else:
        inference = False
        optimizer.zero_grad()
        model.train()

    ce_kwargs = {}
    if fg_weight != 1:
        ce_kwargs['weight'] = torch.FloatTensor([1] + [fg_weight] * (num_classes - 1)).to(device)

    epoch_lossC = 0.
    if loader.dataset._radi_displacement > 0:
        epoch_lossD = 0.
    if loader.dataset._event_team:
        epoch_lossT = 0.

    for batch_idx, batch in tqdm(enumerate(loader)):
        frame = batch['frame'].to(device).float()
        label = batch['label'].to(device)

        frame_original = frame
        label_original = label

        if hasattr(model, 'module') and model.module._double_head or (not hasattr(model, 'module') and model._double_head):
            batch_dataset = batch['dataset']
            label = update_labels_2heads(label, batch_dataset, args.num_classes)

        if 'labelD' in batch.keys():
            # labelD = batch['labelD'].to(device).float()
            labelD = modify_tensor(nn.functional.one_hot(batch['labelD']+4, num_classes=9)).to(device).float()
                    
        if 'labelT' in batch.keys():
            labelT = batch['labelT'].to(device).float()
        
        if 'frame2' in batch.keys():
            frame2 = batch['frame2'].to(device).float()
            label2 = batch['label2'].to(device)
            if 'labelD2' in batch.keys():
                # labelD2 = batch['labelD2'].to(device).float()
                # labelD_dist = torch.zeros((labelD.shape[0], label.shape[1])).to(device)
                labelD2 = modify_tensor(nn.functional.one_hot(batch['labelD2']+4, num_classes=9)).to(device).float()
                labelD_dist = torch.zeros((labelD.shape)).to(device)
                        
            if 'labelT2' in batch.keys():
                labelT2 = batch['labelT2'].to(device).float()

            # l = [random.betavariate(0.2, 0.2) for _ in range(frame2.shape[0])]
            l = [random.betavariate(10, 10) for _ in range(frame2.shape[0])]
            
            label_dist = torch.zeros((label.shape[0], label.shape[1], num_classes)).to(device)

            for i in range(frame2.shape[0]):
                frame[i] = l[i] * frame[i] + (1 - l[i]) * frame2[i]
                lbl1 = label[i]
                lbl2 = label2[i]
                label_dist[i, range(label.shape[1]), lbl1] += l[i]
                label_dist[i, range(label2.shape[1]), lbl2] += 1 - l[i]
                if 'labelD2' in batch.keys():
                    labelD_dist[i] = l[i] * labelD[i] + (1 - l[i]) * labelD2[i]
                if 'labelT2' in batch.keys():
                    noT = (labelT[i] == -1) & (labelT2[i] == -1)
                    bothT = (labelT[i] != -1) & (labelT2[i] != -1)
                    lT = (labelT[i] != -1) & (labelT2[i] == -1)
                    rT = (labelT[i] == -1) & (labelT2[i] != -1)
                    labelT_dist = noT * -1 + bothT * (l[i] * labelT[i] + (1 - l[i]) * labelT2[i]) + lT * labelT[i] + rT * labelT2[i]
                    labelT[i] = labelT_dist

            label = label_dist
            if 'labelD2' in batch.keys():
                labelD = labelD_dist

        label = label.flatten() if len(label.shape) == 2 else label.view(-1, label.shape[-1])

        with torch.amp.autocast("cuda"):
            if inference:
                with torch.no_grad():
                    predDict, y = model(x=frame, y=label, inference=inference)
                    pred = predDict['im_feat']
                    if 'displ_feat' in predDict.keys():
                        predD = predDict['displ_feat']
                    if 'team_feat' in predDict.keys():
                        predT = predDict['team_feat']

                    loss = 0.
                    lossC = 0.
                    if hasattr(model, 'module') and model.module._double_head or (not hasattr(model, 'module') and model._double_head):
                        b, t, c = pred.shape
                        if len(label.shape) == 2:
                            label = label.view(b, t, c)
                        if len(label.shape) == 1:
                            label = label.view(b, t)
                        for i in range(pred.shape[0]):
                            if batch_dataset[i] == 1:
                                if len(label.shape) == 3:
                                    aux_label = label[i][:, :args.num_classes + 1]
                                elif len(label.shape) == 2:
                                    aux_label = label[i]
                                else:
                                    raise NotImplementedError
                                lossC += nn.functional.cross_entropy(pred[i][:, :args.num_classes + 1], aux_label,
                                                                weight=ce_kwargs['weight'][:args.num_classes + 1]) / pred.shape[0]
                            elif batch_dataset[i] == 2:
                                if len(label.shape) == 3:
                                    aux_label = label[i][:, args.num_classes + 1:]
                                elif len(label.shape) == 2:
                                    aux_label = label[i] - (args.num_classes + 1)
                                else:
                                    raise NotImplementedError
                                lossC += nn.functional.cross_entropy(pred[i][:, args.num_classes + 1:], aux_label,
                                                                weight=ce_kwargs['weight'][:args.joint_train['num_classes'] + 1]) / pred.shape[0]
                    else:
                        predictions = pred.reshape(-1, num_classes)
                        lossC += nn.functional.cross_entropy(predictions, label, **ce_kwargs)
                    
                    epoch_lossC += lossC * loss_weights[0]
                    loss += lossC * loss_weights[0]
                    
                    # if 'labelD' in batch.keys():
                    #     lossD = nn.functional.mse_loss(predD, labelD, reduction='none').mean()
                    #     epoch_lossD += lossD * loss_weights[1]
                    #     loss += lossD * loss_weights[1]
                    if 'labelD' in batch.keys(): 
                        lossD = F.binary_cross_entropy_with_logits(predD, labelD)
                        lossD = (lossD).mean()
                        epoch_lossD += lossD * loss_weights[1]
                        loss += lossD * loss_weights[1]
                        
                    if 'labelT' in batch.keys():
                        if len(labelT[labelT != -1]) > 0:
                            lossT = nn.BCEWithLogitsLoss()(predT[labelT != -1], labelT[labelT != -1])
                            epoch_lossT += lossT * loss_weights[2]
                            loss += lossT * loss_weights[2]
            else:
                frame = torch.concat((frame,frame_original), dim=0)
                label_original = nn.functional.one_hot(label_original, num_classes=num_classes)
                label_original = label_original.flatten() if len(label_original.shape) == 2 else label_original.view(-1, label_original.shape[-1])
                label = torch.concat((label, label_original), dim=0)
                predDict, y = model(x=frame, y=label, inference=inference)
                pred = predDict['im_feat']
                if 'displ_feat' in predDict.keys():
                    predD = predDict['displ_feat']
                if 'team_feat' in predDict.keys():
                    predT = predDict['team_feat']

                loss = 0.
                lossC = 0.
                half_batch = frame.shape[0] // 2 
                half_len = label.shape[0] // 2 
                clip_len = label.shape[0] // 4
                
                if hasattr(model, 'module') and model.module._double_head or (not hasattr(model, 'module') and model._double_head):
                    b, t, c = pred.shape
                    if len(label.shape) == 2:
                        label = label.view(b, t, c)
                    if len(label.shape) == 1:
                        label = label.view(b, t)
                    for i in range(pred.shape[0]):
                        if batch_dataset[i] == 1:
                            if len(label.shape) == 3:
                                aux_label = label[i][:, :args.num_classes + 1]
                            elif len(label.shape) == 2:
                                aux_label = label[i]
                            else:
                                raise NotImplementedError
                            lossC += nn.functional.cross_entropy(pred[i][:, :args.num_classes + 1], aux_label,
                                                            weight=ce_kwargs['weight'][:args.num_classes + 1]) / pred.shape[0]
                        elif batch_dataset[i] == 2:
                            if len(label.shape) == 3:
                                aux_label = label[i][:, args.num_classes + 1:]
                            elif len(label.shape) == 2:
                                aux_label = label[i] - (args.num_classes + 1)
                            else:
                                raise NotImplementedError
                            lossC += nn.functional.cross_entropy(pred[i][:, args.num_classes + 1:], aux_label,
                                                            weight=ce_kwargs['weight'][:args.joint_train['num_classes'] + 1]) / pred.shape[0]
                else:
                    predictions = pred.reshape(-1, num_classes)
                    #正常部分
                    lossC += nn.functional.cross_entropy(predictions[half_batch:], label[half_batch:], **ce_kwargs)
                    #regmixup部分
                    for i in range(len(l)):
                        lossC += l[i] * nn.functional.cross_entropy(predictions[i*clip_len:(i+1)*clip_len], 
                                                             label[i*clip_len:(i+1)*clip_len], 
                                                             **ce_kwargs) 
                        
                epoch_lossC += lossC * loss_weights[0]
                loss += lossC * loss_weights[0]
                
                # if 'labelD' in batch.keys():
                #     lossD = nn.functional.mse_loss(predD, labelD, reduction='none').mean()
                #     epoch_lossD += lossD * loss_weights[1]
                #     loss += lossD * loss_weights[1]
                if 'labelD' in batch.keys(): 
                    lossD = F.binary_cross_entropy_with_logits(predD[half_batch:], labelD)
                    epoch_lossD += lossD * loss_weights[1]
                    loss += lossD * loss_weights[1]
                    
                if 'labelT' in batch.keys():
                    if len(labelT[labelT != -1]) > 0:
                        lossT = nn.BCEWithLogitsLoss()(predT[:half_batch][labelT != -1], labelT[labelT != -1])
                        epoch_lossT += lossT * loss_weights[2]
                        loss += lossT * loss_weights[2]
            
            

        if optimizer is not None:
            step(optimizer, scaler, loss / acc_grad_iter, lr_scheduler=lr_scheduler,
                 backward_only=(batch_idx + 1) % acc_grad_iter != 0)

    epoch_loss = epoch_lossC.detach().item()
    if loader.dataset._radi_displacement > 0:
        epoch_loss += epoch_lossD.detach().item()
    if loader.dataset._event_team:
        epoch_loss += epoch_lossT.detach().item()
    
    output = {'loss': epoch_loss / len(loader), 'lossC': epoch_lossC.detach().item() / len(loader)}
    if loader.dataset._radi_displacement > 0:
        output['lossD'] = epoch_lossD.detach().item() / len(loader)
    if loader.dataset._event_team:
        output['lossT'] = epoch_lossT.detach().item() / len(loader)
    
    return output

def main(args):
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '2,3,4,5'
    print('Setting seed to:', args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    config_path = args.model.split('_')[0] + '/' + args.model + '.json'
    config = load_json(os.path.join('config', config_path))
    args = update_args(args, config)

    if args.dataset in ['soccernet', 'soccernetball']:
        global LABELS_SN_PATH, LABELS_SNB_PATH
        LABELS_SN_PATH = load_text(os.path.join('data', 'soccernet', 'labels_path.txt'))[0]
        LABELS_SNB_PATH = load_text(os.path.join('data', 'soccernetball', 'labels_path.txt'))[0]

    assert args.batch_size % args.acc_grad_iter == 0
    if args.crop_dim <= 0:
        args.crop_dim = None

    wandb.login(key='7bd85ff40ccccce23a7ec58e2a434aba12764b77')
    os.makedirs(args.save_dir + '/wandb_logs', exist_ok=True)
    wandb.init(config=args, dir=args.save_dir + '/wandb_logs', project='TDEED-snbas2025', name=args.model + '-' + str(args.seed))

    # # initialize wandb
    # wandb.login()
    # if not os.path.exists(args.save_dir + '/wandb_logs'):
    #     os.makedirs(args.save_dir + '/wandb_logs', exist_ok=True)
    # wandb.init(config = args, dir = args.save_dir + '/wandb_logs', project = 'TDEED-snbas2025', name = args.model + '-' + str(args.seed))


    classes, joint_train_classes, train_data, val_data, val_data_frames = get_datasets(args)
    if args.store_mode == 'store':
        print('Datasets have been stored correctly! Stop training here and rerun.')
        sys.exit('Datasets have correctly been stored! Stop training here and rerun with load mode.')
    else:
        print('Datasets have been loaded from previous versions correctly!')

    def worker_init_fn(id):
        random.seed(id + args.seed * 100)

    loader_batch_size = args.batch_size // args.acc_grad_iter
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = DataLoader(
        train_data, shuffle=True, batch_size=loader_batch_size,
        pin_memory=True, num_workers=args.num_workers, prefetch_factor=2, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(
        val_data, shuffle=False, batch_size=1,
        pin_memory=True, num_workers=args.num_workers, prefetch_factor=2, worker_init_fn=worker_init_fn)

    model = TDEEDModel(device=device, args=args)
    num_classes = model._num_classes

    model = nn.DataParallel(model)
    model = model.to(device)

    if args.joint_train is not None:
        n_classes = [len(classes) // 2 + 1, len(joint_train_classes) // 2 + 1]
        if hasattr(model, 'module'):
            model.module.update_pred_head(n_classes)
            num_classes = np.array(n_classes).sum()
        else:
            model.update_pred_head(n_classes)
            num_classes = np.array(n_classes).sum()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scaler = torch.cuda.amp.GradScaler()

    start_epoch = 0
    if args.resume:
        start_epoch, model, optimizer = load_checkpoint(model, optimizer, scaler, device, args.resume)

    if not args.only_test:
        num_steps_per_epoch = len(train_loader) // args.acc_grad_iter
        num_epochs, lr_scheduler = get_lr_scheduler(args, optimizer, num_steps_per_epoch)

        losses = []
        best_criterion = 0 if args.criterion == 'map' else float('inf')
        print('START TRAINING EPOCHS')
        for epoch_idx in range(start_epoch, num_epochs):
            time_train0 = time.time()
            train_losses = epoch(model, train_loader, device, optimizer, scaler, lr_scheduler,
                                 acc_grad_iter=args.acc_grad_iter, num_classes=num_classes, args=args)
            train_loss = train_losses['loss']
            time_train1 = time.time()
            time_train = time_train1 - time_train0

            time_val0 = time.time()
            val_losses = epoch(model, val_loader, device, num_classes=num_classes, args=args)
            val_loss = val_losses['loss']
            time_val1 = time.time()
            time_val = time_val1 - time_val0

            better = False
            val_mAP = 0
            if args.criterion == 'loss':
                if val_loss < best_criterion:
                    best_criterion = val_loss
                    better = True
            elif args.criterion == 'map' and epoch_idx >= args.start_val_epoch:
                time_map0 = time.time()
                val_mAP = mAPevaluate(model, val_data_frames, classes, printed=True, event_team=args.event_team, metric='at1')
                time_map1 = time.time()
                time_map = time_map1 - time_map0
                if val_mAP > best_criterion:
                    best_criterion = val_mAP
                    better = True

            print('[Epoch {}] Train loss: {:0.5f} Val loss: {:0.5f}'.format(epoch_idx, train_loss, val_loss))
            txt_losses_train = 'Train losses - lossC: {:0.5f} '.format(train_losses['lossC'])
            txt_losses_val = 'Val losses - lossC: {:0.5f} '.format(val_losses['lossC'])
            if 'lossD' in train_losses:
                txt_losses_train += '- lossD: {:0.5f} '.format(train_losses['lossD'])
                txt_losses_val += '- lossD: {:0.5f} '.format(val_losses['lossD'])
            if 'lossT' in train_losses:
                txt_losses_train += '- lossT: {:0.5f} '.format(train_losses['lossT'])
                txt_losses_val += '- lossT: {:0.5f} '.format(val_losses['lossT'])
            print(txt_losses_train)
            print(txt_losses_val)
            if args.criterion == 'map' and epoch_idx >= args.start_val_epoch:
                print('Val mAP: {:0.5f}'.format(val_mAP))
                if better:
                    print('New best mAP epoch!')
            print('Time train: {}min {:.2f}sec'.format(int(time_train // 60), time_train % 60))
            print('Time val: {}min {:.2f}sec'.format(int(time_val // 60), time_val % 60))
            if args.criterion == 'map' and epoch_idx >= args.start_val_epoch:
                print('Time map: {}min {:.2f}sec'.format(int(time_map // 60), time_map % 60))

            log_dict = {'losses/train/loss': train_loss, 'losses/val/loss': val_loss, 'times/time_train': time_train, 'times/time_val': time_val}
            if args.criterion == 'map' and epoch_idx >= args.start_val_epoch:
                log_dict['losses/val/mAP'] = val_mAP
                log_dict['times/time_map'] = time_map
            if args.radi_displacement > 0 and args.event_team:
                log_dict.update({'losses/train/lossC': train_losses['lossC'], 'losses/train/lossD': train_losses['lossD'], 'losses/train/lossT': train_losses['lossT'],
                                 'losses/val/lossC': val_losses['lossC'], 'losses/val/lossD': val_losses['lossD'], 'losses/val/lossT': val_losses['lossT']})
            elif args.radi_displacement > 0:
                log_dict.update({'losses/train/lossC': train_losses['lossC'], 'losses/train/lossD': train_losses['lossD'], 'losses/val/lossC': val_losses['lossC'], 'losses/val/lossD': val_losses['lossD']})
            elif args.event_team:
                log_dict.update({'losses/train/lossC': train_losses['lossC'], 'losses/train/lossT': train_losses['lossT'], 'losses/val/lossC': val_losses['lossC'], 'losses/val/lossT': val_losses['lossT']})
            else:
                log_dict.update({'losses/train/lossC': train_losses['lossC'], 'losses/val/lossC': val_losses['lossC']})
            wandb.log(log_dict)

            losses.append({'epoch': epoch_idx, 'train': train_loss, 'val': val_loss, 'val_mAP': val_mAP})
            if args.save_dir:
                os.makedirs(args.save_dir, exist_ok=True)
                store_json(os.path.join(args.save_dir, 'loss.json'), losses, pretty=True)
                # 保存最新检查点，每次 epoch 结束时更新
                save_checkpoint(model, optimizer, scaler, epoch_idx, args, is_best=better)

    print('START INFERENCE')
    ckpt = torch.load(os.path.join(args.save_dir, 'checkpoint_best.pt'))['model_state_dict']
    # add 'module' to ckpt keys
    for key in list(ckpt.keys()):
        if not key.startswith('module.'):
            ckpt['module.' + key] = ckpt.pop(key)
    model.load_state_dict(ckpt)
    eval_splits = EVAL_SPLITS
    inv_classes = {v: k for k, v in classes.items()}

    for split in eval_splits:
        split_path = os.path.join('data', args.dataset, f'{split}.json')
        stride = STRIDE_SN if args.dataset == 'soccernet' else STRIDE_SNB if args.dataset == 'soccernetball' else STRIDE
        if not os.path.exists(split_path):
            print(f'Split {split} does not exist')
            continue

        split_data = ActionSpotVideoDataset(classes, split_path, args.frame_dir, args.modality,
                                            args.clip_len, overlap_len=args.clip_len // 4 * 3, stride=stride, dataset=args.dataset, event_team=args.event_team)
        pred_file = os.path.join(args.save_dir, f'pred-{split}') if args.save_dir else None

        results = mAPevaluateTest(model, split, split_data, classes, printed=True, event_team=args.event_team, metric='at1', pred_file=pred_file, postprocessing='SNMS')
        if results is None:
            print(f'No results for split {split}')
            print(f'Predictions have been stored in {pred_file}')
            continue

        wandb.log({'test/mAP@1': results['mAP'] * 100})
        wandb.summary['test/mAP@1'] = results['mAP'] * 100
        for j in range(len(classes) // 2):
            wandb.log({'test/classes/mAP@' + inv_classes[j * 2 + 1].split('-')[0]: results['mAP_per_class'][j] * 100})
        if args.event_team:
            wandb.log({'test/mAP@1NoTeam': results['mAP_no_team'] * 100})
            wandb.summary['test/mAP@1NoTeam'] = results['mAP_no_team'] * 100
            for j in range(len(classes) // 2):
                wandb.log({'test/classes/mAP@' + inv_classes[j * 2 + 1].split('-')[0] + 'NoTeam': results['mAP_per_class_no_team'][j] * 100})

    print('CORRECTLY FINISHED TRAINING AND INFERENCE')

if __name__ == '__main__':
    main(get_args())