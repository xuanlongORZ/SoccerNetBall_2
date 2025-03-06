"""
File containing the main model.
"""

#Standard imports
import torch
from torch import nn
import timm
import torchvision.transforms as T
from contextlib import nullcontext
from tqdm import tqdm
import random
import torch.nn.functional as F
import math


#Local imports
from model.modules import BaseRGBModel, EDSGPMIXERLayers, FCLayers, FC2Layers, step, process_prediction, process_predictionTeam, process_double_head
from model.shift import make_temporal_shift

class TDEEDModel(BaseRGBModel):

    class Impl(nn.Module):

        def __init__(self, args = None):
            super().__init__()
            self._modality = args.modality
            assert self._modality == 'rgb', 'Only RGB supported for now'
            in_channels = {'rgb': 3}[self._modality]
            self._temp_arch = args.temporal_arch
            assert self._temp_arch in ['ed_sgp_mixer'], 'Only ed_sgp_mixer supported for now'
            self._radi_displacement = args.radi_displacement
            self._feature_arch = args.feature_arch
            assert 'rny' in self._feature_arch, 'Only rny supported for now'
            self._double_head = False
            self._event_team = args.event_team

            ##self._feature_arch = 'rny002_gsf'
            if self._feature_arch.startswith(('rny002', 'rny008')):

                features = timm.create_model({
                    'rny002': 'regnety_002',
                    'rny008': 'regnety_008',
                }[self._feature_arch.rsplit('_', 1)[0]], pretrained=True)
                
                feat_dim = features.head.fc.in_features

                # Remove final classification layer
                features.head.fc = nn.Identity()
                self._d = feat_dim

            else:
                raise NotImplementedError(args._feature_arch)

            # Add Temporal Shift Modules
            self._require_clip_len = -1
            if self._feature_arch.endswith('_gsm'):
                make_temporal_shift(features, args.clip_len, mode='gsm')
                self._require_clip_len = args.clip_len
            elif self._feature_arch.endswith('_gsf'):
                make_temporal_shift(features, args.clip_len, mode='gsf')
                self._require_clip_len = args.clip_len

            
            ##feature 已经添加了GSM模组，此时为整体头部
            self._features = features
            self._feat_dim = self._d
            feat_dim = self._d

            #Positional encoding learnable
            self.temp_enc = nn.Parameter(torch.normal(mean = 0, std = 1 / args.clip_len, size = (args.clip_len, self._d)))
            
            if self._temp_arch == 'ed_sgp_mixer':
                self._temp_fine = EDSGPMIXERLayers(feat_dim, args.clip_len, num_layers=args.n_layers, ks = args.sgp_ks, k = args.sgp_r, concat = True)
                self._pred_fine = FCLayers(self._feat_dim, args.num_classes+1)
            else:
                raise NotImplementedError(self._temp_arch)
            
            if self._radi_displacement > 0:
                self._pred_displ = FCLayers(self._feat_dim, 1)

            if self._event_team:
                self._pred_team = FCLayers(self._feat_dim, 1)
            
            #Augmentations and crop
            self.augmentation = T.Compose([
                T.RandomApply([T.ColorJitter(hue = 0.2)], p = 0.25),
                T.RandomApply([T.ColorJitter(saturation = (0.7, 1.2))], p = 0.25),
                T.RandomApply([T.ColorJitter(brightness = (0.7, 1.2))], p = 0.25),
                T.RandomApply([T.ColorJitter(contrast = (0.7, 1.2))], p = 0.25),
                T.RandomApply([T.GaussianBlur(5)], p = 0.25),
                # T.RandomHorizontalFlip(),
            ])

            #Standarization
            self.standarization = T.Compose([
                T.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)) #Imagenet mean and std
            ])

            #Croping in case of using it
            self.croping = args.crop_dim
            if self.croping != None:
                self.cropT = T.RandomCrop((self.croping, self.croping))
                self.cropI = T.CenterCrop((self.croping, self.croping))
            else:
                self.cropT = torch.nn.Identity()
                self.cropI = torch.nn.Identity()

        def forward(self, x, y = None, inference=False):
            
            x = self.normalize(x) #Normalize to 0-1
            batch_size, clip_len, channels, height, width = x.shape

            if not inference:
                x.view(-1, channels, height, width)
                if self.croping != None:
                    height = self.croping
                    width = self.croping
                x = self.cropT(x) #same crop for all frames
                x = x.view(batch_size, clip_len, channels, height, width)
                x = self.augment(x) #augmentation per-batch
                x = self.standarize(x) #standarization imagenet stats

            else:
                x = x.view(-1, channels, height, width)
                if self.croping != None:
                    height = self.croping
                    width = self.croping
                x = self.cropI(x) #same center crop for all frames
                x = x.view(batch_size, clip_len, channels, height, width)
                x = self.standarize(x)
            
            #Extract features
            im_feat = self._features(
                x.view(-1, channels, height, width)
            ).reshape(batch_size, clip_len, self._d)

            #Temporal encoding   POSITIONAL ENCODING
            im_feat = im_feat + self.temp_enc.expand(batch_size, -1, -1)

            #Temporal module (SGP-Mixer)
            if self._temp_arch == 'ed_sgp_mixer':
                output_data = {}
                im_feat = self._temp_fine(im_feat)
                if self._radi_displacement > 0:
                    displ_feat = self._pred_displ(im_feat).squeeze(-1)
                    output_data['displ_feat'] = displ_feat
                if self._event_team:
                    team_feat = self._pred_team(im_feat).squeeze(-1)
                    output_data['team_feat'] = team_feat
                im_feat = self._pred_fine(im_feat)

                output_data['im_feat'] = im_feat

                return output_data, y
            
            else:
                raise NotImplementedError(self._temp_arch)
        
        def normalize(self, x):
            return x / 255.
        
        def augment(self, x):
            for i in range(x.shape[0]):
                x[i] = self.augmentation(x[i])
            return x
        

        def standarize(self, x):
            for i in range(x.shape[0]):
                x[i] = self.standarization(x[i])
            return x
        
        def update_pred_head(self, num_classes = [1, 1]):
            self._pred_fine = FC2Layers(self._feat_dim, num_classes)
            self._pred_fine = self._pred_fine.cuda()
            self._double_head = True

        def print_stats(self):
            print('Model params:',
                sum(p.numel() for p in self.parameters()))
            print('  CNN features:',
                sum(p.numel() for p in self._features.parameters()))
            print('  Temporal:',
                sum(p.numel() for p in self._temp_fine.parameters()))
            print('  Head:',
                sum(p.numel() for p in self._pred_fine.parameters()))

    def __init__(self, device='cuda', args=None):
        self.device = device
        self._model = TDEEDModel.Impl(args=args)
        self._model.print_stats()
        self._args = args

        
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            self._model = nn.DataParallel(self._model)
            
        self._model.to(device)
        
        
        self._num_classes = args.num_classes + 1

    def epoch(self, loader, optimizer=None, scaler=None, lr_scheduler=None,
            acc_grad_iter=1, fg_weight=5, loss_weights = [1, 1, 2]):

        if optimizer is None:
            inference = True
            self._model.eval()
        else:
            inference = False
            optimizer.zero_grad()
            self._model.train()

        ce_kwargs = {}
        if fg_weight != 1:
            ce_kwargs['weight'] = torch.FloatTensor(
                [1] + [fg_weight] * (self._num_classes - 1)).to(self.device)

        epoch_lossC = 0.
        if loader.dataset._radi_displacement > 0:
            epoch_lossD = 0.
        if loader.dataset._event_team:
            epoch_lossT = 0.
        with torch.no_grad() if optimizer is None else nullcontext():
            for batch_idx, batch in enumerate(tqdm(loader)):
                frame = batch['frame'].to(self.device).float()
                label = batch['label']
                label = label.to(self.device)

                #update labels for double head
                if self._model.module._double_head:
                    batch_dataset = batch['dataset']
                    label = update_labels_2heads(label, batch_dataset, self._args.num_classes)

                if 'labelD' in batch.keys():
                    labelD = batch['labelD'].to(self.device).float()

                if 'labelT' in batch.keys():
                    labelT = batch['labelT'].to(self.device).float()
                
                if 'frame2' in batch.keys():
                    frame2 = batch['frame2'].to(self.device).float()
                    label2 = batch['label2']
                    label2 = label2.to(self.device)

                    if 'labelD2' in batch.keys():
                        labelD2 = batch['labelD2'].to(self.device).float()
                        labelD_dist = torch.zeros((labelD.shape[0], label.shape[1])).to(self.device)

                    if 'labelT2' in batch.keys():
                        labelT2 = batch['labelT2'].to(self.device).float()

                    l = [random.betavariate(0.2, 0.2) for _ in range(frame2.shape[0])]

                    label_dist = torch.zeros((label.shape[0], label.shape[1], self._num_classes)).to(self.device)

                    for i in range(frame2.shape[0]):
                        frame[i] = l[i] * frame[i] + (1 - l[i]) * frame2[i]
                        lbl1 = label[i]
                        lbl2 = label2[i]

                        label_dist[i, range(label.shape[1]), lbl1] += l[i]
                        label_dist[i, range(label2.shape[1]), lbl2] += 1 - l[i]

                        if 'labelD2' in batch.keys():
                            labelD_dist[i] = l[i] * labelD[i] + (1 - l[i]) * labelD2[i]

                        if 'labelT2' in batch.keys():
                            noT = (labelT[i] == -1) & (labelT2[i] == -1) #no label
                            bothT = (labelT[i] != -1) & (labelT2[i] != -1) #both labels
                            lT = (labelT[i] != -1) & (labelT2[i] == -1) #left label 
                            rT = (labelT[i] == -1) & (labelT2[i] != -1) #right label
                            labelT_dist = noT * -1 + bothT * (l[i] * labelT[i] + (1 - l[i]) * labelT2[i]) + lT * labelT[i] + rT * labelT2[i]
                            labelT[i] = labelT_dist

                    label = label_dist
                    if 'labelD2' in batch.keys():
                        labelD = labelD_dist

                # Depends on whether mixup is used
                label = label.flatten() if len(label.shape) == 2 \
                    else label.view(-1, label.shape[-1])

                with torch.amp.autocast("cuda"):
                    predDict, y = self._model(frame, y = label, inference=inference)#   frame is a list, y is the lable list 

                    pred = predDict['im_feat']
                    if 'displ_feat' in predDict.keys():
                        predD = predDict['displ_feat']
                    if 'team_feat' in predDict.keys():
                        predT = predDict['team_feat']

                    loss = 0.
                    lossC = 0.
                    if self._model.module._double_head:
                        b, t, c = pred.shape

                        if len(label.shape) == 2:
                            label = label.view(b, t, c)
                        if len(label.shape) == 1:
                            label = label.view(b, t)

                        for i in range(pred.shape[0]):
                            if batch_dataset[i] == 1:
                                if len(label.shape) == 3:
                                    aux_label = label[i][:, :self._args.num_classes+1]
                                elif len(label.shape) == 2:
                                    aux_label = label[i]
                                else:
                                    raise NotImplementedError
                                    
                                lossC += F.cross_entropy(pred[i][:, :self._args.num_classes+1], aux_label,
                                                        weight = ce_kwargs['weight'][:self._args.num_classes+1]) / (pred.shape[0])
                                    
                            elif batch_dataset[i] == 2:
                                if len(label.shape) == 3:
                                    aux_label = label[i][:, self._args.num_classes+1:]
                                elif len(label.shape) == 2:
                                    aux_label = label[i] - (self._args.num_classes + 1)
                                else:
                                    raise NotImplementedError
                                    
                                lossC += F.cross_entropy(pred[i][:, self._args.num_classes+1:], aux_label,
                                                        weight = ce_kwargs['weight'][:self._args.joint_train['num_classes']+1]) / (pred.shape[0])

                    else:
                        predictions = pred.reshape(-1, self._num_classes)

                        lossC += F.cross_entropy(
                            predictions, label,
                            **ce_kwargs)    
                    
                    epoch_lossC += lossC * loss_weights[0]
                    loss += lossC * loss_weights[0]
                            
                    if 'labelD' in batch.keys():
                        lossD = F.mse_loss(predD, labelD, reduction = 'none')
                        lossD = (lossD).mean()
                        epoch_lossD += lossD * loss_weights[1]
                        loss += lossD * loss_weights[1]

                    if 'labelT' in batch.keys():
                        if len(labelT[labelT != -1]) > 0:
                            lossT = torch.nn.BCEWithLogitsLoss()(predT[labelT != -1], labelT[labelT != -1])
                            epoch_lossT += lossT * loss_weights[2]
                            loss += lossT * loss_weights[2]


                if optimizer is not None:
                    step(optimizer, scaler, loss / acc_grad_iter,
                        lr_scheduler=lr_scheduler,
                        backward_only=(batch_idx + 1) % acc_grad_iter != 0)

        epoch_loss = epoch_lossC.detach().item() #+ epoch_lossD.detach().item() + epoch_lossT.detach().item()
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
    
    def predict(self, seq, use_amp=True):
        
        if not isinstance(seq, torch.Tensor):
            seq = torch.FloatTensor(seq)
        if len(seq.shape) == 4: # (L, C, H, W)
            seq = seq.unsqueeze(0)
        if seq.device != self.device:
            seq = seq.to(self.device)
        seq = seq.float()

        self._model.eval()
        with torch.no_grad():
            with torch.amp.autocast("cuda") if use_amp else nullcontext():
                predDict, _ = self._model(seq, inference=True)

            pred = predDict['im_feat']
            if 'displ_feat' in predDict.keys():
                predD = predDict['displ_feat']
                if self._model.module._double_head:
                    pred = process_double_head(pred, predD, num_classes = self._args.num_classes+1)
                else:
                    pred = process_prediction(pred, predD)
            if 'team_feat' in predDict.keys():
                predT = predDict['team_feat']
                pred = process_predictionTeam(pred, predT)
            
            pred_cls = torch.argmax(pred, axis=2)

            return pred_cls.cpu().numpy(), pred.cpu().numpy()
        
def update_labels_2heads(labels, datasets, num_classes1 = 1):
    for i in range(len(datasets)):
        if datasets[i] == 2:
            labels[i] = labels[i] + num_classes1 + 1

    return labels
