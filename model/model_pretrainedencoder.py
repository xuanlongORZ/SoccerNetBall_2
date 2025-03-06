"""
File containing the main model.
"""

# Standard imports
import torch
from torch import nn
import timm
import torchvision.transforms as T
from contextlib import nullcontext

# Local imports
from model.modules import BaseRGBModel, EDSGPMIXERLayers, FCLayers, FC2Layers, process_prediction, process_predictionTeam, process_double_head
from model.shift import make_temporal_shift

class TDEEDModel(BaseRGBModel):
    def __init__(self, device='cuda', args=None):
        super().__init__()
        self.device = device
        self._args = args
        self._num_classes = args.num_classes + 1

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

        # Feature extraction
        if self._feature_arch.startswith(('rny002', 'rny008', 'rny160', 'rny120', 'rny064', 'rny040', 'rny016')):
            features = timm.create_model(
                model_name={
                    'rny002': 'regnety_002.pycls_in1k',
                    'rny008': 'regnety_008_tv.tv2_in1k',
                    'rny160': 'regnety_160.sw_in12k_ft_in1k',
                    'rny120': 'regnety_120.sw_in12k_ft_in1k',
                    'rny064': 'regnety_064.ra3_in1k',
                    'rny040': 'regnety_040.ra3_in1k',
                    'rny016': 'regnety_016.tv2_in1k',
                }[self._feature_arch.rsplit('_', 1)[0]], pretrained=False)
            
            # features.load_state_dict(dict_checkpoint, strict=True)
            feat_dim = features.head.fc.in_features
            features.head.fc = nn.Identity()
            self._d = feat_dim
        else:
            raise NotImplementedError(self._feature_arch)

        # Add Temporal Shift Modules
        self._require_clip_len = -1
        if self._feature_arch.endswith('_gsm'):
            make_temporal_shift(features, args.clip_len, mode='gsm')
            self._require_clip_len = args.clip_len
        elif self._feature_arch.endswith('_gsf'):
            make_temporal_shift(features, args.clip_len, mode='gsf')
            self._require_clip_len = args.clip_len

        dict_checkpoint = torch.load({
            'rny002': 'pretrained_ckpts/baseline_ckpt.pt',
        }[self._feature_arch.rsplit('_', 1)[0]])

        for key in list(dict_checkpoint.keys()):
            if key.startswith('_features.'):
                dict_checkpoint[key[10:]] = dict_checkpoint.pop(key)

        self._features = features

        self._features.load_state_dict(dict_checkpoint, strict=False)
        
        # requires_grad = False for all self._features params
        for param in self._features.parameters():
            param.requires_grad = False
            
        self._features.eval()

        self._feat_dim = self._d
        feat_dim = self._d

        # Positional encoding
        self.temp_enc = nn.Parameter(torch.normal(mean=0, std=1 / args.clip_len, size=(args.clip_len, self._d)))
        
        if self._temp_arch == 'ed_sgp_mixer':
            self._temp_fine = EDSGPMIXERLayers(feat_dim, args.clip_len, num_layers=args.n_layers, ks=args.sgp_ks, k=args.sgp_r, concat=True)
            self._pred_fine = FCLayers(self._feat_dim, args.num_classes + 1)
        else:
            raise NotImplementedError(self._temp_arch)
        
        if self._radi_displacement > 0:
            self._pred_displ = FCLayers(self._feat_dim, 1)
        if self._event_team:
            self._pred_team = FCLayers(self._feat_dim, 1)
        
        # Augmentations and crop
        self.augmentation = T.Compose([
            T.RandomApply([T.ColorJitter(hue=0.2)], p=0.25),
            T.RandomApply([T.ColorJitter(saturation=(0.7, 1.2))], p=0.25),
            T.RandomApply([T.ColorJitter(brightness=(0.7, 1.2))], p=0.25),
            T.RandomApply([T.ColorJitter(contrast=(0.7, 1.2))], p=0.25),
            T.RandomApply([T.GaussianBlur(5)], p=0.25),
        ])
        self.standarization = T.Compose([
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        self.croping = args.crop_dim
        if self.croping is not None:
            self.cropT = T.Resize((self.croping, self.croping))
            self.cropI = T.Resize((self.croping, self.croping))
        else:
            self.cropT = torch.nn.Identity()
            self.cropI = torch.nn.Identity()

        # 将模型移动到设备
        self.to(self.device)
        self.print_stats()

    def forward(self, x, y=None, inference=False):
        with torch.no_grad():
            x = self.normalize(x)
            batch_size, clip_len, channels, height, width = x.shape

            if not inference:
                x = x.view(-1, channels, height, width)
                if self.croping is not None:
                    height = self.croping
                    width = self.croping
                x = self.cropT(x)
                x = x.view(batch_size, clip_len, channels, height, width)
                x = self.augment(x)
                x = self.standarize(x)
            else:
                x = x.view(-1, channels, height, width)
                if self.croping is not None:
                    height = self.croping
                    width = self.croping
                x = self.cropI(x)
                x = x.view(batch_size, clip_len, channels, height, width)
                x = self.standarize(x)
            
            im_feat = self._features(x.view(-1, channels, height, width)).reshape(batch_size, clip_len, self._d)
        
        im_feat = im_feat + self.temp_enc.expand(batch_size, -1, -1)

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
    
    def update_pred_head(self, num_classes=[1, 1]):
        self._pred_fine = FC2Layers(self._feat_dim, num_classes)
        self._pred_fine = self._pred_fine.to(self.device)
        self._double_head = True

    def print_stats(self):
        print('Model params:', sum(p.numel() for p in self.parameters()))
        print('  CNN features:', sum(p.numel() for p in self._features.parameters()))
        print('  Temporal:', sum(p.numel() for p in self._temp_fine.parameters()))
        print('  Head:', sum(p.numel() for p in self._pred_fine.parameters()))

    def predict(self, seq, use_amp=True):
        if not isinstance(seq, torch.Tensor):
            seq = torch.FloatTensor(seq)
        if len(seq.shape) == 4:  # (L, C, H, W)
            seq = seq.unsqueeze(0)
        if seq.device != self.device:
            seq = seq.to(self.device)
        seq = seq.float()

        self.eval()
        with torch.no_grad():
            with torch.amp.autocast("cuda") if use_amp else nullcontext():
                predDict, _ = self(seq, inference=True)

            pred = predDict['im_feat']
            if 'displ_feat' in predDict.keys():
                predD = predDict['displ_feat']
                if self._double_head:
                    pred = process_double_head(pred, predD, num_classes=self._args.num_classes + 1)
                else:
                    pred = process_prediction(pred, predD)
            if 'team_feat' in predDict.keys():
                predT = predDict['team_feat']
                pred = process_predictionTeam(pred, predT)
            
            pred_cls = torch.argmax(pred, axis=2)
            return pred_cls.cpu().numpy(), pred.cpu().numpy()

def update_labels_2heads(labels, datasets, num_classes1=1):
    for i in range(len(datasets)):
        if datasets[i] == 2:
            labels[i] = labels[i] + num_classes1 + 1
    return labels