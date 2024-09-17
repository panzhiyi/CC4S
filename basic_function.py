import os
import sys
import time
import shutil
import random
import argparse
import numpy as np
import torchnet as tnt
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.models as models
import torch.backends.cudnn as cudnn
from torch.autograd import Variable, Function
from torch.utils import data
from tensorboardX import SummaryWriter
from PIL import Image

#from diffMap_deeplab.diffMap_layers import *

from IPython.core import debugger
debug = debugger.Pdb().set_trace

class MyArgumentParser():
    def __init__(self, inference=False):
        self.parser = argparse.ArgumentParser(description='PyTorch Hierachy_dif Training')
        self.parser.add_argument('shfilename', metavar='SHFILENAME', help='.sh file name')
        self.parser.add_argument('gpus', metavar='GPUS', help='GPU ID')
        self.parser.add_argument('layers', type=int, metavar='LAYERS', help='the layer number of resnet: 18, 34, 50, 101, 152')
        self.parser.add_argument('dataset_path', metavar='DATASET_PATH', help='path to the dataset(multiple paths are concated by "+")')
        self.parser.add_argument('dataset', metavar='DATASET', help='dataset: VOC2012|PascalContext')
        self.parser.add_argument('numclasses', type=int, metavar='NUMCLASSES', help='number of classes')
        self.parser.add_argument('workers', default=2, type=int, metavar='WORKERS', help='number of dataload worker')
        self.parser.add_argument ( 'model_type', default='RW', metavar='MODEL_TYPE', help='type of model' )
        self.parser.add_argument ( 'shrink_factor', default=1, type=int, metavar='SHRINK',
                                   help='shrink factor of attention map' )
        if inference:
            self.parser.add_argument('checkpoint_path', metavar='CHECKPOINT_PATH', help='path to the checkpoint file')
            self.parser.add_argument('save_path', default='val', metavar='SAVE_PATH', help='path to the inference results')
            self.parser.add_argument('if_mask', default = 'True', metavar='OUT_PUT_TYPE', help='if output the unmask data for next iter')
            self.parser.add_argument('train_path', default='None' ,metavar='TRAIN_PATH', help='path to the dataset training file')
            self.parser.add_argument('confidence', default=1, type=float,metavar='LABLE_CONFIDENCE_LIMIT/CRF_SWITCH', help='Confidence for output label, 1 means 100%; 0.5 means 50%,2 means using crf')
            self.parser.add_argument('data_list',nargs='?',default='train.txt',metavar='DATA_LIST', help='choose trainingg data list (xxx.txt)')
        else:
            self.parser.add_argument('train_path', metavar='TRAIN_PATH', help='path to the dataset training file')
            self.parser.add_argument('batchsize', default=16, type=int, metavar='BATCH_SIZE', help='batchsize')
            self.parser.add_argument('lr', default=2.5e-4, type=float, metavar='LEARNING_RATE', help='learning rate')
            self.parser.add_argument('wdecay', default=0.05, type=float, metavar='WEIGHT_DECAY', help='weight decay')
            self.parser.add_argument('momentum', default=0.9, type=float, metavar='MOMENTUM', help='the momentum of SGD learning algorithm')
            self.parser.add_argument('epochs', default=50, type=int, metavar='EPOCH',help='number of total epochs to run')
            self.parser.add_argument('edgeloss_weight', default=0, type=float, metavar='EDGEYLOSS_WEIGHT',help='edge loss weight')
            self.parser.add_argument ( 'use_boundary', default=0,type = int, metavar='SP_BOUNDARY', help='use sp boundary information or not' )
            self.parser.add_argument('model_path', default= 'None' ,help='pretrain model path')
            self.parser.add_argument('selfloss_type', default='flip', metavar='SELFLOSS_TYPE', help='the type of self supervised operation')
            self.parser.add_argument('selfloss_feature', default='P', metavar='SELFLOSS_FEATURE', help='the input feature of self supervised loss')
            self.parser.add_argument('if_laplaceloss', default='False', metavar='IF_LAPLACELOSS', help='use Laplace_Loss or not,use \'Contrast\' to switch contrstive loss mode')
            self.parser.add_argument('laplace_path', metavar='laplace_PATH', help='path to the Laplace matrix file(.npy file)')
            self.parser.add_argument('spiex_path', metavar='spiex_PATH', help='path to the spiex file(.npy file)')
            self.parser.add_argument('data_list', metavar='DATA_LIST', help='choose trainingg data list (xxx.txt)')
        
        self.parser.add_argument('sleep_time',nargs='?',default=0,type=int,metavar='DELAY_TIMES(s)', help='sleeptime')
        self.parser.add_argument('special_set',nargs='?',default=0,type=int,metavar='SPECIAL', help='special')
        ''' 0b0000001 : reserve 
        0b00000010 : use medial channal of deeplbv3+ as U-Net-like connnect 
        0b00000100 : output_stride = 8 or 16 ; False means 16 True means 8
        0b10000001 : train on context dataset
        '''

    def get_parser(self):
        return self.parser

class EigLoss ( nn.Module ) :
    def __init__(self, eig=True) :
        super ( EigLoss, self ).__init__ ()
        self.eig = eig
        if self.eig == True :
            self.L1Loss = nn.L1Loss ( reduction='mean' )
        self.kld = nn.KLDivLoss ( reduction='mean' )

    def forward(self, f1, f2) :
        f1_softmax = F.softmax ( f1, dim=1 )
        f2_softmax = F.softmax ( f2, dim=1 )
        f1_logsoftmax = F.log_softmax ( f1, dim=1 )
        loss2 = self.kld ( f1_logsoftmax, f2_softmax )

        if self.eig == True :
            loss1 = self.L1Loss ( torch.diagonal ( f1_softmax, dim1=-2, dim2=-1 ).sum ( -1 ),
                                  torch.diagonal ( f2_softmax, dim1=-2, dim2=-1 ).sum ( -1 ) )
            loss = 1e-2 * loss1 + loss2
        else :
            loss = loss2
        return loss


# label = 255 is ambiguious label, and only some gts have this label.
class SegLoss ( nn.Module ) :
    def __init__(self, ignore_label=255, mode=1) :
        super ( SegLoss, self ).__init__ ()
        if mode == 1 :
            self.obj = torch.nn.CrossEntropyLoss ( ignore_index=ignore_label )
        else :
            self.obj = torch.nn.NLLLoss2d ( ignore_index=ignore_label )

    def __call__(self, pred, label) :
        loss = self.obj ( pred, label )
        return loss


class EntropyLoss ( nn.Module ) :
    def __init__(self) :
        super ( EntropyLoss, self ).__init__ ()

    def forward(self, x, mask, mode=1) :
        # mask_size = mask.size()[1:3]
        # x_softmax = F.softmax(x, dim = 1)
        # x_logsoftmax = F.log_softmax(x, dim = 1)
        # x_softmax_up = F.interpolate(x_softmax, size=mask_size, mode='bilinear', align_corners=True)
        # x_logsoftmax_up = F.interpolate(x_logsoftmax, size=mask_size, mode='bilinear', align_corners=True)
        # b = x_softmax_up * x_logsoftmax_up

        if mode == 1 :
            mask = 1.0 - mask / 255
            b = F.softmax ( x, dim=1 ) * F.log_softmax ( x, dim=1 )
            b = torch.sum ( b, dim=1 )
            entropy = b.mul ( mask )
            loss = -1.0 * torch.sum ( entropy ) / torch.sum ( mask )
        else :
            b = F.softmax ( x, dim=1 ) * F.log_softmax ( x, dim=1 )
            b = torch.sum ( b, dim=1 )
            loss = -1.0 * torch.mean ( b )
        return loss


class MSELoss_mask ( nn.Module ) :
    def __init__(self) :
        super ( MSELoss_mask, self ).__init__ ()
        self.criterion_mse = nn.MSELoss ( reduction='none' )
        self.criterion_mse_mean = nn.MSELoss ( reduction='mean' )

    def forward(self, x1, x2, mask=None, mask_type=0) :
        if mask_type == 0 :
            loss = self.criterion_mse_mean ( x1, x2 )
        elif mask_type == 1 :
            mse_loss = self.criterion_mse ( x1, x2 )
            input_size = x1.size ()[2 :4]
            batch_size = x1.size ()[1]
            mask = F.interpolate ( torch.unsqueeze ( mask, 1 ).float (), size=input_size, mode='nearest' )
            mask_ignore = (mask != 255) & (mask != 0)
            mse_mask_loss = mse_loss.mul ( mask_ignore )
            loss = torch.sum ( mse_mask_loss ) / (torch.sum ( mask_ignore ) * batch_size)
        else :
            mse_loss = self.criterion_mse ( x1, x2 )
            input_size = x1.size ()[2 :4]
            batch_size = x1.size ()[1]
            mask = F.interpolate ( torch.unsqueeze ( mask, 1 ), size=input_size, mode='bilinear' )
            mse_mask_loss = mse_loss.mul ( mask )
            loss = torch.sum ( mse_mask_loss ) / (torch.sum ( mask ) * batch_size)
        return loss


class EdgeLoss_entropy ( nn.Module ) :
    def __init__(self, class_num) :
        super ( EdgeLoss_entropy, self ).__init__ ()
        sobel_kernel = np.array ( [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32' )
        sobel_kernel = sobel_kernel.reshape ( (1, 1, 3, 3) )
        sobel_kernel = sobel_kernel.repeat ( class_num, 0 )
        self.weight = torch.from_numpy ( sobel_kernel )

    def forward(self, pred_sg_up, edge_v) :
        pred_sg_softmax = F.softmax ( pred_sg_up )
        edge_pred = F.conv2d ( pred_sg_softmax, self.weight.cuda (), padding=1, groups=21 )
        edge_pred = torch.tanh ( torch.sum ( torch.abs ( edge_pred ), dim=1, keepdim=True ) )
        loss_edge = torch.mean (
            torch.sum ( torch.mul ( edge_pred, torch.abs ( edge_pred - edge_v / 255 ) ), dim=(1, 2) ) / torch.sum (
                edge_pred, dim=(1, 2) ) )
        return loss_edge


class EdgeLoss ( nn.Module ) :
    def __init__(self, delta=0.1, edge_balance=False) :
        super ( EdgeLoss, self ).__init__ ()
        self.edge_balance = edge_balance
        self.delta = delta

    def forward(self, pred_sg_up, edge_v) :
        edge = torch.flatten ( edge_v, start_dim=1 )
        pred_seg_softmax = torch.softmax ( pred_sg_up, 1 )
        pred_seg = torch.flatten ( pred_seg_softmax, start_dim=2 )
        batch_size = pred_seg.size ()[0]
        channel = pred_seg.size ()[1]
        var_term = 0.0
        for i in range ( batch_size ) :
            unique_labels, unique_id, counts = torch.unique ( edge[i], return_inverse=True, return_counts=True )
            num_instances = unique_labels.size ()[0]
            unique_id_repeat = unique_id.unsqueeze ( 0 ).repeat ( channel, 1 )
            segmented_sum = torch.zeros ( channel, num_instances ).cuda ().scatter_add ( dim=1, index=unique_id_repeat,
                                                                                         src=pred_seg[i] )
            mu = torch.div ( segmented_sum, counts )
            mu_expand = torch.gather ( mu, 1, unique_id_repeat )
            tmp_distance = pred_seg[i] - mu_expand
            distance = torch.sum ( torch.abs ( tmp_distance ), dim=0 )
            distance = torch.clamp ( distance - self.delta, min=0.0 )
            if self.edge_balance == False :
                mask = (edge[i] != 0) & (edge[i] != 255)
                l_var = torch.sum ( distance * mask ) / (torch.sum ( mask ) + 1e-5)
            else :
                l_var = torch.zeros ( num_instances ).cuda ().scatter_add ( dim=0, index=unique_id, src=distance )
                l_var = torch.div ( l_var, counts )
                mask = (unique_labels != 0) & (unique_labels != 255)
                l_var = torch.sum ( l_var * mask ) / (torch.sum ( mask ) + 1e-5)
            var_term = var_term + l_var
        loss_edge = var_term / batch_size
        return loss_edge


def ColorMapping(seg):
    colormap = torch.Tensor([[0,0,0], [128,0,0], [0,128,0], [128,128,0], [0,0,128], [128,0,128],
                              [0,128,128], [128,128,128], [64,0,0], [192,0,0], [64,128,0], [192,128,0],
                              [64,0,128], [192,0,128], [64,128,128], [192,128,128], [0, 64,0], [128, 64, 0],
                              [0,192,0], [128,192,0], [0,64,128]])/255.0#total 21 labels
    seg = torch.matmul(seg.transpose(1,2).transpose(2,3), colormap).transpose(3,2).transpose(2,1)
    return seg

def get_flip_transfer(h, w):
    transfer = torch.zeros(h*w, h*w)
    diag_matrix=torch.flip(torch.eye(w),[0])
    for i in range(h):
        transfer[i*w:(i+1)*w,i*w:(i+1)*w]=diag_matrix
    return transfer

def param_restore(model, param_dict):
    new_params = model.state_dict().copy()
    for i in param_dict:
        i_parts = i.split('.')
        #print(i)
        if not i_parts[0]=='fc':
            new_params[i] = param_dict[i]
    model.load_state_dict(new_params)
    return model
    
def param_restore_all(model, param_dict):
    new_params = model.state_dict().copy()
    for i in param_dict:
        i_parts = i.split('.')
        i_parts.pop(0)
        new_params['.'.join(i_parts)] = param_dict[i]
    model.load_state_dict(new_params)
    return model    

def BGR2RGB(img):
    out = torch.zeros(img.size())
    out[:,0,:,:] = img[:,2,:,:]
    out[:,1,:,:] = img[:,1,:,:]
    out[:,2,:,:] = img[:,0,:,:]
    return out


def map_decode(seg, num_label=21):
    seg[seg==255]=0           #######################
    labels = range(num_label)
    batch_size = seg.size(0)
    out = torch.zeros(seg.size()).repeat(1,num_label,1,1)
    for i, label in enumerate(labels):
        #out_slice = out[:,i:i+1,:,:]1
        out[:,i:i+1,:,:] = seg==label
    return out


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def BatchInverse(tensor):
    batch_size = tensor.size()[0]
    tensor_inverse = []
    for i in range(batch_size):
        tensor_inverse += [torch.inverse(tensor[i]).unsqueeze(0)]
    return torch.cat(tensor_inverse, 0)


def save_checkpoint(state, date, is_best, shfilename, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "./runs/{}/".format(date)
    if not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.exists(directory + shfilename):
        shutil.copyfile(shfilename, directory + shfilename)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, directory + 'model_best.pth.tar')
        


def register_checks(model):
    def check_grad(module, grad_input, grad_output):
        # print(module) you can add this to see that the hook is called
        for gi in grad_input:
            if gi is not None:
                if torch.any(torch.isnan(gi)):
                    print('NaN gradient in ' + type(module).__name__)
                    debug()
                if torch.any(torch.isinf(gi)):
                    print('Inf gradient in ' + type(module).__name__)
                    debug()
    model.apply(lambda module: module.register_backward_hook(check_grad))


def get_mask_pallete(npimg, dataset='pascal_voc'):
    """Get image color pallete for visualizing masks"""
    # recovery boundary
    if dataset == 'pascal_voc':
        npimg[npimg==21] = 255
        npimg[npimg==-1] = 255
        colorpallete = vocpallete
    elif dataset == 'pascal_voc_context':
        npimg[npimg==-1] = 255
        colorpallete = vocpallete
    elif dataset == 'ade20k':
        colorpallete = adepallete
    elif dataset == 'cityscapes':
        colorpallete = citypallete  
    elif dataset == 'no_change':
        npimg[npimg==-1] = 255
        out_img = Image.fromarray(npimg.squeeze().astype('uint8'))
        return out_img

    out_img = Image.fromarray(npimg.squeeze().astype('uint8'))
    out_img.putpalette(colorpallete)
    return out_img

def get_confidence_pallete(npimg,max_cls = 21):
    npimg[npimg==-1] = 255
    npimg[npimg==max_cls] = 255
    out_img = Image.fromarray(npimg.squeeze().astype('uint8'))
    return out_img

def get_confidence_pallete_context(npimg,cls):
    npimg[npimg==-1] = 255
    npimg[npimg== cls] = 255
    out_img = Image.fromarray(npimg.squeeze().astype('uint8'))
    return out_img


def _get_voc_pallete(num_cls):
    n = num_cls
    pallete = [0]*(n*3)
    for j in range(0,n):
            lab = j
            pallete[j*3+0] = 0
            pallete[j*3+1] = 0
            pallete[j*3+2] = 0
            i = 0
            while (lab > 0):
                    pallete[j*3+0] |= (((lab >> 0) & 1) << (7-i))
                    pallete[j*3+1] |= (((lab >> 1) & 1) << (7-i))
                    pallete[j*3+2] |= (((lab >> 2) & 1) << (7-i))
                    i = i + 1
                    lab >>= 3
    return pallete

def eq2(imgname,preresult,save_path):
    file_path = '/home/ubuntu/JP/data/VOC2012/spiex/'
    save_path = os.path.join(save_path,imgname)
    imgname = os.path.join(file_path,imgname+ '.mat')
    matimg = h5py.File(imgname)
    #print(np.shape(matimg))
    image_spiex = np.transpose(matimg['var1']).astype('int32')
    N = np.max(image_spiex)
    #print(np.shape(preresult))
    #print(np.shape(image_spiex))
    result = np.zeros((N+1,21))
    for i in range(len(image_spiex)):
        for j in range(len(image_spiex[0])):
            #print([i,j])
            result[image_spiex[i][j]] += preresult[i][j]
    np.save(save_path,result)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return 1
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return 0
    elif v.lower() in ('2','mat','contrast'):
        return 2
    elif v.lower() in ('3','characteristic ','spi','p_contrast','coco'):
        return 3
    elif v.lower() in ('4','coco_traintoval'):
        return 4
    elif v.lower() in ('z_contrast'):
        return int(0b10000011)
    elif v.lower() in ('x_contrast'):
        return int(0b10100011)
    elif v.lower() in ('z_contrast_ur'):
        return int(0b11000011)
    elif v.lower() in ('x_contrast_ur'):
        return int(0b11100011)
    elif v.lower() in ('-1'):
        return -1
    else:
        return 0

def nearest_neighbor_resize(img, new_w, new_h):
    # height and width of the input img
    h, w = img.shape[0], img.shape[1]
    # new image with rgb channel
    ret_img = np.zeros(shape=(new_h, new_w, 3), dtype='uint8')
    # scale factor
    s_h, s_c = (h * 1.0) / new_h, (w * 1.0) / new_w

    # insert pixel to the new img
    for i in range(new_h):
        for j in range(new_w):
            p_x = int(j * s_c)
            p_y = int(i * s_h)

            ret_img[i, j] = img[p_y, p_x]

    return ret_img

def kl_divergence_core(p, q):
    if isinstance(p, torch.Tensor):
        return torch.sum(p * torch.log(p / q))/ (q.shape[0] * q.shape[1])
    else:
        return np.sum(p * np.log(p / q)) / (q.shape[0] * q.shape[1])

def js_divergence_core(p, q):
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence_core(p, m) + 0.5 * kl_divergence_core(q, m)

def divergence_core(ps,qs,T,func):
    div = 0
    l = len(ps)
    for i in range(l):
        s = len(ps[i])
        for j in range(s):
            p = ps[i][j]
            q = qs[i][j]
            q = torch.bmm(torch.bmm(T, q), T)
            t_div = func(p, q)
            div += t_div
    return div

def kl_divergence(ps,qs,T):
    return divergence_core(ps,qs,T,kl_divergence_core)

def js_divergence(ps,qs,T):
    return divergence_core(ps,qs,T,js_divergence_core)

def pre_divergence(x,y):
    prediction1 = np.argmax(x, axis=2)
    prediction2 = np.argmax(y, axis=2)
    div = np.sum(prediction1 != prediction2) / (x.shape[0] * x.shape[1])
    return div

vocpallete = _get_voc_pallete(256)

adepallete = [0,0,0,120,120,120,180,120,120,6,230,230,80,50,50,4,200,3,120,120,80,140,140,140,204,5,255,230,230,230,4,250,7,224,5,255,235,255,7,150,5,61,120,120,70,8,255,51,255,6,82,143,255,140,204,255,4,255,51,7,204,70,3,0,102,200,61,230,250,255,6,51,11,102,255,255,7,71,255,9,224,9,7,230,220,220,220,255,9,92,112,9,255,8,255,214,7,255,224,255,184,6,10,255,71,255,41,10,7,255,255,224,255,8,102,8,255,255,61,6,255,194,7,255,122,8,0,255,20,255,8,41,255,5,153,6,51,255,235,12,255,160,150,20,0,163,255,140,140,140,250,10,15,20,255,0,31,255,0,255,31,0,255,224,0,153,255,0,0,0,255,255,71,0,0,235,255,0,173,255,31,0,255,11,200,200,255,82,0,0,255,245,0,61,255,0,255,112,0,255,133,255,0,0,255,163,0,255,102,0,194,255,0,0,143,255,51,255,0,0,82,255,0,255,41,0,255,173,10,0,255,173,255,0,0,255,153,255,92,0,255,0,255,255,0,245,255,0,102,255,173,0,255,0,20,255,184,184,0,31,255,0,255,61,0,71,255,255,0,204,0,255,194,0,255,82,0,10,255,0,112,255,51,0,255,0,194,255,0,122,255,0,255,163,255,153,0,0,255,10,255,112,0,143,255,0,82,0,255,163,255,0,255,235,0,8,184,170,133,0,255,0,255,92,184,0,255,255,0,31,0,184,255,0,214,255,255,0,112,92,255,0,0,224,255,112,224,255,70,184,160,163,0,255,153,0,255,71,255,0,255,0,163,255,204,0,255,0,143,0,255,235,133,255,0,255,0,235,245,0,255,255,0,122,255,245,0,10,190,212,214,255,0,0,204,255,20,0,255,255,255,0,0,153,255,0,41,255,0,255,204,41,0,255,41,255,0,173,0,255,0,245,255,71,0,255,122,0,255,0,255,184,0,92,255,184,255,0,0,133,255,255,214,0,25,194,194,102,255,0,92,0,255]

citypallete = [
128,64,128,244,35,232,70,70,70,102,102,156,190,153,153,153,153,153,250,170,30,220,220,0,107,142,35,152,251,152,70,130,180,220,20,60,255,0,0,0,0,142,0,0,70,0,60,100,0,80,100,0,0,230,119,11,32,128,192,0,0,64,128,128,64,128,0,192,128,128,192,128,64,64,0,192,64,0,64,192,0,192,192,0,64,64,128,192,64,128,64,192,128,192,192,128,0,0,64,128,0,64,0,128,64,128,128,64,0,0,192,128,0,192,0,128,192,128,128,192,64,0,64,192,0,64,64,128,64,192,128,64,64,0,192,192,0,192,64,128,192,192,128,192,0,64,64,128,64,64,0,192,64,128,192,64,0,64,192,128,64,192,0,192,192,128,192,192,64,64,64,192,64,64,64,192,64,192,192,64,64,64,192,192,64,192,64,192,192,192,192,192,32,0,0,160,0,0,32,128,0,160,128,0,32,0,128,160,0,128,32,128,128,160,128,128,96,0,0,224,0,0,96,128,0,224,128,0,96,0,128,224,0,128,96,128,128,224,128,128,32,64,0,160,64,0,32,192,0,160,192,0,32,64,128,160,64,128,32,192,128,160,192,128,96,64,0,224,64,0,96,192,0,224,192,0,96,64,128,224,64,128,96,192,128,224,192,128,32,0,64,160,0,64,32,128,64,160,128,64,32,0,192,160,0,192,32,128,192,160,128,192,96,0,64,224,0,64,96,128,64,224,128,64,96,0,192,224,0,192,96,128,192,224,128,192,32,64,64,160,64,64,32,192,64,160,192,64,32,64,192,160,64,192,32,192,192,160,192,192,96,64,64,224,64,64,96,192,64,224,192,64,96,64,192,224,64,192,96,192,192,224,192,192,0,32,0,128,32,0,0,160,0,128,160,0,0,32,128,128,32,128,0,160,128,128,160,128,64,32,0,192,32,0,64,160,0,192,160,0,64,32,128,192,32,128,64,160,128,192,160,128,0,96,0,128,96,0,0,224,0,128,224,0,0,96,128,128,96,128,0,224,128,128,224,128,64,96,0,192,96,0,64,224,0,192,224,0,64,96,128,192,96,128,64,224,128,192,224,128,0,32,64,128,32,64,0,160,64,128,160,64,0,32,192,128,32,192,0,160,192,128,160,192,64,32,64,192,32,64,64,160,64,192,160,64,64,32,192,192,32,192,64,160,192,192,160,192,0,96,64,128,96,64,0,224,64,128,224,64,0,96,192,128,96,192,0,224,192,128,224,192,64,96,64,192,96,64,64,224,64,192,224,64,64,96,192,192,96,192,64,224,192,192,224,192,32,32,0,160,32,0,32,160,0,160,160,0,32,32,128,160,32,128,32,160,128,160,160,128,96,32,0,224,32,0,96,160,0,224,160,0,96,32,128,224,32,128,96,160,128,224,160,128,32,96,0,160,96,0,32,224,0,160,224,0,32,96,128,160,96,128,32,224,128,160,224,128,96,96,0,224,96,0,96,224,0,224,224,0,96,96,128,224,96,128,96,224,128,224,224,128,32,32,64,160,32,64,32,160,64,160,160,64,32,32,192,160,32,192,32,160,192,160,160,192,96,32,64,224,32,64,96,160,64,224,160,64,96,32,192,224,32,192,96,160,192,224,160,192,32,96,64,160,96,64,32,224,64,160,224,64,32,96,192,160,96,192,32,224,192,160,224,192,96,96,64,224,96,64,96,224,64,224,224,64,96,96,192,224,96,192,96,224,192,0,0,0]

