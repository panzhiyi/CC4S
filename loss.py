import os
import sys
import time
import shutil
import random
import argparse
import numpy as np
import torchnet as tnt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.models as models
import torch.backends.cudnn as cudnn
from torch.autograd import Variable, Function
from torch.utils import data

from IPython.core import debugger
debug = debugger.Pdb().set_trace

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


class LaplaceLoss ( nn.Module ) :
    def __init__(self) :
        super ( LaplaceLoss1, self ).__init__ ()

    def __call__(self, pred,Laplace_W,Laplace_L,image_spiex) :
        Laplace_L.to(torch.float)
        Laplace_W.to(torch.float)
        [b,x,y] = image_spiex.size()
        N = torch.max(image_spiex)
        result = torch.zeros([b,N+1,21],dtype=torch.float64)
        loss = torch.zeros([b],dtype=torch.float64)
        for k in range(b):
            predk=pred[k].permute(1,2,0).cpu()
            for i in range(N+1):
                maskp = (image_spiex[k] == i)
                mask = predk[maskp]
                if int(maskp.sum()) is not 0:
                    result[k][i] = mask.sum()/maskp.sum()
                else:
                    result[k][i] = 0;

            Laplace = torch.mm(result[k].t() , Laplace_L[k][:N+1,:N+1])
            result2 = torch.zeros([21,21],dtype=torch.float64)
            result2 = torch.mm(Laplace, result[k])
            loss[k] = (2 / torch.norm(Laplace_W[k])) * result2.trace()


        return loss.sum()
class LaplaceLoss_gpu_pzy_ori ( nn.Module ) :
    def __init__(self) :
        super ( LaplaceLoss_gpu_pzy_ori, self ).__init__ ()
        self.spixels_num = 1200
        self.class_num = 21

    def __call__(self, pred,Laplace_W_c,Laplace_L_c,image_spiex,class_num) :
        self.class_num=class_num
        Laplace_L = Laplace_L_c.to(torch.float).cuda()
        Laplace_W = Laplace_W_c.to(torch.float).cuda()
        [batchs,x,y] = image_spiex.size() # batchs - batch number ; x,y - widith&height of picture

        pred = pred.permute(0,2,3,1).reshape(batchs,self.class_num,-1) # pred.size = [batchs * classes * HW]
        image_spiex = image_spiex.view(batchs,-1).unsqueeze(1).repeat(1,self.class_num,1).cuda() # image_spiex.size = [batchs * classes * HW]
        result = torch.zeros([batchs, self.class_num, self.spixels_num],dtype=torch.float).cuda() # result.size = [batch * classes * sp_num]
        result = result.scatter_add_(2, image_spiex, pred) / (image_spiex.sum(2,keepdim=True) + 1e-16)
        
        #result = torch.bmm(pred , mask) / (mask.sum(1,keepdim=True) + 1e-16)
        result2 = torch.bmm( torch.bmm(result , Laplace_L) , result.permute(0,2,1))
        loss = (2 / torch.norm(Laplace_W)) * result2.diagonal(dim1=-2, dim2=-1).sum(0)
        

        return loss.mean()

class LaplaceLoss_gpu_pzy ( nn.Module ) :
    def __init__(self) :
        super ( LaplaceLoss_gpu_pzy, self ).__init__ ()
        self.spixels_num = 1000
        self.class_num = 21

    def __call__(self, pred,Laplace_W_c,Laplace_L_c,image_spiex) :

        Laplace_L = Laplace_L_c.to(torch.float64).cuda()
        Laplace_W = Laplace_W_c.to(torch.float64).cuda()
        [batchs,x,y] = image_spiex.size() # batchs - batch number ; x,y - widith&height of picture

        pred = pred.permute(0,2,3,1).reshape(batchs,self.class_num,-1).to(torch.float64).contiguous() # pred.shape = [batchs * classes * widith * height] , all cal on one cpu
        image_spiex = image_spiex.view(batchs,-1).contiguous()
        
        mask = torch.zeros([batchs,x*y,self.spixels_num],dtype=torch.float64).cuda()
        for i in range(self.spixels_num):
            mask[:,:,i] = (image_spiex == i)
        result = torch.bmm(pred , mask) 
        result = result / (mask.sum() + 1e-16)
        result2 = torch.bmm( torch.bmm(result , Laplace_L) , result.permute(0,2,1))
        loss = torch.zeros([batchs],dtype=torch.float64).cuda()
        for i in range(batchs):
            loss[i] = (2 / torch.norm(Laplace_W)) * result2[i].trace()

        return loss.mean()
