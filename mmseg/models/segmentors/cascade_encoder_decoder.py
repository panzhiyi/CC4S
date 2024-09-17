from torch import nn
import torch
from torch.nn.parameter import Parameter
from torch.nn import functional as Func

from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .encoder_decoder import EncoderDecoder

class RW_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim, shrink_factor):
        
        super(RW_Module, self).__init__()
        self.chanel_in = in_dim
        self.shrink_factor = shrink_factor

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        
    def own_softmax1(self, x):
    
        maxes1 = torch.max(x, 1, keepdim=True)[0]
        maxes2 = torch.max(x, 2, keepdim=True)[0]
        x_exp = torch.exp(x-0.5*maxes1-0.5*maxes2)
        x_exp_sum_sqrt = torch.sqrt(torch.sum(x_exp, 2, keepdim=True))

        return (x_exp/x_exp_sum_sqrt)/torch.transpose(x_exp_sum_sqrt, 1, 2)
    
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        x_shrink = x
        m_batchsize, C, height, width = x.size()
        if self.shrink_factor != 1:
            height = (height - 1) // self.shrink_factor + 1
            width = (width - 1) // self.shrink_factor + 1
            x_shrink = Func.interpolate(x_shrink, size=(height, width), mode='bilinear', align_corners=True)
            
        
        proj_query = self.query_conv(x_shrink).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x_shrink).view(m_batchsize, -1, width*height)
        
        energy = torch.bmm(proj_query, proj_key)

        attention = self.softmax(energy)

        proj_value = self.value_conv(x_shrink).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        
        if self.shrink_factor != 1:
            height = (height - 1) * self.shrink_factor + 1
            width = (width - 1) * self.shrink_factor + 1
            out = Func.interpolate(out, size=(height, width), mode='bilinear', align_corners=True)

        out = self.gamma*out + x
        return out,energy



@SEGMENTORS.register_module()
class CascadeEncoderDecoder(EncoderDecoder):
    """Cascade Encoder Decoder segmentors.

    CascadeEncoderDecoder almost the same as EncoderDecoder, while decoders of
    CascadeEncoderDecoder are cascaded. The output of previous decoder_head
    will be the input of next decoder_head.
    """

    def __init__(self,
                 num_stages,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        self.num_stages = num_stages
        super(CascadeEncoderDecoder, self).__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
        self.PAM_Module = RW_Module(384,shrink_factor = 1)

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        assert isinstance(decode_head, list)
        assert len(decode_head) == self.num_stages
        self.decode_head = nn.ModuleList()
        for i in range(self.num_stages):
            self.decode_head.append(builder.build_head(decode_head[i]))
        self.align_corners = self.decode_head[-1].align_corners
        self.num_classes = self.decode_head[-1].num_classes

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        self.backbone.init_weights(pretrained=pretrained)
        for i in range(self.num_stages):
            self.decode_head[i].init_weights()
        if self.with_auxiliary_head:
            if isinstance(self.auxiliary_head, nn.ModuleList):
                for aux_head in self.auxiliary_head:
                    aux_head.init_weights()
            else:
                self.auxiliary_head.init_weights()

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        # print(img.shape)
        x = self.extract_feat(img)
        # print(x[-1].shape)

        x[-1], P = self.PAM_Module(x[-1]) 
        # P = None

        # print(x[-1].shape)
        # print(P.shape)
        out = self.decode_head[0].forward_test(x, img_metas, self.test_cfg)
        for i in range(1, self.num_stages):
            out = self.decode_head[i].forward_test(x, out, img_metas,
                                                   self.test_cfg)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out,x,P

    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()

        loss_decode = self.decode_head[0].forward_train(
            x, img_metas, gt_semantic_seg, self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode_0'))

        for i in range(1, self.num_stages):
            # forward test again, maybe unnecessary for most methods.
            prev_outputs = self.decode_head[i - 1].forward_test(
                x, img_metas, self.test_cfg)
            loss_decode = self.decode_head[i].forward_train(
                x, prev_outputs, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update(add_prefix(loss_decode, f'decode_{i}'))

        return losses
