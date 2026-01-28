import math
import pdb

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import clip
from Text_Prompt import *
from tools import *
from einops import rearrange, repeat
### impSGN dependancies :
from model.utils.ms_tcn_TP1 import MultiScale_TemporalConv
from model.utils.graphs import Graph
from model.utils.layers import Basic_Layer, Spatial_Graph_Layer
from model.utils.activations import Swish
from model.utils.attentions import ST_Joint_Att





class TextCLIP(nn.Module):
    def __init__(self, model) :
        super(TextCLIP, self).__init__()
        self.model = model

    def forward(self,text):
        return self.model.encode_text(text)

def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


###########################################  implementing SGN model into Model_lst_4part() class ########################################################### 

class Model_SGN(nn.Module):
    def __init__(self, batch_size, num_class=60, seg=64, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3, drop_out=0, bias=True, adaptive=True, head=['ViT-B/32'], k=0):  #, dataset, args
        # super(Model_lst_4part, self).__init__()
        super(Model_SGN, self).__init__()
        
        self.dim1 = 128
        self.seg = seg
        num_joint = num_point
        bs = batch_size

        self.spa = self.one_hot(bs*num_person, num_joint, self.seg)
        self.spa = self.spa.permute(0, 3, 2, 1).cuda()
        self.tem = self.one_hot(bs*num_person, self.seg, num_joint)
        self.tem = self.tem.permute(0, 3, 1, 2).cuda()


        self.tem_embed = embed(self.seg, self.dim1, norm=False, bias=bias)        #64*4       #adapted
        self.spa_embed = embed(num_joint, self.dim1//4, norm=False, bias=bias)    #64 #adapted
        self.joint_embed = embed(3, self.dim1//4, norm=True, bias=bias)           #64 #adapted
        self.dif_embed = embed(3, self.dim1//4, norm=True, bias=bias)             #64 #adapted
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.cnn = local(self.dim1, self.dim1 * 2, bias=bias)
        self.compute_g1 = compute_g_spa(self.dim1 // 2, self.dim1, bias=bias)
        self.gcn1 = gcn_spa(self.dim1 // 2, self.dim1 // 2, bias=bias)
        self.gcn2 = gcn_spa(self.dim1 // 2, self.dim1, bias=bias)
        self.gcn3 = gcn_spa(self.dim1, self.dim1, bias=bias)
        self.fc = nn.Linear(self.dim1 * 2, num_class)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        nn.init.constant_(self.gcn1.w.cnn.weight, 0)
        nn.init.constant_(self.gcn2.w.cnn.weight, 0)
        nn.init.constant_(self.gcn3.w.cnn.weight, 0)

        #########
        self.linear_head = nn.ModuleDict()
        self.logit_scale = nn.Parameter(torch.ones(1,5) * np.log(1 / 0.07))

        self.part_list = nn.ModuleList()

        for i in range(4):
            self.part_list.append(nn.Linear(256,512))

        self.head = head
        if 'ViT-B/32' in self.head:
            self.linear_head['ViT-B/32'] = nn.Linear(256,512)
            conv_init(self.linear_head['ViT-B/32'])
        
        if 'ViT-B/16' in self.head:
            self.linear_head['ViT-B/16'] = nn.Linear(256,512)
            conv_init(self.linear_head['ViT-B/16'])
        if 'ViT-L/14' in self.head:
            self.linear_head['ViT-L/14'] = nn.Linear(256,768)
            conv_init(self.linear_head['ViT-L/14'])
        if 'ViT-L/14@336px' in self.head:
            self.linear_head['ViT-L/14@336px'] = nn.Linear(256,768)
            conv_init(self.linear_head['ViT-L/14@336px'])
        
        if 'RN50x64' in self.head:
            self.linear_head['RN50x64'] = nn.Linear(256,1024)
            conv_init(self.linear_head['RN50x64'])

        if 'RN50x16' in self.head:
            self.linear_head['RN50x16'] = nn.Linear(256,768)
            conv_init(self.linear_head['RN50x16'])

        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x


    def forward(self, x):
        #########
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1) # n, t, vc ->n, t, v, c, -> (n, c, t, v, 1=m)
        N, C, T, V, M = x.size()
        x = rearrange(x, 'n c t v m -> (n m) c v t', m=M, v=V).contiguous()
        # print('**** x = ', x.size())    #[64, 3, 25, 64]  #nm, c, v, t



        ### Dynamic Representation
        dif = x[:, :, :, 1:] - x[:, :, :, 0:-1]
        # print('**** dif = ', dif.size())        #64, 3, 25, 63
        dif = torch.cat([dif.new(dif.size(0), dif.size(1), dif.size(2), 1).zero_(), dif], dim=-1)
        pos = self.joint_embed(x)
        # print('**** pos embed = ', pos.size())      #[64, 64, 25, 64]
        tem1 = self.tem_embed(self.tem)
        # print('**** tem1 embed = ', tem1.size())        #[64, 256, 25, 64])
        spa1 = self.spa_embed(self.spa)
        # print('**** spa1 embed = ', spa1.size())        #[64, 64, 25, 64]
        dif = self.dif_embed(dif)                   
        # print('**** dif embed = ', dif.size())             #[64, 64, 25, 64]
        dy = pos + dif
        ### Joint-level Module
        x= torch.cat([dy, spa1[:dy.size(0), :, :, :]], 1)   # spa1 mdf to spa1[:dy.size(0), :, :, :] so if "dy" won't cause an issue when it takes less samples that spa1
        g = self.compute_g1(x)
        x = self.gcn1(x, g)
        x = self.gcn2(x, g)
        x = self.gcn3(x, g)
        ### Frame-level Module
        x = x + tem1[:x.size(0), :, :, :]     ## tem1 mdf to  so if "x" won't cause an issue when it takes less samples that tem1
        # print('**** x: ', x.size())
        x = self.cnn(x) #n,c: in=256->out=256*2, v, t
        



        ###############
        ## N*M,C,T,V
        x = x.permute(0, 1, 3, 2)
        c_new = x.size(1)

        feature = x.view(N,M,c_new,T//4,V)
        head_list = torch.Tensor([2,3,20]).long()
        hand_list = torch.Tensor([4,5,6,7,8,9,10,11,21,22,23,24]).long()
        foot_list = torch.Tensor([12,13,14,15,16,17,18,19]).long()
        hip_list = torch.Tensor([0,1,2,12,16]).long()
         

        head_feature = self.part_list[0](feature[:,:,:,:,head_list].mean(4).mean(3).mean(1))
        hand_feature = self.part_list[1](feature[:,:,:,:,hand_list].mean(4).mean(3).mean(1))
        foot_feature = self.part_list[2](feature[:,:,:,:,foot_list].mean(4).mean(3).mean(1))
        hip_feature = self.part_list[3](feature[:,:,:,:,hip_list].mean(4).mean(3).mean(1))


        x = x.reshape(N, M, c_new, -1)      #mdf
        x = x.mean(3).mean(1)

        feature_dict = dict()

        for name in self.head:
            feature_dict[name] = self.linear_head[name](x)
        
        x = self.drop_out(x)

        return self.fc(x), feature_dict, self.logit_scale, [head_feature, hand_feature, hip_feature, foot_feature]

    def one_hot(self, bs, spa, tem):

        y = torch.arange(spa).unsqueeze(-1)
        y_onehot = torch.FloatTensor(spa, spa)

        y_onehot.zero_()
        y_onehot.scatter_(1, y, 1)

        y_onehot = y_onehot.unsqueeze(0).unsqueeze(0)
        y_onehot = y_onehot.repeat(bs, tem, 1, 1)

        return y_onehot






class Model_SGN_bone(nn.Module):
    def __init__(self, batch_size, num_class=60, seg=64, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3, drop_out=0, bias=True, adaptive=True, head=['ViT-B/32'], k=1):  #, dataset, args
        super(Model_SGN_bone, self).__init__()
        self.dim1 = 128
        self.seg = seg
        num_joint = num_point
        bs = batch_size
        
        self.spa = self.one_hot(bs*num_person, num_joint, self.seg)
        self.spa = self.spa.permute(0, 3, 2, 1).cuda()
        self.tem = self.one_hot(bs*num_person, self.seg, num_joint)
        self.tem = self.tem.permute(0, 3, 1, 2).cuda()

        self.tem_embed = embed(self.seg, self.dim1, norm=False, bias=bias)        #64*4       #adapted
        self.spa_embed = embed(num_joint, self.dim1//4, norm=False, bias=bias)    #64 #adapted
        self.joint_embed = embed(3, self.dim1//4, norm=True, bias=bias)           #64 #adapted
        self.dif_embed = embed(3, self.dim1//4, norm=True, bias=bias)             #64 #adapted
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.cnn = local(self.dim1, self.dim1 * 2, bias=bias)
        self.compute_g1 = compute_g_spa(self.dim1 // 2, self.dim1, bias=bias)
        self.gcn1 = gcn_spa(self.dim1 // 2, self.dim1 // 2, bias=bias)
        self.gcn2 = gcn_spa(self.dim1 // 2, self.dim1, bias=bias)
        self.gcn3 = gcn_spa(self.dim1, self.dim1, bias=bias)
        self.fc = nn.Linear(self.dim1 * 2, num_class)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        nn.init.constant_(self.gcn1.w.cnn.weight, 0)
        nn.init.constant_(self.gcn2.w.cnn.weight, 0)
        nn.init.constant_(self.gcn3.w.cnn.weight, 0)

        #########
        self.linear_head = nn.ModuleDict()
        self.logit_scale = nn.Parameter(torch.ones(1,5) * np.log(1 / 0.07))

        self.part_list = nn.ModuleList()

        for i in range(4):
            self.part_list.append(nn.Linear(256,512))

        self.head = head
        if 'ViT-B/32' in self.head:
            self.linear_head['ViT-B/32'] = nn.Linear(256,512)
            conv_init(self.linear_head['ViT-B/32'])
        
        if 'ViT-B/16' in self.head:
            self.linear_head['ViT-B/16'] = nn.Linear(256,512)
            conv_init(self.linear_head['ViT-B/16'])
        if 'ViT-L/14' in self.head:
            self.linear_head['ViT-L/14'] = nn.Linear(256,768)
            conv_init(self.linear_head['ViT-L/14'])
        if 'ViT-L/14@336px' in self.head:
            self.linear_head['ViT-L/14@336px'] = nn.Linear(256,768)
            conv_init(self.linear_head['ViT-L/14@336px'])
        
        if 'RN50x64' in self.head:
            self.linear_head['RN50x64'] = nn.Linear(256,1024)
            conv_init(self.linear_head['RN50x64'])

        if 'RN50x16' in self.head:
            self.linear_head['RN50x16'] = nn.Linear(256,768)
            conv_init(self.linear_head['RN50x16'])

        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x


    def forward(self, x):
        #########
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1) # n, t, vc ->n, t, v, c, -> (n, c, t, v, 1=m)
        N, C, T, V, M = x.size()
        x = rearrange(x, 'n c t v m -> (n m) c v t', m=M, v=V).contiguous()
        # print('**** x = ', x.size())    #[64, 3, 25, 64]  #nm, c, v, t
        ### Dynamic Representation
        dif = x[:, :, :, 1:] - x[:, :, :, 0:-1]
        # print('**** dif = ', dif.size())        #64, 3, 25, 63
        dif = torch.cat([dif.new(dif.size(0), dif.size(1), dif.size(2), 1).zero_(), dif], dim=-1)
        pos = self.joint_embed(x)
        # print('**** pos embed = ', pos.size())      #[64, 64, 25, 64]
        tem1 = self.tem_embed(self.tem)
        # print('**** tem1 embed = ', tem1.size())        #[64, 256, 25, 64])
        spa1 = self.spa_embed(self.spa)
        # print('**** spa1 embed = ', spa1.size())        #[64, 64, 25, 64]
        dif = self.dif_embed(dif)                   
        # print('**** dif embed = ', dif.size())             #[64, 64, 25, 64]
        dy = pos + dif
        ### Joint-level Module
        x= torch.cat([dy, spa1[:dy.size(0), :, :, :]], 1)   # spa1 mdf to spa1[:dy.size(0), :, :, :] so if "dy" won't cause an issue when it takes less samples that spa1
        g = self.compute_g1(x)
        x = self.gcn1(x, g)
        x = self.gcn2(x, g)
        x = self.gcn3(x, g)
        ### Frame-level Module
        x = x + tem1[:x.size(0), :, :, :]     ## tem1 mdf to  so if "x" won't cause an issue when it takes less samples that tem1
        # print('**** x: ', x.size())
        x = self.cnn(x) #n,c: in=256->out=256*2, v, t
        
        

        ###############
        ## N*M,C,T,V
        x = x.permute(0, 1, 3, 2)
        c_new = x.size(1)

        feature = x.view(N,M,c_new,T//4,V)
        head_list = torch.Tensor([2,3]).long()
        hand_list = torch.Tensor([4,5,6,7,8,9,10,11,20,22,23,24]).long()
        foot_list = torch.Tensor([12,13,14,15,16,17,18,19]).long()
        hip_list = torch.Tensor([0,1,12,16]).long()
        head_feature = self.part_list[0](feature[:,:,:,:,head_list].mean(4).mean(3).mean(1))
        hand_feature = self.part_list[1](feature[:,:,:,:,hand_list].mean(4).mean(3).mean(1))
        foot_feature = self.part_list[2](feature[:,:,:,:,foot_list].mean(4).mean(3).mean(1))
        hip_feature = self.part_list[3](feature[:,:,:,:,hip_list].mean(4).mean(3).mean(1))


        x = x.reshape(N, M, c_new, -1)      #mdf
        x = x.mean(3).mean(1)

        feature_dict = dict()

        for name in self.head:
            feature_dict[name] = self.linear_head[name](x)
        
        x = self.drop_out(x)

        return self.fc(x), feature_dict, self.logit_scale, [head_feature, hand_feature, hip_feature, foot_feature]



    def one_hot(self, bs, spa, tem):

        y = torch.arange(spa).unsqueeze(-1)
        y_onehot = torch.FloatTensor(spa, spa)

        y_onehot.zero_()
        y_onehot.scatter_(1, y, 1)

        y_onehot = y_onehot.unsqueeze(0).unsqueeze(0)
        y_onehot = y_onehot.repeat(bs, tem, 1, 1)

        return y_onehot



class Model_SGN_ucla(nn.Module):
    def __init__(self, batch_size, num_class=60, seg=64, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3, drop_out=0, bias=True, adaptive=True, head=['ViT-B/32'], k=1):  #, dataset, args
        super(Model_SGN_ucla, self).__init__()
        
        self.dim1 = 128
        self.seg = seg
        num_joint = num_point
        bs = batch_size
        
        self.spa = self.one_hot(bs*num_person, num_joint, self.seg)
        self.spa = self.spa.permute(0, 3, 2, 1).cuda()
        self.tem = self.one_hot(bs*num_person, self.seg, num_joint)
        self.tem = self.tem.permute(0, 3, 1, 2).cuda()

        self.tem_embed = embed(self.seg, self.dim1, norm=False, bias=bias)        #64*4       #adapted
        self.spa_embed = embed(num_joint, self.dim1//4, norm=False, bias=bias)    #64 #adapted
        self.joint_embed = embed(3, self.dim1//4, norm=True, bias=bias)           #64 #adapted
        self.dif_embed = embed(3, self.dim1//4, norm=True, bias=bias)             #64 #adapted
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.cnn = local(self.dim1, self.dim1 * 2, bias=bias)
        self.compute_g1 = compute_g_spa(self.dim1 // 2, self.dim1, bias=bias)
        self.gcn1 = gcn_spa(self.dim1 // 2, self.dim1 // 2, bias=bias)
        self.gcn2 = gcn_spa(self.dim1 // 2, self.dim1, bias=bias)
        self.gcn3 = gcn_spa(self.dim1, self.dim1, bias=bias)
        self.fc = nn.Linear(self.dim1 * 2, num_class)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        nn.init.constant_(self.gcn1.w.cnn.weight, 0)
        nn.init.constant_(self.gcn2.w.cnn.weight, 0)
        nn.init.constant_(self.gcn3.w.cnn.weight, 0)

        #########
        self.linear_head = nn.ModuleDict()
        self.logit_scale = nn.Parameter(torch.ones(1,5) * np.log(1 / 0.07))

        self.part_list = nn.ModuleList()

        for i in range(4):
            self.part_list.append(nn.Linear(256,512))

        self.head = head
        if 'ViT-B/32' in self.head:
            self.linear_head['ViT-B/32'] = nn.Linear(256,512)
            conv_init(self.linear_head['ViT-B/32'])
        
        if 'ViT-B/16' in self.head:
            self.linear_head['ViT-B/16'] = nn.Linear(256,512)
            conv_init(self.linear_head['ViT-B/16'])
        if 'ViT-L/14' in self.head:
            self.linear_head['ViT-L/14'] = nn.Linear(256,768)
            conv_init(self.linear_head['ViT-L/14'])
        if 'ViT-L/14@336px' in self.head:
            self.linear_head['ViT-L/14@336px'] = nn.Linear(256,768)
            conv_init(self.linear_head['ViT-L/14@336px'])
        
        if 'RN50x64' in self.head:
            self.linear_head['RN50x64'] = nn.Linear(256,1024)
            conv_init(self.linear_head['RN50x64'])

        if 'RN50x16' in self.head:
            self.linear_head['RN50x16'] = nn.Linear(256,768)
            conv_init(self.linear_head['RN50x16'])

        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x


    def forward(self, x):
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1) # n, t, vc ->n, t, v, c, -> (n, c, t, v, 1=m)
        N, C, T, V, M = x.size()
        x = rearrange(x, 'n c t v m -> (n m) c v t', m=M, v=V).contiguous()

        ### Dynamic Representation
        dif = x[:, :, :, 1:] - x[:, :, :, 0:-1]
        dif = torch.cat([dif.new(dif.size(0), dif.size(1), dif.size(2), 1).zero_(), dif], dim=-1)
        pos = self.joint_embed(x)
        tem1 = self.tem_embed(self.tem)
        spa1 = self.spa_embed(self.spa)
        dif = self.dif_embed(dif)                   
        dy = pos + dif
        ### Joint-level Module
        x= torch.cat([dy, spa1[:dy.size(0), :, :, :]], 1)   # spa1 mdf to spa1[:dy.size(0), :, :, :] so if "dy" won't cause an issue when it takes less samples that spa1
        g = self.compute_g1(x)
        x = self.gcn1(x, g)
        x = self.gcn2(x, g)
        x = self.gcn3(x, g)
        ### Frame-level Module
        x = x + tem1[:x.size(0), :, :, :]     ## tem1 mdf to  so if "x" won't cause an issue when it takes less samples that tem1
        # print('**** x: ', x.size())
        x = self.cnn(x) #n,c: in=256->out=256*2, v, t
        
        ## N*M,C,T,V
        x = x.permute(0, 1, 3, 2)
        c_new = x.size(1)

        feature = x.view(N,M,c_new,T//4,V)
        head_list = torch.Tensor([2,3]).long()
        hand_list = torch.Tensor([10,11,6,7,8,9,4,5]).long()
        foot_list = torch.Tensor([16,17,18,19,12,13,14,15]).long()
        hip_list = torch.Tensor([0,1,12,16]).long()
        head_feature = self.part_list[0](feature[:,:,:,:,head_list].mean(4).mean(3).mean(1))
        hand_feature = self.part_list[1](feature[:,:,:,:,hand_list].mean(4).mean(3).mean(1))
        foot_feature = self.part_list[2](feature[:,:,:,:,foot_list].mean(4).mean(3).mean(1))
        hip_feature = self.part_list[3](feature[:,:,:,:,hip_list].mean(4).mean(3).mean(1))


        x = x.reshape(N, M, c_new, -1)      #mdf
        x = x.mean(3).mean(1)

        feature_dict = dict()

        for name in self.head:
            feature_dict[name] = self.linear_head[name](x)
        
        x = self.drop_out(x)

        return self.fc(x), feature_dict, self.logit_scale, [head_feature, hand_feature, hip_feature, foot_feature]

    def one_hot(self, bs, spa, tem):

        y = torch.arange(spa).unsqueeze(-1)
        y_onehot = torch.FloatTensor(spa, spa)

        y_onehot.zero_()
        y_onehot.scatter_(1, y, 1)

        y_onehot = y_onehot.unsqueeze(0).unsqueeze(0)
        y_onehot = y_onehot.repeat(bs, tem, 1, 1)

        return y_onehot



class Model_SGN_bone_ucla(nn.Module):
    def __init__(self, batch_size, num_class=60, seg=64, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3, drop_out=0, bias=True, adaptive=True, head=['ViT-B/32'], k=1):  #, dataset, args
        super(Model_SGN_bone_ucla, self).__init__()
        
        self.dim1 = 128
        self.seg = seg
        num_joint = num_point
        bs = batch_size
        
        self.spa = self.one_hot(bs*num_person, num_joint, self.seg)
        self.spa = self.spa.permute(0, 3, 2, 1).cuda()
        self.tem = self.one_hot(bs*num_person, self.seg, num_joint)
        self.tem = self.tem.permute(0, 3, 1, 2).cuda()

        self.tem_embed = embed(self.seg, self.dim1, norm=False, bias=bias)        #64*4       #adapted
        self.spa_embed = embed(num_joint, self.dim1//4, norm=False, bias=bias)    #64 #adapted
        self.joint_embed = embed(3, self.dim1//4, norm=True, bias=bias)           #64 #adapted
        self.dif_embed = embed(3, self.dim1//4, norm=True, bias=bias)             #64 #adapted
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.cnn = local(self.dim1, self.dim1 * 2, bias=bias)
        self.compute_g1 = compute_g_spa(self.dim1 // 2, self.dim1, bias=bias)
        self.gcn1 = gcn_spa(self.dim1 // 2, self.dim1 // 2, bias=bias)
        self.gcn2 = gcn_spa(self.dim1 // 2, self.dim1, bias=bias)
        self.gcn3 = gcn_spa(self.dim1, self.dim1, bias=bias)
        self.fc = nn.Linear(self.dim1 * 2, num_class)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        nn.init.constant_(self.gcn1.w.cnn.weight, 0)
        nn.init.constant_(self.gcn2.w.cnn.weight, 0)
        nn.init.constant_(self.gcn3.w.cnn.weight, 0)

        #########
        self.linear_head = nn.ModuleDict()
        self.logit_scale = nn.Parameter(torch.ones(1,5) * np.log(1 / 0.07))

        self.part_list = nn.ModuleList()

        for i in range(4):
            self.part_list.append(nn.Linear(256,512))

        self.head = head
        if 'ViT-B/32' in self.head:
            self.linear_head['ViT-B/32'] = nn.Linear(256,512)
            conv_init(self.linear_head['ViT-B/32'])
        
        if 'ViT-B/16' in self.head:
            self.linear_head['ViT-B/16'] = nn.Linear(256,512)
            conv_init(self.linear_head['ViT-B/16'])
        if 'ViT-L/14' in self.head:
            self.linear_head['ViT-L/14'] = nn.Linear(256,768)
            conv_init(self.linear_head['ViT-L/14'])
        if 'ViT-L/14@336px' in self.head:
            self.linear_head['ViT-L/14@336px'] = nn.Linear(256,768)
            conv_init(self.linear_head['ViT-L/14@336px'])
        
        if 'RN50x64' in self.head:
            self.linear_head['RN50x64'] = nn.Linear(256,1024)
            conv_init(self.linear_head['RN50x64'])

        if 'RN50x16' in self.head:
            self.linear_head['RN50x16'] = nn.Linear(256,768)
            conv_init(self.linear_head['RN50x16'])

        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x


    def forward(self, x):
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1) # n, t, vc ->n, t, v, c, -> (n, c, t, v, 1=m)
        N, C, T, V, M = x.size()
        x = rearrange(x, 'n c t v m -> (n m) c v t', m=M, v=V).contiguous()

        ### Dynamic Representation
        dif = x[:, :, :, 1:] - x[:, :, :, 0:-1]
        dif = torch.cat([dif.new(dif.size(0), dif.size(1), dif.size(2), 1).zero_(), dif], dim=-1)
        pos = self.joint_embed(x)
        tem1 = self.tem_embed(self.tem)
        spa1 = self.spa_embed(self.spa)
        dif = self.dif_embed(dif)                   
        dy = pos + dif
        ### Joint-level Module
        x= torch.cat([dy, spa1[:dy.size(0), :, :, :]], 1)   # spa1 mdf to spa1[:dy.size(0), :, :, :] so if "dy" won't cause an issue when it takes less samples that spa1
        g = self.compute_g1(x)
        x = self.gcn1(x, g)
        x = self.gcn2(x, g)
        x = self.gcn3(x, g)
        ### Frame-level Module
        x = x + tem1[:x.size(0), :, :, :]     ## tem1 mdf to  so if "x" won't cause an issue when it takes less samples that tem1
        # print('**** x: ', x.size())
        x = self.cnn(x) #n,c: in=256->out=256*2, v, t
        
        ## N*M,C,T,V
        x = x.permute(0, 1, 3, 2)
        c_new = x.size(1)

        feature = x.view(N,M,c_new,T//4,V)
        head_list = torch.Tensor([2]).long()
        hand_list = torch.Tensor([7,8,9,10,3,4,5,6]).long()
        foot_list = torch.Tensor([11,12,13,14,15,16,17,18]).long()
        hip_list = torch.Tensor([0,1,11,15]).long()
        head_feature = self.part_list[0](feature[:,:,:,:,head_list].mean(4).mean(3).mean(1))
        hand_feature = self.part_list[1](feature[:,:,:,:,hand_list].mean(4).mean(3).mean(1))
        foot_feature = self.part_list[2](feature[:,:,:,:,foot_list].mean(4).mean(3).mean(1))
        hip_feature = self.part_list[3](feature[:,:,:,:,hip_list].mean(4).mean(3).mean(1))


        x = x.reshape(N, M, c_new, -1)      #mdf
        x = x.mean(3).mean(1)

        feature_dict = dict()

        for name in self.head:
            feature_dict[name] = self.linear_head[name](x)
        
        x = self.drop_out(x)

        return self.fc(x), feature_dict, self.logit_scale, [head_feature, hand_feature, hip_feature, foot_feature]

    def one_hot(self, bs, spa, tem):

        y = torch.arange(spa).unsqueeze(-1)
        y_onehot = torch.FloatTensor(spa, spa)

        y_onehot.zero_()
        y_onehot.scatter_(1, y, 1)

        y_onehot = y_onehot.unsqueeze(0).unsqueeze(0)
        y_onehot = y_onehot.repeat(bs, tem, 1, 1)

        return y_onehot




############ implementing SGN model class-dependencies ##############################
class norm_data(nn.Module):
    def __init__(self, dim= 64):
        super(norm_data, self).__init__()

        self.bn = nn.BatchNorm1d(dim* 25)

    def forward(self, x):
        bs, c, num_joints, step = x.size()
        x = x.view(bs, -1, step)
        x = self.bn(x)
        x = x.view(bs, -1, num_joints, step).contiguous()
        return x

class embed(nn.Module):
    def __init__(self, dim = 3, dim1 = 128, norm = True, bias = False):
        super(embed, self).__init__()

        if norm:
            self.cnn = nn.Sequential(
                norm_data(dim),
                cnn1x1(dim, 64, bias=bias),
                nn.ReLU(),
                cnn1x1(64, dim1, bias=bias),
                nn.ReLU(),
            )
        else:
            self.cnn = nn.Sequential(
                cnn1x1(dim, 64, bias=bias),
                nn.ReLU(),
                cnn1x1(64, dim1, bias=bias),
                nn.ReLU(),
            )

    def forward(self, x):
        x = self.cnn(x)
        return x

class cnn1x1(nn.Module):
    def __init__(self, dim1 = 3, dim2 =3, bias = True):
        super(cnn1x1, self).__init__()
        self.cnn = nn.Conv2d(dim1, dim2, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.cnn(x)
        return x

class local(nn.Module):
    def __init__(self, dim1 = 3, dim2 = 3, bias = False):
        super(local, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool2d((25, 64//4))                               # (v=25, and T//4 = 64//4 = 16 )# modified so it want cause an issue in the hierarchical model
        self.cnn1 = nn.Conv2d(dim1, dim1, kernel_size=(1, 3), padding=(0, 1), bias=bias)
        self.bn1 = nn.BatchNorm2d(dim1)
        self.relu = nn.ReLU()
        self.cnn2 = nn.Conv2d(dim1, dim2, kernel_size=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(dim2)
        self.dropout = nn.Dropout2d(0.2)

    def forward(self, x1):
        x1 = self.maxpool(x1)   
        x = self.cnn1(x1)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.cnn2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

class gcn_spa(nn.Module):
    def __init__(self, in_feature, out_feature, bias = False):
        super(gcn_spa, self).__init__()
        self.bn = nn.BatchNorm2d(out_feature)
        self.relu = nn.ReLU()
        self.w = cnn1x1(in_feature, out_feature, bias=False)
        self.w1 = cnn1x1(in_feature, out_feature, bias=bias)


    def forward(self, x1, g):
        x = x1.permute(0, 3, 2, 1).contiguous()
        x = g.matmul(x)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.w(x) + self.w1(x1)
        x = self.relu(self.bn(x))
        return x

class compute_g_spa(nn.Module):
    def __init__(self, dim1 = 64 *3, dim2 = 64*3, bias = False):
        super(compute_g_spa, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.g1 = cnn1x1(self.dim1, self.dim2, bias=bias)
        self.g2 = cnn1x1(self.dim1, self.dim2, bias=bias)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1):

        g1 = self.g1(x1).permute(0, 3, 2, 1).contiguous()
        g2 = self.g2(x1).permute(0, 3, 1, 2).contiguous()
        g3 = g1.matmul(g2)
        g = self.softmax(g3)
        return g

#############################################################################################################################################################



########################################### impSGN to match Model_SGN() class: ###############################################

class Model_impSGN(nn.Module):
    """Adapted impSGN model to match Model_SGN interface for GAP/CTR-GCN."""

    def __init__(self, batch_size, num_class=60, seg=64, num_point=25, num_person=2, graph=None, graph_args=dict(), head=['ViT-B/32'], args=None, drop_out=0, bias=True, k=1):
        super(Model_impSGN, self).__init__()

        # Basic dims
        self.dim1 = 256 #128 #256 
        self.seg = seg
        self.num_point = num_point
        self.batch_size = batch_size

        # --- One-hot spatial and temporal embeddings ---
        self.spa = self.one_hot(batch_size*num_person, num_point, seg).permute(0, 3, 2, 1).cuda()
        self.tem = self.one_hot(batch_size*num_person, seg, num_point).permute(0, 3, 1, 2).cuda()

        # --- Feature embeddings ---
        self.tem_embed = embed(seg, self.dim1*2, norm=False, bias=bias)
        self.spa_embed = embed(num_point, self.dim1//4, norm=False, bias=bias)  #mdf: 64
        self.joint_embed = embed(6, self.dim1//4, norm=True, bias=bias)         #mdf: 64
        self.velocity_embed = embed(6, self.dim1//4, norm=True, bias=bias)      #mdf: 64
        self.bone_embed = embed(6, self.dim1//4, norm=True, bias=bias)          #mdf: 64
        self.velocity_prime_embed = embed(6, self.dim1//4, norm=True, bias=bias)#mdf: 64

        # CNN & pooling
        self.cnn = local(self.dim1*2, self.dim1*2, bias=bias)
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(self.dim1*2, num_class)

        # STB modules (stabilization)
        self.stab_modules = nn.ModuleList((
            Spatial_Temporal_Att_Block(self.dim1//2, self.dim1, bias=bias),              
            Spatial_Temporal_Att_Block(self.dim1, self.dim1, bias=bias),
            Spatial_Temporal_Att_Block(self.dim1, self.dim1*2, bias=bias),
        ))

        # Linear heads for text features
        self.linear_head = nn.ModuleDict()
        self.logit_scale = nn.Parameter(torch.ones(1,5) * np.log(1 / 0.07))
        self.part_list = nn.ModuleList()

        for i in range(4):
            # self.part_list.append(nn.Linear(256,512)) #org params 
            self.part_list.append(nn.Linear(self.dim1*2,512))   #mdf params for a self.dim1=256

        self.head = head
        if 'ViT-B/32' in self.head:
            self.linear_head['ViT-B/32'] = nn.Linear(self.dim1*2,512)  #the org params for self.dim1=128 => nn.Linear(256,512)
            conv_init(self.linear_head['ViT-B/32'])
        
        if 'ViT-B/16' in self.head:
            self.linear_head['ViT-B/16'] = nn.Linear(self.dim1*2,512)   #the org params for self.dim1=128 => nn.Linear(256,512)
            conv_init(self.linear_head['ViT-B/16'])
        if 'ViT-L/14' in self.head:
            self.linear_head['ViT-L/14'] = nn.Linear(self.dim1*2,768)   #the org params for self.dim1=128 => nn.Linear(256,768)
            conv_init(self.linear_head['ViT-L/14'])
        if 'ViT-L/14@336px' in self.head:
            self.linear_head['ViT-L/14@336px'] = nn.Linear(self.dim1*2,768) #the org params for self.dim1=128 => nn.Linear(256,768)
            conv_init(self.linear_head['ViT-L/14@336px'])
        
        if 'RN50x64' in self.head:
            self.linear_head['RN50x64'] = nn.Linear(self.dim1*2,1024)   #the org params for self.dim1=128 => nn.Linear(256,1024)
            conv_init(self.linear_head['RN50x64'])

        if 'RN50x16' in self.head:
            self.linear_head['RN50x16'] = nn.Linear(self.dim1*2,768)    #the org params for self.dim1=128 => nn.Linear(256,768)
            conv_init(self.linear_head['RN50x16'])

        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x



        # Connection for bones / velocities
        self.connect_joint = np.array(
            [2, 2, 21, 3, 21, 5, 6, 7, 21, 9, 10, 11, 1, 13, 14, 15, 1, 17, 18, 19, 2, 23, 8, 25, 12]) - 1
        self.velocity_vect = [1, 2, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45,
                              47]
        self.velocity_mx_prime = torch.Tensor(self.velocity_vect).repeat(batch_size*2, 3, seg, 1).permute(0, 1, 3, 2).cuda()    #"batch_size*2" =32*2  instead of "batch_size"=32 like in impSGN, because the data here takes into consideration M=2 unlike impSGN that mobines it in the data pre-processing part.


    def forward(self, x):
        """Forward pass returns same interface as Model_SGN:
        output, feature_dict, logit_scale, part_feature_list
        """      
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1) # n, t, vc ->n, t, v, c, -> (n, c, t, v, 1=m)
        N, C, T, V, M = x.size()
        x = rearrange(x, 'n c t v m -> (n m) c v t', m=M, v=V).contiguous()

        # Dynamic representation
        pos, velocity, bone, velocity_prime = self.multi_input_impSGN_2(x, self.connect_joint, self.velocity_mx_prime)
        pos = self.joint_embed(pos)
        velocity = self.velocity_embed(velocity)
        bone = self.bone_embed(bone)
        velocity_prime = self.velocity_prime_embed(velocity_prime)

        tem1 = self.tem_embed(self.tem)
        spa1 = self.spa_embed(self.spa)

        dy = pos + velocity + bone + velocity_prime


        # Joint-level module
        x = torch.cat([dy, spa1[:dy.size(0), :, :, :]], 1)   # spa1 mdf to spa1[:dy.size(0), :, :, :] so if "dy" won't cause an issue when it takes less samples that spa1

        for stab in self.stab_modules:
            x = stab(x)

        # Frame-level module
        x = x + tem1[:x.size(0), :, :, :]     ## tem1 mdf to  so if "x" won't cause an issue when it takes less samples that tem1
        x = self.cnn(x)

        x = x.permute(0, 1, 3, 2)
        c_new = x.size(1)
        feature = x.view(N,M,c_new,T//4,V)
        # Define body parts indices
        head_list = torch.LongTensor([2,3,20])
        hand_list = torch.LongTensor([4,5,6,7,8,9,10,11,21,22,23,24])
        foot_list = torch.LongTensor([12,13,14,15,16,17,18,19])
        hip_list = torch.LongTensor([0,1,2,12,16])
        # print('******** feature: ', feature.size())                 #[32, 2, 512, 16, 25]
        # print('******** feature 2: ', feature[:,:,:,:,head_list].mean(4).mean(3).mean(1).size())    #[32, 512]
        # print('******** part_list: ', self.part_list[0])
        # assert False, 'aaaaaaaaaaaaa'

        head_feature = self.part_list[0](feature[:,:,:,:,head_list].mean(4).mean(3).mean(1))
        hand_feature = self.part_list[1](feature[:,:,:,:,hand_list].mean(4).mean(3).mean(1))
        foot_feature = self.part_list[2](feature[:,:,:,:,foot_list].mean(4).mean(3).mean(1))
        hip_feature = self.part_list[3](feature[:,:,:,:,hip_list].mean(4).mean(3).mean(1))
        # Flatten and classify
        x = x.reshape(N, M, c_new, -1)      #mdf
        x = x.mean(3).mean(1)

        feature_dict = dict()

        for name in self.head:
            feature_dict[name] = self.linear_head[name](x)
        
        x = self.drop_out(x)

        return self.fc(x), feature_dict, self.logit_scale, [head_feature, hand_feature, hip_feature, foot_feature]

    def one_hot(self, bs, spa, tem):
        y = torch.arange(spa).unsqueeze(-1)
        y_onehot = torch.FloatTensor(spa, spa).zero_()
        y_onehot.scatter_(1, y, 1)
        y_onehot = y_onehot.unsqueeze(0).unsqueeze(0).repeat(bs, tem, 1, 1)
        return y_onehot

    def multi_input_impSGN_2(self, data, conn, velocity_mx_prime):
        """Compute joint, velocity, bone, velocity_prime"""
        N, C, V, T = data.size()
        joint = torch.zeros((N, C*2, V, T), device=data.device)
        velocity = torch.zeros((N, C*2, V, T), device=data.device)
        bone = torch.zeros((N, C*2, V, T), device=data.device)
        velocity_prime = torch.zeros((N, C*2, V, T), device=data.device)

        joint[:, :C, :, :] = data
        for i in range(V):
            joint[:, C:, i, :] = data[:, :, i, :] - data[:, :, 1, :]

        for i in range(T-2):
            velocity[:, :C, :, i] = data[:, :, :, i+1] - data[:, :, :, i]
            velocity[:, C:, :, i] = data[:, :, :, i+2] - data[:, :, :, i]

        velocity_prime[:, :C, :, :] = torch.mul(velocity[:, :C, :, :], velocity_mx_prime[:velocity.size(0), :, :, :])
        velocity_prime[:, C:, :, :] = torch.mul(velocity[:, C:, :, :], velocity_mx_prime[:velocity.size(0), :, :, :])

        for i in range(len(conn)):
            bone[:, :C, i, :] = data[:, :, i, :] - data[:, :, conn[i], :]
        bone_length = torch.sqrt(torch.sum(bone[:, :C, :, :]**2, dim=1, keepdim=True)) + 1e-6
        for i in range(C):
            bone[:, C+i, :, :] = torch.acos(bone[:, i, :, :] / bone_length[:, 0, :, :])

        return joint, velocity, bone, velocity_prime

###########################################  implementing impSGNv1 ###########################################################

'''class impSGN(nn.Module):
    def __init__(self, num_classes, dataset, seg, args, bias=True):
        super(impSGN, self).__init__()

        self.dim1 = 256
        self.dataset = dataset
        self.seg = seg
        num_joint = 25
        bs = args.batch_size
        self.connect_joint = np.array(
            [2, 2, 21, 3, 21, 5, 6, 7, 21, 9, 10, 11, 1, 13, 14, 15, 1, 17, 18, 19, 2, 23, 8, 25, 12]) - 1
        self.velocity_vect = [1, 2, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45,
                              47]
        self.velocity_mx_prime = torch.Tensor(self.velocity_vect).repeat(bs, 3, seg, 1).permute(0, 1, 3, 2).cuda()

        if args.train:
            self.spa = self.one_hot(bs, num_joint, self.seg)
            self.spa = self.spa.permute(0, 3, 2, 1).cuda()
            self.tem = self.one_hot(bs, self.seg, num_joint)
            self.tem = self.tem.permute(0, 3, 1, 2).cuda()
        else:
            self.spa = self.one_hot(32 * 5, num_joint, self.seg)
            self.spa = self.spa.permute(0, 3, 2, 1).cuda()
            self.tem = self.one_hot(32 * 5, self.seg, num_joint)
            self.tem = self.tem.permute(0, 3, 1, 2).cuda()
            self.velocity_mx_prime = torch.Tensor(self.velocity_vect).repeat(32 * 5, 3, seg, 1).permute(0, 1, 3,
                                                                                                        2).cuda()

        ### changed for state04
        self.tem_embed = embed(self.seg, self.dim1 * 2, norm=False, bias=bias)  
        self.spa_embed = embed(num_joint, 64, norm=False, bias=bias)
        self.joint_embed = embed(dim=6, dim1=64, norm=True, bias=bias)
        self.velocity_embed = embed(6, 64, norm=True, bias=bias)
        self.bone_embed = embed(6, 64, norm=True, bias=bias)
        self.velocity_prime_embed = embed(6, 64, norm=True, bias=bias)

        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.cnn = local(self.dim1 * 2, self.dim1 * 2, bias=bias)  
  
        self.fc = nn.Linear(self.dim1 * 2, num_classes)
        self.stab_modules = nn.ModuleList((
            Spatial_Temporal_Att_Block(128, 256, bias=bias),
            Spatial_Temporal_Att_Block(256, 256, bias=bias),
            Spatial_Temporal_Att_Block(256, 256 * 2, bias=bias),
        ))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))



    def forward(self, input):

        # Dynamic Representation
        bs, step, dim = input.size()
        num_joints = dim // 3
        input = input.view((bs, step, num_joints, 3))
        input = input.permute(0, 3, 2, 1).contiguous()

        pos, velocity, bone, velocity_prime = self.multi_input_impSGN_2(input, self.connect_joint, self.velocity_mx_prime)
        pos = self.joint_embed(pos)  # (bs, 64, v, t)
        velocity = self.velocity_embed(velocity)  # (bs, 64, v, t)
        bone = self.bone_embed(bone)  # (bs, 64, v, t)
        velocity_prime = self.velocity_prime_embed(velocity_prime)  # (bs, 64, v, t)

        tem1 = self.tem_embed(self.tem)
        spa1 = self.spa_embed(self.spa)

        dy = pos + velocity + bone + velocity_prime

        # Joint-level Module
        input = torch.cat([dy, spa1], 1)
        ### add STB_modules:
        for stab in self.stab_modules:
            input = stab(input)

        # Frame-level Module
        input = input + tem1
        input = self.cnn(input)
        # Classification
        output = self.maxpool(input)
        output = torch.flatten(output, 1)
        output = self.fc(output)

        return output

    def one_hot(self, bs, spa, tem):

        y = torch.arange(spa).unsqueeze(-1)
        y_onehot = torch.FloatTensor(spa, spa)

        y_onehot.zero_()
        y_onehot.scatter_(1, y, 1)

        y_onehot = y_onehot.unsqueeze(0).unsqueeze(0)
        y_onehot = y_onehot.repeat(bs, tem, 1, 1)

        return y_onehot

    ##### added
    def multi_input_impSGN_2(self, data, conn, velocity_mx_prime):
        ### this function "multi_input_impSGN_2" contains "velocity_mx_prime" compared with "multi_input_SGN"
        N, C, V, T = data.size()
        if torch.cuda.is_available():  # 'GPU'
            joint = torch.zeros((N, C * 2, V, T)).cuda()
            velocity = torch.zeros((N, C * 2, V, T)).cuda()
            bone = torch.zeros((N, C * 2, V, T)).cuda()
            ### added : Bakir idea
            velocity_prime = torch.zeros((N, C * 2, V, T)).cuda()
        else:  # 'CPU'
            joint = torch.zeros((N, C * 2, V, T))
            velocity = torch.zeros((N, C * 2, V, T))
            bone = torch.zeros((N, C * 2, V, T))
            velocity_prime = torch.zeros((N, C * 2, V, T))

        joint[:, :C, :, :] = data
        for i in range(V):
            joint[:, C:, i, :] = data[:, :, i, :] - data[:, :, 1, :]
        for i in range(T - 2):
            velocity[:, :C, :, i] = data[:, :, :, i + 1] - data[:, :, :, i]
            velocity[:, C:, :, i] = data[:, :, :, i + 2] - data[:, :, :, i]
        velocity_prime[:, :C, :, :] = torch.mul(velocity[:, :C, :, :], velocity_mx_prime)
        velocity_prime[:, C:, :, :] = torch.mul(velocity[:, C:, :, :], velocity_mx_prime)

        for i in range(len(conn)):
            bone[:, :C, i, :] = data[:, :, i, :] - data[:, :, conn[i], :]
        bone_length = 0
        for i in range(C):
            bone_length += bone[:, i, :, :] ** 2
        bone_length = torch.sqrt(bone_length) + 0.0001
        for i in range(C):
            bone[:, C + i, :, :] = torch.acos(
                bone[:, i, :, :] / bone_length)  # acos(): torch=1.3.0 | arcoos(): torch>1.3.0
        return joint, velocity, bone, velocity_prime

'''
##############################################

class norm_act(nn.Module):
    def __init__(self, dim=256):
        super(norm_act, self).__init__()

        self.bn = nn.BatchNorm1d(
            dim * 25)  # dim * 25: he did it because he will use x.view(bs, -1, step) {-1= (c=dim) * (25-joint) }
        self.relu = nn.ReLU()

    def forward(self, x):
        bs, c, num_joints, step = x.size()
        x = x.contiguous()  # since we use graph.A, there are 2 types, we use contiguous() to harmonize types.
        x = x.view(bs, -1, step)
        x = self.bn(x)
        x = x.view(bs, c, num_joints, step).contiguous()
        x = self.relu(x)
        return x


class SpatialTemporalBlock(nn.Module):
    def __init__(self, in_channels=256 // 2, out_channels=256, bias=True):
        super(SpatialTemporalBlock, self).__init__()
        self.dim1 = 256
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.compute_g1 = compute_g_spa(self.in_channels, self.dim1, bias=bias)
        self.gcn1 = gcn_spa(self.in_channels, self.dim1 // 2, bias=bias)
        self.gcn2 = gcn_spa(self.dim1 // 2, self.dim1, bias=bias)
        self.gcn3 = gcn_spa(self.dim1, self.out_channels, bias=bias)
        ### added:
        self.bn_relu = norm_act(self.out_channels)
        self.output_feature = 42 * 6
        self.mstcn = MultiScale_TemporalConv(self.out_channels,
                                             self.output_feature)  # 42*6=252   #the output_features must be divided by 6 {6: 4 Branches + 2}
        self.upsampling = nn.Conv2d(in_channels=self.output_feature, out_channels=self.out_channels, kernel_size=1,
                                    bias=bias)
        self.residual = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1,
                                  bias=bias)

        nn.init.constant_(self.gcn1.w.cnn.weight, 0)
        nn.init.constant_(self.gcn2.w.cnn.weight, 0)
        nn.init.constant_(self.gcn3.w.cnn.weight, 0)

    def forward(self, input):
        ### input: bs,c:128=dim1/2, v, t
        res = self.residual(input)  # c: 128 -> 256

        g = self.compute_g1(input)  # bs, c:256=dim1, v, t
        input = self.gcn1(input, g)
        input = self.gcn2(input, g)
        input = self.gcn3(input, g)  # bs, c:256

        ### added:
        input = self.bn_relu(input)
        input = input + res
        ### add ms_tcn:
        input = self.mstcn(input)  # output: (bs, 42*6, v, t)
        input = self.upsampling(input)  # (bs, 42*6+4=256, v, t)
        input = input + res
        return input  # bs, out_channels=256, v, t


class Spatial_A_Temporal_att_Block(nn.Module):
    def __init__(self, in_channels=256 // 2, out_channels=256, bias=True):
        super(Spatial_A_Temporal_att_Block, self).__init__()
        self.dim1 = 256
        self.in_channels = in_channels
        self.out_channels = out_channels

        graph = Graph('ntu')  # , max_hop=1)
        kwargs = {
            'A': torch.tensor(graph.A).type(torch.FloatTensor),
            'edge': False,
            'bias': bias,
            'act': Swish(),
            'amp_ratio': 2,
        }
        self.SGL = Spatial_Graph_Layer(self.in_channels, self.out_channels, 1, **kwargs)

        ### added:
        self.bn_relu = norm_act(self.out_channels)
        self.output_feature = 42 * 6
        self.mstcn = MultiScale_TemporalConv(self.out_channels,
                                             self.output_feature)  # 42*6=252   #the output_features must be divided by 6 {6: 4 Branches + 2}

        self.attention = ST_Joint_Att(self.output_feature, reduct_ratio=2, **kwargs)

        self.upsampling = nn.Conv2d(in_channels=self.output_feature, out_channels=self.out_channels, kernel_size=1,
                                    bias=bias)
        self.residual = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1,
                                  bias=bias)

    def forward(self, input):
        ### input: bs,c:128=dim1/2, v, t
        res = self.residual(input)  # c: 128 -> 256

        input = input.permute(0, 1, 3, 2)  # (bs, c, t, v)
        input = self.SGL(input)  # (bs, 256, t, v)
        input = input.permute(0, 1, 3, 2)  # (bs, c, v, t)
        input = self.bn_relu(input)
        input = input + res

        input = self.mstcn(input)  # output: (bs, 42*6, v, t)
        input = input.permute(0, 1, 3, 2)
        input = self.attention(input)
        input = input.permute(0, 1, 3, 2)
        input = self.upsampling(input)  # (bs, 42*6+4=256, v, t)

        input = input + res
        return input  # bs, out_channels=256, v, t


class Spatial_Temporal_Att_Block(nn.Module):
    def __init__(self, in_channels=256 // 2, out_channels=256, bias=True):
        super(Spatial_Temporal_Att_Block, self).__init__()
        self.dim1 = 256
        self.in_channels = in_channels
        self.out_channels = out_channels

        graph = Graph('ntu')  # , max_hop=1)
        kwargs = {
            'A': torch.tensor(graph.A).type(torch.FloatTensor),
            'edge': False,
            'bias': bias,
            'act': Swish(),
            'amp_ratio': 2,
        }

        self.compute_g1 = compute_g_spa(self.in_channels, self.dim1, bias=bias)
        self.gcn1 = gcn_spa(self.in_channels, self.dim1 // 2, bias=bias)
        self.gcn2 = gcn_spa(self.dim1 // 2, self.dim1, bias=bias)
        self.gcn3 = gcn_spa(self.dim1, self.out_channels, bias=bias)
        ### added:
        self.bn_relu = norm_act(self.out_channels)
        self.output_feature = 42 * 6
        self.mstcn = MultiScale_TemporalConv(self.out_channels,
                                             self.output_feature)  # 42*6=252   #the output_features must be divided by 6 {6: 4 Branches + 2}
        self.upsampling = nn.Conv2d(in_channels=self.output_feature, out_channels=self.out_channels, kernel_size=1,
                                    bias=bias)
        # self.attention = ST_Joint_Att(self.out_channels, reduct_ratio=2, **kwargs)
        self.residual = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1,
                                  bias=bias)

        nn.init.constant_(self.gcn1.w.cnn.weight, 0)
        nn.init.constant_(self.gcn2.w.cnn.weight, 0)
        nn.init.constant_(self.gcn3.w.cnn.weight, 0)

    def forward(self, input):
        ### input: bs,c:128=dim1/2, v, t
        res = self.residual(input)  # c: 128 -> 256

        g = self.compute_g1(input)  # bs, c:256=dim1, v, t
        input = self.gcn1(input, g)
        input = self.gcn2(input, g)
        input = self.gcn3(input, g)  # bs, c:256

        ### added:
        input = self.bn_relu(input)
        input = input + res
        ### add ms_tcn:
        input = self.mstcn(input)  # output: (bs, 42*6, v, t)
        input = self.upsampling(input)  # (bs, 42*6+4=256, v, t)
        # res2 = input
        # input = self.attention(input)
        # input = input + res2
        input = input + res
        return input  # bs, out_channels=256, v, t




##############################################################################################################################































