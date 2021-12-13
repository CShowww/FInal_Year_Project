import torch 
import torch.nn as nn
import torch.nn.functional as F 

import numpy as np

#一阶段
from backbone import *
from utils import *
from roi_align.roi_align import RoIAlign      # RoIAlign module
from roi_align.roi_align import CropAndResize # crop_and_resize module
import os
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2'

class Basenet_volleyball(nn.Module):
    """
    main module of base model for the volleyball
    """
    def __init__(self, cfg):
        super(Basenet_volleyball, self).__init__()
        self.cfg=cfg
        
        NFB=self.cfg.num_features_boxes#1024
        D=self.cfg.emb_features#1056
        K=self.cfg.crop_size[0]#5
        

        if cfg.backbone=='inv3':
            self.backbone=MyInception_v3(transform_input=False,pretrained=True)
        elif cfg.backbone=='vgg16':
            self.backbone=MyVGG16(pretrained=True)
        elif cfg.backbone=='vgg19':
            self.backbone=MyVGG19(pretrained=True)
        else:
            assert False
        
        self.roi_align=RoIAlign(*self.cfg.crop_size)
        
        
        self.fc_emb = nn.Linear(K*K*D,NFB)
        self.dropout_emb = nn.Dropout(p=self.cfg.train_dropout_prob)
        
        self.fc_actions=nn.Linear(NFB,self.cfg.num_actions)
        self.fc_activities=nn.Linear(NFB,self.cfg.num_activities)

        
        
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
        
        

    def savemodel(self,filepath):
        state = {
            'backbone_state_dict': self.backbone.state_dict(),
            'fc_emb_state_dict':self.fc_emb.state_dict(),
            'fc_actions_state_dict':self.fc_actions.state_dict(),
            'fc_activities_state_dict':self.fc_activities.state_dict()
        }
        
        torch.save(state, filepath)
        print('model saved to:',filepath)
        

    def loadmodel(self,filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        self.fc_emb.load_state_dict(state['fc_emb_state_dict'])
        self.fc_actions.load_state_dict(state['fc_actions_state_dict'])
        self.fc_activities.load_state_dict(state['fc_activities_state_dict'])
        print('Load model states from: ',filepath)
        
        #print("我正在执行base_model下的loadmodel函数")


    #两块GPU来处理8个数据
    def forward(self,batch_data):
        #print("我正在执行base_model下的Basenet_Volleyball的前向函数");
        images_in, boxes_in = batch_data
        #print(len(boxes_in))
      
        
        '''
        images_in:[4,1,3,720,1280]
        boxed_in:[4,1,12,4]
        '''
        #print(boxes_in)
        #input()
        #print(batch_data)
        #print("----------------------------")
        #print("看看batch_data的形状")
        #print(len(batch_data))#len(batch_data)=2
        #print("-----------------")
        
        #print("看看images_in的形状")
        #print(images_in.shape)#[4,1,3,720,1280]--->一块GPU上，batch=4,t=1(只有一帧),c=3,h=720,w=1280
        #print("------------")
        
        #print("看看images_in的形状")
        #print(boxes_in[0][0][0])#[4,1,12,4]--->一块GPU上，batch=4,每一个batch里面有一张图片，一张图片里面有12个bb，最后一个4是四个点，x1,x2,y1,y2，即坐标
        
        # read config parameters
        B=images_in.shape[0]#4
        T=images_in.shape[1]#1
        
        #print(B)
        #print(T)
        H, W=self.cfg.image_size
        OH, OW=self.cfg.out_size#87,157
        N=self.cfg.num_boxes#12
        NFB=self.cfg.num_features_boxes#1024
        
        # Reshape the input data
        images_in_flat=torch.reshape(images_in,(B*T,3,H,W))  #B*T, 3, H, W-->[4,3,720,1280]
        
        boxes_in_flat=torch.reshape(boxes_in,(B*T*N,4))  #B*T*N, 4--->[48,4]
        #print(boxes_in_flat)
        boxes_idx=[i * torch.ones(N, dtype=torch.int)   for i in range(B*T) ]#产生4组，每组大小为12的list
    
        boxes_idx=torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, N，垂直叠起来,[4,12]
        boxes_idx_flat=torch.reshape(boxes_idx,(B*T*N,))  #B*T*N,长度为48的一维数组[48]
        #print(boxes_idx_flat)
        
        
        # Use backbone to extract features of images_in
        # Pre-precess first
        #print(images_in_flat[2])
        images_in_flat=prep_images(images_in_flat)#[4,3,720,1280],将像素值归一化，0到1之间
        #print(images_in_flat[2])
        outputs=self.backbone(images_in_flat)#长度为2,1个大小为[4,288,87,157],一个大小为[4,768,87,157]
        
        #print("一维的长度是："+str(len(outputs[0])))
        #print("二维的长度是："+str(len(outputs[1][0])))
        #print("三维的长度是："+str(len(outputs[0][0][0])))
        #print("四维的长度是："+str(len(outputs[0][0][0][0])))        
        # Build multiscale features
        features_multiscale=[]#长度也是2,一个的大小为[4,288,87,157],一个大小为[4,768,87,157]，多尺度
        for features in outputs:
            #print(torch.Size([OH,OW]))
            if features.shape[2:4]!=torch.Size([OH,OW]):
                features=F.interpolate(features,size=(OH,OW),mode='bilinear',align_corners=True)
            features_multiscale.append(features)
        #print(len(features_multiscale[0]))
        #print(len(features_multiscale[1]))

        features_multiscale=torch.cat(features_multiscale,dim=1)  #B*T, D, OH, OW--->[4,1056,87,157]
        #print(len(features_multiscale))
        
        
        
        # ActNet
        boxes_in_flat.requires_grad=False
        boxes_idx_flat.requires_grad=False
        #features_multiscale.requires_grad=False
        
    
        # RoI Align
        #features_multiscale:[4,1056,87,157]
        #boxes_in_flat:[48]
        #boxes_idx_flat：[48,4]
        #roi-align的操作是：在对应的区域上进行max 或者average pooling操作
        boxes_features=self.roi_align(features_multiscale,
                                            boxes_in_flat,
                                            boxes_idx_flat)  #B*T*N, D, K, K,--->48,1056,5,5
   
        
        
        boxes_features=boxes_features.reshape(B*T*N,-1) # B*T*N, D*K*K--->[48,1056*5*5]
        
            
        # Embedding to hidden state
        boxes_features=self.fc_emb(boxes_features)  # B*T*N, NFB-->(K*K*D,NFB)--->[48,1024]
        boxes_features=F.relu(boxes_features)
        boxes_features=self.dropout_emb(boxes_features)
       
    
        boxes_states=boxes_features.reshape(B,T,N,NFB)#[4,1,12,1024]
        
        # Predict actions
        boxes_states_flat=boxes_states.reshape(-1,NFB)  #B*T*N, NFB,[48,1024]

        actions_scores=self.fc_actions(boxes_states_flat)  #B*T*N, actn_num[48,9]
        #print(actions_scores)
        #input()
        # Predict activities
        boxes_states_pooled,_=torch.max(boxes_states,dim=2)  #B, T, NFB[4,1,1024]
        boxes_states_pooled_flat=boxes_states_pooled.reshape(-1,NFB)  #B*T, NFB[4,1024]
        
        activities_scores=self.fc_activities(boxes_states_pooled_flat)  #B*T, acty_num[4,8]
        
        if T!=1:
            actions_scores=actions_scores.reshape(B,T,N,-1).mean(dim=1).reshape(B*N,-1)
            activities_scores=activities_scores.reshape(B,T,-1).mean(dim=1)
            
        return actions_scores, activities_scores
        
        
class Basenet_collective(nn.Module):
    """
    main module of base model for collective dataset
    """
    def __init__(self, cfg):
        super(Basenet_collective, self).__init__()
        self.cfg=cfg
        
        D=self.cfg.emb_features
        K=self.cfg.crop_size[0]
        NFB=self.cfg.num_features_boxes
        NFR, NFG=self.cfg.num_features_relation, self.cfg.num_features_gcn
        
        self.backbone=MyInception_v3(transform_input=False,pretrained=True)
#         self.backbone=MyVGG16(pretrained=True)
        
        if not self.cfg.train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad=False
        
        self.roi_align=RoIAlign(*self.cfg.crop_size)
        
        self.fc_emb_1=nn.Linear(K*K*D,NFB)
        self.dropout_emb_1 = nn.Dropout(p=self.cfg.train_dropout_prob)
#         self.nl_emb_1=nn.LayerNorm([NFB])
        
        
        self.fc_actions=nn.Linear(NFB,self.cfg.num_actions)
        self.fc_activities=nn.Linear(NFB,self.cfg.num_activities)
        
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def savemodel(self,filepath):
        state = {
            'backbone_state_dict': self.backbone.state_dict(),
            'fc_emb_state_dict':self.fc_emb_1.state_dict(),
            'fc_actions_state_dict':self.fc_actions.state_dict(),
            'fc_activities_state_dict':self.fc_activities.state_dict()
        }
        
        torch.save(state, filepath)
        print('model saved to:',filepath)
        

    def loadmodel(self,filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        self.fc_emb_1.load_state_dict(state['fc_emb_state_dict'])
        print('Load model states from: ',filepath)
        
                
    def forward(self,batch_data):
        images_in, boxes_in, bboxes_num_in = batch_data
    
        # read config parameters
        B=images_in.shape[0]
        T=images_in.shape[1]
        H, W=self.cfg.image_size
        OH, OW=self.cfg.out_size
        MAX_N=self.cfg.num_boxes
        NFB=self.cfg.num_features_boxes
        NFR, NFG=self.cfg.num_features_relation, self.cfg.num_features_gcn
        EPS=1e-5
        
        D=self.cfg.emb_features
        K=self.cfg.crop_size[0]
        
        # Reshape the input data
        images_in_flat=torch.reshape(images_in,(B*T,3,H,W))  #B*T, 3, H, W
        boxes_in=boxes_in.reshape(B*T,MAX_N,4)
                
        # Use backbone to extract features of images_in
        # Pre-precess first
        images_in_flat=prep_images(images_in_flat)
        outputs=self.backbone(images_in_flat)
            
        
        # Build multiscale features
        features_multiscale=[]
        for features in outputs:
            if features.shape[2:4]!=torch.Size([OH,OW]):
                features=F.interpolate(features,size=(OH,OW),mode='bilinear',align_corners=True)
            features_multiscale.append(features)
        
        features_multiscale=torch.cat(features_multiscale,dim=1)  #B*T, D, OH, OW
        

        boxes_in_flat=torch.reshape(boxes_in,(B*T*MAX_N,4))  #B*T*MAX_N, 4
            
        boxes_idx=[i * torch.ones(MAX_N, dtype=torch.int)   for i in range(B*T) ]
        boxes_idx=torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, MAX_N
        boxes_idx_flat=torch.reshape(boxes_idx,(B*T*MAX_N,))  #B*T*MAX_N,

        # RoI Align
        boxes_in_flat.requires_grad=False
        boxes_idx_flat.requires_grad=False
        boxes_features_all=self.roi_align(features_multiscale,
                                            boxes_in_flat,
                                            boxes_idx_flat)  #B*T*MAX_N, D, K, K,
        
        boxes_features_all=boxes_features_all.reshape(B*T,MAX_N,-1)  #B*T,MAX_N, D*K*K
        
        # Embedding 
        boxes_features_all=self.fc_emb_1(boxes_features_all)  # B*T,MAX_N, NFB
        boxes_features_all=F.relu(boxes_features_all)
        boxes_features_all=self.dropout_emb_1(boxes_features_all)
        
    
        actions_scores=[]
        activities_scores=[]
        bboxes_num_in=bboxes_num_in.reshape(B*T,)  #B*T,
        for bt in range(B*T):
        
            N=bboxes_num_in[bt]
            boxes_features=boxes_features_all[bt,:N,:].reshape(1,N,NFB)  #1,N,NFB
    
            boxes_states=boxes_features  

            NFS=NFB

            # Predict actions
            boxes_states_flat=boxes_states.reshape(-1,NFS)  #1*N, NFS
            actn_score=self.fc_actions(boxes_states_flat)  #1*N, actn_num
            actions_scores.append(actn_score)

            # Predict activities
            boxes_states_pooled,_=torch.max(boxes_states,dim=1)  #1, NFS
            boxes_states_pooled_flat=boxes_states_pooled.reshape(-1,NFS)  #1, NFS
            acty_score=self.fc_activities(boxes_states_pooled_flat)  #1, acty_num
            activities_scores.append(acty_score)

        actions_scores=torch.cat(actions_scores,dim=0)  #ALL_N,actn_num
        activities_scores=torch.cat(activities_scores,dim=0)   #B*T,acty_num
        
#         print(actions_scores.shape)
#         print(activities_scores.shape)
       
        return actions_scores, activities_scores
        