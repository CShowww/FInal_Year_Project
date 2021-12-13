import torch 
import torch.nn as nn
import torch.nn.functional as F 

import numpy as np
#GCN
import copy
from backbone import *
from utils import *
from roi_align.roi_align import RoIAlign      # RoIAlign module
from roi_align.roi_align import CropAndResize # crop_and_resize module
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

class GCN_Module(nn.Module):
    def __init__(self, cfg):
        super(GCN_Module, self).__init__()
        
        self.cfg=cfg
        
        NFR =cfg.num_features_relation#256
        
        NG=cfg.num_graph#16
        N=cfg.num_boxes#12
        T=cfg.num_frames#3
        
        NFG=cfg.num_features_gcn#=self.num_features_boxes 1024
        NFG_ONE=NFG #1024
        
        #可以自动识别nn.ModuleList中的参数而普通的list则不可以
        #16*2个全连接层，每一个是[1024,256]
        #fc_rn_theta_list和fc_rn_phi_list是外观关系（Appearance relation）
        #目的都是将每个box的图特征通道数NFG转换成设定的关系特征通道数NFR
        self.fc_rn_theta_list=torch.nn.ModuleList([ nn.Linear(NFG,NFR) for i in range(NG) ])
        self.fc_rn_phi_list=torch.nn.ModuleList([ nn.Linear(NFG,NFR) for i in range(NG) ])
        
        #16个全连接层，但是不会学习偏差
        #1024*1024
        self.fc_gcn_list=torch.nn.ModuleList([ nn.Linear(NFG,NFG_ONE,bias=False) for i in range(NG) ])
        
        self.fc_gcn_knn_list = torch.nn.ModuleList([ nn.Linear(NFG,NFG_ONE,bias=False) for i in range(N) ])
        
        if cfg.dataset_name=='volleyball':
            #[36,1024]
            #这个LayerNorm的归一化，并不是将数据限定在0-1之间，也没有进行一个类似于高斯分布一样的分数,只是将其进行了一个处理，对应的数值得到了一些变化，相同数值的变化也是相同的。
            #nl_gcn_list,采用了按层归一化**（LayerNomalization）**，即对每一层的输入进行归一化操作。
            self.nl_gcn_list=torch.nn.ModuleList([ nn.LayerNorm([T*N,NFG_ONE]) for i in range(NG) ])
        else:
            self.nl_gcn_list=torch.nn.ModuleList([ nn.LayerNorm([NFG_ONE]) for i in range(NG) ])
        
            

        
    def forward(self,graph_boxes_features,boxes_in_flat):
        """
        graph_boxes_features  [B,T*N,NFG]
        #[1,4,1024]
        #[4,4]
        """
        # 分给两块GPU做，一块GPU处理B=1;
        # GCN graph modeling
        # Prepare boxes similarity relation
        
        B,N,NFG=graph_boxes_features.shape
    
 
        #input()
        #print(NFG)
        #input()
        #print("GCN_Module的前馈函数了！")
        
        NFR=self.cfg.num_features_relation#256
        NG=self.cfg.num_graph#16
        NFG_ONE=NFG#1024
        
        OH, OW=self.cfg.out_size#87，157
        pos_threshold=self.cfg.pos_threshold#0.2
        
        # Prepare position mask
        # 找到每个bboxes的中心坐标
        graph_boxes_positions=boxes_in_flat  #B*T*N, 4 [4,4]
 
        graph_boxes_positions[:,0]=(graph_boxes_positions[:,0] + graph_boxes_positions[:,2]) / 2 
        graph_boxes_positions[:,1]=(graph_boxes_positions[:,1] + graph_boxes_positions[:,3]) / 2 
        graph_boxes_positions=graph_boxes_positions[:,:2].reshape(B,N,2)  #B, T*N, 2 
        
    
        graph_boxes_distances=calc_pairwise_distance_3d(graph_boxes_positions,graph_boxes_positions)  #
       
    
        
        position_mask=( graph_boxes_distances > (pos_threshold*OW) )#小于阈值的为0，大于阈值的为1
        
      
        
        
        relation_graph=None
        graph_boxes_features_list=[]
        
        for i in range(NG):
        
            graph_boxes_features_theta=self.fc_rn_theta_list[i](graph_boxes_features)  #B,N,NFR[1,4,256]

            graph_boxes_features_phi=self.fc_rn_phi_list[i](graph_boxes_features)  #B,N,NFR[1,4,256]


#             graph_boxes_features_theta=self.nl_rn_theta_list[i](graph_boxes_features_theta)
#             graph_boxes_features_phi=self.nl_rn_phi_list[i](graph_boxes_features_phi)

            similarity_relation_graph=torch.matmul(graph_boxes_features_theta,graph_boxes_features_phi.transpose(1,2))  #B,N,N

            similarity_relation_graph=similarity_relation_graph/np.sqrt(NFR)

            similarity_relation_graph=similarity_relation_graph.reshape(-1,1)  #B*N*N, 1
            
        
        
            # Build relation graph
            relation_graph=similarity_relation_graph

            relation_graph = relation_graph.reshape(B,N,N)#[1,36,36]
            
            
            relation_graph[position_mask]=-float('inf')
            
            
            relation_graph = torch.softmax(relation_graph,dim=2)       
        
          

            one_graph_boxes_features=self.fc_gcn_list[i]( torch.matmul(relation_graph,graph_boxes_features) )  #B, N, NFG_ONE ,参数共享
            
           
           
            
            #归一化one_graph_boxes_features
            one_graph_boxes_features=self.nl_gcn_list[i](one_graph_boxes_features)
            one_graph_boxes_features=F.relu(one_graph_boxes_features)
            
            
            
            graph_boxes_features_list.append(one_graph_boxes_features)
            
           
        graph_boxes_features=torch.sum(torch.stack(graph_boxes_features_list),dim=0) #B, N, NFG
        
        
        #print(graph_boxes_features.shape)#[1,4,1024]
        return graph_boxes_features,relation_graph



class GCNnet_volleyball(nn.Module):
    """
    main module of GCN for the volleyball dataset
    """
    def __init__(self, cfg):
        super(GCNnet_volleyball, self).__init__()
        self.cfg=cfg
        
        T, N=self.cfg.num_frames, self.cfg.num_boxes#T=3,N=12
  
        D=self.cfg.emb_features#1056
  
        K=self.cfg.crop_size[0]#5
  
        NFB=self.cfg.num_features_boxes#1024
   
        NFR, NFG=self.cfg.num_features_relation, self.cfg.num_features_gcn#256,1024

        NG=self.cfg.num_graph#16
        
        
        if cfg.backbone=='inv3':
            self.backbone=MyInception_v3(transform_input=False,pretrained=True)
        elif cfg.backbone=='vgg16':
            self.backbone=MyVGG16(pretrained=True)
        elif cfg.backbone=='vgg19':
            self.backbone=MyVGG19(pretrained=False)
        else:
            assert False
        
        if not cfg.train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad=False
        
        self.roi_align=RoIAlign(*self.cfg.crop_size)
        
        self.fc_emb_1=nn.Linear(K*K*D,NFB)#[5*5*1056,1024]
        self.nl_emb_1=nn.LayerNorm([NFB])#[1024]
        
        
        
        self.gcn_list = torch.nn.ModuleList([ GCN_Module(self.cfg)  for i in range(self.cfg.gcn_layers) ])  #一层GCN  
        
        
        self.dropout_global=nn.Dropout(p=self.cfg.train_dropout_prob)
    
        self.fc_actions=nn.Linear(NFG,self.cfg.num_actions) #[1024,9]
        self.fc_activities=nn.Linear(NFG,self.cfg.num_activities)#[1024,8]
        
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
                    
    def loadmodel(self,filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        self.fc_emb_1.load_state_dict(state['fc_emb_state_dict'])
        print('Load model states from: ',filepath)
        




        
        
              
    def forward(self,batch_data):
        images_in, boxes_in = batch_data
        
        
        
        #print("修改部分")
        torch.cuda.synchronize()
        start = time.time()
        '''
        images_in:[1,3,3,720,1280]
        boxed_in:[1,3,12,4]
        '''
        # read config parameters
        B=images_in.shape[0]#1
        T=images_in.shape[1]#3
        

        
        H, W=self.cfg.image_size#720,1280
        OH, OW=self.cfg.out_size
        N=self.cfg.num_boxes
        NFB=self.cfg.num_features_boxes#1024
        NFR, NFG=self.cfg.num_features_relation, self.cfg.num_features_gcn#256,1024
        NG=self.cfg.num_graph
        
        D=self.cfg.emb_features
        K=self.cfg.crop_size[0]
        
        
        if not self.training:
            B=B*3
            T=T//3
            images_in.reshape( (B,T)+images_in.shape[2:] )
            boxes_in.reshape(  (B,T)+boxes_in.shape[2:] )
        
        
        # Reshape the input data
        images_in_flat=torch.reshape(images_in,(B*T,3,H,W))  #B*T, 3, H, W-->[3,3,720,1280]
        boxes_in_flat=torch.reshape(boxes_in,(B*T*N,4))  #B*T*N, 4--->[36,4]

        boxes_idx=[i * torch.ones(N, dtype=torch.int)   for i in range(B*T) ]
        boxes_idx=torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, N
        boxes_idx_flat=torch.reshape(boxes_idx,(B*T*N,))  #B*T*N,一个一维数组[36]
        
        #print(boxes_idx_flat)
        #input()
        
        # Use backbone to extract features of images_in
        # Pre-precess first
        images_in_flat=prep_images(images_in_flat)
        outputs=self.backbone(images_in_flat)
            
        
        # Build  features
        assert outputs[0].shape[2:4]==torch.Size([OH,OW])
        features_multiscale=[]
        for features in outputs:
            if features.shape[2:4]!=torch.Size([OH,OW]):
                features=F.interpolate(features,size=(OH,OW),mode='bilinear',align_corners=True)
            features_multiscale.append(features)
        
        features_multiscale=torch.cat(features_multiscale,dim=1)  #B*T, D, OH, OW-->[1,1056,87,157]
        
        
        # RoI Align
        boxes_in_flat.requires_grad=False
        boxes_idx_flat.requires_grad=False
        boxes_features=self.roi_align(features_multiscale,
                                            boxes_in_flat,
                                            boxes_idx_flat)  #B*T*N, D, K, K,-->[1*3*12,1056,5,5]
        
        boxes_features=boxes_features.reshape(B,T,N,-1)  #B,T,N, D*K*K--->[1,3,12,1056*5*5]
        
        
        # Embedding 
        boxes_features=self.fc_emb_1(boxes_features)  # B,T,N, NFB-->[1,3,12,1024]
        boxes_features=self.nl_emb_1(boxes_features)
        boxes_features=F.relu(boxes_features)
        



        
        boxes_knn_original = boxes_features  # B,T,N,D
        boxes_knn_trans = torch.transpose(boxes_knn_original, 3, 2)  # B,T,D,N
        boxes_knn_modified = torch.unsqueeze(boxes_knn_original, dim=3)  # B,T,N,K,D
        
        boxes_knn = torch.matmul(boxes_knn_original, boxes_knn_trans)
        
        _, pre = boxes_knn.topk(3, dim=3, largest=True)
        
        list = []
        

        #0.16s
        #KNN
        boxes_knn = []
        for i in range(B):#1
            for j in range(T):#3
                for m in range(N):#12
                    top1, top2, top3 = pre[i][j][m][0], pre[i][j][m][1], pre[i][j][m][2]
                    data_top1, data_top2, data_top3 = boxes_knn_modified[i][j][top1], boxes_knn_modified[i][j][top2], \
                                                      boxes_knn_modified[i][j][top3]
                                                      
                    loc_m,loc_top1,loc_top2,loc_top3 = boxes_in_flat[m+j*N],boxes_in_flat[top1+j*N],boxes_in_flat[top2+j*N],boxes_in_flat[top3+j*N]
                    
                    list.append(torch.cat((data_top1,data_top2,data_top3), dim=0))
                    boxes_knn.append(torch.cat((loc_m,loc_top1,loc_top2,loc_top3), dim=0).reshape(4,4))
                    
                # print(list[0],list[1],list[2],list[3],list[4],list[5],list[6],list[7])
               

        
                
        for i in range(1,len(list)):
            list[0] = torch.cat((list[0],list[i]),dim=0)
        w = list[0]  # [3*12*3,1024]
        #print(w.shape)
        
        
        w = w.reshape(B, T, N, 3, 1024)  #1,3,12,3,1024      
        boxes_knn_modified = torch.cat((boxes_knn_modified, w), dim=3)  # 1,3,12,4,1024
        
        
        final_boxes_features = torch.zeros(B,T,N,NFB)
        final_boxes_features = final_boxes_features.cuda()
        
        for i in range(B):
            for j in range(T):
                for m in range(N):
                    final_boxes_features[i][j][m] = (boxes_knn_modified[i][j][m][0] + boxes_knn_modified[i][j][m][1] + 
                                                         boxes_knn_modified[i][j][m][2] + boxes_knn_modified[i][j][m][3]) / 4
                                                         

        graph_boxes_features=final_boxes_features.reshape(B,T*N,NFG)
  
        
        
#         visual_info=[]
        for i in range(len(self.gcn_list)):
            graph_boxes_features,relation_graph=self.gcn_list[i](graph_boxes_features,boxes_in_flat)
        

        #print("转换后")   
        #print(final_graph_boxes_features.shape)
        
        #input()

            
        graph_boxes_features=graph_boxes_features.reshape(B,T,N,NFG) 
        boxes_states = self.dropout_global(graph_boxes_features)
        boxes_states=self.dropout_global(boxes_states)
        
        
        NFS=NFG
        # Predict actions
        boxes_states_flat=boxes_states.reshape(-1,NFS)  #B*T*N, NFS
        actions_scores=self.fc_actions(boxes_states_flat)  #B*T*N, actn_num
        
        # Predict activities
        boxes_states_pooled,_=torch.max(boxes_states,dim=2)  
        boxes_states_pooled_flat=boxes_states_pooled.reshape(-1,NFS)  
        
        activities_scores=self.fc_activities(boxes_states_pooled_flat)  #B*T, acty_num
        
        # Temporal fusion
        actions_scores=actions_scores.reshape(B,T,N,-1)
        actions_scores=torch.mean(actions_scores,dim=1).reshape(B*N,-1)
        activities_scores=activities_scores.reshape(B,T,-1)
        activities_scores=torch.mean(activities_scores,dim=1).reshape(B,-1)
        
        if not self.training:
            B=B//3
            actions_scores=torch.mean(actions_scores.reshape(B,3,N,-1),dim=1).reshape(B*N,-1)
            activities_scores=torch.mean(activities_scores.reshape(B,3,-1),dim=1).reshape(B,-1)


        torch.cuda.synchronize()
        end = time.time()
        print("K帧图片处理时间："+str(end-start))
        return actions_scores, activities_scores
       
        

        
class GCNnet_collective(nn.Module):
    """
    main module of GCN for the collective dataset
    """
    def __init__(self, cfg):
        super(GCNnet_collective, self).__init__()
        self.cfg=cfg
        
        D=self.cfg.emb_features
        K=self.cfg.crop_size[0]
        NFB=self.cfg.num_features_boxes
        NFR, NFG=self.cfg.num_features_relation, self.cfg.num_features_gcn
        
        self.backbone=MyInception_v3(transform_input=False,pretrained=True)
        
        if not self.cfg.train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad=False
        
        self.roi_align=RoIAlign(*self.cfg.crop_size)
        
        self.fc_emb_1=nn.Linear(K*K*D,NFB)
        self.nl_emb_1=nn.LayerNorm([NFB])
        
        
        self.gcn_list = torch.nn.ModuleList([ GCN_Module(self.cfg)  for i in range(self.cfg.gcn_layers) ])    
        
        
        self.dropout_global=nn.Dropout(p=self.cfg.train_dropout_prob)
    
        self.fc_actions=nn.Linear(NFG,self.cfg.num_actions)
        self.fc_activities=nn.Linear(NFG,self.cfg.num_activities)
        
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

#         nn.init.zeros_(self.fc_gcn_3.weight)
        

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
        
        D=self.cfg.emb_features
        K=self.cfg.crop_size[0]
        
        if not self.training:
            B=B*3
            T=T//3
            images_in.reshape( (B,T)+images_in.shape[2:] )
            boxes_in.reshape(  (B,T)+boxes_in.shape[2:] )
            bboxes_num_in.reshape((B,T))
        
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
        boxes_features_all=self.nl_emb_1(boxes_features_all)
        boxes_features_all=F.relu(boxes_features_all)
        
        
        boxes_features_all=boxes_features_all.reshape(B,T,MAX_N,NFB)
        boxes_in=boxes_in.reshape(B,T,MAX_N,4)
        
    
        actions_scores=[]
        activities_scores=[]
        bboxes_num_in=bboxes_num_in.reshape(B,T)  #B,T,
        
        for b in range(B):
            
            N=bboxes_num_in[b][0]
            
            boxes_features=boxes_features_all[b,:,:N,:].reshape(1,T*N,NFB)  #1,T,N,NFB
        
            boxes_positions=boxes_in[b,:,:N,:].reshape(T*N,4)  #T*N, 4
        
            # GCN graph modeling
            for i in range(len(self.gcn_list)):
                graph_boxes_features,relation_graph=self.gcn_list[i](boxes_features,boxes_positions)
        
        
            # cat graph_boxes_features with boxes_features
            boxes_features=boxes_features.reshape(1,T*N,NFB)
            boxes_states=graph_boxes_features+boxes_features  #1, T*N, NFG
            boxes_states=self.dropout_global(boxes_states)
            

            NFS=NFG
        
            boxes_states=boxes_states.reshape(T,N,NFS)
        
            # Predict actions
            actn_score=self.fc_actions(boxes_states)  #T,N, actn_num
            

            # Predict activities
            boxes_states_pooled,_=torch.max(boxes_states,dim=1)  #T, NFS
            acty_score=self.fc_activities(boxes_states_pooled)  #T, acty_num
            
            
            # GSN fusion
            actn_score=torch.mean(actn_score,dim=0).reshape(N,-1)  #N, actn_num
            acty_score=torch.mean(acty_score,dim=0).reshape(1,-1)  #1, acty_num
            
            
            actions_scores.append(actn_score)  
            activities_scores.append(acty_score)
            
            

        actions_scores=torch.cat(actions_scores,dim=0)  #ALL_N,actn_num
        activities_scores=torch.cat(activities_scores,dim=0)   #B,acty_num
        
        
        if not self.training:
            B=B//3
            actions_scores=torch.mean(actions_scores.reshape(-1,3,actions_scores.shape[1]),dim=1)
            activities_scores=torch.mean(activities_scores.reshape(B,3,-1),dim=1).reshape(B,-1)
       
        
#         print(actions_scores.shape)
#         print(activities_scores.shape)
       
        return actions_scores, activities_scores