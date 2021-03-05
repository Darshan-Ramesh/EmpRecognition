import torch.nn as nn
import config as cfg
import torch.nn.functional as F
from torchvision import datasets,models
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization,training
from head import ArcFace, CirriculumFace
import pickle
from backbone.irse import Backbone
from backbone.resnet import resnet50
from backbone.senet import senet50

class MyResnet(nn.Module):
    def __init__(self,model,classify=False):
        super(MyResnet,self).__init__()
        self.model = model
        self.dropout = nn.Dropout(p=0.8)
        
        if cfg.MODEL == 'Inception' or cfg.MODEL == 'resnet50':
            self.last_linear = nn.Linear(2048*1*1,512,bias=False)
        
        elif cfg.MODEL == 'resnet34':
            self.last_linear = nn.Linear(512*1*1,512,bias=False)
            
        self.last_bn = nn.BatchNorm1d(512)
        
        self.classify = classify
        
        if self.classify:
            self.logits = nn.Linear(512,cfg.CLASSES)
        
        
    def forward(self,x):
        #follows avgpooling2d-> dropout -> linear -> bn1d ->linear (withclasses,logits)
        
        #output from resnet's avgpool2d (2048*1*1)
        x = self.model(x)
        x = self.dropout(x)
        x = self.last_linear(x.view(x.size(0),-1))
        x = self.last_bn(x)
        
        if self.classify:
            x = self.logits(x)
        else:
            x = x
        
        return x

# def load_pretrained_weights(model_path,model):
    
#     with open(model_path,'rb') as f:
#         weights = pickle.load(f, encoding='latin1')
        
#     model_dict = model.state_dict()
    
#     for name,param in weights.items():
#         if not "fc" in name and name in model_dict:
#             try:
#                 model_dict[name].copy_(torch.from_numpy(param))
#             except Exception:
#                 raise RuntimeError("Run time error")
#         elif "fc" in name:
#             break
#         else:
#             raise KeyError(f'Unexpected key in {name} in the state_dictionary!')
        
#     print(f'[INFO] Updated the dicts using pre-trained weights')
#     model.load_state_dict(model_dict)
        

def get_model(model_name = 'Inception'):
    
    if model_name == 'Inception':
        #define model
        print(f'[INFO] Loading the {model_name}')
        model = InceptionResnetV1(pretrained='vggface2',classify=cfg.CLASSIFY,num_classes=cfg.CLASSES,
                                  dropout_prob=0.4)   
           
        if cfg.HEAD == 'ArcFace':
            print(f'[INFO] Using this mdoel as a feature extractor...')
            model =InceptionResnetV1(pretrained='vggface2', classify=cfg.CLASSIFY, device=cfg.DEVICE,
                                     dropout_prob=0.4)
            
            print(f'[INFO] Loading the {cfg.HEAD} as a head...')
            model_head = ArcFace(512,cfg.CLASSES)
            print(f'[INFO] Loaded model and the model head...')
        
        elif cfg.HEAD == 'CirriculumFace':
            print(f'[INFO] Loading the {model_name}')
            print(f'[INFO] Using this mdoel as a feature extractor...')
            model =InceptionResnetV1(pretrained='vggface2', classify=cfg.CLASSIFY, device=cfg.DEVICE,
                                     dropout_prob=0.8)
            
            print(f'[INFO] Loading the {cfg.HEAD} as a head...')
            model_head = CirriculumFace(512,cfg.CLASSES)
            print(f'[INFO] Loaded model and the model head...')
            
        else:
            model_head = None
            
            print(f'[INFO] Loaded the CNN model...')
            print(f'[INFO] Not using any heads...using the linear layer for classification..')
            
        #training only the class specific layer
        if cfg.RESUME:
            print(f'[INFO] Loading the prev checkpoint - {cfg.PREV_LOAD_CHECKPOINT}')
            
            model = torch.load(cfg.PREV_LOAD_CHECKPOINT)
            
            for p in model.parameters():
                p.requires_grad = False
                
            for p in model.logits.parameters():
                p.requires_grad = True
                print('[INFO] Not training the linear classifier of the model')
                
            #Used to extract normalized embeddings
            # for traning last conv
            for p in model.block8.conv2d.parameters():
                p.requires_grad = True
                print(f'[INFO] Last Conv Layer params - {p.requires_grad}')
            
            #[13.adaptivepool]->[14.dropout]->[15.last_linear]->[16.last_bn]->[17.logit]
            for p in model.last_bn.parameters():
                p.requires_grad = True
            
            for p in model.last_linear.parameters():
                p.requires_grad = True

        else:
        
            for p in model.parameters():
                p.requires_grad = False
            
            #[13.adaptivepool]->[14.dropout]->[15.last_linear]->[16.last_bn]->[17.logit]
            for p in model.last_bn.parameters():
                p.requires_grad = True
            
            for p in model.last_linear.parameters():
                p.requires_grad = True
                
            for p in model.logits.parameters():
                p.requires_grad = True
                
            
        print("-"*100)
        for param in model.parameters():
            print(param.requires_grad)
        print("-"*100)
        print('\n')
        
        print("-"*100)
        print(model)
        print("-"*100)

        #customise model
        # in_features = model.fc.in_features
        # model.fc = nn.Sequential(nn.Dropout(p=0.7),
        #                          nn.Linear(in_features=in_features,out_features=len(train_dataset.classes)))
        return model,model_head
        
    elif model_name == 'resnet50' or model_name == 'senet50':

        # model = resnet50(classify=True,num_classes=cfg.CLASSES,dropout_prob=0.4)
        if cfg.RESUME:
            print(f'[INFO] Loading from prev checkpoint - {cfg.PREV_LOAD_CHECKPOINT}')
            model = torch.load(cfg.PREV_LOAD_CHECKPOINT)
        
        print('-'*100)
        print('Model params...')        
        for p in model.parameters():
            p.requires_grad = False
        print('-'*100)
            
        # for idx,child in enumerate(model.children()):
        #     if idx > 9:
        #         for p in child.parameters():
        #             p.requires_grad = False
        
        for p in model.layer4[2].conv3.parameters():
            p.requires_grad = True
        
        for p in model.layer4[2].bn3.parameters():
            p.requires_grad = True
            
            
        if cfg.HEAD:
            pass
        else:
            model_head = None
        
        for p in model.parameters():
            print(p.requires_grad)
            
        print("-"*100)
        print(f'[INFO] Model - {model_name}')
        print("-"*100)
        print('\n')
        
        print("-"*100)
        print(model)
        print("-"*100)
    
        
        return model,model_head
                    
        
        # if cfg.HEAD == 'ArcFace':
           
        #     print(f'[INFO] Loading the {cfg.HEAD} as a head...')
        #     model_head = ArcFace(512,cfg.CLASSES)
        #     print(f'[INFO] Loaded model and the model head...')
        # else:
        #     model_head = None
        
        # resnet = resnet50()
        # model = nn.Sequential(*list(resnet.children())[:-1])
        
        # for p in model.parameters():
        #     p.requires_grad = False
        #     print(p.requires_grad)

        # #check how this pretrained model's image has been normalized
        # model = MyResnet(model,classify=cfg.CLASSIFY)
            
        # for i,child in enumerate(model.children()):
        #     if i == 0:
        #         for p in child[7][2].bn3.parameters():
        #          p.requires_grad = True
                 
        #         for p in child[7][2].conv3.parameters():  
        #             p.requires_grad = True
            
        # print('-'*100)
        # print('\t\t Model')
        # print(model)
        # print('-'*100)
        # print(f'[INFO] Last_linear - {model.last_linear.weight.requires_grad}')
        # print(f'[INFO] Last BN - {model.last_bn.weight.requires_grad}')
        
        # if cfg.CLASSIFY:
        #     print(f'[INFO] Logits - {model.logits.weight.requires_grad}')
        
        # return model,model_head
    
    elif model_name == 'IR50':
        
        #ir_50
        input_size = [112,112]
        model = Backbone(input_size, 50, 'ir')
        
        for params in model.parameters():
            params.requires_grad = False
        
        #train the last layer
        for p in model.output_layer.parameters():
            p.requires_grad = True
        
        # #train last conv + BN layer
        # for p in model.body[23].res_layer[4].parameters():
        #     p.requires_grad = True   #BN layer
        
        # for p in model.body[23].res_layer[3].parameters():
        #     p.requires_grad = True  #Conv layer
            
        for p in model.parameters():
            print(p.requires_grad)
            
        print('-'*100)
        print('\t\t Model')
        print(model)
        print('-'*100)
        
        if cfg.HEAD == 'ArcFace':
           
            print(f'[INFO] Loading the {cfg.HEAD} as a head...')
            model_head = ArcFace(512,cfg.CLASSES)
            print(f'[INFO] Loaded model and the model head...')
        
        elif cfg.HEAD == 'CirriculumFace':
            print(f'[INFO] Loading the {cfg.HEAD} as a head...')
            model_head = CirriculumFace(512,cfg.CLASSES)
            print(f'[INFO] Loaded model and the model head...')

        return model,model_head

    elif model_name == 'resnet34':
        print('*'*100)
        print(f'[INFO] Using {model_name} as the model..')
        print('*'*100)
        
        model = models.resnet34(pretrained=True)
        model = nn.Sequential(*list(model.children())[:-1])
        
        for p in model.parameters():
            p.requires_grad = False
            
        # model.fc = nn.Linear(in_features=model.fc.in_features,out_features=cfg.CLASSES,bias=False)
        if cfg.HEAD == 'ArcFace':
            model = MyResnet(model,cfg.CLASSIFY)
            model_head = ArcFace(512,cfg.CLASSES)
        else:
            model = MyResnet(model,cfg.CLASSIFY)
            model_head = None
        
        #training last conv layer also
        for mod,modules in enumerate(model.children()):
            if mod == 0:
                for m,module in enumerate(modules.children()):
                    if m == 7:
                        for p in module[2].conv2.parameters():
                            p.requires_grad = True
                            print(p.requires_grad)
                        
                        for p in module[2].bn2.parameters():
                            p.requires_grad = True
                            print(p.requires_grad)
                            
        print(model)
        
        for p in model.parameters():
            print(p.requires_grad)
        
        return model, model_head
    
    elif model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
        
        for p in model.parameters():
            p.requires_grad = False
            
        model.fc = nn.Linear(in_features=model.fc.in_features,out_features=cfg.CLASSES,bias=False)
        
        model_head = None
        
        print('-'*100)
        print(model)
        print('-'*100)
        print(f'[INFO] FC layer - {model.fc.requires_grad}')
        
        
        
        return model,model_head
        
    
        

# --------------------------------------------------------------
#testing
# model = models.resnet50(pretrained=True)
# model = nn.Sequential(*list(model.children())[:-1])
# my_model = MyResnet(model)
# my_model = my_model.cuda()

# # model_name = 'IR50'
# model,_ = get_model(cfg.MODEL)
# # print(model.output_layer.requires_grad)

# #random image
# image = torch.randn(4,3,224,224).cuda()
# print(image.shape)
# model = model.cuda()
# logit = model(image)
# print(logit.shape)
# --------------------------------------------------------------
