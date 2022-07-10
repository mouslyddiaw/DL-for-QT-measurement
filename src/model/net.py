"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

def choose_model(model_dir, params):
    if "cnn" in model_dir:
        model, loss_fn = AttnCNNv2(), Loss_MSE_CE(2)
    if "resnet" in model_dir:
        model, loss_fn =  KanResWide(), Loss_MSE_CE(2)
    if "unet" in model_dir:
        model, loss_fn =  UNET_1D(), Loss_MSE_CE_Dice(3) 
    if params.cuda:
        model = model.cuda()  
    model.apply(reset_weights) 
    return model, loss_fn  

def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children(): 
        if 'BatchNorm1d' in str(layer):
            continue
        if hasattr(layer, 'reset_parameters'):
            #print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()
            torch.nn.init.xavier_uniform_(layer.weight , gain= torch.sqrt(torch.tensor(2.0)))

##### Basic CNN + attention
class AttnCNNv2(torch.nn.Module):  
    def __init__(self):
        super(AttnCNNv2, self).__init__()
        self.conv_block1 = torch.nn.Sequential(   
            
            torch.nn.Conv1d(in_channels=1, out_channels=8, kernel_size=16  ), 
            torch.nn.BatchNorm1d(8 ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Dropout(p=0.2),
            
            torch.nn.Conv1d(in_channels=8, out_channels=8, kernel_size=8  ), 
            torch.nn.BatchNorm1d(8 ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Dropout(p=0.2),
            torch.nn.MaxPool1d(2, stride=2),
        )
            
        self.conv_block2 = torch.nn.Sequential(    
            torch.nn.Conv1d(in_channels=8, out_channels=16, kernel_size=16  ), 
            torch.nn.BatchNorm1d(16 ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Dropout(p=0.2),
            
            torch.nn.Conv1d(in_channels=16, out_channels=16, kernel_size=8  ), 
            torch.nn.BatchNorm1d(16 ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Dropout(p=0.2),
            torch.nn.MaxPool1d(2, stride=2),
        ) 
        
        self.conv_block3 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=16, out_channels=32, kernel_size=16  ), 
            torch.nn.BatchNorm1d(32 ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Dropout(p=0.2),
            
            torch.nn.Conv1d(in_channels=32, out_channels=32, kernel_size=8  ), 
            torch.nn.BatchNorm1d(32 ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Dropout(p=0.2),
            torch.nn.MaxPool1d(2, stride=1),
        )   
            
        self.conv_block4 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=32, out_channels=64, kernel_size=16   ), 
            torch.nn.BatchNorm1d(64 ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Dropout(p=0.2),
            
            torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=8  ), 
            torch.nn.BatchNorm1d(64 ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Dropout(p=0.2),
            torch.nn.MaxPool1d(2, stride=1), 
        ) 
        
        self.conv_block5 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=64, out_channels=128, kernel_size=16 ), 
            torch.nn.BatchNorm1d(128 ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Dropout(p=0.2),
            
            torch.nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8 ), 
            torch.nn.BatchNorm1d(128 ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Dropout(p=0.2),
            torch.nn.MaxPool1d(2, stride=2),
        )   
            
        self.conv_block6 = torch.nn.Sequential(
         torch.nn.Conv1d(in_channels=128, out_channels=256, kernel_size=16 ), 
            torch.nn.BatchNorm1d(256 ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Dropout(p=0.2),
            
            torch.nn.Conv1d(in_channels=256, out_channels=256, kernel_size=8 ), 
            torch.nn.BatchNorm1d(256 ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Dropout(p=0.2),
            torch.nn.MaxPool1d(2, stride=2), 
        )
            
        self.avg_pool  = torch.nn.AdaptiveAvgPool1d(1)#torch.nn.Conv1d(in_channels=256, out_channels=256, kernel_size=5 )
        
        self.attn1 = AttentionBlock(16, 256, 16, 133/5)
        self.attn2 = AttentionBlock(32, 256, 32, 110/5)
        self.attn3 = AttentionBlock(128, 256, 128, 32/5)
          
        self.fc1 =  torch.nn.Linear(in_features=432, out_features=250) 
        #self.fc2 =  torch.nn.Linear(in_features=432, out_features=1) 
        
    def forward(self, x):
        x = self.conv_block1(x.unsqueeze(1)) 
        
        block2 = self.conv_block2(x) 
        
        block3 = self.conv_block3(block2) 
        
        block4 = self.conv_block4(block3) 
        
        block5 = self.conv_block5(block4) 
        
        block6 = self.conv_block6(block5) 
        
       
        g = self.avg_pool(block6) 
        
        a1, g1 = self.attn1(block2, block6)
        a2, g2 = self.attn2(block3, block6)
        
        a3, g3 = self.attn3(block5, block6) 
         
        g_hat = torch.cat((g, g1, g2, g3), dim=1)  

        out = g_hat.view(g_hat.size()[0], g_hat.size()[1]*g_hat.size()[2])
       
        
        logits = self.fc1(out) 
        qt = None #self.fc2(out)   

        return (logits, qt, a1, a2, a3)

class AttentionBlock(torch.nn.Module):
    def __init__(self, in_features_l, in_features_g, attn_features, up_factor):
        super(AttentionBlock, self).__init__()
        self.up_factor = up_factor
        self.W_l = torch.nn.Conv1d(in_channels=in_features_l, out_channels=attn_features, kernel_size=1, padding=0, bias=False)
        self.W_g = torch.nn.Conv1d(in_channels=in_features_g, out_channels=attn_features, kernel_size=1, padding=0, bias=False)
        self.phi = torch.nn.Conv1d(in_channels=attn_features, out_channels=1, kernel_size=1, padding=0, bias=True)
        
        self.avg_pool = torch.nn.AdaptiveAvgPool1d(1)#torch.nn.AvgPool1d(attn_features)
        
    def forward(self, l, g): 
        N, C, W  = l.size()
        l_ = self.W_l(l)
        g_ = self.W_g(g) 
         
        if self.up_factor > 1:
            g_ = F.interpolate(g_, scale_factor=self.up_factor, mode='linear', align_corners=False)
        
        c = self.phi(F.relu(l_ + g_)) # batch_sizex1xWxH
        
        # compute attn map
         
        a = torch.sigmoid(c)
        
        # re-weight the local feature
        f = torch.mul(c.expand_as(l), l) # batch_sizexCxWxH
        
        #f = torch.transpose(f, 1, 2) 
        output = self.avg_pool(f) # global average pooling
        
        return a, output

##### ResNet
class KanResWide(torch.nn.Module): #input: template
    def __init__(self):
        super(KanResWide, self).__init__()
        self.kanres_init = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1, out_channels=64, kernel_size=7, padding=3 ), #padding=floor(kernel_size/2)
            torch.nn.BatchNorm1d(64) ,
            torch.nn.ReLU(inplace=False),
            torch.nn.Dropout(p=0.2),
            
            torch.nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1  ),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(inplace=False),
            torch.nn.Dropout(p=0.2),
            
            torch.nn.AvgPool1d(2),  
        )
        
        self.layer1 = kanres_module()
        
        self.layer2 = kanres_module()
        
        self.layer3 = kanres_module()
        
        self.layer4 = kanres_module()
        
        self.layer5 = kanres_module()
        
        self.layer6 = kanres_module()
        
        self.layer7 = kanres_module()
        
        self.layer8 = kanres_module()
        
        self.avg_pool = torch.nn.AdaptiveAvgPool1d(1)
          
        self.fc1  =  torch.nn.Linear(in_features=32, out_features=250)  
        #self.fc2 =  torch.nn.Linear(in_features=32, out_features=1) 
     
    def forward(self, x):
        x = self.kanres_init(x.unsqueeze(1))  
        
        x =  torch.add(x, self.layer1(x))
        x =  torch.add(x, self.layer2(x)) 
        x =  torch.add(x, self.layer3(x))
        x =  torch.add(x, self.layer4(x)) 
        x =  torch.add(x, self.layer5(x))
        x =  torch.add(x, self.layer6(x)) 
        x =  torch.add(x, self.layer7(x))
        x =  torch.add(x, self.layer8(x)) 
        
         
        x = self.avg_pool(x)
         
        
        x = x.view(x.size()[0], x.size()[1]*x.size()[2])
        
        logits = self.fc1(x)  
        qt = None #self.fc2(x)  

        return (logits, qt)

def kanres_module():
    module = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=32, out_channels=64, kernel_size=49, padding= 24  ),
            torch.nn.BatchNorm1d(64) ,
            torch.nn.ReLU(inplace=False),
            torch.nn.Dropout(p=0.2),
            
            torch.nn.Conv1d(in_channels=64, out_channels=32, kernel_size=49, padding= 24 ),
            torch.nn.BatchNorm1d(32) ,
            torch.nn.ReLU(inplace=False), 
            torch.nn.Dropout(p=0.2),
        ) 
    return module

##### U-Net
class UNET_1D(torch.nn.Module):
    def __init__(self):
        super(UNET_1D, self).__init__()
        
        num_features=8
        input_dim=1
            
        self.down_layer_1 = conv_step(input_dim=input_dim, num_features=num_features)
        self.down_layer_2 = conv_step(num_features, num_features*2)
        self.down_layer_3 = conv_step(num_features*2, num_features*4)
        self.down_layer_4 = conv_step(num_features*4, num_features*8 )
        self.down_layer_5 = conv_step(num_features*8, num_features*16 )
        self.down_layer_6 = conv_step(num_features*16, num_features*32 )
        
        self.up_conv_1 = up_conv(num_features*32)
        self.up_conv_2 = up_conv(num_features*16)
        self.up_conv_3 = up_conv(num_features*8)
        self.up_conv_4 = up_conv(num_features*4)
        self.up_conv_5 = up_conv(num_features*2)
        
        self.up_layer_1 = conv_step(num_features*32, num_features*16 )
        self.up_layer_2 = conv_step(num_features*16, num_features*8 )
        self.up_layer_3 = conv_step(num_features*8, num_features*4)
        self.up_layer_4 = conv_step(num_features*4, num_features*2 )
        self.up_layer_5 = conv_step(num_features*2, num_features )
        
        self.maxpool1 = torch.nn.MaxPool1d(2, stride=2)
        self.maxpool2 = torch.nn.MaxPool1d(2, stride=1)
        
        self.avg_pool  = torch.nn.AdaptiveAvgPool1d(1)
        self.fc1 =  torch.nn.Linear(in_features=256, out_features=250)
        #self.fc2 =  torch.nn.Linear(in_features=256, out_features=1)
        
        self.sigmoid = torch.nn.Sigmoid()
        
        self.final = torch.nn.Conv1d(num_features, input_dim, 1)
        
        #self.bn = torch.nn.BatchNorm1d(input_dim)
      
        
    def forward(self,x):
        #x = self.bn(x) 
        input_size = x.size()
        
        """ Contracting """
        
        out_1 = self.down_layer_1(x.unsqueeze(1))
        x = self.maxpool1(out_1)
        
        out_2 = self.down_layer_2(x)
        x = self.maxpool1(out_2)
        
        out_3 = self.down_layer_3(x)
        x = self.maxpool2(out_3)
        
        out_4 = self.down_layer_4(x)
        x = self.maxpool2(out_4)
        
        out_5 = self.down_layer_5(x)
        x = self.maxpool1(out_5)
        
        end = self.down_layer_6(x)
        x = self.maxpool1(end)
        
        g = self.avg_pool(x) 
        logits = self.fc1(g.view(g.size()[0], g.size()[1]*g.size()[2])) 
        qt = None #self.fc2(g.view(g.size()[0], g.size()[1]*g.size()[2])) 
        
        """ Expanding """ 
        x = self.up_conv_1(end)
        x = F.interpolate(x, size=out_5.size()[2], mode='linear')
        x = torch.cat([out_5,x],dim = 1)  
        x = self.up_layer_1(x)
        
        x = self.up_conv_2(x)  
        x = F.interpolate(x, size=out_4.size()[2], mode='linear')
        x = torch.cat([out_4,x],dim = 1)  
        x = self.up_layer_2(x)
        
        x = self.up_conv_3(x)
        x = F.interpolate(x, size=out_3.size()[2], mode='linear')
        x = torch.cat([out_3,x],dim = 1)
        x = self.up_layer_3(x)
        
        x = self.up_conv_4(x)
        x = F.interpolate(x, size=out_2.size()[2], mode='linear')
        x = torch.cat([out_2,x],dim = 1)
        x = self.up_layer_4(x)
        
        x = self.up_conv_5(x)
        x = F.interpolate(x, size=out_1.size()[2], mode='linear')
        x = torch.cat([out_1,x],dim = 1)
        x = self.up_layer_5(x)
         
        x = F.interpolate(x, size=input_size[1], mode='linear') 
         
        mask = self.sigmoid(self.final(x))
        
        return (logits, qt, mask, g)

class conv_step(torch.nn.Module):
    def __init__(self, input_dim, num_features ):
        super(conv_step, self).__init__()
         
        self.conv_1 = torch.nn.Conv1d(input_dim,num_features, kernel_size=16)
        #self.bn = torch.nn.BatchNorm1d(num_features)
        self.conv_2 = torch.nn.Conv1d(num_features,num_features, kernel_size=8)
        self.relu = torch.nn.ReLU()
        
    def forward(self,x):
        x = self.conv_1(x)
        # x = self.bn(x)
        x = self.relu(x)
        x = self.conv_2(x)
        # x = self.bn(x)
        x = self.relu(x)
        return x

class up_conv(torch.nn.Module):
    def __init__(self, input_dim):
        super(up_conv, self).__init__()
        #self.upsample = torch.nn.Upsample(scale_factor = 2)
        self.conv = torch.nn.Conv1d(input_dim, input_dim // 2, 3 , padding=1)
        #self.bn = torch.nn.BatchNorm1d(input_dim // 2)
        self.relu = torch.nn.ReLU()
        
    def forward(self, x): 
        #x = self.upsample(x) 
        x = self.conv(x) 
        x = self.relu(x)
        # x = self.bn(x)
        return x

### Losses
class Loss_MSE_CE(torch.nn.Module):
    def __init__(self, task_num):
        super(Loss_MSE_CE, self).__init__()
        self.task_num = task_num
        self.log_vars = torch.nn.Parameter(torch.zeros((task_num))) 

    def forward(self, logits, qt_predictions, target_classes, target_intervals):

        mse, crossEntropy = torch.nn.MSELoss( ), torch.nn.CrossEntropyLoss() 
        qt_predictions = torch.tensor([compute_qt(classify_qt(output)) for output in logits]).cuda()  
        
        loss0 = mse(qt_predictions, target_intervals)
        loss1 = crossEntropy(logits , target_classes.long()) 

        precision0 = torch.exp(-self.log_vars[0])
        loss0 = precision0*loss0 + self.log_vars[0]

        precision1 = torch.exp(-self.log_vars[1])
        loss1 = precision1*loss1 + self.log_vars[1]
        
        return loss0+loss1  

class Loss_MSE_CE_Dice(torch.nn.Module):
    def __init__(self, task_num):
        super(Loss_MSE_CE_Dice, self).__init__()
        self.task_num = task_num
        self.log_vars = torch.nn.Parameter(torch.zeros((task_num)))

    def forward(self, logits, qt_predictions, target_classes, target_intervals, mask_preds, mask_targets):

        mse, crossEntropy = torch.nn.MSELoss( ), torch.nn.CrossEntropyLoss()
         
        qt_predictions = torch.tensor([compute_qt(classify_qt(output)) for output in logits]).cuda()   
         
        loss0 = mse(qt_predictions, target_intervals)
        loss1 = crossEntropy(logits, target_classes.long())  
        loss2 = dice_coeff(mask_preds.squeeze(1), mask_targets)
        
        precision0 = torch.exp(-self.log_vars[0])
        loss0 = precision0*loss0 + self.log_vars[0]

        precision1 = torch.exp(-self.log_vars[1])
        loss1 = precision1*loss1 + self.log_vars[1]
        
        precision2 = torch.exp(-self.log_vars[2])
        loss2 = precision2*loss2 + self.log_vars[2]

        return loss0 + loss1 + loss2   

def dice_coeff(input, target): 
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])
    return 1 - (s / (i + 1))      

class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target

def compute_qt(qt_class):    
     intervals = range(150, 752, 2) 
     qt_value = 0 
     for index, (lower, upper) in enumerate(zip(intervals[:-1],intervals[1:] )): 
         if qt_class == index:
             qt_value = (lower+upper)/2
             break
     return torch.tensor(qt_value)

def classify_qt(outputs, dim=0):   
    m = torch.nn.Softmax(dim=dim)
    qt_class = torch.argmax(m(outputs)).item()    
    return qt_class 

####### Metrics

def error(outputs_qt, labels): 
    preds_qt = [compute_qt(classify_qt(probas)).item() for probas in outputs_qt]
    errors = [(ref - pred).item() for ref, pred in zip(labels, preds_qt)] 
    return np.array(errors)

def get_dices(outputs_mask, labels_mask): 
    dices = [DiceCoeff().forward(ref, mask).item() for ref, mask in zip(labels_mask.squeeze(1), outputs_mask)] 
    return np.array(dices)

def absolute_error(outputs_qt, labels): 
    return np.abs(error(outputs_qt, labels))
 

# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'error': error,
    'absolute error': absolute_error,
    'dice coeff': get_dices
}
