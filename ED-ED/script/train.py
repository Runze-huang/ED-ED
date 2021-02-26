import json
import argparse
import pickle
import yaml
import torch
import sys
import os
import time
import numpy as np
from model import VAE
from torch import optim
from torch.utils.data import DataLoader
from torch.distributions import kl_divergence, Normal
from torch.nn import functional as F
from torch.optim.lr_scheduler import ExponentialLR
#from tensorboardX import SummaryWriter
sys.path.append("./") 
from src.dataset import Dataset

def txt_save(content,filename,mode='a'):
    # Try to save a list variable in txt file.
    file = open(filename,mode)
    for i in range(len(content)):
        file.write(str(i+1)+"  "+str(content[i])+'\n')
    file.close()

def load_data(x_data,source_data,length_data, batch_size):
    data_loader = None
    if x_data != '':     
        X = pickle.load(open(x_data, 'rb')) 
        source = pickle.load(open(source_data, 'rb'))
        length = pickle.load(open(length_data, 'rb'))
        data = Dataset(X,source,length)  
        data_loader = DataLoader(data, batch_size=batch_size, shuffle = True)
    return data_loader

class MinExponentialLR(ExponentialLR):  
    def __init__(self, optimizer, gamma, minimum, last_epoch=-1): 
        self.min = minimum
        super(MinExponentialLR, self).__init__(optimizer, gamma, last_epoch=-1)
    def get_lr(self):
        return [
            max(base_lr * self.gamma**self.last_epoch, self.min)
            for base_lr in self.base_lrs
            ]

def std_normal(shape):
    N = Normal(torch.zeros(shape), torch.ones(shape))
    if torch.cuda.is_available():
        N.loc = N.loc.cuda()
        N.scale = N.scale.cuda()
    return N


def loss_function(recon,
                  target_tensor): 
    CE = F.nll_loss(
        recon.view(-1, recon.size(-1)), #（batch*32，130）
        target_tensor,  #（batch*32）
        reduction='mean')

    return CE
    
def train(start_epoch, step, end_epoch, train_data):
    train_loss = []
    for epoch in range(start_epoch, end_epoch):
        batch_loss = []
        #print(start_epoch, end_epoch)
        small = 100
        big = 0
        batch_total_loss = []
        for idx, data in enumerate(train_data):  
            infor = data[0]
            source = data[1]
            length = data[2].int()
            '''
            indexs = 76
            print(infor[indexs])
            print(torch.sum(infor[indexs,:34]))
            print(source[indexs,25:35])
            print(length[indexs])
            return
            '''
            ''' 
            batch = batch.reshape(-1,36)[:,:-2]
            note = batch.nonzero()[:,1]
            mins = torch.min(note).item()
            maxs = torch.max(note).item()
            if mins < small: small = mins
            if maxs > big : big = maxs
            #print(mins,maxs)
            #print(note[:100])
            '''
            target_tensor = source.view(-1, source.size(-1)).max(-1)[1]  
            assert(target_tensor.shape[0] == source.shape[0] * longest)
            optimizer.zero_grad() #梯度清零
            recon = model(infor, source, length)
            loss = loss_function( recon, target_tensor )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1) 
            optimizer.step()
            step += 1
            print('Epoch: %d  iteration: %d  loss: %.5f ' %( epoch, step, loss.item() ))
            #writer.add_scalar('batch_loss', loss.item(), step)
            if model_params['decay'] > 0:
                scheduler.step()
            batch_loss.append(loss)

        train_loss.append(round(torch.mean(torch.tensor(batch_loss)).item(),5))
        if epoch >= end_epoch -10 :
            epoch_num  = 'epoch_{}'.format(epoch)
            model_path = os.path.join(path,epoch_num)
            if os.path.exists(model_path):
               shutil.rmtree(model_path)
            os.makedirs(model_path)
            torch.save(model.cpu().state_dict(), os.path.join(model_path,save_name))
            if torch.cuda.is_available():
                model.cuda()
            print('Model saved!')

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='myconf.yml')
parser.add_argument('--model_type', type=str, default='lstm')
parser.add_argument('--resume', type=bool, default=False)    
args = parser.parse_args()

# some initialization
model_params = None
data_params = None
with open(args.config, 'r') as config_file:   
    config = yaml.load(config_file, Loader=yaml.FullLoader)
    model_params = config['model']
    data_params = config['data']
    preprocessor_params = config['preprocessor']
batch_size = model_params['batch_size']
longest = preprocessor_params['longest']
data_loader = load_data(data_params['X_train_data'],   # pickles_49/X_train.pickle
                        data_params['source_train_data'],
                        data_params['length_train_data'],
                        batch_size)    
if not os.path.isdir('logs'):
    os.mkdir('logs')
save_name = '{}.pt'.format(model_params['name']) 
#writer = SummaryWriter('logs/{}'.format(model_params['name'])) 
model = VAE(model_params['roll_dim'], model_params['hidden_dim'], model_params['infor_dim'],  
           model_params['time_step'],12)

if model_params['if_parallel']: 
      model = torch.nn.DataParallel(model, device_ids=[0, 1])
optimizer = optim.Adam(model.parameters(), lr=model_params['lr'])
if model_params['decay'] > 0:
    scheduler = MinExponentialLR(optimizer, gamma=model_params['decay'], minimum=1e-5)
if torch.cuda.is_available():
    print('Using: ', torch.cuda.get_device_name(torch.cuda.current_device()))
    model.cuda()
else:
    print('CPU mode')

step = 0
start_epoch= 1
epochs= model_params['n_epochs']
model.train() 
date_time = time.strftime('%Y_%m_%d_%H_%M', time.localtime())
path = os.path.join("params",date_time)
if not os.path.isdir(path): 
    os.mkdir(path)
print(path)
train(start_epoch, step, start_epoch+epochs, data_loader)