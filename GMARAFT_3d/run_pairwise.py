__author__ = "Semih Tarik Uenal"

from network.model import GMARAFT_Denoiser
from network_3d.model import GMARAFT_Denoiser3D
from train.trainer import Trainer
from loader.loader_vibe_alldata import VibeDatasetPairwise
from loader.loader_vibe_3d import VibeDatasetPairwise3D
import json
import torch.utils.data as data
import os
import multiprocessing

cwd  = os.getcwd()
if cwd == '/code':
    cwd = '/z0043wnf/GMRAFT/'
device = "cuda"
json_file_path = os.path.join (cwd,"configs", "train_vibe_pairwise3d.json")

with open(json_file_path, 'r') as file:
    config = json.load(file)
config['cwd'] = cwd
print(config['cwd'] )
config['data_loader']['data_list'] = os.path.join(cwd,config['data_loader']['data_list'])
config['data_loader']['data_dir'] = os.path.join(cwd,config['data_loader']['data_dir'])
## load model
model = GMARAFT_Denoiser3D().to(device)
model.cuda()
model.train()

## read data
mode = 'debug' if config['debug'] else 'train'
train_dataset = VibeDatasetPairwise3D(config['data_loader'], mode=mode)
train_loader = data.DataLoader(train_dataset,
                                   batch_size=config['data_loader']['batch_size'],
                                   pin_memory=True,
                                   shuffle=True,
                                   num_workers=config['data_loader']['num_workers'],
                                   drop_last=True)
print('Loader has %d vibe image pairs' % len(train_dataset))
print('steps per epoch', len(train_loader))

## run training
trainer = Trainer(config, model=model, data_loader=train_loader)
# trainer = Trainer( model=model, data_loader=train_loader)
trainer.run()
