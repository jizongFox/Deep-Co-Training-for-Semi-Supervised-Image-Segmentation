from generalframework.models import Segmentator
import yaml, os
import torch
import torch.nn as nn
with open('../config.yaml','r') as f:
    config = yaml.load(f.read())

print(config)

model = Segmentator(arch_dict=config['Arch'],optim_dict=config['Optim'],scheduler_dict=config['Scheduler'])
img = torch.randn(1,1,224,224)
model.predict(img)
target = torch.randint(0,2,(1,224,224))
model.update(img,target,nn.CrossEntropyLoss())

