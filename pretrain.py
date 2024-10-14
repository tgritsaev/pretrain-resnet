import argparse
import csv
import time
import os
import wandb
import random
from tqdm import trange

from PIL import Image

import numpy as np

import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F

import sys


class PretrainDataset(Dataset):
    
    def __init__(self, path):
        self.path = path
        cut = None if not args.debug else 128
        self.len = len(os.listdir(path)) if not args.debug else cut
        angles = [0, 90, 180, 270]
        # Load the dataset on Disk
        imgs = [Image.open(f"{self.path}/{i + 1}.jpg") for i in range(self.len)]
        self.rotated_tensor_imgs = [
            torch.stack([F.rotate(transforms.ToTensor()(img), angle) for angle in angles])
            for img in imgs[:cut]
        ]
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        return self.rotated_tensor_imgs[idx]
    
class FineTuningDataset(Dataset):
    
    def __init__(self, path):
        self.path = path
        
        self.targets = []
        self.tensor_imgs = []
        for i, class_name in enumerate(CLASS_NAMES):
            for fname in os.listdir(f"{path}/{class_name}"):
                self.tensor_imgs.append(transforms.ToTensor()(Image.open(f"{self.path}/{class_name}/{fname}")))
                self.targets.append(i)
                
        self.len = len(self.targets)
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        return self.tensor_imgs[idx], self.targets[idx]
    

parser = argparse.ArgumentParser()
parser.add_argument("--pretrain-epochs", default=100, type=int)
parser.add_argument("--ft-epochs", default=1, type=int)

parser.add_argument("-b", "--batch-size", default=128, type=int)
parser.add_argument("-p", "--pt-learning-rate", default=0.1, type=float)
parser.add_argument("-g", "--gamma", default=0.95, type=float)
parser.add_argument("-f", "--ft-learning-rate", default=1e-4, type=float)

parser.add_argument("-n", "--num-workers", default=8, type=int)
parser.add_argument("-d", "--debug", action="store_true")
parser.add_argument("-c", "--cpu_only", action="store_true")
args = parser.parse_args()

CLASS_NAMES = sorted(os.listdir("data/train/labeled"))

run_name = f"bs={args.batch_size}-ptlr={args.pt_learning_rate}-g={args.gamma}-ftlr={args.ft_learning_rate}"
wandb.init(project="Pretrain ResNet-18, HW-2. LSDL 2024, CUB.", name=run_name)

device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu_only else "cpu")

model = models.resnet18(num_classes=10).to(device)
pretrain_head = nn.Linear(512, 4).to(device)

intermediate_outputs = {}
# Hook function to save intermediate output
def hook_fn(module, input, output):
    intermediate_outputs[module] = output
model.avgpool.register_forward_hook(hook_fn) 

print(model)
print(f"device: {device}")
ce_loss = nn.CrossEntropyLoss()

# 1. Pretrain Phase
start_time = time.time()
save_dir = f"checkpoints/{run_name}"
os.makedirs(save_dir, exist_ok=True)

pt_optimizer = torch.optim.SGD(
    [v for k, v in model.named_parameters() if k != "fc"] + list(pretrain_head.parameters()),
    lr=args.pt_learning_rate,
    momentum=0.9,
)
pt_scheduler = torch.optim.lr_scheduler.ExponentialLR(pt_optimizer, gamma=args.gamma)

pt_dataset = PretrainDataset("data/train/unlabeled")
pt_dataloader = DataLoader(pt_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

ft_dataset = FineTuningDataset("data/train/labeled")
ft_dataloader = DataLoader(ft_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

test_path = "data/test"
test_images = [transforms.ToTensor()(Image.open(f"{test_path}/{i}.jpg")) for i in range(len(os.listdir(test_path)))]

pt_total_steps = 0

for epoch in trange(args.pretrain_epochs):
    begin_epoch_time = time.time()
    model.train()
    for i, batch in enumerate(pt_dataloader):
        pt_optimizer.zero_grad()
        rotation_preds = model(torch.flatten(batch, 0, 1).to(device))
        target = torch.arange(batch.shape[0] * 4, device=device) % 4
        
        loss = ce_loss(rotation_preds, target)
        loss.backward()
        pt_optimizer.step()
        
        pt_total_steps += 1
        
        if i > 0 and i % 10 == 0:
            wandb.log({"train_loss": loss.item()}, step=pt_total_steps)
    
    pt_scheduler.step()
    wandb.log({"pt_lr": pt_scheduler.get_last_lr()[0]}, step=pt_total_steps)
    torch.save(
        {    
            'model_state_dict': model.state_dict(),
            'pt_optimizer_state_dict': pt_optimizer.state_dict(),
            'pt_scheduler_state_dict': pt_scheduler.state_dict(),
        }, 
        f"{save_dir}/pt-{epoch}.pt"
    )
       
    ft_model = models.resnet18(num_classes=10)
    ft_model.load_state_dict(model.state_dict())
    ft_optimizer = torch.optim.SGD(ft_model.parameters(), lr=args.ft_learning_rate, momentum=0.9)
    ft_model.train()
    for (batch, target) in ft_dataloader:
        pt_optimizer.zero_grad()
        
        class_preds = model(batch.to(device))
        
        loss = ce_loss(class_preds, target.to(device))
        loss.backward()
        ft_optimizer.step()
        
    ft_model.eval()
    correct_pred_cnt = 0
    with torch.no_grad():
        for (batch, target) in ft_dataloader:
            class_preds = model(batch.to(device))
            correct_pred_cnt += torch.sum(class_preds.argmax(-1) == target.to(device))
            
    wandb.log({"ft_accuracy": correct_pred_cnt / len(ft_dataset)}, step=pt_total_steps) 
    
    test_preds = []
    with torch.no_grad():
        for i, image in enumerate(test_images):
            pred_idx = model(image.unsqueeze(0).to(device)).squeeze(0).argmax().to("cpu")
            test_preds.append((f"{i}.jpg", CLASS_NAMES[pred_idx]))
            
    with open(f"{save_dir}/test-{epoch}.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header
        writer.writerow(['id', 'class'])
        
        # Write the data rows
        writer.writerows(test_preds)
    
    print(f"Spent time on {i} epoch: {time.strftime('%H:%M:%S', time.gmtime(time.time() - begin_epoch_time))}")
        
# # ft_dataset = 
# # test_dataset = 