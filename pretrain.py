import argparse
import csv
import time
import os
import glob
import wandb
from tqdm import trange

from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F


class PretrainDataset(Dataset):
    
    def __init__(self, path):
        self.path = path
        cut = None if not args.debug else 128
        self.len = len(os.listdir(path)) if not args.debug else cut
        angles = [0, 90, 180, 270]
        # Load the dataset on Disk
        tensor_imgs = []
        for i in range(self.len):
            with Image.open(f"{self.path}/{i + 1}.jpg") as img:
                tensor_imgs.append(TRANSFORM(img))
        self.rotated_tensor_imgs = [
            torch.stack([F.rotate(tensor_img, angle) for angle in angles])
            for tensor_img in tensor_imgs[:cut]
        ]
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        return self.rotated_tensor_imgs[idx]
    
class FineTuningDataset(Dataset):
    
    def __init__(self, path):
        self.path = path
        
        targets = []
        tensor_imgs = []
        for i, class_name in enumerate(CLASS_NAMES):
            for fname in os.listdir(f"{path}/{class_name}"):
                with Image.open(f"{self.path}/{class_name}/{fname}") as img:
                    tensor_imgs.append(TRANSFORM(img))
                targets.append(i)
        
        if args.debug:
            self.targets = []
            self.tensor_imgs = []
            mask = torch.rand(len(targets)) < 0.01
            for i in range(len(targets)):
                if mask[i]:
                    self.targets.append(targets[i])
                    self.tensor_imgs.append(tensor_imgs[i])
        else:
            self.targets = targets
            self.tensor_imgs = tensor_imgs
        self.len = len(self.targets)        
        print(f"Fine-tuning dataset size: {self.len}")
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        return self.tensor_imgs[idx], self.targets[idx]


class Logger:
    
    def __init__(self, run_name):
        self.is_debug = args.debug
        if args.debug:
            print(f"Run name: {run_name}")
        else:
            wandb.init(project="Pretrain ResNet-18, HW-2. LSDL 2024, CUB.", name=run_name)
        
        
    def log(self, d, step, is_image=False):
        if self.is_debug:
            if not is_image:
                for k, v in d.items():
                    print(f"{k}: {v}")
        else:
            wandb.log(d, step=step)
    

parser = argparse.ArgumentParser()
parser.add_argument("--pretrain-epochs", default=100, type=int)
parser.add_argument("--ft-epochs", default=300, type=int)

parser.add_argument("-b", "--batch-size", default=128, type=int)
parser.add_argument("-p", "--pt-learning-rate", default=0.1, type=float)
parser.add_argument("--pt-gamma", default=0.95, type=float)
parser.add_argument("-f", "--ft-learning-rate", default=1e-2, type=float)
parser.add_argument("--ft-gamma", default=0.99, type=float)
parser.add_argument("--weight-decay", default=5e-4, type=float)

parser.add_argument("-n", "--num-workers", default=8, type=int)
parser.add_argument("-d", "--debug", action="store_true")
parser.add_argument("-c", "--cpu-only", action="store_true")
args = parser.parse_args()

CLASS_NAMES = sorted(os.listdir("data/train/labeled"))
mean_stats = [0.485, 0.456, 0.406]
std_stats = [0.229, 0.224, 0.225]
TRANSFORM = transforms.Compose([
    transforms.ToTensor(),  # Converts image to PyTorch tensor with values in [0, 1]
    transforms.Normalize(mean=mean_stats, std=std_stats)  # Normalize the tensor
])
DENORMALIZE_TRANSFORM = transforms.Normalize(
    mean=[-m/s for m, s in zip(mean_stats, std_stats)],  # Subtract mean / std
    std=[1/s for s in std_stats]                   # Divide by std
)

run_name = f"bs={args.batch_size}-ptlr={args.pt_learning_rate}-g={args.pt_gamma}-ftlr={args.ft_learning_rate}-g={args.ft_gamma}"
logger = Logger(run_name)

device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu_only else "cpu")

model = models.resnet18(num_classes=10).to(device)
pretrain_head = nn.Linear(512, 4).to(device)
pretrain_head.train()

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output
    return hook
model.avgpool.register_forward_hook(get_activation("avgpool"))

print(model)
print(f"device: {device}")
ce_loss = nn.CrossEntropyLoss()

# 1. Pretrain Phase
start_time = time.time()
save_dir = f"checkpoints/{run_name}"
os.makedirs(save_dir, exist_ok=True)
files = glob.glob(f"{save_dir}/*")
for f in files:
    os.remove(f)

pt_optimizer = torch.optim.SGD(
    [v for k, v in model.named_parameters() if not k.startswith("fc.")] + list(pretrain_head.parameters()),
    lr=args.pt_learning_rate,
    momentum=0.9,
    weight_decay=args.weight_decay,
)
print()
pt_scheduler = torch.optim.lr_scheduler.ExponentialLR(pt_optimizer, gamma=args.pt_gamma)

pt_dataset = PretrainDataset("data/train/unlabeled")
pt_dataloader = DataLoader(pt_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

ft_dataset = FineTuningDataset("data/train/labeled")
ft_dataloader = DataLoader(ft_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

test_path = "data/test"
test_images = []
for i in range(len(os.listdir(test_path))):
    with Image.open(f"{test_path}/{i}.jpg") as img:
        test_images.append(TRANSFORM(img))

pt_total_steps = 0

for epoch in trange(args.pretrain_epochs):
    begin_epoch_time = time.time()
    model.train()
    for i, batch in enumerate(pt_dataloader):
        pt_optimizer.zero_grad()
        batch = torch.flatten(batch, 0, 1).to(device)
        _ = model(batch)
        rotation_preds = pretrain_head(activation["avgpool"].squeeze())
        target = torch.arange(batch.shape[0], device=device) % 4
        loss = ce_loss(rotation_preds, target)
        loss.backward()
        pt_optimizer.step()
        
        pt_total_steps += 1
        
        if pt_total_steps > 0 and pt_total_steps % 40 == 0:
            logger.log({"train_loss": loss.item()}, pt_total_steps)
            
            log_images_n = 4
            denormalized_images = [DENORMALIZE_TRANSFORM(img) for img in batch[:log_images_n]]
            grid = make_grid(denormalized_images, nrow=log_images_n)
            grid_np = grid.permute(1, 2, 0).cpu().numpy()
            logger.log({"rotations": [wandb.Image(grid_np, caption=f"Predictions: {rotation_preds.argmax(1)[:log_images_n]}")]}, pt_total_steps, True) 
    logger.log({"pt_lr": pt_scheduler.get_last_lr()[0]}, pt_total_steps)
    pt_scheduler.step()
    torch.save(
        {    
            'model_state_dict': model.state_dict(),
            'pt_optimizer_state_dict': pt_optimizer.state_dict(),
            'pt_scheduler_state_dict': pt_scheduler.state_dict(),
        }, 
        f"{save_dir}/pt-{epoch}.pt"
    )
    
    is_final_epoch = epoch + 1 == args.pretrain_epochs
    for ftlr_multiplier in [1, 0.1, 0.01]:
        ft_model = models.resnet18(num_classes=10)
        ft_model.load_state_dict(model.state_dict())
        ft_model = ft_model.to(device)
        ft_optimizer = torch.optim.SGD(ft_model.parameters(), lr=args.ft_learning_rate * ftlr_multiplier, momentum=0.9, weight_decay=args.weight_decay)
        ft_scheduler = torch.optim.lr_scheduler.ExponentialLR(ft_optimizer, gamma=args.ft_gamma)
        for ft_epoch in range(args.ft_epochs if is_final_epoch else 1):
            ft_model.train()
            for (batch, target) in ft_dataloader:
                ft_optimizer.zero_grad()
                
                class_preds = ft_model(batch.to(device))
                
                loss = ce_loss(class_preds, target.to(device))
                loss.backward()
                ft_optimizer.step()
            if is_final_epoch:
                logger.log({"final_ft_lr": ft_scheduler.get_last_lr()[0]}, ft_epoch)
            ft_scheduler.step()
                
            ft_model.eval()
            correct_pred_cnt = 0
            with torch.no_grad():
                for (batch, target) in ft_dataloader:
                    class_preds = ft_model(batch.to(device))
                    correct_pred_cnt += torch.sum(class_preds.argmax(-1) == target.to(device))
            
                log_images_n = 8
                denormalized_images = [DENORMALIZE_TRANSFORM(img) for img in batch[:log_images_n]]
                grid = make_grid(denormalized_images, nrow=log_images_n)
                grid_np = grid.permute(1, 2, 0).cpu().numpy()
                class_name_preds = [CLASS_NAMES[class_pred] for class_pred in class_preds.argmax(-1)[:log_images_n]]
                if is_final_epoch:
                    logger.log({f"final_test_images_ftlrmul={ftlr_multiplier}": [wandb.Image(grid_np, caption=f"Predictions: {class_name_preds}")]}, ft_epoch, True)  
                else:
                    logger.log({f"test_images_ftlrmul={ftlr_multiplier}": [wandb.Image(grid_np, caption=f"Predictions: {class_name_preds}")]}, pt_total_steps, True)  
                
            if is_final_epoch:
                logger.log({f"final_ft_accuracy_ftlrmul={ftlr_multiplier}": correct_pred_cnt / len(ft_dataset)}, ft_epoch)
            else:
                logger.log({f"ft_accuracy_ftlrmul={ftlr_multiplier}": correct_pred_cnt / len(ft_dataset)}, pt_total_steps)
            
            test_preds = []
            with torch.no_grad():
                for i, image in enumerate(test_images):
                    pred_idx = ft_model(image.unsqueeze(0).to(device)).squeeze(0).argmax().to("cpu")
                    test_preds.append((f"{i}.jpg", CLASS_NAMES[pred_idx]))
            
            fname = f"{save_dir}/" + (f"final_test_{ft_epoch}" if is_final_epoch else f"test_{epoch}") + f"_ftlrmul={ftlr_multiplier}.csv"
            with open(fname, mode='w', newline='') as fout:
                writer = csv.writer(fout)
                
                # Write the header
                writer.writerow(['id', 'class'])
                
                # Write the data rows
                writer.writerows(test_preds)
    
    print(f"Spent time on {i} epoch: {time.strftime('%H:%M:%S', time.gmtime(time.time() - begin_epoch_time))}")