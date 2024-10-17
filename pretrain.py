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
    
    def __init__(self, path, transforms_set):
        super().__init__()
        self.path = path
        self.transforms_set = transforms_set
        self.angles = [0, 90, 180, 270]
        
        # Get list of image file paths
        image_files = os.listdir(path)
        if args.debug:
            image_files = image_files[:128]
        self.image_paths = [os.path.join(self.path, f"{i + 1}.jpg") for i in range(len(image_files))]
        self.len = len(self.image_paths)
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        # Open the image
        with Image.open(self.image_paths[idx]) as img:
            img = img.convert('RGB')
            # Apply transformations
            transformed_img = self.transforms_set(img)
            # Rotate the image by the specified angles and stack
            rotated_tensor_imgs = torch.stack([F.rotate(transformed_img, angle) for angle in self.angles])
        return rotated_tensor_imgs
    
    
class FineTuningDataset(Dataset):
    
    def __init__(self, path, transforms_set):
        super().__init__()
        self.path = path
        self.transforms_set = transforms_set
        
        self.image_paths = []
        self.targets = []
        for i, class_name in enumerate(CLASS_NAMES):
            class_dir = os.path.join(self.path, class_name)
            for fname in os.listdir(class_dir):
                self.image_paths.append(os.path.join(class_dir, fname))
                self.targets.append(i)
        
        if args.debug:
            # Select a small subset of data for debugging
            mask = torch.rand(len(self.targets)) < 0.01
            self.image_paths = [self.image_paths[i] for i in range(len(self.targets)) if mask[i]]
            self.targets = [self.targets[i] for i in range(len(self.targets)) if mask[i]]
        
        self.len = len(self.targets)
        print(f"Fine-tuning dataset size: {self.len}")
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        # Open the image
        with Image.open(self.image_paths[idx]) as img:
            img = img.convert('RGB')
            # Apply transformations
            img = self.transforms_set(img)
        target = self.targets[idx]
        return img, target



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
parser.add_argument("-f", "--ft-learning-rate", default=0.1, type=float)
parser.add_argument("--ft-gamma", default=0.99, type=float)
parser.add_argument("--weight-decay", default=5e-4, type=float)

parser.add_argument("-n", "--num-workers", default=8, type=int)
parser.add_argument("-d", "--debug", action="store_true")
parser.add_argument("-c", "--cpu-only", action="store_true")
args = parser.parse_args()

CLASS_NAMES = sorted(os.listdir("data/train/labeled"))
print(f"class names: {CLASS_NAMES}, len: {len(CLASS_NAMES)}")
mean_stats = [0.485, 0.456, 0.406]
std_stats = [0.229, 0.224, 0.225]
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), ratio=(0.75, 1.33)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),  # Converts image to PyTorch tensor with values in [0, 1]
    transforms.Normalize(mean=mean_stats, std=std_stats),  # Normalize the tensor,
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
])
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # Converts image to PyTorch tensor with values in [0, 1]
    transforms.Normalize(mean=mean_stats, std=std_stats)  # Normalize the tensor
])
denormalize_transform = transforms.Normalize(
    mean=[-m/s for m, s in zip(mean_stats, std_stats)],  # Subtract mean / std
    std=[1/s for s in std_stats]                   # Divide by std
)

run_name = f"AUGM--pt-epochs={args.pretrain_epochs}--bs={args.batch_size}-ptlr={args.pt_learning_rate}-g={args.pt_gamma}-ftlr={args.ft_learning_rate}-g={args.ft_gamma}"
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
pt_scheduler = torch.optim.lr_scheduler.ExponentialLR(pt_optimizer, gamma=args.pt_gamma)

pt_dataset = PretrainDataset("data/train/unlabeled", train_transforms)
pt_dataloader = DataLoader(pt_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

ft_train_dataset = FineTuningDataset("data/train/labeled", train_transforms)
ft_train_dataloader = DataLoader(ft_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

ft_test_dataset = FineTuningDataset("data/train/labeled", test_transforms)
ft_test_dataloader = DataLoader(ft_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

test_path = "data/test"
test_images = []
for i in range(len(os.listdir(test_path))):
    with Image.open(f"{test_path}/{i}.jpg") as img:
        test_images.append(test_transforms(img))

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
            denormalized_images = [denormalize_transform(img) for img in batch[:log_images_n]]
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
        ft_model = models.resnet18(num_classes=10).to(device)
        ft_model.load_state_dict(model.state_dict())
        # ft_model.fc.reset_parameters()
        ft_optimizer = torch.optim.SGD(ft_model.parameters(), lr=args.ft_learning_rate * ftlr_multiplier, momentum=0.9, weight_decay=args.weight_decay)
        ft_scheduler = torch.optim.lr_scheduler.ExponentialLR(ft_optimizer, gamma=args.ft_gamma)
        for ft_epoch in range(args.ft_epochs if is_final_epoch else 3):
            ft_model.train()
            for (batch, target) in ft_train_dataloader:
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
                for (batch, target) in ft_test_dataloader:
                    class_preds = ft_model(batch.to(device))

                    correct_pred_cnt += torch.sum(class_preds.argmax(-1) == target.to(device))
            
                log_images_n = 8
                denormalized_images = [denormalize_transform(img) for img in batch[:log_images_n]]
                grid = make_grid(denormalized_images, nrow=log_images_n)
                grid_np = grid.permute(1, 2, 0).cpu().numpy()
                class_name_preds = [CLASS_NAMES[class_pred] for class_pred in class_preds.argmax(-1)[:log_images_n]]
                if is_final_epoch:
                    logger.log({
                        f"final_test_images_ftlrmul={ftlr_multiplier}": [wandb.Image(grid_np, caption=f"Predictions: {class_name_preds}")]}, 
                        pt_total_steps, 
                        True,
                    )  
                    pt_total_steps += 1
                elif ft_epoch == 2:
                    logger.log({f"test_images_ftlrmul={ftlr_multiplier}": [wandb.Image(grid_np, caption=f"Predictions: {class_name_preds}")]}, pt_total_steps, True)  
                
            if is_final_epoch:
                logger.log({f"final_ft_accuracy_ftlrmul={ftlr_multiplier}": correct_pred_cnt / len(ft_train_dataset)}, pt_total_steps)
            elif ft_epoch == 2:
                logger.log({f"ft_accuracy_ftlrmul={ftlr_multiplier}": correct_pred_cnt / len(ft_train_dataset)}, pt_total_steps)
            
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