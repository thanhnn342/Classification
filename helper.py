import torch
import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 
from torchvision.datasets import ImageFolder
import os 
from datetime import datetime

now = datetime.now()

def read_heic(main_folder):
    register_heif_opener()
    img = Image.open(main_folder)
    img = np.asarray(img)
    return img


def conv_image(tensor_image, prev_norm=True):

    tensor_image = tensor_image.cpu()

    if (len(tensor_image.shape)) == 4:
        tensor_image = tensor_image.squeeze(0)  # Remove Batch dimension

    # Change from [C,H,W] to [H,W,C]
    np_image = tensor_image.permute(1, 2, 0).numpy()

    if prev_norm:
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.5, 0.5, 0.5])
        np_image = np_image * std + mean

    np_image = np.clip(np_image, 0, 1)  # Display in range [0,1]

    return np_image


def count_param(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_sample(data_path_train, data_path_validation):

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(227, scale=(0.6, 1.0)),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.05),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    eval_transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    train_data = ImageFolder(data_path_train, transform=train_transform)
    validation_data = ImageFolder(data_path_validation, transform=eval_transform)

    return train_data, validation_data

def get_loader(train_data, validation_data, BATCH_SIZE):
   
    trainloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    validationloader = DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    return trainloader, validationloader

def init_training(model, save_path = None):
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    else: 
        save_path = f'model_loss/{now.year}_{now.month}_{now.day}/{model.name}'
        os.makedirs(save_path, exist_ok=True)

    logFile = os.path.join(save_path, 'loss.txt')
    
    print("Saving model at", save_path)
    return save_path, logFile

def save_model(epoch, model, optimizer, train_losses, validation_losses, save_path, save_every = 5, save_best = True):
    
    state = {'epoch': epoch, 
             'state_dict': model.state_dict(), 
             'optimizer': optimizer.state_dict(),
             'train_loss': train_losses,
             'validation_loss': validation_losses}
    
    if epoch % save_every == 0:
        torch.save(
                state,
                os.path.join(
                    save_path,
                    f"{model.name}_{epoch}_epoch.pt",
                )
            )
    
    if save_best and validation_losses[-1] == min(validation_losses):
        torch.save(
                state,
                os.path.join(
                    save_path,
                    f"Best_ckpt_{model.name}.pt",
                ),
            )
        
def save_plot(epoch, losses_train, losses_validation, acc_train, acc_validation, save_path):

    plt.figure(figsize=(12, 6))

    plt.subplot(1,2,1)
    plt.plot(losses_train, label="Train Loss")
    plt.plot(losses_validation, label="Validation Loss")
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss value", fontsize=12)
    plt.legend(loc="best")
    plt.title(f"Epoch: {epoch}")

    plt.subplot(1,2,2)
    plt.plot(acc_train, label="Train Accuracy", c = 'g', alpha = 0.7)
    plt.plot(acc_validation, label="validation Accuracy", c = 'r', alpha = 0.7)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Accuracy value", fontsize=12)
    plt.legend(loc="best")
    plt.title(f"Epoch: {epoch}")

    plt.tight_layout()
    plt.savefig(os.path.join(save_path,f"Loss Plot.png"))
    plt.close()