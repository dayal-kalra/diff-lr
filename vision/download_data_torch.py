import pickle
import os
import numpy as np
import torch
import torchvision
from torchvision.datasets import CIFAR10, CIFAR100
import requests
import zipfile
from PIL import Image

def download_cifar10():
    """Download CIFAR-10 and save in required format"""
    print("Downloading CIFAR-10...")
    os.makedirs('data/cifar-10', exist_ok=True)
    
    # Download datasets
    train_dataset = CIFAR10(root='temp', train=True, download=True)
    test_dataset = CIFAR10(root='temp', train=False, download=True)
    
    # Convert to numpy arrays
    x_train = np.array([np.array(img) for img, _ in train_dataset])
    y_train = np.array([label for _, label in train_dataset])
    x_test = np.array([np.array(img) for img, _ in test_dataset])
    y_test = np.array([label for _, label in test_dataset])
    
    # Normalize to [0, 1]
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    
    # Save train data
    with open('data/cifar-10/cifar-10.train', 'wb') as f:
        pickle.dump((x_train, y_train), f)
    
    # Save test data
    with open('data/cifar-10/cifar-10.test', 'wb') as f:
        pickle.dump((x_test, y_test), f)
    
    print(f"CIFAR-10 saved: {x_train.shape} train, {x_test.shape} test")

def download_cifar100():
    """Download CIFAR-100 and save in required format"""
    print("Downloading CIFAR-100...")
    os.makedirs('data/cifar-100', exist_ok=True)
    
    # Download datasets
    train_dataset = CIFAR100(root='temp', train=True, download=True)
    test_dataset = CIFAR100(root='temp', train=False, download=True)
    
    # Convert to numpy arrays
    x_train = np.array([np.array(img) for img, _ in train_dataset])
    y_train = np.array([label for _, label in train_dataset])
    x_test = np.array([np.array(img) for img, _ in test_dataset])
    y_test = np.array([label for _, label in test_dataset])
    
    # Normalize to [0, 1]
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    
    # Save train data
    with open('data/cifar-100/cifar-100.train', 'wb') as f:
        pickle.dump((x_train, y_train), f)
    
    # Save test data
    with open('data/cifar-100/cifar-100.test', 'wb') as f:
        pickle.dump((x_test, y_test), f)
    
    print(f"CIFAR-100 saved: {x_train.shape} train, {x_test.shape} test")

def download_tiny_imagenet():
    """Download Tiny ImageNet and save in required format"""
    print("Downloading Tiny ImageNet...")
    os.makedirs('data/tiny-imagenet', exist_ok=True)
    
    # Download Tiny ImageNet
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = "data/tiny-imagenet/tiny-imagenet-200.zip"
    
    if not os.path.exists(zip_path):
        print("Downloading zip file...")
        response = requests.get(url, stream=True)
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    
    # Extract
    extract_path = "data/tiny-imagenet"
    if not os.path.exists(f"{extract_path}/tiny-imagenet-200"):
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
    
    # Process data
    data_dir = f"{extract_path}/tiny-imagenet-200"
    
    # Load training data
    train_dir = f"{data_dir}/train"
    x_train, y_train = [], []
    class_to_idx = {}
    
    for i, class_name in enumerate(sorted(os.listdir(train_dir))):
        class_to_idx[class_name] = i
        class_path = f"{train_dir}/{class_name}/images"
        for img_name in os.listdir(class_path):
            img_path = f"{class_path}/{img_name}"
            img = np.array(Image.open(img_path).convert('RGB'))
            x_train.append(img)
            y_train.append(i)
    
    # Load validation data (use as test set)
    val_dir = f"{data_dir}/val"
    x_test, y_test = [], []
    
    # Read val annotations
    val_annotations = {}
    with open(f"{val_dir}/val_annotations.txt", 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            val_annotations[parts[0]] = class_to_idx[parts[1]]
    
    val_images_dir = f"{val_dir}/images"
    for img_name in sorted(os.listdir(val_images_dir)):
        img_path = f"{val_images_dir}/{img_name}"
        img = np.array(Image.open(img_path).convert('RGB'))
        x_test.append(img)
        y_test.append(val_annotations[img_name])
    
    # Convert to numpy and normalize
    x_train = np.array(x_train).astype(np.float32) / 255.0
    y_train = np.array(y_train)
    x_test = np.array(x_test).astype(np.float32) / 255.0
    y_test = np.array(y_test)
    
    # Save train data
    with open('data/tiny-imagenet/tiny-imagenet.train', 'wb') as f:
        pickle.dump((x_train, y_train), f)
    
    # Save test data  
    with open('data/tiny-imagenet/tiny-imagenet.test', 'wb') as f:
        pickle.dump((x_test, y_test), f)
    
    print(f"Tiny ImageNet saved: {x_train.shape} train, {x_test.shape} test")

if __name__ == "__main__":
    download_cifar10()
    #download_cifar100()
    #download_tiny_imagenet()
    print("All datasets downloaded!")
    print("\nYou can now load them using:")
    print("load_image_data('data', 'cifar-10', flatten=False)")
    #print("load_image_data('data', 'cifar-100', flatten=False)")
    #print("load_image_data('data', 'tiny-imagenet', flatten=False)")
