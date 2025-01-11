import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
import torchvision
from CustomImageDataset import CustomImageDataset
from MyLeNetCNN import MyLeNetCNN
from skimage.feature import local_binary_pattern

def extract_CNN_features(net:nn.Module, num_classes, image_path, transforms, train_test, data_struct, palmar_dorsal, batch_size=32):
    features = torch.tensor([])

    net = modify_net(net, num_classes)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    net.eval()
    
    with torch.no_grad():
        dataset = CustomImageDataset(image_dir=image_path, data_structure= data_struct, train_test=train_test, palmar_dorsal=palmar_dorsal, transform=transforms, action=False)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for image, _ in data_loader: 
            image = image.to(device)    
            features = torch.cat((features, net(image)))
    
    return features

def modify_net(net:nn.Module, num_classes:int=189):
    if isinstance(net, MyLeNetCNN):
        net.fc = nn.Linear(400, num_classes)
        net.relu = nn.Identity()
        net.fc1 = nn.Identity()
        net.relu1 = nn.Identity()
        net.fc2 = nn.Identity()
    elif isinstance(net, torchvision.models.AlexNet):
        net.classifier[1] = nn.Linear(in_features=9216, out_features=num_classes, bias=True)
        for i in range(2, len(net.classifier)):
            net.classifier[i] = nn.Identity()
    return net

def extract_LBP_features(image_path:str, data_struct:dict, palmar_dorsal:str, train_test:str, num_points:int, radius:int, method:str, batch_size:int, transforms):
    features = []

    dataset = CustomImageDataset(image_dir=image_path, data_structure = data_struct, train_test=train_test, palmar_dorsal=palmar_dorsal, transform=transforms, action=False)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for images, _ in data_loader:
        for image in images:
            lbp = local_binary_pattern(np.array(image, dtype=np.int64).reshape(1200,1600), num_points, radius, method=method)       
            n_bins = int(lbp.max() + 1)                  
            (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_bins), range=(0, n_bins))                                                 
                            
            hist = hist.astype("float")                         
            hist /= (hist.sum() + 10e-6)
            features.append(hist)
        
    return features

