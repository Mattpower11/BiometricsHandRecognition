import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
import torchvision
from CustomImageDataset import CustomImageDataset
from MyLeNetCNN import MyLeNetCNN
from skimage.feature import local_binary_pattern

def extract_CNN_features(net:nn.Module, image_path, transforms, train_test, data_struct, palmar_dorsal, tot_exp, batch_size=32):
    features = torch.tensor([])
    tot_labels = torch.tensor([])

    net = modify_net(net)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    net.eval()
    
    with torch.no_grad():

        for exp in range(tot_exp):

            dataset = CustomImageDataset(image_dir=image_path, data_structure= data_struct, id_exp=exp, train_test=train_test, palmar_dorsal=palmar_dorsal, transform=transforms, action=False)
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            for data in data_loader:     
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                
                features = torch.cat((features, net(images)))
                tot_labels = torch.cat((tot_labels, labels))

    return features, tot_labels

def modify_net(net:nn.Module):
    if isinstance(net, MyLeNetCNN):
        net.relu = nn.Identity()
        net.fc1 = nn.Identity()
        net.relu1 = nn.Identity()
        net.fc2 = nn.Identity()
    elif isinstance(net, torchvision.models.AlexNet):
        net.classifier[1] = nn.Linear(in_features=9216, out_features=189, bias=True)
        for i in range(2, len(net.classifier)):
            net.classifier[i] = nn.Identity()
    return net

def extract_LBP_features(image_path:str, data_struct:dict, exp:int, palmar_dorsal:str, train_test:str, num_points:int, radius:int, method:str, batch_size:int, transforms):
    features = []

    dataset_train = CustomImageDataset(image_dir=image_path, data_structure = data_struct, id_exp=exp, train_test=train_test, palmar_dorsal=palmar_dorsal, transform=transforms)
    data_loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)

    for data in data_loader_train:
        print(data[0])
        for image in data[0]:
            print(image)
            features.append(local_binary_pattern( (np.asarray(image)).reshape(2, 960000), num_points, radius, method))
    return features

