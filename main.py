import numpy as np
from sklearn.svm import SVC
import torch
import torch.nn as nn
from CNNTrainingTest import testCNN, trainingCNN
from FeatureExtractor import extract_CNN_features, extract_HOG_features, extract_LBP_features
from MyLeNetCNN import MyLeNetCNN
from PrepareData import prepare_data_CNN, prepare_data_SVC
import torchvision
import torchvision.models.feature_extraction as feature_extraction
from PerformanceEvaluation import *
from SVCTrainingTest import SVC_Testing, SVC_Training, find_best_match, find_weights
from StreamEvaluation import streamEvaluationCNN
from CustomTransform import buildAlexNetTransformations, buildHOGTransformations, buildLBPTransformations, buildLeNetTransformations
from StreamEvaluation import streamEvaluationSVC


# Set number of experiments
num_exp = 10
image_path = '/home/mattpower/Downloads/Hands'
csv_path = '/home/mattpower/Documents/backup/Magistrale/Sapienza/ComputerScience/Biometrics Systems/Progetto/BiometricsHandRecognition/HandInfo.csv'
num_train = 100
num_test = 50
'''
# Prepare data
data_struct = prepare_data_CNN(csv_path=csv_path, num_exp=num_exp, num_train=num_train, num_test=num_test, action=False)


# Create the networks
leNet = MyLeNetCNN(num_classes=2)
alexNet1 = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1)
alexNet2 = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1)

# Customize AlexNet1
# Update the final layer to output 2 classes
num_features = alexNet1.classifier[6].in_features
alexNet1.classifier[6] = nn.Linear(num_features, 2)

# Freeze all layers except the newly added fully connected layer
for param in alexNet1.parameters():
    param.requires_grad = False
for param in alexNet1.classifier[6].parameters():
    param.requires_grad = True

# Customize AlexNet2
# Update the final layer to output 2 classes
num_features = alexNet2.classifier[6].in_features
alexNet2.classifier[6] = nn.Linear(num_features, 2)

# Freeze all layers except the newly added fully connected layer
for param in alexNet2.parameters():
    param.requires_grad = False
for param in alexNet2.classifier[6].parameters():
    param.requires_grad = True

# Set the networks
net_palmar = alexNet1
net_dorsal = alexNet2

weight_palmar = 0.4
weight_dorsal = 0.6

# Build the tranformations for the networks
palmar_transforms = buildAlexNetTransformations()
if isinstance(net_palmar, MyLeNetCNN):
        palmar_transforms = buildLeNetTransformations()
elif isinstance(net_palmar, torchvision.models.AlexNet):
        palmar_transforms = buildAlexNetTransformations()

dorsal_transforms = buildAlexNetTransformations()
if isinstance(net_dorsal, MyLeNetCNN):
        dorsal_transforms = buildLeNetTransformations()
elif isinstance(net_dorsal, torchvision.models.AlexNet):
        dorsal_transforms = buildAlexNetTransformations()

transforms = [
    palmar_transforms,
    dorsal_transforms
]

# Weights for the fusion
weights_palmar_dorsal = [weight_palmar, weight_dorsal]



# Training the networks
print('Begin Palm Training\n')
train_loss_p, train_labels_p = trainingCNN(net=net_palmar, transforms=transforms, data_struct=data_struct, image_path=image_path, palmar_dorsal='palmar', tot_exp=num_exp)
print('\nFinished Palm Training\n')
print('Begin Dorsal Training\n')
train_loss_d, train_labels_d = trainingCNN(net=net_dorsal, transforms=transforms, data_struct=data_struct, image_path=image_path, palmar_dorsal='dorsal', tot_exp=num_exp)
print('\nFinished Dorsal Training\n')

# Test the networks
print('Begin Palm Testing')
test_labels_p, palmar_predicted = testCNN(net=net_palmar, transforms=transforms, data_struct=data_struct, image_path=image_path, palmar_dorsal='palmar', tot_exp=num_exp)
print('Finished Palm Testing\n')
print('Begin Dorsal Testing')
test_labels_d, dorsal_predicted = testCNN(net=net_dorsal, transforms=transforms, data_struct=data_struct, image_path=image_path, palmar_dorsal='dorsal', tot_exp=num_exp)
print('Finished Dorsal Testing\n')

# Evaluate the unified network
print("Begin Unified Network Testing")
un_labels, un_predicted  = streamEvaluationCNN(net1=net_palmar, net2=net_dorsal, transforms=transforms, weights_palmar_dorsal=weights_palmar_dorsal, data_struct=data_struct, image_path=image_path, tot_exp=num_exp)
print("Finished Unified Network Testing\n")

'''
'''
# Performance evaluation
calculate_confusion_matrix(test_labels_p, palmar_predicted)
calculate_confusion_matrix(test_labels_d, dorsal_predicted)
calculate_confusion_matrix(un_labels, un_predicted)

# Calculate the loss plot
calculate_loss_plot(train_loss_p)
calculate_loss_plot(train_loss_d)
'''
'''
# Print the performance metrics
print("\nPerformance Metrics\n")

print(f"\nPalmar Network= {type(net_palmar).__name__}")
print(f"Dorsal Network= {type(net_dorsal).__name__}\n")

print("\nAccuracy Palmar Network: ", calculate_accuracy(y_true=test_labels_p, y_pred=palmar_predicted))
print("Precision Palmar Network: ", calculate_precision(y_true=test_labels_p, y_pred=palmar_predicted))
print("Recall Palmar Network: ", calculate_recall(y_true=test_labels_p, y_pred=palmar_predicted))
print("F1 Score Palmar neNetworkt: ", calculate_f1_score(y_true=test_labels_p, y_pred=palmar_predicted),"\n")

print("\nAccuracy Dorsal Network: ", calculate_accuracy(y_true=test_labels_d, y_pred=dorsal_predicted))
print("Precision Dorsal Network: ", calculate_precision(y_true=test_labels_d, y_pred=dorsal_predicted))
print("Recall Dorsal Network: ", calculate_recall(y_true=test_labels_d, y_pred=dorsal_predicted))
print("F1 Score Dorsal Network: ", calculate_f1_score(y_true=test_labels_d, y_pred=dorsal_predicted),"\n")

print("\nAccuracy Unified Network: ", calculate_accuracy(y_true=un_labels, y_pred=un_predicted))
print("Precision Unified Network: ", calculate_precision(y_true=un_labels, y_pred=un_predicted))
print("Recall Unified Network: ", calculate_recall(y_true=un_labels, y_pred=un_predicted))
print("F1 Score Unified Network: ", calculate_f1_score(y_true=un_labels, y_pred=un_predicted),"\n")
'''


transformsLBP = [
    buildLBPTransformations(),
    buildLBPTransformations()
]

transformsHOG = [
    
    buildHOGTransformations(ksize=(5,5), sigma=1.0),
    buildHOGTransformations(ksize=(5,5), sigma=1.0)
]

transformsCNN = [
    buildAlexNetTransformations(),
    buildAlexNetTransformations()
]

svcLBP_p = SVC(kernel='poly', degree=5, decision_function_shape='ovr', class_weight='balanced', probability=True)
svcLBP_d = SVC(kernel='poly', degree=5, decision_function_shape='ovr', class_weight='balanced', probability=True)
svcHOG_p = SVC(kernel='poly', degree=5, decision_function_shape='ovr', class_weight='balanced', probability=True)
svcHOG_d = SVC(kernel='poly', degree=5, decision_function_shape='ovr', class_weight='balanced', probability=True)

#svcCNN = SVC(kernel='poly', degree=5, decision_function_shape='ovr', class_weight='balanced')

# Number of subjects and images
num_sub = 5
num_img = 10
threshold = 0.5

# Prepare data
result_dict = prepare_data_SVC(csv_path=csv_path, num_img=num_img, num_sub=num_sub)

# ------------------- LBP features extractor ---------------

# LBP parameters
'''
radius = 1
num_points = 8 * radius
method = 'uniform'


feature_train_p = extract_LBP_features(image_path=image_path, data_struct=result_dict, palmar_dorsal='palmar', train_test='train', num_points=num_points, radius=radius, method=method, batch_size=32, transforms=transformsLBP)
feature_test_p = extract_LBP_features(image_path=image_path, data_struct=result_dict, palmar_dorsal='palmar', train_test='test', num_points=num_points, radius=radius, method=method, batch_size=32, transforms=transformsLBP)



SVC_Training(model=svcLBP_p, train_features=feature_train_p, labels=result_dict['train']['person_id'])
prob_matrix_LBP_p = SVC_Testing(model=svcLBP_p, test_features=feature_test_p)
#print(f"Accuracy LBP palmar: {calculate_accuracy(y_true=result_dict['test']['person_id'], y_pred=predicted)}")
print(prob_matrix_LBP_p)
#print(svcLBP_p.classes_)

feature_train_d= extract_LBP_features(image_path=image_path, data_struct=result_dict, palmar_dorsal='dorsal', train_test='train', num_points=num_points, radius=radius, method=method, batch_size=32, transforms=transformsLBP)
feature_test_d = extract_LBP_features(image_path=image_path, data_struct=result_dict, palmar_dorsal='dorsal', train_test='test', num_points=num_points, radius=radius, method=method, batch_size=32, transforms=transformsLBP)


SVC_Training(model=svcLBP_d, train_features=feature_train_d, labels=result_dict['train']['person_id'])
prob_matrix_LBP_d = SVC_Testing(model=svcLBP_d, test_features=feature_test_d)
#print(f"Accuracy LBP dorsal: {calculate_accuracy(y_true=result_dict['test']['person_id'], y_pred=predicted)}")
print(prob_matrix_LBP_d)
#print(svcLBP_d.classes_)

# ------------------- HOG features extractor ---------------
# HOG parameters

orientations = 9
pixels_per_cell = 16
cells_per_block = 1
batch_size = 32
block_norm = 'L2-Hys'

feature_train_p = extract_HOG_features(image_path=image_path, data_struct=result_dict, palmar_dorsal='palmar', train_test='train', orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, batch_size=batch_size, block_norm=block_norm, transforms=transformsHOG)
feature_test_p = extract_HOG_features(image_path=image_path, data_struct=result_dict, palmar_dorsal='palmar', train_test='test', orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, batch_size=batch_size, block_norm=block_norm, transforms=transformsHOG)

SVC_Training(model=svcHOG_p, train_features=feature_train_p, labels=result_dict['train']['person_id'])
prob_matrix_HOG_p = SVC_Testing(model=svcHOG_p, test_features=feature_test_p)
print(svcHOG_p.classes_)


feature_train_d= extract_HOG_features(image_path=image_path, data_struct=result_dict, palmar_dorsal='dorsal', train_test='train', orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, batch_size=batch_size, block_norm=block_norm, transforms=transformsHOG)
feature_test_d = extract_HOG_features(image_path=image_path, data_struct=result_dict, palmar_dorsal='dorsal', train_test='test', orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, batch_size=batch_size, block_norm=block_norm, transforms=transformsHOG)


SVC_Training(model=svcHOG_d, train_features=feature_train_d, labels=result_dict['train']['person_id'])
prob_matrix_HOG_d = SVC_Testing(model=svcHOG_d, test_features=feature_test_d)

list_prob_matrix_palmar= np.array(object=[prob_matrix_LBP_p, prob_matrix_HOG_p])
list_prob_matrix_dorsal= np.array(object=[prob_matrix_LBP_d, prob_matrix_HOG_d])
print(svcHOG_d.classes_)

predicted = streamEvaluationSVC(list_prob_matrix_palmar=list_prob_matrix_palmar, list_prob_matrix_dorsal=list_prob_matrix_dorsal, classes=svcHOG_d.classes_)

calculate_confusion_matrix(y_true=result_dict['test']['person_id'], y_pred=predicted)
print(f"Accuracy SVC: {calculate_accuracy(y_true=result_dict['test']['person_id'], y_pred=predicted)}")
'''

'''
feature_train = extract_CNN_features(net=alexNet1, num_classes=num_sub, image_path=image_path, transforms=transformsCNN, train_test='train', data_struct=result_dict, palmar_dorsal='palmar', batch_size=32)
feature_test = extract_CNN_features(net=alexNet1, num_classes=num_sub, image_path=image_path, transforms=transformsCNN, train_test='test', data_struct=result_dict, palmar_dorsal='palmar', batch_size=32)
SVC_Training(model=svcCNN, train_features=feature_train, labels=result_dict['train']['person_id'])

predicted = SVC_Testing(model=svcCNN, test_features=feature_test)
print(f"Accuracy CNN: {calculate_accuracy(y_true=result_dict['test']['person_id'], y_pred=predicted)}")
'''