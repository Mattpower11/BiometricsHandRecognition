from sklearn.svm import SVC
import torch
import torch.nn as nn
from CNNTrainingTest import testCNN, trainingCNN
from FeatureExtractor import extract_CNN_features, extract_LBP_features
from MyLeNetCNN import MyLeNetCNN
from PrepareData import prepare_data
import torchvision
import torchvision.models.feature_extraction as feature_extraction
from PerformanceEvaluation import *
from SVCTrainingTest import find_best_match, find_weights
from StreamEvaluation import streamEvaluation
from CustomTransform import buildAlexNetTransformations, buildLBPTransformations, buildLeNetTransformations


# Set number of experiments
num_exp = 10
image_path = 'C:/Users/aless/OneDrive/Desktop/Hands/Hands'
csv_path = 'BiometricsHandRecognition/HandInfo.csv'
num_train = 30
num_test = 50

# Prepare data
data_struct = prepare_data(csv_path=csv_path, num_exp=num_exp, num_train=num_train, num_test=num_test, action=False)


# Create the networks
leNet = MyLeNetCNN(num_classes=2)
alexNet1 = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1)
alexNet2 = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1)
'''
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
net_palmar = leNet
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
'''

'''
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
un_labels, un_predicted  = streamEvaluation(net1=net_palmar, net2=net_dorsal, transforms=transforms, weights_palmar_dorsal=weights_palmar_dorsal, data_struct=data_struct, image_path=image_path, tot_exp=num_exp)
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



feature_train_p, labels_train_p = extract_CNN_features(net=net_palmar, data_struct=data_struct, image_path=image_path, transforms=transforms, train_test='train', palmar_dorsal='palmar', tot_exp=num_exp)
feature_test_p, labels_test_p = extract_CNN_features(net=net_palmar, data_struct=data_struct, image_path=image_path, transforms=transforms, train_test='test', palmar_dorsal='palmar', tot_exp=num_exp)

print(feature_train_p.shape)
print(feature_test_p.shape)

palmar_classifier = SVC(kernel='poly', degree=5, decision_function_shape='ovr', class_weight='balanced')
dorsal_classifier = SVC(kernel='poly', degree=5, decision_function_shape='ovr', class_weight='balanced')


SVCTraining(model=palmar_classifier, train_features=feature_train_p, labels=labels_train_p)
predicted_p = SVCTesting(model=palmar_classifier, test_features=feature_test_p)


#print(predicted_p)
#print(labels_test_p)
print("\nAccuracy SVM: ", calculate_accuracy(y_true=torch.detach(labels_test_p).numpy(), y_pred=predicted_p))


feature_train_d, labels_train_d = extract_features(net=net_dorsal, data_struct=data_struct, image_path=image_path, transforms=transforms, train_test='train', palmar_dorsal='dorsal', tot_exp=num_exp)
feature_test_d, labels_test_d = extract_features(net=net_dorsal, data_struct=data_struct, image_path=image_path, transforms=transforms, train_test='test', palmar_dorsal='dorsal', tot_exp=num_exp)

print(feature_train_d.shape)
print(feature_test_d.shape)

SVCTraining(model=dorsal_classifier, train_features=feature_train_d, labels=labels_train_d)
predicted_d = SVCTesting(model=dorsal_classifier, test_features=feature_test_d)

#print(predicted_d)
#print(labels_test_d)
print("\nAccuracy SVM: ", calculate_accuracy(y_true=torch.detach(labels_test_d).numpy(), y_pred=predicted_d))
'''

transforms = [
    buildLBPTransformations(),
    buildLBPTransformations()
]

#variable2 = np.asarray(variable1, dtype="object")

model_features = extract_LBP_features(image_path= image_path, data_struct=data_struct, exp=0, palmar_dorsal='palmar', train_test='train', num_points=8, radius=1, method='uniform', batch_size=32, transforms=transforms)
query_features = extract_LBP_features(image_path= image_path, data_struct=data_struct, exp=1, palmar_dorsal='palmar', train_test='train', num_points=8, radius=1, method='uniform', batch_size=32, transforms=transforms)

predicted, _ = find_best_match(model_images=model_features, query_images=query_features, dist_type='ce', hist_type='grayvalue', num_bins=8) #was euclidean e grayvalue

print(f"Accuracy LBP: {calculate_accuracy(y_true=torch.tensor(data_struct[1]["train"]["labels_id"]), y_pred=torch.tensor(predicted))}")
