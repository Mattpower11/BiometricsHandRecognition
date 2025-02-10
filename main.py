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
from CustomTransform import CustomAlexNetTransform, CustomHOGTransform, CustomLBPCannyTransform, CustomLBPTransform, buildCustomTransform, buildCustomTransformHogExtended, buildCustomTransformPalmExtended, buildHistogramTransformations
from StreamEvaluation import streamEvaluationSVC
from utility import compute_dynamic_threshold, compute_stream_dynamic_threshold


# Set number of experiments
num_exp = 5
image_path = '/home/mattpower/Downloads/PalmCutHands'
csv_path = 'HandInfo.csv'
num_train = 30
num_test = 10
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
    buildCustomTransformPalmExtended(transform=CustomLBPCannyTransform, isPalm=True),
    buildCustomTransform(transform=CustomLBPTransform),
]

transformsHOG = [
    buildCustomTransformHogExtended(transform=CustomHOGTransform, ksize=(3,3), sigma=1),
    buildCustomTransformHogExtended(transform=CustomHOGTransform, ksize=(3,3), sigma=1)
]

transformsCNN = [
    buildCustomTransform(transform=CustomAlexNetTransform),
    buildCustomTransform(transform=CustomAlexNetTransform)
]                        
                
transformsHistograms = [
    buildHistogramTransformations(),
    buildHistogramTransformations()
]

svcLBP_p = SVC(kernel='poly', degree=5, decision_function_shape='ovr', class_weight='balanced', probability=True)
svcLBP_d = SVC(kernel='poly', degree=5, decision_function_shape='ovr', class_weight='balanced', probability=True)
svcHOG_p = SVC(kernel='poly', degree=5, decision_function_shape='ovr', class_weight='balanced', probability=True)
svcHOG_d = SVC(kernel='poly', degree=5, decision_function_shape='ovr', class_weight='balanced', probability=True)

#svcCNN = SVC(kernel='poly', degree=5, decision_function_shape='ovr', class_weight='balanced')

# Number of subjects and images
num_sub = 15
num_img = 10
isClosedSet = True
num_impostors = 4
# threshold = 0.5 Use of dynamic threshold
# Percentile for the dynamic threshold
percentile = 5

# Prepare data
result_dict = prepare_data_SVC(csv_path=csv_path, num_img=num_img, num_sub=num_sub, isClosedSet=isClosedSet, num_impostors=num_impostors)

# ------------------- LBP features extractor ---------------

# LBP parameters

radius = 1
num_points = 8 * radius
method = 'uniform'

feature_train_p = extract_LBP_features(image_path=image_path, data_struct=result_dict, palmar_dorsal='palmar', train_test='train', num_points=num_points, radius=radius, method=method, batch_size=32, transforms=transformsLBP)
feature_test_p = extract_LBP_features(image_path=image_path, data_struct=result_dict, palmar_dorsal='palmar', train_test='test', num_points=num_points, radius=radius, method=method, batch_size=32, transforms=transformsLBP)

max_length = max(len(x) for x in feature_train_p)
feature_train_p = [np.pad(x, (0, max_length - len(x)), 'constant') for x in feature_train_p]


train_prob_matrix_LBP_p = SVC_Training(model=svcLBP_p, train_features=feature_train_p, labels=result_dict['train']['person_id'])


if not isClosedSet:
    # Calulate dynamic threshold
    threshold = compute_dynamic_threshold(train_data=feature_train_p,model=svcLBP_p, percentile=percentile)
    max_length = max(len(x) for x in feature_test_p)
    feature_test_p = [np.pad(x, (0, max_length - len(x)), 'constant') for x in feature_test_p]

    test_prob_matrix_LBP_p, predicted_labels_LBP_p = SVC_Testing(model=svcLBP_p, test_features=feature_test_p, threshold=threshold)
else:
    max_length = max(len(x) for x in feature_test_p)
    feature_test_p = [np.pad(x, (0, max_length - len(x)), 'constant') for x in feature_test_p]


    test_prob_matrix_LBP_p, predicted_labels_LBP_p = SVC_Testing(model=svcLBP_p, test_features=feature_test_p)


print(f"Accuracy LBP palmar: {calculate_accuracy(y_true=result_dict['test']['person_id'], y_pred=predicted_labels_LBP_p)}")
#print(prob_matrix_LBP_p)
#print(svcLBP_p.classes_)



feature_train_d= extract_LBP_features(image_path=image_path, data_struct=result_dict, palmar_dorsal='dorsal', train_test='train', num_points=num_points, radius=radius, method=method, batch_size=32, transforms=transformsLBP)
feature_test_d = extract_LBP_features(image_path=image_path, data_struct=result_dict, palmar_dorsal='dorsal', train_test='test', num_points=num_points, radius=radius, method=method, batch_size=32, transforms=transformsLBP)


train_prob_matrix_LBP_d = SVC_Training(model=svcLBP_d, train_features=feature_train_d, labels=result_dict['train']['person_id'])

# Calulate dynamic threshold
if not isClosedSet:
    threshold = compute_dynamic_threshold(train_data=feature_train_d,model=svcLBP_d, percentile=percentile)
    test_prob_matrix_LBP_d, predicted_labels_LBP_d = SVC_Testing(model=svcLBP_d, test_features=feature_test_d, threshold=threshold)
else:
    test_prob_matrix_LBP_d, predicted_labels_LBP_d = SVC_Testing(model=svcLBP_d, test_features=feature_test_d)

print(f"Accuracy LBP dorsal: {calculate_accuracy(y_true=result_dict['test']['person_id'], y_pred=predicted_labels_LBP_d)}")


#print(prob_matrix_LBP_d)
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

train_prob_matrix_HOG_p = SVC_Training(model=svcHOG_p, train_features=feature_train_p, labels=result_dict['train']['person_id'])

# Calulate dynamic threshold
if not isClosedSet:
    threshold = compute_dynamic_threshold(train_data=feature_train_p,model=svcHOG_p, percentile=percentile)
    test_prob_matrix_HOG_p, predicted_labels_HOG_p = SVC_Testing(model=svcHOG_p, test_features=feature_test_p, threshold=threshold)
else:
    test_prob_matrix_HOG_p, predicted_labels_HOG_p = SVC_Testing(model=svcHOG_p, test_features=feature_test_p)

print(f"Accuracy HOG palmar: {calculate_accuracy(y_true=result_dict['test']['person_id'], y_pred=predicted_labels_HOG_p)}")
#print(svcHOG_p.classes_)


feature_train_d= extract_HOG_features(image_path=image_path, data_struct=result_dict, palmar_dorsal='dorsal', train_test='train', orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, batch_size=batch_size, block_norm=block_norm, transforms=transformsHOG)
feature_test_d = extract_HOG_features(image_path=image_path, data_struct=result_dict, palmar_dorsal='dorsal', train_test='test', orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, batch_size=batch_size, block_norm=block_norm, transforms=transformsHOG)


train_prob_matrix_HOG_d = SVC_Training(model=svcHOG_d, train_features=feature_train_d, labels=result_dict['train']['person_id'])

# Calulate dynamic threshold
if not isClosedSet:
    threshold = compute_dynamic_threshold(train_data=feature_train_d,model=svcHOG_d, percentile=percentile)
    test_prob_matrix_HOG_d, predicted_labels_HOG_d = SVC_Testing(model=svcHOG_d, test_features=feature_test_d, threshold=threshold)
else:
    test_prob_matrix_HOG_d, predicted_labels_HOG_d = SVC_Testing(model=svcHOG_d, test_features=feature_test_d)

print(f"Accuracy HOG dorsal: {calculate_accuracy(y_true=result_dict['test']['person_id'], y_pred=predicted_labels_HOG_d)}")


# ------------------- Multibiometric system ---------------

# Create the list of test probability matrices
list_test_prob_matrix_palmar= np.array(object=[test_prob_matrix_LBP_p, test_prob_matrix_HOG_p])
list_test_prob_matrix_dorsal= np.array(object=[test_prob_matrix_LBP_d, test_prob_matrix_HOG_d])

if not isClosedSet:
    # Create the list of train probability matrices
    list_train_prob_matrix_palmar= np.array(object=[train_prob_matrix_LBP_p, train_prob_matrix_HOG_p])
    list_train_prob_matrix_dorsal= np.array(object=[train_prob_matrix_LBP_d, train_prob_matrix_HOG_d])

    # Calulate dynamic threshold
    threshold = compute_stream_dynamic_threshold(list_prob_matrix_palmar=list_train_prob_matrix_palmar, list_prob_matrix_dorsal=list_train_prob_matrix_dorsal, percentile=percentile)
    tot_prob_matrix, predicted = streamEvaluationSVC(list_prob_matrix_palmar=list_test_prob_matrix_palmar, list_prob_matrix_dorsal=list_test_prob_matrix_dorsal, classes=svcHOG_d.classes_, threshold=threshold, isClosedSet=isClosedSet)
else:
    tot_prob_matrix, predicted = streamEvaluationSVC(list_prob_matrix_palmar=list_test_prob_matrix_palmar, list_prob_matrix_dorsal=list_test_prob_matrix_dorsal, classes=svcHOG_d.classes_, isClosedSet=isClosedSet)

print(f"Accuracy multibiometric system: {calculate_accuracy(y_true=result_dict['test']['person_id'], y_pred=predicted)}")

# ------------------ Performance evaluation -----------------

#print(result_dict['test']['person_id'])

true_labels = np.array(result_dict['test']['person_id'])
gallery_labels = np.unique(np.array(result_dict['test']['person_id']))

if isClosedSet:
    calculate_CMC_plot(score_matrix=test_prob_matrix_LBP_p, true_labels=true_labels, gallery_labels=gallery_labels, type_feature_extractor='LBP', palm_dorsal='palmar')
    calculate_CMC_plot(score_matrix=test_prob_matrix_HOG_p, true_labels=true_labels, gallery_labels=gallery_labels, type_feature_extractor='HOG', palm_dorsal='palmar')
    calculate_CMC_plot(score_matrix=test_prob_matrix_LBP_d, true_labels=true_labels, gallery_labels=gallery_labels, type_feature_extractor='LBP', palm_dorsal='dorsal')
    calculate_CMC_plot(score_matrix=test_prob_matrix_HOG_d, true_labels=true_labels, gallery_labels=gallery_labels, type_feature_extractor='HOG', palm_dorsal='dorsal')
    calculate_CMC_plot(score_matrix=tot_prob_matrix, true_labels=true_labels, gallery_labels=gallery_labels, type_feature_extractor='Multibiometric', palm_dorsal='')

    calculate_confusion_matrix(y_true=result_dict['test']['person_id'], y_pred=predicted)
else: 
    LBP_p_far_values = calculate_FAR_plot(predicted_scores=test_prob_matrix_LBP_p, true_labels=true_labels, type_feature_extractor='LBP', palm_dorsal='palmar')
    LBP_d_far_values = calculate_FAR_plot(predicted_scores=test_prob_matrix_LBP_d, true_labels=true_labels, type_feature_extractor='LBP', palm_dorsal='dorsal')
    HOG_p_far_values = calculate_FAR_plot(predicted_scores=test_prob_matrix_HOG_p, true_labels=true_labels, type_feature_extractor='HOG', palm_dorsal='palmar')
    HOG_d_far_values = calculate_FAR_plot(predicted_scores=test_prob_matrix_HOG_d, true_labels=true_labels, type_feature_extractor='HOG', palm_dorsal='dorsal')
    tot_far_values = calculate_FAR_plot(predicted_scores=tot_prob_matrix, true_labels=true_labels, type_feature_extractor='Multibiometric', palm_dorsal='')

    LBP_p_frr_values = calculate_FRR_plot(predicted_scores=test_prob_matrix_LBP_p, true_labels=true_labels, type_feature_extractor='LBP', palm_dorsal='palmar')
    LBP_d_frr_values = calculate_FRR_plot(predicted_scores=test_prob_matrix_LBP_d, true_labels=true_labels, type_feature_extractor='LBP', palm_dorsal='dorsal')
    HOG_p_frr_values = calculate_FRR_plot(predicted_scores=test_prob_matrix_HOG_p, true_labels=true_labels, type_feature_extractor='HOG', palm_dorsal='palmar')
    HOG_d_frr_values = calculate_FRR_plot(predicted_scores=test_prob_matrix_HOG_d, true_labels=true_labels, type_feature_extractor='HOG', palm_dorsal='dorsal')
    tot_frr_values = calculate_FRR_plot(predicted_scores=tot_prob_matrix, true_labels=true_labels, type_feature_extractor='Multibiometric', palm_dorsal='')

    plot_FAR_FRR(far_values=LBP_p_far_values, frr_values=LBP_p_frr_values, type_feature_extractor='LBP', palm_dorsal='palmar')
    plot_FAR_FRR(far_values=LBP_d_far_values, frr_values=LBP_d_frr_values, type_feature_extractor='LBP', palm_dorsal='dorsal')
    plot_FAR_FRR(far_values=HOG_p_far_values, frr_values=HOG_p_frr_values, type_feature_extractor='HOG', palm_dorsal='palmar')
    plot_FAR_FRR(far_values=HOG_d_far_values, frr_values=HOG_d_frr_values, type_feature_extractor='HOG', palm_dorsal='dorsal')
    plot_FAR_FRR(far_values=tot_far_values, frr_values=tot_frr_values, type_feature_extractor='Multibiometric', palm_dorsal='')

'''
feature_train = extract_CNN_features(net=alexNet1, num_classes=num_sub, image_path=image_path, transforms=transformsCNN, train_test='train', data_struct=result_dict, palmar_dorsal='palmar', batch_size=32)
feature_test = extract_CNN_features(net=alexNet1, num_classes=num_sub, image_path=image_path, transforms=transformsCNN, train_test='test', data_struct=result_dict, palmar_dorsal='palmar', batch_size=32)
SVC_Training(model=svcCNN, train_features=feature_train, labels=result_dict['train']['person_id'])

predicted = SVC_Testing(model=svcCNN, test_features=feature_test)
print(f"Accuracy CNN: {calculate_accuracy(y_true=result_dict['test']['person_id'], y_pred=predicted)}")


predicted_palmar = find_best_match(dist_type="l2", hist_type="gb", num_bins=64, image_path=image_path, data_struct=result_dict, palmar_dorsal="palmar", transforms=transformsHistograms)
predicted_dorsal = find_best_match(dist_type="l2", hist_type="gb", num_bins=64, image_path=image_path, data_struct=result_dict, palmar_dorsal="dorsal", transforms=transformsHistograms)
'''