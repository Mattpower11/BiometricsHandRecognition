import numpy as np
from sklearn.svm import SVC
from FeatureExtractor import extract_HOG_features, extract_LBP_features
from PrepareData import  prepare_data_SVC
from PerformanceEvaluation import *
from SVCTrainingTest import SVC_Testing, SVC_Training
from CustomTransform import *
from StreamEvaluation import streamEvaluationSVC
from utility import compute_dynamic_threshold, compute_stream_dynamic_threshold

# Parameters
image_path = '/home/mattpower/Downloads/PalmCutHands'
csv_path = 'HandInfo.csv'
num_sub = 15
num_img = 5
perc_test_data = 0.3
isClosedSet = False
num_impostors = 4

# Percentile for the dynamic threshold
percentile = 5

transformsLBP = [
    buildCustomTransformPalmExtended(transform=CustomLBPCannyTransform, isPalm=True),
    buildCustomTransform(transform=CustomLBPTransform),
]

transformsHOG = [
    buildCustomTransformHogPalmExtended(transform=CustomHOGTransform, ksize=(3,3), sigma=1, isPalm=True),
    buildCustomTransformHogExtended(transform=CustomHOGTransform, ksize=(3,3), sigma=1)
]                       
  
svcLBP_p = SVC(kernel='poly', degree=5, decision_function_shape='ovr', class_weight='balanced', probability=True)
svcLBP_d = SVC(kernel='poly', degree=5, decision_function_shape='ovr', class_weight='balanced', probability=True)
svcHOG_p = SVC(kernel='poly', degree=5, decision_function_shape='ovr', class_weight='balanced', probability=True)
svcHOG_d = SVC(kernel='poly', degree=5, decision_function_shape='ovr', class_weight='balanced', probability=True)


# Prepare data
result_dict = prepare_data_SVC(csv_path=csv_path, num_img=num_img, num_sub=num_sub, isClosedSet=isClosedSet, num_impostors=num_impostors, perc_test=perc_test_data)

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
