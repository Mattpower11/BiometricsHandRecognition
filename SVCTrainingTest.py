import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.svm import SVC

def SVC_Training(model: SVC, train_features, labels):
    # Standardize features 
    scaler = StandardScaler() 
    X_train = scaler.fit_transform(train_features) 
    
    # Train the model
    model.fit(X_train, labels)

    train_prob =  model.predict_proba(X_train)

    return train_prob

def SVC_Testing(model: SVC, test_features, threshold = 0):
    # Standardize features 
    scaler = StandardScaler() 
    X_test = scaler.fit_transform(test_features) 
    
    # Test the model
    prob_matrix = model.predict_proba(X_test)

    # Predict the labels
    if threshold == 0:
        predicted_labels = model.classes_[prob_matrix.argmax(axis=1)]
    else:
        predicted_labels = np.where(prob_matrix.max(axis=1) >= threshold, model.classes_[prob_matrix.argmax(axis=1)], -1)

    return prob_matrix, predicted_labels
