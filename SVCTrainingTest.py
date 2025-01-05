from sklearn.discriminant_analysis import StandardScaler
from sklearn.svm import SVC

def SVCTraining(model: SVC, train_features, labels):
    # Standardize features 
    scaler = StandardScaler() 
    X_train = scaler.fit_transform(train_features) 
    
    # Train the model
    model.fit(X_train, labels)

def SVCTesting(model: SVC, test_features):
    # Standardize features 
    scaler = StandardScaler() 
    X_test = scaler.fit_transform(test_features) 
    
    # Test the model
    predicted = model.predict(X_test)

    return predicted
