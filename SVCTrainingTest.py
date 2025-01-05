import numpy as np
import pandas as pd
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

def find_weights(csv_path:str):
    # Load the data from csv metadata file
    df = pd.read_csv(csv_path)

    # Create a data structure to store the images' name and the corresponding label
    weights = {} 
    
    print("Finding Weights\n")

    person_id_list = df['id'].unique()

    for i in range(0, len(person_id_list)):
        '''
            Exclude people have no palm and back images to extract
        '''
        if (len(df.loc[(df['id'] == person_id_list[i]) & (df['aspectOfHand'].str.contains('palmar'))]) == 0 or len(df.loc[(df['id'] == person_id_list[i]) & (df['aspectOfHand'].str.contains('dorsal'))]) == 0):
            person_id_list = np.delete(person_id_list, np.where(person_id_list == person_id_list[i])[0])
        
    num_couples_per_id = [] 
    for i in range(0, len(person_id_list)):
        num_couples_per_id.append(min(len(df.loc[(df['id'] == person_id_list[i]) & (df['aspectOfHand'].str.contains('palmar'))]), len(df.loc[(df['id'] == person_id_list[i]) & (df['aspectOfHand'].str.contains('dorsal'))])))
           
    weight_values = (sum(num_couples_per_id) - np.array(num_couples_per_id))/(sum(num_couples_per_id)*len(person_id_list))

    weights.update(dict(zip(person_id_list, weight_values)))

    print("Finding Weights Completed\n")
    return weights