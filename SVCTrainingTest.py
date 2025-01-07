from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.svm import SVC

from CustomImageDataset import CustomImageDataset
from utility import compute_histograms, get_dist_by_name, is_grayvalue_hist

def SVC_CNN_Training(model: SVC, train_features, labels):
    # Standardize features 
    scaler = StandardScaler() 
    X_train = scaler.fit_transform(train_features) 
    
    # Train the model
    model.fit(X_train, labels)

def SVC_CNN_Testing(model: SVC, test_features):
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


def find_best_match(
    model_images: List[np.ndarray],
    query_images: List[np.ndarray],
    dist_type: str,
    hist_type: str,
    num_bins: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function to find the best match for each image in the
    query folder.

    Args:
        model_images (List[str]): list of model LBP images.
        query_images (List[str]): list of query LBP images.
        dist_type (str): a string to represent the name of the distance you want to use. Should be one among "l2", "intersect", "chi2".
        hist_type (str): a string to represent the name of the histogram you want to use. Should be one among "grayvalue", "rgb", "rg", "dxdy".
        num_bins (int): number of bins for the gray_scale histogram.

    Returns:
        best_match (np.ndarray): list containing in each position the index of the retrieved best matching image.
        D (np.ndarray): Matrix with |model_images| rows and |query_images| columns containing the scores of each matching.
    """

    #inizializzazione della matrice D
    D = np.zeros((len(model_images), len(query_images))) #righe = model, colonne = query

    #inizializzazione della lista di best_match
    best_match = []

    #calcolo degli istogrammi
    model_hist = compute_histograms(model_images, hist_type, is_grayvalue_hist(hist_type), num_bins)
    query_hist = compute_histograms(query_images, hist_type, is_grayvalue_hist(hist_type), num_bins)

    #calcolo della distanza tra gli istogrammi
    for i, model in enumerate(model_hist):
        for j, query in enumerate(query_hist):
            D[i, j] = get_dist_by_name(model, query, dist_type)
    
    #calcolo del best match
    for i in range(D.shape[1]): #prende le colonne
        best_match.append(np.argmin(D[:, i])) #argmin in quanto il distance based matching é basato sulla distanza minima tra i due istogrammi


    return best_match, D