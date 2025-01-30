from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from CustomImageDataset import CustomImageDataset
from utility import compute_histograms, get_dist_by_name, is_grayvalue_hist

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
    dist_type: str,
    hist_type: str,
    num_bins: int,
    image_path: str,
    data_struct: dict,
    palmar_dorsal: str,
    transforms: list,
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
    D = np.zeros((len(data_struct["train"]["images"]), len(data_struct["test"]["images"]))) #righe = model, colonne = query

    #inizializzazione della lista di best_match
    best_match = []
    model_hist = []
    query_hist = []

    #calcolo degli istogrammi
    dataset = CustomImageDataset(image_dir=image_path, data_structure=data_struct, train_test="train", palmar_dorsal=palmar_dorsal, action=False, transform=transforms)
    model_hist.append(compute_histograms(dataset, hist_type, is_grayvalue_hist(hist_type), num_bins))

    print(model_hist)

    dataset = CustomImageDataset(image_dir=image_path, data_structure=data_struct, train_test="test", palmar_dorsal=palmar_dorsal, action=False, transform=transforms)
    query_hist.append(compute_histograms(dataset, hist_type, is_grayvalue_hist(hist_type), num_bins))

    print(query_hist)

    m_hist = model_hist[0]
    q_hist = query_hist[0]

    # Array di indici di bin
    bins = np.arange(len(m_hist))  #if num_bins=8, bins = [0,1,...,7]

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.bar(bins, m_hist, width=0.8, color="blue", alpha=0.7)
    plt.title("Model Histogram")
    plt.xlabel("Bin index")
    plt.ylabel("Frequency (or Probability)")

    plt.subplot(1, 2, 2)
    plt.bar(bins, q_hist, width=0.8, color="red", alpha=0.7)
    plt.title("Query Histogram")
    plt.xlabel("Bin index")
    plt.ylabel("Frequency (or Probability)")

    plt.tight_layout()
    plt.show()

    #calcolo della distanza tra gli istogrammi
    for i, model in enumerate(model_hist):
        for j, query in enumerate(query_hist):
            D[i, j] = get_dist_by_name(model, query, dist_type)
    
    #calcolo del best match
    for i in range(D.shape[1]): #prende le colonne
        best_match.append(np.argmin(D[:, i])) #argmin in quanto il distance based matching é basato sulla distanza minima tra i due istogrammi

    return best_match, D

