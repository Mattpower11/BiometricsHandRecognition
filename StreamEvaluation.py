import numpy as np

def streamEvaluationSVC(list_prob_matrix_palmar:np.array, list_prob_matrix_dorsal:np.array, classes:np.array, isClosedSet:bool, threshold:float= 0):
    # Sum the probabilities of all the images
    sum_prob_palm = np.sum(list_prob_matrix_palmar, axis=0)
    
    sum_prob_dorsal = np.sum(list_prob_matrix_dorsal, axis=0)
    
    tot_prob_matrix = sum_prob_palm * 0.6 + sum_prob_dorsal * 0.4
    
    if isClosedSet:
        predicted = classes[np.argmax(tot_prob_matrix, axis=1)]
    else:
        predicted = np.where(tot_prob_matrix.max(axis=1) >= threshold, classes[np.argmax(tot_prob_matrix, axis=1)], -1) 
    
    return tot_prob_matrix,  predicted
    

