import numpy as np
import torch.nn as nn

def compute_dynamic_threshold(model:nn.Module ,train_data: np.ndarray, percentile: float) -> float:
    train_prob = model.predict_proba(train_data)

    max_train_probs = np.max(train_prob, axis=1)

    # Set threshold dynamically (e.g., 5th percentile)
    threshold = np.percentile(max_train_probs, percentile)

    return threshold

def compute_stream_dynamic_threshold(list_prob_matrix_palmar:np.array, list_prob_matrix_dorsal:np.array, percentile: float) -> float:
    # Sum the probabilities of all the images
    sum_prob_palm = np.sum(list_prob_matrix_palmar, axis=0)
    sum_prob_dorsal = np.sum(list_prob_matrix_dorsal, axis=0)
    tot_prob_matrix = sum_prob_palm * 0.6 + sum_prob_dorsal * 0.4

    max_train_probs = np.max(tot_prob_matrix, axis=1)

    # Set threshold dynamically (e.g., 5th percentile)
    threshold = np.percentile(max_train_probs, percentile)

    return threshold

    """
    Compute the cross-entropy between two histograms.

    Args:
        h1 (np.ndarray): First input histogram.
        h2 (np.ndarray): Second input histogram.

    Returns:
        float: Cross-entropy between h1 and h2.
    """
    #####################################################
    ##                 YOUR CODE HERE                  ##
    #####################################################
    # Compute the cross-entropy between histograms
    ce = -np.sum(h1 * np.log(h2 + 1e-6))

    return ce