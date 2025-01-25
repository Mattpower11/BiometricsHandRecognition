import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_accuracy(y_pred, y_true):
    return sum(y_pred == y_true) / len(y_true)

def calculate_confusion_matrix(y_pred, y_true):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Plot confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def calculate_f1_score(y_pred, y_true):

    cm = confusion_matrix(y_true, y_pred)
    # Compute F1 score
    FP = cm[0, 1]
    FN = cm[1, 0]
    TP = cm[1, 1]
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * (precision * recall) / (precision + recall)

    return f1

def calculate_precision(y_pred, y_true):

    cm = confusion_matrix(y_true, y_pred)
    # Compute precision
    FP = cm[0, 1]
    TP = cm[1, 1]
    precision = TP / (TP + FP)

    return precision

def calculate_recall(y_pred, y_true):
    
    cm = confusion_matrix(y_true, y_pred)

    # Compute recall
    FN = cm[1, 0]
    TP = cm[1, 1]
    recall = TP / (TP + FN)

    return recall

def calculate_loss_plot(train_loss):
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss, 'bo-', label='Training loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def calculate_FAR_plot(predicted_scores: np.ndarray, true_labels: np.ndarray, num_impostors: int, type_feature_extractor: str, palm_dorsal: str):
    # Calcolo di FAR per diverse soglie
    thresholds = np.linspace(0, 1, 1000)  # Soglie tra 0 e 1
    far_values = []

    for threshold in thresholds:
        predicted_labels = predicted_scores >= threshold  # Etichette predette
        false_positives = np.sum((predicted_labels == 1) & (true_labels == 0))
        far = false_positives / num_impostors if num_impostors > 0 else 0
        far_values.append(far)

    # Creazione del grafico
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, far_values, label=f"{type_feature_extractor} {palm_dorsal} FAR", color='blue')
    plt.title('False Alarm Rate (FAR) vs Threshold', fontsize=16)
    plt.xlabel('Threshold', fontsize=14)
    plt.ylabel('False Alarm Rate (FAR)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.show()

    
def calculate_FRR_plot(predicted_scores: np.ndarray, true_labels: np.ndarray, num_genuines: int, type_feature_extractor: str, palm_dorsal: str):
    # Calcolo di FAR per diverse soglie
    thresholds = np.linspace(0, 1, 1000)  # Soglie tra 0 e 1
    frr_values = []

    for threshold in thresholds:
        predicted_labels = predicted_scores <= threshold  # Etichette predette
        false_negatives = np.sum((predicted_labels == 0) & (true_labels == 1))
        frr = false_negatives / num_genuines if num_genuines > 0 else 0
        frr_values.append(frr)

    # Creazione del grafico
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, frr_values, label=f"{type_feature_extractor} {palm_dorsal} FRR", color='red')
    plt.title('False Rejection Rate (FRR) vs Threshold', fontsize=16)
    plt.xlabel('Threshold', fontsize=14)
    plt.ylabel('False Rejection Rate (FRR)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.show()



def calculate_CMC_plot(rank_matrix: np.ndarray, type_feature_extractor: str, palm_dorsal: str):
    # Compute Cumulative Match Characteristic (CMC)
    cmc = np.zeros(rank_matrix.shape[1])
    for i in range(rank_matrix.shape[1]):
        cmc[i] = rank_matrix[:, i].sum() / rank_matrix.shape[0]

    # Plot CMC
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, rank_matrix.shape[1] + 1), cmc, 'bo-', label=f"{type_feature_extractor} {palm_dorsal} CMC")
    plt.title('Cumulative Match Characteristic')
    plt.xlabel('Rank')
    plt.ylabel('CMC')
    plt.legend()
    plt.show()
