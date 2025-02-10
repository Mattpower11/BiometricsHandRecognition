import numpy as np
from sklearn.metrics import auc, confusion_matrix, roc_curve
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


def calculate_FAR_plot(predicted_scores: np.ndarray, true_labels: np.ndarray, type_feature_extractor: str, palm_dorsal: str):
    # Normalizzazione delle etichette: 1 = genuine, 0 = impostor
    true_labels_binary = np.where(true_labels != -1, 1, 0)
    predicted_score = np.max(predicted_scores, axis=1)

    # Numero reale di immagini di impostori
    num_impostor_images = np.sum(true_labels_binary == 0)
    
    # Calcolo di FAR per diverse soglie
    thresholds = np.linspace(0, 1, 1000)  # Soglie tra 0 e 1
    far_values = []

    for threshold in thresholds:
        predicted_labels = predicted_score >= threshold  # Etichette predette
        false_positives = np.sum((predicted_labels == 1) & (true_labels_binary == 0))
        far = false_positives / num_impostor_images if num_impostor_images > 0 else 0
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

    return far_values

    
def calculate_FRR_plot(predicted_scores: np.ndarray, true_labels: np.ndarray, type_feature_extractor: str, palm_dorsal: str):
    # Normalizzazione delle etichette: 1 = genuine, 0 = impostor
    true_labels_binary = np.where(true_labels != -1, 1, 0)
    predicted_score = np.max(predicted_scores, axis=1)

    # Numero reale di immagini genuine
    num_genuine_images = np.sum(true_labels_binary)

    # Calcolo di FAR per diverse soglie
    thresholds = np.linspace(0, 1, 1000)  # Soglie tra 0 e 1
    frr_values = []

    for threshold in thresholds:
        predicted_labels = predicted_score >= threshold  # Etichette predette
        false_negatives = np.sum((predicted_labels == 0) & (true_labels_binary == 1))
        frr = false_negatives / num_genuine_images if num_genuine_images > 0 else 0
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

    return frr_values

def plot_FAR_FRR(far_values: np.array, frr_values: np.array, type_feature_extractor: str, palm_dorsal: str):  
    # Definiamo le soglie tra 0 e 1 (1000 valori per precisione)
    thresholds = np.linspace(0, 1, 1000)  

    # Calcolo dell'EER: punto in cui FAR ≈ FRR
    eer_index = np.nanargmin(np.abs(np.array(far_values) - np.array(frr_values)))  
    eer_threshold = thresholds[eer_index]
    eer_value = far_values[eer_index]  # EER = FAR (o FRR) al punto di intersezione

    # Plot delle curve
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, far_values, label=f"{type_feature_extractor} {palm_dorsal} FAR (False Acceptance Rate)", color='blue')
    plt.plot(thresholds, frr_values, label=f"{type_feature_extractor} {palm_dorsal} FRR (False Rejection Rate)", color='red')
    
    # Evidenziamo l'EER con un punto e una linea tratteggiata
    plt.axvline(eer_threshold, color='green', linestyle='--', label=f"EER = {eer_value:.3f} @ threshold {eer_threshold:.3f}")
    plt.scatter(eer_threshold, eer_value, color='green', s=100, label="Equal Error Rate (EER)")

    # Etichette e legenda
    plt.xlabel("Threshold", fontsize=14)
    plt.ylabel("Rate", fontsize=14)
    plt.title("FAR vs FRR con evidenziazione EER", fontsize=16)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Mostra il grafico
    plt.show()

    print(f"Equal Error Rate (EER): {eer_value:.3f} a soglia {eer_threshold:.3f}")

def calculate_CMC(score_matrix: np.ndarray, true_labels: np.ndarray, gallery_labels: np.ndarray):
    num_tests, num_gallery = score_matrix.shape
    ranks = np.zeros(num_gallery)
    num_samples_skipped = 0  # Contatore per test senza corrispondenza

    for i in range(num_tests):
        #print(f"Test {i}")

        # Estrai la riga i (punteggi per il test i)
        scores = score_matrix[i]  
        #print(scores)

        # Ordina i punteggi in ordine decrescente
        sorted_indices = np.argsort(scores)[::-1]
        #print(sorted_indices)

        # Ordina le etichette della galleria
        sorted_labels = gallery_labels[sorted_indices] 
             
        correct_rank = np.where(sorted_labels == true_labels[i])[0]

        # Se l'identità non è nella galleria
        if correct_rank.size == 0:  
            print(f"Errore: impossibile trovare {true_labels[i]} in sorted_labels!")
            num_samples_skipped += 1
            continue  # Salta all'iterazione successiva

        # Seleziona il primo rank corretto
        correct_rank = correct_rank[0]

        # Incrementa dal rank corretto fino alla fine
        ranks[correct_rank:] += 1

    # Normalizza la curva CMC in percentuale
    cmc_curve = ranks / (num_tests - num_samples_skipped)
    #print(f"Numero di campioni ignorati: {num_samples_skipped}")

    return cmc_curve


def calculate_CMC_plot(score_matrix: np.ndarray, true_labels: np.ndarray, gallery_labels: np.ndarray, type_feature_extractor: str, palm_dorsal: str):
    cmc_curve = calculate_CMC(score_matrix, true_labels, gallery_labels)
    ranks = np.arange(1, len(cmc_curve) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(ranks, cmc_curve, marker='o', linestyle='-', color='green', label=f"CMC {type_feature_extractor} {palm_dorsal}")
    plt.title("Cumulative Match Characteristic (CMC) Curve", fontsize=16)
    plt.xlabel("Rank", fontsize=14)
    plt.ylabel("Recognition Rate", fontsize=14)
    # Mostra solo alcuni tick
    plt.xticks(ranks[::max(len(ranks)//10, 1)])  
    # Da 0 a 1 con step di 0.1
    plt.yticks(np.linspace(0, 1, 11))  
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=12)
    plt.show()


def plot_OSROC_curve(true_labels: np.ndarray, predicted_scores: np.ndarray, known_classes: np.ndarray, type_feature_extractor: str, palm_dorsal: str):
    # Creiamo le etichette binarie per l'open set
    is_known = np.isin(true_labels, known_classes)  # 1 se è una classe nota, 0 se è un impostore

    # Score più alto tra le classi conosciute per ogni campione
    max_scores = predicted_scores.max(axis=1)

    # Calcoliamo la OSROC Curve
    fpr, tpr, _ = roc_curve(is_known, max_scores)
    
    # Area sotto la curva (AUC-OSROC)
    roc_auc = auc(fpr, tpr)  

    # Disegniamo la OSROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'{type_feature_extractor} {palm_dorsal} OSROC Curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Linea casuale
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=14)
    plt.ylabel('True Positive Rate (TPR)', fontsize=14)
    plt.title('Open-Set ROC Curve (OSROC)', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
