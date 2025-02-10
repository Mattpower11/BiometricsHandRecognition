import numpy as np
from skimage.feature import hog
from torch.utils.data import DataLoader
from CustomImageDataset import CustomImageDataset
from skimage.feature import local_binary_pattern

def extract_LBP_features(image_path:str, data_struct:dict, palmar_dorsal:str, train_test:str, num_points:int, radius:int, method:str, batch_size:int, transforms):
    features = []

    dataset = CustomImageDataset(image_dir=image_path, data_structure = data_struct, train_test=train_test, palmar_dorsal=palmar_dorsal, transform=transforms, action=False)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for images, _ in data_loader:
        for image in images:
            if palmar_dorsal == 'dorsal':
                lbp = local_binary_pattern(np.array(image, dtype=np.int64).reshape(1600,1200), num_points, radius, method=method) 
            else:
                lbp = local_binary_pattern(np.array(image, dtype=np.int64).reshape(150,150), num_points, radius, method=method) 
                     
            n_bins = int(lbp.max() + 1)                  
            (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_bins), range=(0, n_bins))                                                 
                            
            hist = hist.astype("float")                         
            hist /= (hist.sum() + 10e-6)
            features.append(hist)
        
    return features


def extract_HOG_features(image_path: str, data_struct: dict, palmar_dorsal: str, train_test: str, orientations: int, pixels_per_cell: int, cells_per_block: int, batch_size: int, block_norm="L2-Hys", transforms=None):
    features = []

    dataset = CustomImageDataset(image_dir=image_path, data_structure=data_struct, train_test=train_test, palmar_dorsal=palmar_dorsal, action=False, transform=transforms)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for images, _ in data_loader:
        for image in images:
            image = np.array(image)  # Convert PIL Image to NumPy array
            image = np.transpose(image, (1, 2, 0))

            # Extract HOG features
            hog_features = hog(
                image,
                orientations=orientations,
                pixels_per_cell=(pixels_per_cell, pixels_per_cell),
                cells_per_block=(cells_per_block, cells_per_block),
                transform_sqrt=True,
                block_norm=block_norm,
                channel_axis=-1
            )
            features.append(hog_features)
        
    return features
