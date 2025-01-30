import cv2
import numpy as np
from PIL import Image
from torch import Size 
from torchvision import transforms

# Custom transformation for AlexNet
class CustomAlexNetTransform:
    def __call__(self, pil_image):
        # Convert PIL -> RGB -> NumPy
        pil_image = pil_image.convert('RGB')
        np_image = np.array(pil_image, dtype=np.uint8)

        # Optional: normalizing to [0..1] before blur
        np_image_norm = imageNormalization(np_image)

        # Blur
        blurred = cv2.GaussianBlur(np_image_norm, (7, 7), 0)

        # Resize to 224×224
        resized = cv2.resize(blurred, (224, 224))

        # Convert back to 0..255
        final_8u = restoreOriginalPixelValue(resized)  # shape: (224, 224, 3)

        # Return PIL image (mode='RGB')
        return Image.fromarray(final_8u, mode='RGB')


# Custom transformation for LeNet
class CustomLeNetTransform:
    def __call__(self, pil_image):
        # Convert PIL -> RGB -> NumPy
        pil_image = pil_image.convert('RGB')
        np_image = np.array(pil_image, dtype=np.uint8)

        # Convert to GRAY correctly
        gray_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)

        # Equalize hist
        contrast = cv2.equalizeHist(gray_image)

        # Normalize [0..255] -> [0..1]
        norm = imageNormalization(contrast)

        # Resize to 32×32
        resized = cv2.resize(norm, (32, 32))

        # Convert back to [0..255] uint8
        final_8u = restoreOriginalPixelValue(resized)  # shape: (32, 32)

        # Return PIL image (mode='L' = single channel)
        return Image.fromarray(final_8u, mode='L')


# Custom transformation for LeNet
class CustomLBPTransform:
    counter = 0
    def __call__(self, pil_image):
        # 1. Converti l'immagine in scala di grigi
        grigio = cv2.cvtColor(np.array(pil_image), cv2.COLOR_BGR2GRAY)

        # 2. Applica un filtro per ridurre il rumore
        sfocata = cv2.GaussianBlur(grigio, (5, 5), 0)

        # 3. Rileva i bordi con Canny
        bordi = cv2.Canny(sfocata, 50, 150)

        # 4. Trova i contorni
        contorni, _ = cv2.findContours(bordi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 5. Trova il contorno più grande (presumibilmente quello del soggetto)
        contorno_piu_grande = max(contorni, key=cv2.contourArea)

        # 6. Estrai il rettangolo di delimitazione del contorno
        x, y, w, h = cv2.boundingRect(contorno_piu_grande)

        # 7. Ritaglia l'immagine
        immagine_ritagliata = grigio[y:y+h, x:x+w]

        immagine_ritagliata = cv2.resize(np.array(immagine_ritagliata), (1600, 1200))
        CustomLBPTransform.counter += 1
        cv2.imwrite(f"./img/immagine_salvata{CustomLBPTransform.counter}.png", immagine_ritagliata)
        return immagine_ritagliata
    
# Custom transformation for HOG
class CustomHOGTransform:
    def __init__(self, ksize:Size, sigma:float):
        self.ksize=ksize
        self.sigma=sigma

    def __call__(self, pil_image):
        # Convert PIL -> RGB -> NumPy
        image = pil_image.convert('RGB')
        image = np.array(pil_image, dtype=np.uint8)
        gaussian_blurred = cv2.GaussianBlur(image, self.ksize, self.sigma) 
        
        image = cv2.resize(gaussian_blurred, (1024, 1024))
        return Image.fromarray(image, mode='RGB')
    
# To normalize one image [values range 0:1]
def imageNormalization(image: np.ndarray):
    # E.g., convert from [0..255] to [0..1] float
    return image.astype(np.float32) / 255.0

# To restore the original pixel scale -> cast on int 
def restoreOriginalPixelValue(image: np.ndarray):
    # Convert from [0..1] float back to [0..255] uint8
    return (image * 255).astype(np.uint8)

# Build AlexNet trasformations
def buildAlexNetTransformations():
    return transforms.Compose([
        CustomAlexNetTransform(),
        transforms.ToTensor(),         
    ])

# Build LeNet trasformations
def buildLeNetTransformations():
    return transforms.Compose([
        CustomLeNetTransform(),
        transforms.ToTensor(),          
    ])

# Build LBP trasformations
def buildLBPTransformations():
    return transforms.Compose([
        CustomLBPTransform(),
        transforms.ToTensor(),          
    ])

# Build HOG trasformations
def buildHOGTransformations(ksize:Size, sigma:float):
    return transforms.Compose([
        CustomHOGTransform(ksize=ksize, sigma=sigma),
        transforms.ToTensor(),          
    ])

# Build a histogram transformation
def buildHistogramTransformations():
    return transforms.Compose([
        transforms.ToTensor(),          
    ])