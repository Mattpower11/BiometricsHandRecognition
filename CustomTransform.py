import cv2
import numpy as np
from PIL import Image
from torch import Size 
from torchvision import transforms

# Custom transformation for HOG
class CustomHOGTransform:
    def __init__(self, ksize:Size, sigma:float, isPalm:bool=False):
        self.ksize=ksize
        self.sigma=sigma
        self.isPalm=isPalm

    def __call__(self, pil_image):
        # Convert PIL -> RGB -> NumPy
        if self.isPalm:
            pil_image = pil_image.resize((150, 150))
        else:
            pil_image = pil_image.resize((1024, 1024))
        image = pil_image.convert('RGB')
        image = np.array(pil_image, dtype=np.uint8)
        gaussian_blurred = cv2.GaussianBlur(image, self.ksize, self.sigma) 
        return Image.fromarray(gaussian_blurred, mode='RGB')
    
# Custom transformation for LBP
class CustomLBPTransform:
    def __init__(self, isPalm:bool=False):
        self.isPalm = isPalm

    def __call__(self, image):
        if self.isPalm:
            image = cv2.resize(np.array(image), (150, 150))
        grigio = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
        contrasto = cv2.equalizeHist(np.array(grigio))
        return Image.fromarray(contrasto, mode='L')

class CustomLBPCannyTransform:
    def __init__(self, isPalm:bool=False):
        self.isPalm = isPalm

    def __call__(self, image):
        if self.isPalm:
            image = cv2.resize(np.array(image), (150, 150))
        grigio = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        contrasto = cv2.equalizeHist(np.array(grigio))
        canny = cv2.Canny(contrasto, 50, 200, apertureSize=7, L2gradient = True)
        return Image.fromarray(canny, mode='L')

# To normalize one image [values range 0:1]
def imageNormalization(image: np.ndarray):
    # E.g., convert from [0..255] to [0..1] float
    return image.astype(np.float32) / 255.0

# To restore the original pixel scale -> cast on int 
def restoreOriginalPixelValue(image: np.ndarray):
    # Convert from [0..1] float back to [0..255] uint8
    return (image * 255).astype(np.uint8)

def buildCustomTransform(transform:type):
    return transforms.Compose([
        transform(),
        transforms.ToTensor(),          
    ])

def buildCustomTransformHogExtended(transform:type, ksize:Size, sigma:float):
    return transforms.Compose([
        transform(ksize=ksize, sigma=sigma),
        transforms.ToTensor(),          
    ])

def buildCustomTransformPalmExtended(transform:type, isPalm:bool):
    return transforms.Compose([
        transform(isPalm=isPalm),
        transforms.ToTensor(),          
    ])

def buildCustomTransformHogPalmExtended(transform:type, ksize:Size, sigma:float, isPalm:bool):
    return transforms.Compose([
        transform(ksize=ksize, sigma=sigma, isPalm=isPalm),
        transforms.ToTensor(),          
    ])