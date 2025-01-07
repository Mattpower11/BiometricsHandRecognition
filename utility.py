import math
from typing import List, Tuple

import numpy as np
from skimage.color import rgb2gray
from scipy.ndimage import convolve1d

def compute_histograms(
    image_list: List[np.ndarray], hist_type: str, hist_isgray: bool, num_bins: int
) -> List[np.ndarray]:
    """
    Compute the histogram for each image in image_list.

    Args:
        image_list (List[np.ndarray]): list of images.
        hist_type (str): histogram type.
        hist_isgray (bool): True if the histogram is in gray_scale, False otherwise.
        num_bins (int): number of bins for the gray_scale histogram.

    Returns:
        image_hist (List[np.ndarray]): list of histograms.
    """
    #####################################################
    ##                 YOUR CODE HERE                  ##
    #####################################################

    #per ogni immagine nella lista di immagini converte in gray se necessario e calcola l'istogramma con la funzione del professore
    #ritorna la lista di istogrammi delle immagini

    image_hist = [] 
    for img in image_list: 
        #if hist_isgray:
        #    img = rgb2gray(img)
        hist = get_hist_by_name(img, num_bins, hist_type)
        image_hist.append(hist)

    return image_hist


def is_grayvalue_hist(hist_name: str) -> bool:
    """
    Handle function to discriminate when your input
    function is in gray_scale or colors.

    Args:
        hist_name (str): histogram name.

    Returns:
        bool: True if the histogram is in gray_scale, False otherwise.
    """
    if hist_name == "grayvalue" or hist_name == "dxdy":
        return True
    elif hist_name == "rgb" or hist_name == "gb":
        return False
    else:
        assert False, "unknown histogram type"

def get_hist_by_name(img: np.ndarray, num_bins_gray: int, hist_name: str) -> np.ndarray:
    """
    Handle function to get the correct historgam function
    by his name.

    Args:
        img (np.ndarray): input image.
        num_bins_gray (int): number of bins for the gray_scale histogram.

    Returns:
        np.ndarray: histogram.
    """
    # if hist_name == "rgb":
    #     return rgb_hist(img, num_bins_gray)
    if hist_name == "gb":
        return gb_hist(img, num_bins_gray)
    elif hist_name == "dxdy":
        return hist_dxdy(img, num_bins_gray)
    else:
        assert False, "unknown hist type: %s" % hist_name

def gb_hist(img_color_double: np.ndarray, num_bins: int = 5) -> np.ndarray:
    """
    Compute the *joint* histogram for the G and B color channels in the image.
    The histogram should be normalized so that sum of all values equals 1,
    assume that values in each channel vary between 0 and 255

    Args:
        img_color_double (np.ndarray): Input color image.
        num_bins (int): Number of bins used to discretize each channel, total number of bins in the histogram should be num_bins^2.

    Returns:
        hists (np.ndarray): Joint histogram.

    E.g. hists[0,9] contains the number of image_color pixels such that:
        - their G values fall in bin 0
        - their B values fall in bin 9
    """

    G_channel = img_color_double[:, :, 1]  
    B_channel = img_color_double[:, :, 2] 
    
    bin_edges = np.linspace(0, 256, num_bins + 1)  # 256 to include 255 in the last bin
    
    G_digitized = np.digitize(G_channel, bin_edges) - 1  # bin index starts from 0
    B_digitized = np.digitize(B_channel, bin_edges) - 1
    
    # initialize histogram
    hists = np.zeros((num_bins, num_bins))
    
    # populate histogram
    for i in range(G_channel.shape[0]):
        for j in range(G_channel.shape[1]):
            G_bin = G_digitized[i, j]
            B_bin = B_digitized[i, j]
            hists[G_bin, B_bin] += 1
    
    # normalization
    hists /= np.sum(hists)

    hists = hists.flatten()
    
    return hists

def gauss_dxdy(img: np.ndarray, sigma: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function applies the first derivative of the 1D Gaussian operator to the image in the x and y directions.

    Args:
        img (np.ndarray): the input image.
        sigma (float): the standard deviation of the Gaussian filter.

    Returns:
        img_Dx (np.ndarray): the image after applying the first derivative of the 1D Gaussian operator in the x direction.
        img_Dy (np.ndarray): the image after applying the first derivative of the 1D Gaussian operator in the y direction.
    """
    #####################################################
    ##                 FIX THIS CODE                   ##
    #####################################################
    Gx, _ = gauss(sigma, filter_size=6*sigma)
    Dx, _ = gaussdx(sigma)

    img_Dx = convolve1d(convolve1d(img, Dx, axis=1),Gx, axis=0) #1st line with error
    img_Dy = convolve1d(convolve1d(img, Dx, axis=0),Gx, axis=1) #2nd line with error



    return img_Dx, img_Dy

def gaussdx(sigma: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function computes the first derivative of the 1D Gaussian operator.

    Args:
        sigma (float) : the standard deviation of the Gaussian filter

    Returns:
        Dx (np.ndarray): the first derivative of the 1D Gaussian operator
        x (np.ndarray): the indexes of the 1D Gaussian operator
    """
    #####################################################
    ##              FIX THIS CODE (2 ERRORS)           ##
    #####################################################
    
    sigma = math.ceil(sigma)
    filter_size = 3 * sigma + 1

    # Generate the index x
    zero_pos = 3 * filter_size  # the center of the filter

    x = np.arange(-zero_pos/3 +1, zero_pos/3)  # indexes from -3*sigma to 3*sigma

    # Compute the Gaussian curve with std-dev sigma at the indexes x
    Dx = -(x * np.exp(-(x**2) / (2.0 * sigma**2)) / (math.sqrt(2.0 * np.pi) * sigma**3)) # error missing - and **3

    return Dx, x

def gauss(sigma: float, filter_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the Gaussian filter.

    Args:
        sigma (int): Standard deviation.
        filter_size (int): Size of the filter.

    Returns:
        Gx (np.ndarray): Gaussian filter.
        x (np.ndarray): Array of integer values.
    """
    #x = np.arange(-filter_size/2, math.floor(filter_size/2) if filter_size%2==1 else math.floor(filter_size/2)+1 ,1, dtype=int)                                                              
    
    x = np.arange(-filter_size//2 + 1, filter_size//2 + 1, 1, dtype=int)

    Gx = (1/(np.sqrt(2*np.pi)*sigma))*np.exp(-((x**2)/(2*(sigma**2)))) 

    print(x.size)
    
    return Gx, x

def hist_dxdy(img_gray, num_bins=5):
    """
    This function computes the *joint* histogram of Gaussian partial derivatives of the image in x and y direction.
    Set sigma to 3.0 and cap the range of derivative values is in the range [-6, 6].
    The histogram should be normalized so that the sum of all values equals 1.

    Args:
        img_gray: the input image
        num_bins: number of bins used to discretize each dimension, total number of bins in the histogram should be num_bins^2

    Returns:
        hists: the joint normalized histogram of Gaussian partial derivatives of the image in x and y direction
    """

    assert len(img_gray.shape) == 2, "image dimension mismatch"
    assert img_gray.dtype == "float", "incorrect image type"

    #####################################################
    ##                 FIX THIS CODE                   ##
    #####################################################
    
    # Compute the first derivatives of img_gray
    sigma = 3.0
    img_dx, img_dy = gauss_dxdy(img_gray, sigma)

    # Set the min_der and max_der to -6 and 6, which defines the ranges for quantization
    min_der, max_der = (-6, 6)

    # Flatten the 2D derivative images to 1D arrays
    img_dx = img_dx.reshape(-1)
    img_dy = img_dy.reshape(-1)

    # Clip the min and max values to min_der and max_der respectively
    img_dx = np.clip(img_dx, min_der, max_der) #errore 1 rimosso + max_der perché non serve
    img_dy = np.clip(img_dy, min_der, max_der)  

    # Define the range for quantization
    hists = np.zeros((num_bins, num_bins), dtype=int)
    bin_range = (max_der - min_der) / num_bins
    
    # Quantize image derivative values into bins
    bin_dx = np.floor((img_dx - min_der) / bin_range).astype(int)  # normalizzato range bin
    bin_dy = np.floor((img_dy - min_der) / bin_range).astype(int)  # normalizzato range bin
    bin_dx = np.clip(bin_dx, 0, num_bins - 1)
    bin_dy = np.clip(bin_dy, 0, num_bins - 1)

    # Compute the joint histogram
    for i in range(bin_dx.size):
        hists[bin_dx[i], bin_dy[i]] += 1

    # Normalize the histogram so that the sum equals 1
    hists = (hists.flatten() / hists.flatten().sum()).astype(float) #errore 2 normalizzato


    return hists

def get_dist_by_name(x: np.ndarray, y: np.ndarray, dist_name: str) -> float:
    """
    Handle function to get the correct distance function
    by his name.

    Args:
        x (np.ndarray): input histogram.
        y (np.ndarray): input histogram.

    Returns:
        float: distance.
    """
    if dist_name == "chi2":
        return hist_chi2(x, y)
    elif dist_name == "intersect":
        return 1 - hist_intersect(x, y)
    elif dist_name == "l2":
        return hist_l2(x, y)
    elif dist_name == "ce":
        return hist_ce(x, y)
    elif dist_name == "all":
        pass
    else:
        assert False, "unknown distance: %s" % dist_name

def hist_intersect(h1: np.ndarray, h2: np.ndarray) -> float:
    """
    Compute the intersection between histograms x and y.
    Check that the distance range is [0,1].

    Args:
        h1 (np.ndarray): Input histogram.
        h2 (np.ndarray): Input histogram.

    Returns:
        x (float): Intersection distance between histograms x and y.
    """

    #####################################################
    ##                 YOUR CODE HERE                  ##
    #####################################################
    # Compute the intersection between histograms
    x = 1/2 * (np.sum(np.minimum(h1, h2))/np.sum(h1) + np.sum(np.minimum(h1, h2))/np.sum(h2))

    return x

def hist_l2(h1: np.ndarray, h2: np.ndarray) -> float:
    """
    Compute the L2 between x and y histograms.
    Check that the distance range in [0,sqrt(2)].

    Args:
        h1 (np.ndarray): Input histogram.
        h2 (np.ndarray): Input histogram.

    Returns:
        x (float): L2 distance between x and y histograms.
    """

    #####################################################
    ##                 YOUR CODE HERE                  ##
    #####################################################
    # Compute the L2 distance between histograms
    x = np.sqrt(np.sum((h1 - h2)**2))

    return x

def hist_chi2(h1: np.ndarray, h2: np.ndarray) -> float:
    """
    Compute chi2 between x and y.
    Check that the distance range in [0,Inf].

    Args:
        h1 (np.ndarray): Input histogram.
        h2 (np.ndarray): Input histogram.

    Returns:
        x (float): Chi2 distance between x and y.
    """

    #####################################################
    ##                 YOUR CODE HERE                  ##
    #####################################################
    # Compute the chi2 distance between histograms
    x = np.sum(((h1-h2)**2)/(h1+h2 + 1e-10))

    return x

def hist_ce(h1: np.ndarray, h2: np.ndarray) -> float:
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