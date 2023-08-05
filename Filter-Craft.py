
# # FILTER CRAFT: Creating Instagram style photo filters


import numpy as np
import cv2
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().run_line_magic('matplotlib', 'inline')

path = "scene.jpg"

img = mpimg.imread(path)

plt.rcParams['figure.figsize'] = [10,15]
plt.imshow(img)

#creating mapping function
def mapping_function(x,y):
    spl = UnivariateSpline(x,y)
    return spl(range(256))

def apply_warm(image):
    increase = mapping_function([0,64,128,192,256],[0,70,140,210,256])
    decrease = mapping_function([0,64,128,192,256],[0,40,90,150,256])
    red, green, blue = cv2.split(image)
    red = cv2.LUT(red, increase).astype(np.uint8)
    blue = cv2.LUT(blue, decrease).astype(np.uint8)
    image = cv2.merge((red, green, blue))
    return image

def apply_cool(image):
    increase = mapping_function([0,64,128,192,256],[0,70,140,210,256])
    decrease = mapping_function([0,64,128,192,256],[0,40,90,150,256])
    red, green, blue = cv2.split(image)
    red = cv2.LUT(red, decrease).astype(np.uint8)
    blue = cv2.LUT(blue, increase).astype(np.uint8)
    image = cv2.merge((red, green, blue))
    return image

def apply_grayscale(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def apply_sepia(image):
    sepia_filter = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
    sepia_image = cv2.transform(image, sepia_filter)
    sepia_image = np.clip(sepia_image, 0, 255).astype(np.uint8)
    return sepia_image

def apply_blur(image, kernel_size=(5, 5)):
    blurred_image = cv2.GaussianBlur(image, kernel_size, 0)
    return blurred_image

def apply_edge_detection(image, low_threshold=50, high_threshold=150):
    edges = cv2.Canny(image, low_threshold, high_threshold)
    edges_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return edges_image

def apply_emboss(image):
    kernel = np.array([[0, -1, -1],
                       [1, 0, -1],
                       [1, 1, 0]])
    embossed_image = cv2.filter2D(image, -1, kernel)
    return embossed_image


from ipywidgets import interact, interactive, fixed
import ipywidgets as widgets

def choice(x, img):
    if x == 'Warm':
        return plt.imshow(apply_warm(img))
    if x == 'Cool':
        return plt.imshow(apply_cool(img))
    if x == 'GrayScale':
        return plt.imshow(apply_grayscale(img))
    if x == 'Sepia':
        return plt.imshow(apply_sepia(img))
    if x == 'Blur':
        return plt.imshow(apply_blur(img))
    if x == 'Edge Detection':
        return plt.imshow(apply_edge_detection(img))
    if x == 'Emboss':
        return plt.imshow(apply_emboss(img))
    if x == 'No Filter':
        return plt.imshow(img)

interact(choice, x= widgets.Dropdown(options=['No Filter','Warm','Cool','Sepia','Blur','Edge Detection','Emboss','GrayScale'], description='Filter'), img=fixed(img))




