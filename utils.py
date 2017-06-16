import numpy as np

##
## color space 
## 
def rgb2ycbcr(im):
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = im.dot(xform.T)
    ycbcr[:,:,[1,2]] += 128
    return np.uint8(ycbcr)

def ycbcr2rgb(im):
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im.astype(np.float)
    rgb[:,:,[1,2]] -= 128
    return np.uint8(rgb.dot(xform.T))


##
## VGG
## 
def preprocess_vgg(input):
    rn_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)
    preproc = lambda x: (x - rn_mean)[:, :, :, ::-1]
    return preproc
