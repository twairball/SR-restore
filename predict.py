import tensorflow as tf
import numpy as np
from keras.models import load_model

from models import create_espcnn_model
from train import get_filenames, get_images, list_filenames

from utils import ycbcr2rgb, rgb2ycbcr

from scipy.misc import imsave

# lr_path = ''
# output_path = ''
# weights_file = 'results/espcnn_weights_20170613_182006.h5'


def pipeline(input_path, output_path, weights_path, scale=4, batch_size=32):
    # read input images
    X_filenames = list_filenames(input_path)

    X = get_images(X_filenames)
    input_shape = X[0].shape

    # model
    model = create_espcnn_model(input_shape, scale=scale)
    model.load_weights(weights_path)

    # predict
    preds = model.predict(X, batch_size=batch_size)

    # output filepaths
    Y_filenames = list_filenames(input_path, full_path=False)
    Y_filenames = [output_path + f for f in Y_filenames]

    # save images
    for index, out_filename in enumerate(Y_filenames):
        print("[%d] saving to: %s" % (index, out_filename))
        out_img = ycbcr2rgb(preds[index] * 255.)  # rescale from normalized array
        imsave(out_filename, out_img, format='png')


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description="Output model prediction images")
    parser.add_argument("input_path", type=str, help="Path to input images.")
    parser.add_argument("--weights", type=str,  help="Model weights filepath.")
    parser.add_argument("--output", type=str, help="Model save path.")
    parser.add_argument("--scale", type=int, default=4, help="Upscale factor. Default=4.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size. Default=32")

    args = parser.parse_args()

    input_path = args.input_path
    weights_path = args.weights
    output_path = args.output
    scale = args.scale
    batch_size = args.batch_size


    pipeline(input_path,
            output_path=output_path,
            weights_path=weights_path,
            scale=scale,
            batch_size=batch_size)
