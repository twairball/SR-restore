import tensorflow as tf
import numpy as np
from keras.models import load_model

from models import create_espcnn_model, create_srcnn_model, create_resnet_up_model, create_espcnn_bn_model
from train import get_filenames, get_images, list_filenames

from utils import ycbcr2rgb, rgb2ycbcr

from scipy.misc import imsave

def pipeline(input_path, output_path, weights_path, network='espcnn', scale=4, batch_size=32, limit=None):
    # read input images
    X_filenames = list_filenames(input_path)

    X = get_images(X_filenames)
    input_shape = X[0].shape

    # model
    if (network == 'srcnn'):
        model = create_srcnn_model(input_shape, scale=scale)
    elif (network == 'resnet_up'):
        model = create_resnet_up_model(input_shape, scale=scale)
    elif (network == 'espcnn_bn'):
        model = create_espcnn_bn_model(input_shape, scale=scale)
    else:
        model = create_espcnn_model(input_shape, scale=scale)
    
    model.load_weights(weights_path)

    # predict
    preds = model.predict(X, batch_size=batch_size)

    # output filepaths
    Y_filenames = list_filenames(input_path, full_path=False)
    Y_filenames = [output_path + f for f in Y_filenames]

    # limit predictions
    if limit:
        Y_filenames = Y_filenames[:limit]

    # save images
    for index, out_filename in enumerate(Y_filenames):
        print("[%d] saving to: %s" % (index, out_filename))
        # no need to rescale since we are not normalizing for tanh
        out_img = ycbcr2rgb(preds[index])
        imsave(out_filename, out_img, format='png')


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description="Output model prediction images")
    parser.add_argument("input_path", type=str, help="Path to input images.")
    parser.add_argument("--network", type=str, default="espcnn", help="Network architecture, [srcnn|espcnn|resnet_up]. Default=espncnn")
    parser.add_argument("--weights", type=str,  help="Model weights filepath.")
    parser.add_argument("--output", type=str, help="Model save path.")
    parser.add_argument("--scale", type=int, default=4, help="Upscale factor. Default=4.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size. Default=32")
    parser.add_argument("--limit", type=int, help="Limit prediction to first N files.")

    args = parser.parse_args()

    input_path = args.input_path
    network = args.network
    output_path = args.output
    weights_path = args.weights
    scale = args.scale
    batch_size = args.batch_size
    limit = args.limit

    pipeline(input_path,
            network=network,
            output_path=output_path,
            weights_path=weights_path,
            scale=scale,
            batch_size=batch_size,
            limit=limit)
