from scipy.misc import imsave

from sr.train import get_images, list_filenames
from sr.utils import ycbcr2rgb
from .models import restore_cnn_bn_model, restore_cnn_model

def pipeline(input_path, output_path, weights_path, network='cnn', batch_size=32, limit=None):
    # read input images
    X_filenames = list_filenames(input_path)

    X = get_images(X_filenames)
    input_shape = X[0].shape

    # model
    if (network == "cnn_bn"):
        model = restore_cnn_bn_model(input_shape)
    else:
        model = restore_cnn_model(input_shape)

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
    parser.add_argument("--network", type=str, default="espcnn",
                        help="Network architecture, [srcnn|espcnn|resnet_up]. Default=espncnn")
    parser.add_argument("--weights", type=str, help="Model weights filepath.")
    parser.add_argument("--output", type=str, help="Model save path.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size. Default=32")
    parser.add_argument("--limit", type=int, help="Limit prediction to first N files.")

    args = parser.parse_args()

    input_path = args.input_path
    network = args.network
    output_path = args.output
    weights_path = args.weights
    batch_size = args.batch_size
    limit = args.limit

    pipeline(input_path,
             network=network,
             output_path=output_path,
             weights_path=weights_path,
             batch_size=batch_size,
             limit=limit)
