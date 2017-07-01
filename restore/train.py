# -*- coding: utf-8 -*-

from .models import restore_cnn_model, restore_cnn_bn_model
from sr.train import Pipeline, get_image_shape, get_count, steps_for_batch_size, image_pair_generator, PSNRLoss

class RestorePipeline(Pipeline):

    def run(self, epochs=100, batch_size=32, save=True):
        orig_path = self.root_dir + 'original/'
        enh_path = self.root_dir + 'enhanced/'
        input_shape = get_image_shape(orig_path)
        output_shape = get_image_shape(enh_path)
        image_count = get_count(orig_path)

        print("[TRAIN] orig %s ==> enhanced %s. (%s images)" % (input_shape, output_shape, image_count))

        # model
        if (self.network == "cnn_bn"):
            model = restore_cnn_bn_model(input_shape)
        else:
            model = restore_cnn_model(input_shape)

        model.compile(loss='mse', optimizer='adam', metrics=[PSNRLoss])

        # callbacks
        callbacks = self.get_callbacks()

        # train
        gen = image_pair_generator(enh_path, orig_path)
        steps = steps_for_batch_size(enh_path, batch_size)
        model.fit_generator(gen, steps, epochs=epochs, callbacks=callbacks)

        # save
        if (save):
            model_path = self.results_dir + "model.h5"
            model.save(model_path)



if __name__ == '__main__':

    import argparse
    import timeit
    parser = argparse.ArgumentParser(description="Train RESTORE model.")
    parser.add_argument("image_path", type=str, help="Path to input images, expects sub-directories /path/original/ and /path/enhanced/.")
    parser.add_argument("--results", type=str, default="results/restore/", help="Results base dir, will create subdirectories e.g. /results/model_timestamp/")
    parser.add_argument("--network", type=str, default="cnn", help="Network architecture. Default=cnn")
    parser.add_argument("--epochs", type=int, default=100, help="Epochs. Default=100")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size. Default=32")

    args = parser.parse_args()

    image_path = args.image_path
    results_path = args.results
    network = args.network
    epochs = args.epochs
    batch_size = args.batch_size

    # training pipeline
    p = RestorePipeline(image_path, results_path, network=network)

    start_time = timeit.default_timer()
    p.run(epochs=epochs, batch_size=batch_size)
    duration = timeit.default_timer() - start_time
    print("[RESTORE Train] time taken: %s" % duration)