# SR-Restore

Super resolution and Image restoration.

## Install

Requires `tensorflow==1.1.0` and `keras==2.0.2`

    $ pip install -r ./requirements.txt


## Super Resolution example

````
    # Resize low-res and high-res images for Super Resolution
    $ python -m sr.images data/images/raw/ \
        --lr data/images/lr \
        --hr data/images/hr \
        --lr_size 72x72 \
        --hr_size 288x288

    # Train Super Resolution model.
    # Expects images in /path/images/lr/ and /path/images/hr/
    $ python -m sr.train data/images \
        --results results/ \
        --network espcnn \
        --scale 4 \
        --epochs 100 \
        --batch_size 32

    # Predict high-res images using trained model weights.
    # (Optional) use `--limit` option to limit num of images to perform on.
    $ python -m sr.predict data/images/lr/ \
        --output results/output/ \
        --weights results/espcnn_20170701_190534/weights.h5 \
        --network espcnn \
        --scale 4 \
        --batch_size 32 \
        --limit 20
````


## Image Restore example


````
    # Resize a directory of images to single size
    # We need /image_path/original/ and /image_path/enhanced/
    $ python -m restore.images data/images/raw_original \
        --output data/images/original \
        --size 288x288

    $ python -m restore.images data/images/raw_enhanced \
        --output data/images/enhanced \
        --size 288x288


    # Train Restore model.
    # Expects images in /path/images/original/ and /path/images/enhanced/
    # Saves model weights, checkpoints, and tensorboard logs to path specified in results.
    $ python -m sr.train data/images \
        --results results/ \
        --network cnn \
        --epochs 100 \
        --batch_size 32


    # Predict original images using trained model weights.
    # (Optional) use `--limit` option to limit num of images to perform on.
    $ python -m sr.predict data/images/enhanced/ \
        --output results/output/ \
        --weights results/cnn_20170701_190534/weights.h5 \
        --network cnn \
        --batch_size 32 \
        --limit 20

````


## Dataset

celebA: (link)


## References

    TODO