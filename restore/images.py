from PIL import Image
import os
from resizeimage import resizeimage

from sr.images import resize_images, str_to_size


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Resize a directory of images. ")
    parser.add_argument("image_path", type=str, help="Path to input images")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--size", type=str, help="Target image size. Default=288x288")

    args = parser.parse_args()

    image_path = args.image_path
    size = str_to_size(args.size)

    resize_images(image_path, size=size)
