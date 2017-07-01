from PIL import Image
import os
from resizeimage import resizeimage

def str_to_size(size_str):
    """
    Convert string '72x72' to array [72, 72].
    :param size_str: String
    :return: Array of Int
    """
    return [int(s) for s in size_str.split('x')]


def resize_images(image_path, output_path, size=[288,288], filter=None):
    """
    Resize directory of images to target size
    :param image_path:
    :param output_path:
    :param size:
    :return:
    """
    # create target dirs
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # get source images
    file_names = [f for f in os.listdir(image_path)]
    print("Total images found: %d" % len(file_names))

    # keep count
    skipped_count = 0
    proc_count = 0

    # filter out images below certain size.
    # this avoids upscaling images above their original size.
    _filter = filter if filter else size

    for f in file_names:
        with open(image_path + f, 'r+b') as file_handle:
            with Image.open(file_handle) as image:
                w, h = image.size

                # skip images below filter size
                if (w < _filter[0]) or (h < _filter[1]):
                    print("skipping image: %s, size: %d, %d" % (f, w, h))
                    skipped_count = skipped_count + 1
                    continue
                img_out = resizeimage.resize_cover(image, size)
                img_out.save(output_path + f, image.format)
                proc_count = proc_count + 1

    print("[DONE] skipped: %d, processed: %d" % (skipped_count, proc_count))

def make_lr_hr_images(image_path="data/celeba_sample/celeba_1000/",
                lr_path="data/celeba_sample/lr/",
                hr_path="data/celeba_sample/hr/",
                lr_size=[72, 72],
                hr_size=[288, 288]):
    """
    Process images and output to low-res (lr) and high-res (hr).

    :param image_path:
    :param lr_path:
    :param hr_path:
    :param lr_size: String e.g. "72x72"
    :param hr_size: String e.g. "288x288"
    :return:
    """

    print("Processing low-res images...")
    resize_images(image_path, lr_path, size=lr_size, filter=hr_size)

    print("\n\nProcessing high-res images...")
    resize_images(image_path, hr_path, size=hr_size, filter=hr_size)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess images to /path/lr/ and /path/hr/.")
    parser.add_argument("image_path", type=str, help="Path to input images")
    parser.add_argument("--lr", type=str, help="Path to output low-res (lr) images")
    parser.add_argument("--hr", type=str, help="Path to output high-res (hr) images")
    parser.add_argument("--lr_size", type=str, help="low-res size")
    parser.add_argument("--hr_size", type=str, help="high-res size")

    args = parser.parse_args()

    image_path = args.image_path
    lr_path = args.lr
    hr_path = args.hr
    lr_size = str_to_size(args.lr_size)
    hr_size = str_to_size(args.hr_size)

    make_lr_hr_images(image_path,
                lr_path=lr_path,
                hr_path=hr_path,
                lr_size=lr_size,
                hr_size=hr_size)