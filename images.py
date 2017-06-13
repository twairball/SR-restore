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

def make_images(image_path="data/celeba_sample/celeba_1000/",
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
    # create target dirs
    if not os.path.exists(lr_path):
        os.makedirs(lr_path)

    if not os.path.exists(hr_path):
        os.makedirs(hr_path)

    # get source images
    file_names = [f for f in os.listdir(image_path)]
    print("Total images found: %d" % len(file_names))

    # keep count
    skipped_count = 0
    proc_count = 0

    for f in file_names:
        with open(image_path + f, 'r+b') as file_handle:
            with Image.open(file_handle) as image:
                w, h = image.size
                if (w < 288) or (h < 288):
                    print("skipping image: %s, size: %d, %d" % (f, w, h))
                    skipped_count = skipped_count + 1
                    continue
                lr = resizeimage.resize_cover(image, lr_size)
                hr = resizeimage.resize_cover(image, hr_size)
                lr.save(lr_path + f, image.format)
                hr.save(hr_path + f, image.format)
                proc_count = proc_count + 1

    print("[DONE] lr: %s, hr: %s" % (lr_size, hr_size))
    print("[DONE] skipped: %d, processed: %d" % (skipped_count, proc_count))


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

    make_images(image_path,
                lr_path=lr_path,
                hr_path=hr_path,
                lr_size=lr_size,
                hr_size=hr_size)