import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def verify_train_images(source_dir: str = "data_containers/train", target_dir: str = None, save_output: bool = True):
    """Verify the training image-mask pair to avoid mismatch of images/masks.
    The image-mask pairs are randomly sampled, so that for every run a corresponding image-mask pair is checked.

    Parameters:
        source_dir: path containing the train images and masks
        target_dir: path to save the output
        save_output: flag to save output or display it. Default is True

    Return:
        None
    """
    if target_dir is not None:
        os.makedirs(target_dir, exist_ok=True)
    else:
        target_dir = os.getcwd()

    images = os.listdir(os.path.join(source_dir, "images"))
    masks = os.listdir(os.path.join(source_dir, "masks"))
    idx = np.random.randint(len(images))
    print(idx)
    img_path = os.path.join(source_dir, "images", images[idx])
    mask_path = os.path.join(source_dir, "masks", masks[idx])

    assert (
            ".".join(images[idx].split(".")) == ".".join(masks[idx].split("."))
    ), "Incorrect image-mask pair sampled!"
    out_fname = ".".join(images[idx].split("."))

    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    msk = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

    stack = np.hstack((img, msk))
    plt.imshow(stack, "gray")
    plt.title(f"Image-mask sampled, ({out_fname})")
    plt.axis("off")

    if save_output:
        plt.savefig(f"{target_dir}/{out_fname}.png")
    else:
        plt.show()


if __name__ == "__main__":
    verify_train_images(source_dir="../../databank/patches", save_output=False)
    # verify_train_images(source_dir="../../data/train", save_output=False)
