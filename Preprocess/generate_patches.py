"""Split images into patches and combined patches back into single image.
written on 2021.07.07 by Ekwonu Michael Chukwuemeka"""

import cv2
import os
from tqdm.auto import tqdm


def patch_images(img_dir: str,
                 target_dir: str,
                 patch_step: int,
                 index: int = 0,
                 Gray: bool = True,
                 verbose: bool = False
                 ):
    """
	Generate patches from image.
	
	Parameters:
		img_dir: str, input image directory.
		target_dir: str, Output directory to save image patches.
		patch_step: int, step size spacing between patches.
		index: int, value to save images with grids specifying patches generated,
			when handling sequence of images.
		Gray: bool, flag to load image as grayscale else image is loaded as colored.
			Default is True.
		verbose: bool, flag to save input image with grids of extracted patches.
	"""
    os.makedirs(target_dir, exist_ok=True)

    if Gray:
        img = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(img_dir)

    img_grid = img.copy()
    img_height, img_width = img.shape[0], img.shape[1]

    for y in range(0, img_height, patch_step):
        for x in range(0, img_width, patch_step):
            if (img_height - y) < patch_step or (img_width - x) < patch_step:
                break

            y1 = y + patch_step
            x1 = x + patch_step

            if x1 >= img_width and y1 >= img_height:
                x1 = img_width - 1
                y1 = img_height - 1
                tiles = img.copy()[y:y + patch_step, x:x + patch_step]
                cv2.imwrite(f"{target_dir}/{x}.{y}.{index}.png", tiles)

            elif y1 >= img_height:
                y1 = img_height - 1
                tiles = img.copy()[y:y + patch_step, x:x + patch_step]
                cv2.imwrite(f"{target_dir}/{x}.{y}.{index}.png", tiles)

            elif x1 >= img_width:
                x1 = img_width - 1
                tiles = img.copy()[y:y + patch_step, x:x + patch_step]
                cv2.imwrite(f"{target_dir}/{x}.{y}.{index}.png", tiles)

            else:
                tiles = img.copy()[y:y + patch_step, x:x + patch_step]
                cv2.imwrite(f"{target_dir}/{x}.{y}.{index}.png", tiles)

            if verbose:
                cv2.imwrite(
                    f"{target_dir}/img.{index}.png",
                    cv2.rectangle(img_grid, (x, y), (x1, y1), (0, 255, 0), 1)
                )


def generate_patches(root_dir: str = "Test",
                     target_dir: str = "generated_patches",
                     patch_step: int = 128,
                     unpatch: bool = False,
                     Gray: bool = True,
                     verbose: bool = False,
                     ):
    """
	Generate patches from sequence of images.

	Parameters:
		root_dir: str, parent directory containing all images.
		target_dir: str, Output directory to save image patches.
		patch_step: int, step size spacing between patches.
		unpatch: bool, Required to separate generated patches of sequence of images into different
			subfolders. This is necessary, if patches of images are to be recovered as a single image.
		Gray: bool, flag to load image as grayscale else image is loaded as colored.
				Default is True.
		verbose: bool, flag to save input image with grids of extracted patches.
	"""
    parent_dir = target_dir

    if unpatch:
        if os.path.isfile(root_dir):
            idx = 0
            target_dir = f"{idx}"
            path = root_dir
            patch_images(
                path, target_dir=f"{parent_dir}/{target_dir}", patch_step=patch_step, index=idx,
                verbose=verbose, Gray=Gray
            )
        else:
            root_dir = next(os.walk(root_dir))[0]
            files = next(os.walk(root_dir))[2]

            for idx, img in tqdm(enumerate(files), desc="Creating patches", total=len(files)):
                path = os.path.join(root_dir, img)
                target_dir = f"{idx}"
                patch_images(
                    path, target_dir=f"{parent_dir}/{target_dir}", patch_step=patch_step, index=idx,
                    verbose=verbose, Gray=Gray
                )

    else:
        if os.path.isfile(root_dir):
            idx = 0
            path = root_dir
            patch_images(
                path, target_dir=target_dir, patch_step=patch_step, index=idx, verbose=verbose,
                Gray=Gray
            )
        else:
            root_dir = next(os.walk(root_dir))[0]
            files = next(os.walk(root_dir))[2]

            for idx, img in tqdm(enumerate(files), desc="Creating patches", total=len(files)):
                path = os.path.join(root_dir, img)
                patch_images(
                    path, target_dir=target_dir, patch_step=patch_step, index=idx, verbose=verbose,
                    Gray=Gray
                )


if __name__ == "__main__":
    generate_patches(root_dir="image_containers", patch_step=64, target_dir="database_mains/images")
    generate_patches(root_dir="mask_containers", patch_step=64, target_dir="database_mains/masks")
