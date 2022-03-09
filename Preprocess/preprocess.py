import cv2
import os
from tqdm.auto import tqdm

from utils import isempty


class Preprocess:
    def __init__(self, source_dir: str, target_dir: str) -> None:
        self.source_dir = source_dir
        self.target_dir = target_dir

        os.makedirs(target_dir, exist_ok=True)

    def crop_images(self, height: tuple = (288, 800), width: tuple = (200, 512), Grayscale: bool = True):

        y1, h = height
        x1, w = width

        isempty(source_dir)  # Sanity check if folder is not empty

        for root, subdirs, files in os.walk(self.source_dir):
            for i, file in tqdm(enumerate(files), desc=f"==> Cropping and saving images in {self.target_dir}"):
                if file.endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg")):
                    img_dir = os.path.join(root, file)
                    filename = file[6:9]

                    if Grayscale:
                        img = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
                    else:
                        img = cv2.imread(img_dir)

                    img = img[y1:y1 + h, x1:x1 + w]
                    cv2.imwrite(f"{self.target_dir}/{filename}.{i}.png", img)

    def create_masks(self, threshold: int = 255, kernel_size: int = 5):
        """Binarize images and generates masks"""

        isempty(source_dir)  # Sanity check if folder is not empty
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        for i, file in tqdm(enumerate(os.listdir(source_dir)),
                            desc=f"==> Generating masks and saving images in {target_dir}"):
            filename = os.path.splitext(file)[0]
            if file.endswith((".tif", ".tiff", ".png", ".jpeg")):
                img_dir = os.path.join(self.source_dir, file)
                img = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
                _, mask = cv2.threshold(img, threshold // 2, threshold, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
                mask_op = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask_cls = cv2.morphologyEx(mask_op, cv2.MORPH_CLOSE, kernel)
                cv2.imwrite(f"{self.target_dir}/{filename}.png", mask_cls)

    def bicenter_crop(self, image, dim: tuple):
        """Center crop image.

		Parameter:
			image: (ndarrays) image to be cropped.
			dim: (tuple), height and width of the cropped image.
		"""
        center_y, center_x = image.shape
        width, height = dim
        x = center_x // 2 - width // 2
        y = center_y - height

        return image[y:y + height, x:x + width]

    def uniform_crop_images(self, dim):
        for root, subdirs, files in os.walk(self.source_dir):
            for i, file in tqdm(enumerate(files),
                                desc=f"Cropping and saving images in {os.path.realpath(self.target_dir)}"):
                img_dir = os.path.join(root, file)
                img = cv2.imread(img_dir, cv2.IMREAD_UNCHANGED)
                img_crp = self.bicenter_crop(img, dim)
                cv2.imwrite(f"{self.target_dir}/{i+1}.png", img_crp)


if __name__ == "__main__":
    # crop_images(source_dir=".././Experiments_2019/Experiment_2/Reference", target_dir="image_containers")
    # create_masks(source_dir="image_containers", target_dir="mask_containers")
    Preprocess(source_dir="../predictions/raw_images",
               target_dir="../predictions/test_images").uniform_crop_images(dim=(512, 512))
