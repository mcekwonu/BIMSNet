import os
import sys
import shutil
from tqdm import tqdm


def rename_images(source_dir, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    for count, img in tqdm(enumerate(sorted(os.listdir(source_dir), key=len)),
                           desc=f"Renaming and copying images to "
                                f"{os.path.abspath(target_dir)}"):
        if os.path.isfile(os.path.join(source_dir, img)):
            filename, file_ext = os.path.splitext(img)
            shutil.copy(
                os.path.join(source_dir, img),
                os.path.join(target_dir, str(count) + file_ext)
            )


if __name__ == "__main__":
    rename_images(source_dir="../../databank/data_containers/test/images", target_dir="../data/images")
    rename_images(source_dir="../../databank/data_containers/test/masks", target_dir="../data/masks")
