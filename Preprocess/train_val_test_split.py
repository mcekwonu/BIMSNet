import argparse
import splitfolders


def train_val_test(source_dir: str,
                   target_dir: str,
                   split_ratio: tuple = (0.8, 0.15, 0.05),
                   seed: int = 11,
                   ):
    """
	Create train, validation and test subdirectories of images and masks.
	
	Parameters:
		source_dir: str, Input directory containing subfolders of images and masks.
		target_dir: str, Output directory to contain the train, test and validation subdirectories with images.
			and masks subdirectories
		split_ratio: tuple, split size for train, test and val images.
		seed: int, seed value for shuffling the items. Default is 11.
	
	Returns:
		None
	"""
    assert (
            len(split_ratio) == 2 or len(split_ratio) == 3
    ), f"{split_ratio} must be a contain two or three values!"

    if len(split_ratio) == 2:
        return splitfolders.ratio(source_dir, target_dir, seed=seed, ratio=split_ratio)

    elif len(split_ratio) == 3:
        return splitfolders.ratio(source_dir, target_dir, seed=seed, ratio=split_ratio)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="Train, Test and Validation folders split")
    # parser.add_argument("--source_dir", type=str, help="Input directory contain subfolders of images and masks")
    # parser.add_argument("--target_dir", type=str, help="Output directory")
    # parser.add_argument("--split_ratio", type=tuple, help="split ratio for specifying size of train, val and test.
    # It can be a tuple of two or three values.")
    # parser.add_argument("--seed", type=int, default=11, help="seed value for shuffling")
    # args = parser.parse_args()

    train_val_test(source_dir="../../databank/patches",
                   target_dir="../../databank/data", split_ratio=(0.8, 0.19, 0.01))
