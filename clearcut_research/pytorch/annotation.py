import os
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
import albumentations as alb
from random import shuffle
from tqdm import tqdm
import argparse

def parse_args():

    parser = argparse.ArgumentParser(
        description='Annotate image mini-tiles for training.'
    )
    parser.add_argument(
    	'--train_size', '-trs', type=float, default=0.7, dest='train_size')
    parser.add_argument(
    	'--test_size', '-ts', type=float, default=0.2, dest='test_size')
    parser.add_argument(
    	'--val_size', '-vs', type=float, default=0.1, dest='val_size')
    parser.add_argument(
    	'--source_path', '-s', dest='source_path', required=True)


    return parser.parse_args()


def albumentations_transform(image):

	augment = alb.Compose[Compose([
            RandomRotate90(),
            Flip(),
            OneOf([
                RandomSizedCrop(
                    min_max_height=(int(self.image_size * 0.7), self.image_size),
                    height=self.image_size, width=self.image_size)
            ], p=0.4),
            CLAHE(clip_limit=2),
            ToTensor()
        ])]

	return augment(image)


def generate_transforms(image):

	transform = transforms.Compose([
		transforms.ToPILImage(),
		transforms.Resize((CROP_SIZE, CROP_SIZE)),
		transforms.RandomHorizontalFlip(p=0.5),
		transforms.RandomRotation(degrees=(-180, 180)),
		transforms.RandomVerticalFlip(p=0.5)])

	return transform(image)


def save_new_image(folder, filename):
	path = os.path.join(folder, filename)
	image = generate_transforms(np.array(Image.open(img_path)))
	image.save(os.path.join(folder, 'transformed_'+filename))


# Annotate all cut images with masks. Note: flags masks in this
# function track how many pixels are tagged as mask
def annotate(paths, DIR, masks=True, MASK_DIR=None):

	if masks:
		if MASK_DIR is not None:
			MASK_DIR = MASK_DIR
	df = pd.DataFrame(columns=['dataset_folder',
		'name','position', 'mask_pxl'])

	for file in tqdm(paths):

		d = dict()
		original_img = Image.open(file)
		original_array = np.asarray(original_img)
		if masks:
			mask_img = Image.open(file.replace('images', 'masks'))
			mask_array = np.asarray(mask_img)
		fn = file.split('/')[-1].split('.')[0]
		d['dataset_folder'] = d['name'] = '_'.join(fn.split('_')[:-2])
		d['position'] = fn.split('_')[-2]+'_'+fn.split('_')[-1]
		if masks:
			d['mask_pxl'] = np.count_nonzero(mask_array)
		else:
			d['mask_pxl'] = 0
		df = df.append(d, ignore_index=True)

	return df

def make_sets():
	if not os.path.isdir(args.source_path):
		os.system(f'mkdir -p {args.source_path}')
	all_paths = []
	for d in os.walk(args.source_path):
		if 'images' in d[0]:
			if len(d[2]) > 0:
				for f in d[2]:
					all_paths.append(os.path.join(d[0], f))
	shuffle(all_paths)
	train_ind= int(len(all_paths)*args.train_size)
	test_ind = int(len(all_paths)*args.test_size)
	val_ind = len(all_paths) - train_ind - test_ind
	train_set = all_paths[:train_ind]
	test_set = all_paths[train_ind:(len(all_paths)-val_ind)]
	val_set = all_paths[-val_ind:]

	print('Getting data together...')
	annotate(train_set, DIR=args.source_path).to_csv('train_df.csv')
	print('Train set has been written')
	annotate(test_set, DIR=args.source_path).to_csv('test_df.csv')
	print('Test set has been written')
	annotate(val_set, DIR=args.source_path).to_csv('val_df.csv')
	print('Validation set has been written')

if __name__ == '__main__':
	args = parse_args()
	make_sets()