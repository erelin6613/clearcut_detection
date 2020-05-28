import os
import numpy as np
import rasterio as rio
import rasterio.plot as rioplot
import rasterio.mask as riomask
from rasterio.windows import Window
import geopandas as gpd
from PIL import Image
from shapely.ops import cascaded_union
import argparse
from tqdm import tqdm
from clearcut_research.pytorch.cut_images import *
from cut_images import *
import matplotlib.pyplot as plt


def parse_args():

    parser = argparse.ArgumentParser(
        description='Generate empty images where model misspredicted erosions'
    )
    parser.add_argument(
    	'--tiff_file', '-tf', dest='tiff_file', required=True)
    parser.add_argument(
    	'--pred_shapes', '-ps', dest='pred_shapes', required=True)
    parser.add_argument(
    	'--true_shapes', '-ts', dest='true_shapes', required=True)
    parser.add_argument(
    	'--crop_size', '-cs', type=int, default=224, dest='crop_size')
    parser.add_argument(
    	'--new_dir', '-nd', dest='new_dir', default='misspredicted')

    return parser.parse_args()

def get_pred_n_true(cut_function, true_shapes, pred_shapes):

	true_arr, _ = cut_function(base_dir=base_dir_true, 
			band=band, CROP_SIZE=crop_size, 
			geoms=true_shapes, no_cut=True, 
			get_index=False)
	pred_arr, _ = cut_function(base_dir=base_dir_pred, 
			band=band, CROP_SIZE=crop_size, 
			geoms=pred_shapes, no_cut=True, 
			get_index=False)
	return true_arr, pred_arr, true_shapes, pred_shapes

def save_img(array, path):
	img = Image.fromarray(array)
	img.save(path)

def save_empty(save_function):

	def wrapper():
		img = np.zeros(array.shape)
		save_img(img, path)
	return wrapper

@save_empty
def save_mask(array, path):
	print('replaced and written:', path)

def relabel(true_arr, pred_arr, img_arr, crop_size, new_dir):

	if not os.path.isdir(new_dir):
		os.system(f'mkdir -p {new_dir}')

	for i in tqdm(range(0, true_arr.shape[0] // crop_size)):
		for j in range(0, true_arr.shape[1] // crop_size):
			i_size, j_size = i*crop_size, j*crop_size

			true_chunk = true_arr[i_size:i_size+crop_size, j_size:j_size+crop_size]#*255
			pred_chunk = pred_arr[i_size:i_size+crop_size, j_size:j_size+crop_size]#*255
			img_chunk = img_arr[:, i_size:i_size+crop_size, j_size:j_size+crop_size]

			if not os.path.isdir(os.path.join(new_dir, 'true_mask')):
				os.system('mkdir -p {}'.format(os.path.join(new_dir, 'true_mask')))
			if not os.path.isdir(os.path.join(new_dir, 'predicted_mask')):
				os.system('mkdir -p {}'.format(os.path.join(new_dir, 'predicted_mask')))
			if not os.path.isdir(os.path.join(new_dir, 'images')):
				os.system('mkdir -p {}'.format(os.path.join(new_dir, 'images')))

			true_path = os.path.join(new_dir, 'true_mask', str(i_size)+'_'+str(j_size)+'.png')
			predicted_path = os.path.join(new_dir, 'predicted_mask', str(i_size)+'_'+str(j_size)+'.png')
			image_path = os.path.join(new_dir, 'images', str(i_size)+'_'+str(j_size)+'.png')
			if np.argmax(true_chunk.ravel()) == 0 and np.argmax(pred_chunk.ravel()) != 0:
				continue
			else:
				true_chunk *= 255
				pred_chunk *= 255
			img_t = np.ma.transpose(img_chunk, [1, 2, 0])
			save_img(true_chunk.astype(np.uint8), true_path)
			save_img(pred_chunk.astype(np.uint8), predicted_path)
			save_img(img_t.astype(np.uint8), image_path)



def estimate_borders(mask_arr, crop_size):

	pos_x_start = pos_x_stop = pos_y_start = pos_y_stop = 0
	print(mask_arr.shape)

	for i in range(0, mask_arr.shape[0]):
		for j in range(0, mask_arr.shape[1]):
			img_arr = mask_arr[i:i+crop_size, j:j+crop_size]
			if np.argmax(img_arr.ravel()) == 0:
				zero_arr = True
				continue
			zero_arr = False

			pos_x_start = j if pos_x_start == 0 else pos_x_start
			pos_x_stop = j + crop_size
		if not zero_arr:
			pos_y_start = i if pos_y_start == 0 else pos_y_start
			pos_y_stop = i + crop_size

	return (pos_x_start, pos_y_start, pos_x_stop, pos_y_stop)

def mask_n_img(file, geoms, crop_size):
	file = rio.open(file, driver='GTiff')
	profile = file.profile
	file_array = file.read()
	geoms = get_geoms(geoms, file.crs)
	mask_arr, mask_transform, window = riomask.raster_geometry_mask(file, geoms, invert=True)
	file.close()
	return window, mask_arr*1, file_array

def generate_misspreds():

	args = parse_args()
	print('Loading images...')
	size, mask_arr, img_arr = mask_n_img(args.tiff_file, 
		args.true_shapes, args.crop_size)
	size, pred_arr, img_arr = mask_n_img(args.tiff_file, 
		args.pred_shapes, args.crop_size)
	relabel(mask_arr, pred_arr, img_arr, args.crop_size, args.new_dir)



if __name__ == '__main__':
	generate_misspreds()