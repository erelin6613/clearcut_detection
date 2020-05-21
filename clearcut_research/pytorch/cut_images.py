import os
import numpy as np
import rasterio as rio
import rasterio.plot as rioplot
import rasterio.mask as riomask
from rasterio.windows import Window
import geopandas as gpd
from PIL import Image
from shapely.ops import cascaded_union
from threading import Thread
from clearcut_research.pytorch.index_research import get_metrics

"""
	The main pipeline creates neseccary index (NRG)
	and cuts images with masks. By default images
	with masks are saved.
"""


def parse_args():

    parser = argparse.ArgumentParser(
        description='Generate NRG index and cut images for training'
    )
    parser.add_argument(
    	'--source_path', '-s', dest='source_path', required=True)
    parser.add_argument(
    	'--d_crs', '-crs', default='EPSG:32636', dest='crs')
    parser.add_argument(
    	'--crop_size', '-cs', type=int, default=224, dest='crop_size')
    parser.add_argument(
    	'--polygons_dir', '-pd', dest='polygons_dir')

    return parser.parse_args()

def overlap_mask(shapes, filename, profile, 
	tiff_map, target_shape):

	transform, crs = tiff_map.profile['transform'], tiff_map.profile['crs']
	mask_arr = riomask.raster_geometry_mask(tiff_map, 
		shapes, invert=True)
	mask_arr = mask_arr[0]*255
	i=0
	j=0
	while i < mask_arr.shape[0]:
		while j < mask_arr.shape[1]:
			img_arr = mask_arr[i:i+target_shape[0], j:j+target_shape[1]]
			if np.argmax(img_arr.ravel()) == 0 or img_arr.shape != (target_shape[0], target_shape[1]):
				j += target_shape[0]
				continue
			img_arr = np.array(img_arr, dtype='uint8')
			img_arr = Image.fromarray(img_arr)
			img_arr.save(os.path.join('mask', f'{filename}_{i}_{j}.png'))
			j += target_shape[0]
		i += target_shape[1]
		j=0

def save_divided_imgs(image, filename, CROP_SIZE):
	"""
	image should be an array of size
	(channels, height, width).
	Use if saveing images in tiff format
	"""
	i=0
	j=0
	img = image.read()
	while i < img.shape[1]:
		while j < img.shape[2]:
			meta = image.profile
			meta.update(driver='GTiff')
			meta.update(dtype=rio.float32)
			with rio.open(f'{filename}_{i}_{j}.tif', 'w', **meta) as dst:
				dst.meta['nodata'] = -999
				dst.meta['max'] = 1
				dst.meta['min'] = 0
				dst.write(img.astype(rio.float32), 
					window=Window(i, j, i+CROP_SIZE, j+CROP_SIZE))
			j += CROP_SIZE
		i += CROP_SIZE
		j=0

def get_geoms(shape_paths, crs):

	geoms = []
	for file in shape_paths:
		p = gpd.read_file(file).to_crs(crs)
		for shape in p.geometry:
			geoms.append(shape)
	geoms = cascaded_union(geoms)
	return geoms


def get_masks_n_imgs(base_dir, band, CROP_SIZE,
					geoms=None, no_cut=False):
	"""
	The data structure is expected to resemble with what it
	has been tested:

	BASE_DIR
		|
		|_WV
		| |
		| |_20190427T083601
		| |	|
		| | |_20190427T083601_agriculture.jp2
		| |	|_20190427T083601_false_color.jp2
		| |	...
		| |_20190427T083601
		| | |
		| | |_20190427T083601_geology.jp2
		| | ...
		|_XA
		| |
		| |_20190427T083601
		| | |
		....
	"""

	tiles = []
	for d in os.walk(base_dir):
		if len(d[2]) > 0:
			l=[i for i in d[2] if '_'+str(band) in i]
			if len(l) > 0:
				tiles.append(os.path.join(d[0], l[0]))

	for tile in tiles:
		i=j=0
		file = rio.open(tile, driver='GTiff')
		file_array = file.read()
		mask_arr, mask_transform, window = riomask.raster_geometry_mask(file, geoms, invert=True)
		transform, crs = file.profile['transform'], file.profile['crs']
		file.close()
		if no_cut:
			mask_arr.save(os.path.join('mask', filename))
			tag_arr = np.ma.transpose(file_array, [1, 2, 0])
			tag_img.save(os.path.join(band, filename))
			return

		while i < file_array.shape[1]:
			while j < file_array.shape[2]:
				img_arr = mask_arr[i:i+CROP_SIZE, j:j+CROP_SIZE]*255
				if np.argmax(img_arr.ravel()) == 0 or img_arr.shape != (CROP_SIZE, CROP_SIZE):	
					print('No overlaping masks found in tile {}'
						.format((i, i+CROP_SIZE, j, j+CROP_SIZE)))
					j += CROP_SIZE
					continue
				base_name = os.path.join(tile.split('/')[0], 
					tile.split('/')[1], tile.split('/')[2], 
					tile.split('/')[3])
				print(base_name)
				if not os.path.isdir(os.path.join(base_name, 'masks')):
					os.system('mkdir -p {}'.format(os.path.join(base_name, 'masks')))
				if not os.path.isdir(os.path.join(base_name, 'images')):
					os.system('mkdir -p {}'.format(os.path.join(base_name, 'images')))
				img_filename = '{}/images/{}_{}_{}.png'.format(base_name,
					tile.split('/')[3].split('.')[0], i, j)
				mask_filename = '{}/masks/{}_{}_{}.png'.format(base_name,
					tile.split('/')[3].split('.')[0], i, j)
				print(mask_filename, 'will be added to masks and images')
				img_arr = Image.fromarray(np.uint8(img_arr))
				img_arr.save(mask_filename)
				tag_arr = file_array[:, i:i+CROP_SIZE, j:j+CROP_SIZE]
				tag_arr = np.ma.transpose(tag_arr, [1, 2, 0])
				tag_img = Image.fromarray(tag_arr.astype(np.uint8))
				tag_img.save(img_filename)

				j += CROP_SIZE
			i += CROP_SIZE
			j = 0


def get_imgs_for_preds(base_dir, band, CROP_SIZE, no_cut=False):

	tiles = []
	os.system('mkdir -p {}'.format(os.path.join(base_dir, band)))
	for d in os.walk(base_dir):
		if len(d[-1]) > 0:
			#print(d)
			l=[i for i in d[2]]
			if len(l) > 0:
				for each in l:
					tiles.append(os.path.join(d[0], each))
	get_metrics(base_dir)
	if no_cut:
		return None, None
	tiles = []
	for d in os.walk(base_dir):
		if len(d[-1]) > 0:
			l=[i for i in d[2]]
			if len(l) > 0:
				for each in l:
					tiles.append(os.path.join(d[0], each))

	for tile in tiles:
		if band not in tile:
			continue
		print(f'Working on tile {tile} ...')
		i=j=0
		new_dir = tile.split('.')[0]
		if os.path.isdir(new_dir):
			file = rio.open(tile)
			meta = file.meta
			print('Images are cut earlier')
			return meta, new_dir
		os.system(f'mkdir {new_dir}')
		try:
			assert tile.split('.')[-1] == 'jp2'
			file = rio.open(tile, driver='JP2OpenJPEG')
			meta = file.meta
			file_array = file.read()
		except Exception:
			try:
				assert tile.split('.')[-1] == 'tiff'
				file = rio.open(tile, driver='GTiff')
				meta = file.meta
				file_array = file.read()
			except Exception:
				continue

		while i < file_array.shape[1]:
			while j < file_array.shape[2]:

				filename = new_dir+'/'+'{}_{}_{}.png'.format(new_dir.split('/')[-1], i, j)
				if not os.path.isfile(filename):
					tag_arr = file_array[:, i:i+CROP_SIZE, j:j+CROP_SIZE]
					if tag_arr.shape[-1] != CROP_SIZE or tag_arr.shape[-2] != CROP_SIZE:
						j+=CROP_SIZE
						continue
					tag_arr = np.ma.transpose(tag_arr, [1, 2, 0])
					tag_img = Image.fromarray(tag_arr.astype(np.uint8))
					tag_img.save(filename)
					print(f'{filename} saved')
				else:
					pass

				j += CROP_SIZE
			i += CROP_SIZE
			j = 0
	return meta, new_dir

def convert_to_tif(jp2_path):
	tif_name = jp2_path.replace('.jp2', '.tif').replace(
		'B02_10m', 'b2').replace('B03_10m', 'b3').replace(
		'B04_10m', 'b4').replace('B08_10m', 'b8')
	with rio.open(jp2_path) as src:
		img = src.read(1)
		img = np.nan_to_num(img)
		mean_ = img.mean()
		std_ = img.std()
		min_ = max(img.min(), mean_ - 2 * std_)
		max_ = min(img.max(), mean_ + 2 * std_)
	os.system(f'gdal_translate -ot Byte -scale {min_} {max_} 0 255 \
			{jp2_path} {tif_name}')
	os.system(f'rm {jp2_path}')


def merge_nrg(b3, b4, b8):
	out_file = b3.split('_b3')[0]+'_nrg.tif'
	os.system(f'gdal_merge.py -separate -o {out_file} {b3} {b4} {b8}')

def prepare_from_dir(directory):

	for f in os.listdir(directory):
		print(os.path.join(directory, f))
		convert_to_tif(os.path.join(directory, f))

	for f in os.listdir(directory):
		if '_b3' in f:
			b3 = os.path.join(directory, f)
		if '_b4' in f:
			b4 = os.path.join(directory, f)
		if '_b8' in f:
			b8 = os.path.join(directory, f)
	assert b3 is not None
	assert b4 is not None
	assert b8 is not None
	merge_nrg(b3, b4, b8)



if __name__ == '__main__':

	args = parse_args()
	prepare_from_dir(args.source_path)

	shape_paths = [os.path.join(
		args.polygons_dir, os.listdir(args.polygons_dir)[i]) 
		for i in range(len(os.listdir(args.polygons_dir)))]

	geoms = get_geoms(shape_paths, crs=args.crs)

	for f in os.walk(args.source_path):
		if len(f[2]) > 0:
			get_masks_n_imgs(f[0], 'nrg', geoms=geoms, CROP_SIZE=args.crop_size)