import os
import numpy as np
import rasterio as rio
import rasterio.plot as rioplot
import rasterio.mask as riomask
import geopandas as gpd
from PIL import Image
from tqdm import tqdm
import argparse

"""
	Generate indexes to explore which of those might
	help with the case. For now the best metric has
	been proven to be NRG index. Other index are
	predefined in STACKING_INDEXES. 

	Along side the main pipeline also calculates
	NDVI and SAVI indexes.
"""


STACKING_INDEXES = {
	'swir': ['B12', 'B8A', 'B4'],
	'geology': ['B12', 'B11', 'B02'],
	'agriculture': ['B11', 'B8A', 'B02'],
	'false_color': ['B8A', 'B04', 'B03']
}

STACKING_INDEXES_10m = {
	'nrg': ['B08', 'B04', 'B03']
}

def parse_args():

    parser = argparse.ArgumentParser(
    	description='Generate indexes for research.\
    	See README to find out needed files and structure')
    parser.add_argument(
    	'--d_crs', '-crs', default='EPSG:32636', dest='crs')
    parser.add_argument(
    	'--source_path', '-s', dest='source_path', required=True)


    return parser.parse_args()


def stack_layer(directory, bands, name, 
	scale=False):
	files = []
	for d in os.listdir(directory):
		try:
			if d.split('_')[-2] in bands:
				files.append(os.path.join(directory, d))
		except Exception as e:
			print('Preparing layer', name, e)
	print('stacking:', files)

	try:
		f_name = os.path.join(directory, 
			files[0].split('/')[-1].split(
				'.')[0]+'_'+name+'.'+files[0].split('/')[-1].split('.')[1])
		f_name = f_name.replace('_B02_10m', '')
		f_name = f_name.replace('_B03_10m', '')
		f_name = f_name.replace('_B04_10m', '')
		f_name = f_name.replace('_B08_10m', '')
		if not os.path.isfile(f_name):
			print('Stacking layers...')
			raster_prof = rio.open(files[0]).profile
			to_stack = []
			for file in files:
				to_stack.append(rio.open(file).read(1))

			array = np.stack(to_stack, axis=0)
			if len(to_stack) != 3:
				raise Warning(f'Warning: found {len(to_stack)} bands to stack')
			raster_prof['count'] = len(to_stack)
			with rio.open(f_name, 'w', **raster_prof) as dst:
				dst.write(array)
		else:
			print(f'The layer {name} exists, skipping')
			return
	except Exception as e:
		print('Error:', e)

def get_nvdi(red_file, nir_file, 
	save=True, outname='NDVI'):

	if save:
		if not os.path.isfile(outname):
			pass
		else:
			print('The band NVDI already exists, skipping')
			return

	if red_file.endswith('.jp2') and nir_file.endswith('.jp2'):
		red = rio.open(red_file, driver='JP2OpenJPEG')
		nir = rio.open(nir_file, driver='JP2OpenJPEG')
	elif (red_file.endswith('.tif') and nir_file.endswith('.tif')) or \
	(red_file.endswith('.tiff') and nir_file.endswith('.tiff')):
		red = rio.open(red_file, driver='GTiff')
		nir = rio.open(nir_file, driver='GTiff')
	else:
		raise Exception('Bands images must be of the same format')
	meta = red.meta
	red = red.read()
	nir = nir.read()
	ndvi = (nir.astype(float)-red.astype(float))/(nir+red)
	ndvi = ndvi.astype(rio.float32)

	if save:
		if not os.path.isfile(outname):
			ndvi = ndvi
			meta.update(driver='GTiff')
			meta.update(dtype=rio.float32)
			with rio.open(outname, 'w', **meta) as dst:
				dst.meta['nodata'] = -999
				dst.meta['max'] = 1
				dst.meta['min'] = 0
				dst.write(ndvi.astype(rio.float32))
		else:
			print('The band NVDI already exists, skipping')
		try:
			scale_img(outname, min_value=0, max_value=255, output_type='Byte')
		except Exception as e:
			print(e)

	return np.nan_to_num(ndvi, nan=-999)


def get_savi(red_file, nir_file, L=0.5,
	save=True, outname='SAVI'):

	if save:
		if not os.path.isfile(outname):
			pass
		else:
			print('The band SAVI already exists, skipping')
			return

	if red_file.endswith('.jp2') and nir_file.endswith('.jp2'):
		red = rio.open(red_file, driver='JP2OpenJPEG')
		nir = rio.open(nir_file, driver='JP2OpenJPEG')
	elif (red_file.endswith('.tif') and nir_file.endswith('.tif')) or \
	(red_file.endswith('.tiff') and nir_file.endswith('.tiff')):
		red = rio.open(red_file, driver='GTiff')
		nir = rio.open(nir_file, driver='GTiff')
	else:
		raise Exception('Bands images must be of the same format')
	meta = red.meta
	red = red.read()
	nir = nir.read()
	L = np.full(red.shape, L)
	savi = ((1+L)*(nir.astype(float)-red.astype(float)))/(nir+red+L)
	savi = savi.astype(rio.float32)

	if save:
		if not os.path.isfile(outname):
			savi = savi
			meta.update(driver='GTiff')
			meta.update(dtype=rio.float32)
			with rio.open(outname, 'w', **meta) as dst:
				dst.meta['nodata'] = -999
				dst.meta['max'] = 1
				dst.meta['min'] = 0
				dst.write(savi.astype(rio.float32))
		try:
			scale_img(outname, min_value=0, max_value=255, output_type='Byte')
		except Exception as e:
			print(e)

	return np.nan_to_num(savi, nan=-999)


def get_mi(b8a_file, b11_file,
	save=True, outname='MI'):

	if b8a_file.endswith('.jp2') and b11_file.endswith('.jp2'):

		b8a = rio.open(b8a_file, driver='JP2OpenJPEG')
		b11 = rio.open(b11_file, driver='JP2OpenJPEG')
	elif (b8a_file.endswith('.tif') and b11_file.endswith('.tif')) or \
	(b8a_file.endswith('.tiff') and b11_file.endswith('.tiff')):
		b8a = rio.open(b8a_file, driver='GTiff')
		b11 = rio.open(b11_file, driver='GTiff')
	else:
		raise Exception('Bands images must be of the same format')
	meta = b8a.meta
	b8a = b8a.read()
	b11 = b11.read()
	mi = ((b8a.astype(float)-b11.astype(float)))/(b8a+b11)
	mi = mi.astype(rio.float32)

	if save:
		mi = mi
		meta.update(driver='GTiff')
		meta.update(dtype=rio.float32)
		with rio.open(outname+'.tif', 'w', **meta) as dst:
			dst.meta['nodata'] = -999
			dst.meta['max'] = 1
			dst.meta['min'] = 0
			dst.write(mi.astype(rio.float32))

	return np.nan_to_num(mi, nan=-999)


def get_metrics(date_dir):

	for ind in STACKING_INDEXES_10m.keys():
		stack_layer(date_dir, STACKING_INDEXES_10m[ind], ind)
	for each in os.listdir(date_dir):
		print(each)
		if 'B04' in each.split('_'):
			red = os.path.join(date_dir, each)
		if 'B08' in each.split('_'):
			nir = os.path.join(date_dir, each)
		if 'B03' in each.split('_'):
			b03 = os.path.join(each)
		if 'B02' in each.split('_'):
			b02 = os.path.join(each)
	base_name = os.path.join(date_dir, os.listdir(date_dir)[0].split('.')[0].replace(
		'_B02_10m', '').replace('_B03_10m', '').replace(
		'_B04_10m', '').replace('_B08_10m', ''))
	get_nvdi(red, nir, True, base_name+'_ndvi'+'.tif')
	get_savi(red, nir, 0.5, True, base_name+'_savi'+'.tif')


def merge(save_path, *images):
	os.system(f'gdal_merge.py -separate -o {save_path} {" ".join(images)}')


def scale_img(img_file, min_value=0, max_value=255, output_type='Byte'):
	with rio.open(img_file) as src:
		img = src.read(1)
		img = np.nan_to_num(img)
		mean_ = img.mean()
		std_ = img.std()
		min_ = max(img.min(), mean_ - 2 * std_)
		max_ = min(img.max(), mean_ + 2 * std_)
		os.system(
			f"gdal_translate -ot {output_type} \
			-scale {min_} {max_} {min_value} {max_value} \
			{img_file} {f'{os.path.splitext(img_file)[0]}_scaled.tif'}")


def convert_to_tif(date_dir):

	for file in os.listdir('../'+date_dir+'/10m'):
		img_path = os.path.join('../'+date_dir+'/10m', file)
		if img_path.endswith('.jp2'):
			geo_path = img_path.replace('.jp2', '.tif')
			print('Converting to tif')
			os.system(f'gdalwarp -of GTiff -overwrite -ot Byte -t_srs EPSG:{args.crs} ' \
				f'-wm 4096 -multi -wo NUM_THREADS=ALL_CPUS ' \
				f'-co COMPRESS=DEFLATE -co PREDICTOR=2 {img_path} {geo_path}')
			os.system(f'rm {img_path}')

if __name__ == '__main__':
	args = parse_args()
	get_metrics(args.source_path)