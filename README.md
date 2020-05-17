# Clearcut_detection  

## Project structure info
 * `clearcut_detection_backend` - web-service for clearcut detection
 * `clearcut_research` - investigation about model approach, model training and model evaluation of clearcut detection
 
 ## Launch requirements:  
To start a web-service, do next:
* `cd clearcut_detection_backend/`  
* create peps_download_config.ini based on the peps_download_config.ini.example and setup secure params
* update `gcp_config.ini`, `AREA_TILE_SET` value is responsible for region that should be fetched and processed.
 In order to use Google Cloud Storage you need to generate service account key and put this file inside /clercut_detection_backend folder, file name should be specified inside gcp_config file(key.json by default).
 Update it if needed. To get tiles Ids you can use https://mappingsupport.com/p2/gissurfer.php?center=14SQH05239974&zoom=4&basemap=USA_basemap   
* create django.env file based on django.env.dist 
* create model.env file based on model.env.dist should be inside /model folder, use same service account key generated earlier  
* put unet_v4.pth in to  clearcut_detection_backend/model/unet_v4.pth (trained model can be obtained from maintainers)
* Run `docker-compose -f docker-compose.dev.yml up` in order to run docker for backend and frontend development.  
* Run `docker-compose -f docker-compose-stage.yml up` for deployment.

## Credential setup

This project needs several secure credentials, for peps.cnes.fr and sentinel-hub. 
For correct setup, you need to create peps_download_config.ini 
(it could be done by example peps_download_config.ini.example) and feel auth, 
password and sentinel_id parameters. 

## Swagger:  
After the app has been launched, swagger for api can be used. Go to http://localhost/api/swagger to access swagger with full description of api endpoints.

## Model Development Guide
### Data downloading
1) Create an account on https://peps.cnes.fr/rocket/#/home

2) Specify params in config file clearcut_detection_backend/peps_download_config.ini

3) Download an archive `python clearcut_detection_backend/peps_download.py`

4) Unzip the archive

5) Merge bands `python clearcut_detection_backend/prepare_tif.py --data_folder … --save_path …`

### Data preparation
1) Create folder in clearcut_research where is stored data:
   * Source subfolder stores raw data that has to be preprocess
   * Input subfolder stores data that is used in training and evaluation
   * Polygons subfolder stores markup

2) The source folder contains folders for each image that you downloaded. In that folder you have to store TIFF images of channels (in our case RGB, B2 and B8) named as f”{image_folder}\_{channel}.tif”.

3) If you have already merged bands to a single TIFF, you can just move it to input folder. But you have to create the folder (it can be empty) for this image in the source folder.

4) The polygons folder contains markup that you apply to all images in input folder.

#### Example of data folder structure:
```
data
├── input
│   ├── image0.tif
│   └── image1.tif
├── polygons
│   └── markup.geojson
└── source
    ├── image0
    │   ├── image0_b2.tif
    │   ├── image0_b8.tif
    │   └── image0_rgb.tif
    └── image1
        ├── iamge1_b2.tif
        ├── image1_b8.tif
        └── image1_rgb.tif
```
5) Run preprocessing on this data. You can specify other params if it necessary (add --no_merge if you have already merged channels with prepare_tif.py script).
```
python preprocessing.py \
 --polys_path ../data/polygons/markup.geojson \
 --tiff_path ../data/source
 --save_path ../data/input
```

#### Example of input folder structure after preprocessing:
```
input
├── image0
│   ├── geojson_polygons
│   ├── image0.png
│   ├── image_pieces.csv
│   ├── images
│   ├── instance_masks
│   └── masks
├── image0.tif
├── image1
│   ├── geojson_polygons
│   ├── image1.png
│   ├── image_pieces.csv
│   ├── images
│   ├── instance_masks
│   └── masks
└── image1.tif
```
6) Run data division script with specified split_function (default=’geo_split’) to create train/test/val datasets.
```
python generate_data.py --markup_path ../data/polygons/markup.geojson
```

### Model training
1) If it necessary specify augmentation in clearcut_research/pytorch/dataset.py

2) Specify hyperparams in clearcut_research/pytorch/train.py

3) Run training `python train.py`

### Model evaluation
1) Generate predictions 
```
python prediction.py \
 --data_path ../data/input \
 --model_weights_path … \
 --test_df ../data/test_df.csv \
 --save_path ../data
```  
2) Run evaluation
```
python evaluation.py \
 --datasets_path ../data/input \
 --prediction_path ../data/predictions \
 --test_df_path ../data/test_df.csv \
 --output_name …
```
3) Run clearcut_research/notebooks/tf_records_visualizer.ipynb to view results of evaluation.

4) Run clearcut_research/notebooks/f1_score.ipynb to get metrics for the whole image.


### Soil Erosion branch

Generic pipeline of deployemnt is inherented from the main branch of Clearcut project. However, there are key differences.
1) At this point the project is not designed to update frequently since soil erosion as a rule does not show patterns of rapid growth, rather it is steady and gradual process.
2) The model currently deployed has been trained with different feature input (namely the merge of green, red and near-infrared bands)

#### Research indexes
The script index_research.py can aid for further research of different metrics and input features. Currently this script generates previously mentioned NRG merge but with customization is capable of generating other layers such as SWIR, NDVI, SAVI etc.

#### Data preparation for soil erosion
The pipeline has been designed to work with Sentinel-2 obtained bands (10m resolution). To train model you will need the folder containing files *_B03_10m.jp2, *_B04_10m.jp2, *_B08_10m.jp2. To generate merged layer and cut images the file cut_images.py should be run:
```
python cut_images.py --source_path ../data/source
```
Afterwards it should be split and annotates:
```
python annoation.py --source_path ../data/source --train_size 0.7 --test_size 0.1 --val_size 0.2
```
Once it is done the data is ready to train as described previosly.