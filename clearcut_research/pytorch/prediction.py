import argparse
import os

import cv2 as cv
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch import nn
from tqdm import tqdm

from clearcut_research.pytorch.models.utils import get_model
from clearcut_research.pytorch.utils import get_filepath, count_channels, read_tensor, filter_by_channels


def predict(
        data_path, model_weights_path, network,
        test_df_path, save_path, size, channels
):
    model = get_model(network)
    model.encoder.conv1 = nn.Conv2d(
        count_channels(args.channels), 64, kernel_size=(7, 7),
        stride=(2, 2), padding=(3, 3), bias=False
    )

    checkpoint = torch.load(model_weights_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    test_df = pd.read_csv(test_df_path)

    predictions_path = os.path.join(save_path, "predictions")

    if not os.path.exists(predictions_path):
        os.makedirs(predictions_path, exist_ok=True)
        print("Prediction directory created.")

    for _, image_info in tqdm(test_df.iterrows()):
        filename = '_'.join([image_info['name'], image_info['position']])
        image_path = get_filepath(
            data_path, image_info['dataset_folder'],
            'images', filename,
            file_type='png'
        )

        image_tensor = filter_by_channels(
            read_tensor(image_path),
            channels
        )
        if image_tensor.ndim == 2:
            image_tensor = np.expand_dims(image_tensor, -1)

        image = transforms.ToTensor()(image_tensor)

        prediction = model.predict(image.view(1, count_channels(channels), size, size))

        result = prediction.view(size, size).detach().numpy()

        cv.imwrite(
            get_filepath(predictions_path, filename, file_type='png'),
            result * 255
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description='Script for making predictions on test images of dataset.')
    parser.add_argument('--network', '-n', default='unet50')
    parser.add_argument('--data_path', '-dp', required=True, help='Path to directory with datasets')
    parser.add_argument('--model_weights_path', '-mwp', required=True, help='Path to file with model weights')
    parser.add_argument('--test_df', '-td', required=True, help='Path to test dataframe')
    parser.add_argument('--save_path', '-sp', required=True, help='Path to save predictions')
    parser.add_argument('--size', '-s', default=224, type=int, help='Image size')
    parser.add_argument('--channels', '-ch', default=['nrg'], nargs='+', help='Channels list')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    predict(
        args.data_path, args.model_weights_path,
        args.network, args.test_df, args.save_path,
        args.size, args.channels
    )

    #if args.production == 1:
        # I surrender for now do predictions my way for now but not forever
        # print('_'.join(args.channels))
        # meta, new_dir = get_imgs_for_preds(args.data_path, 
        #     '_'.join(args.channels), args.size, True)
        #df = pd.DataFrame(columns=['dataset_folder',
        #'name','position', 'mask_pxl'])
        # all_paths = []
        #for d in dirs:
        # for f in list(os.listdir(new_dir)):
        #    all_paths.append(os.path.join(new_dir, f))
        #print(all_paths)
        # just for development stage
        # all_paths = all_paths[:100]
        # print(new_dir)
        #df = df.append(annotate(all_paths, new_dir, masks=False, MASK_DIR=None),
        #    ignore_index=True)
        #print(df.head())
        # just for development stage
        #df.to_csv(args.test_df)
        #predict(new_dir, args.model_weights_path,
        #    args.network, args.test_df, args.save_path,
        #    args.size, args.channels)
        #pull_together(os.path.join(args.save_path, "predictions"))
    #predict(
    #    args.data_path, args.model_weights_path,
    #    args.network, args.test_df, args.save_path,
    #    args.size, args.channels
    #)
