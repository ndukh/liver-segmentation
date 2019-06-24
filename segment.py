import argparse
import os
from keras.models import load_model
from tqdm import tqdm
from utils import dice_coef_loss, dice_coef
from data_processing import get_filepaths, save_img, get_data_generator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', default='sample/input',
                        help=f'path to the folder with input DICOM images')
    parser.add_argument('-o', '--output', default='sample/output',
                        help=f'folder where the output masks will be saved')

    return parser.parse_args()


def main():
    args = parse_args()

    input_path = args.input
    output_path = args.output

    model_path = 'model.keras'
    model = load_model(model_path, custom_objects={'dice_coef_loss': dice_coef_loss,
                                                   'dice_coef': dice_coef})
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    print('model is loaded, predicting...')
    predict_all(model, input_path + '/', output_path + '/')
    print('all the images have been processed, prediction is finished')


def predict_all(model, in_path, out_path):
    filepaths_df = get_filepaths(in_path)
    length = filepaths_df.shape[0]
    data_generator = get_data_generator(filepaths_df)

    outpaths = filepaths_df[0].str.replace(in_path, out_path, 1) + '.png'
    outpathdirs = outpaths.str.rsplit('/', 1, True)[0]
    for unique_dirpath in outpathdirs.unique():
        if not os.path.exists(unique_dirpath):
            os.makedirs(unique_dirpath)

    for i, image in enumerate(tqdm(data_generator, position=0, total=length)):
        if i >= length:
            break

        prediction = model.predict(image).astype('uint8')
        for p in prediction:
            save_img(p.squeeze(), outpaths.iloc[i])

if __name__ == "__main__":
    main()
