from train import Train
from test import Test
from config import DatasetName, ModelArch
from pca_utilities import PCAUtility

if __name__ == '__main__':
    '''use the pretrained model'''
    tester = Test()
    tester.test_model(ds_name=DatasetName.w300,
                      pretrained_model_path='./pre_trained_models/ACRLoss/mnv2.h5')

    '''training model from scratch'''

    #   pretrain prerequisites
    #       1- PCA calculation:
    pca_calc = PCAUtility()
    pca_calc.create_pca_from_npy(dataset_name=DatasetName.w300,
                                 labels_npy_path='./data/w300/normalized_labels/',
                                 pca_percentages=90)

    #  Train:
    trainer = Train(arch=ModelArch.MNV2,
                    dataset_name=DatasetName.w300,
                    save_path='./')
