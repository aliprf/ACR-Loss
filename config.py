class DatasetName:
    w300 = '300W'
    cofw = 'COFW'


class ModelArch:
    MNV2 = 'mobileNetV2'
    EFNB0 = 'EfficientNet-B0'
    EFNB3 = 'EfficientNet-B3'


class DatasetType:
    data_type_train = 0
    data_type_test = 1


class LearningConfig:
    batch_size = 3
    epochs = 150


class InputDataSize:
    image_input_size = 224


class W300Conf:
    W300W_prefix_path = './300W/'

    train_annotation = W300W_prefix_path + 'train_set/annotations/'
    train_image = W300W_prefix_path + 'train_set/images/'

    test_annotation_path = W300W_prefix_path + 'test_set/annotations/'
    test_image_path = W300W_prefix_path + 'test_set/images/'
    num_of_landmarks = 68

class CofwConf:
    cofw_prefix_path = './cofw/'

    train_annotation = cofw_prefix_path + 'train_set/annotations/'
    train_image = cofw_prefix_path + 'train_set/images/'

    test_annotation_path = cofw_prefix_path + 'test_set/annotations/'
    test_image_path = cofw_prefix_path + 'test_set/images/'
    num_of_landmarks = 29