from imageai.Prediction import ImagePrediction
import os
from tensorflow.keras.applications.resnet50 import ResNet50
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

execution_path = os.getcwd()  # Путь к папке с проектом
im_path = os.path.abspath('/images/img_1.jpg')  # Путь к папке с рисункамии

prediction = ImagePrediction()

"""
# Тип модели экземпляра класса для распознования изображения SqueezeNet
prediction.setModelTypeAsSqueezeNet()

# Тип модели экземпляра класса для распознования изображения ResNet
prediction.setModelTypeAsResNet()

# Тип модели экземпляра класса для распознования изображения InceptionV3
prediction.setModelTypeAsInceptionV3()

# Тип модели экземпляра класса для распознования изображения DenseNet
prediction.setModelTypeAsDenseNet()
"""

# Тип модели экземпляра класса для распознования изображения ResNet
prediction.setModelTypeAsResNet()


prediction.setModelPath(os.path.join(execution_path,
                                     "resnet50_weights_tf_dim_ordering_tf_kernels.h5"))

prediction.loadModel()

predictions, probabilities = prediction.predictImage(im_path, result_count=5)
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction, " : ", eachProbability)
