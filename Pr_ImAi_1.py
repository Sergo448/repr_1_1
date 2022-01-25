# ImAi_1
# Для анализа уже готовыми сетями одного изоражения или группы изображений
from imageai.Prediction import ImagePrediction
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Путь к папке с проектом и к анализируемому изображению\группе изображений
execution_path = os.getcwd()

# im_path = os.path.abspath('P:\\Python Projects\\Learning_projects\\images\\kate_2.jpg')


# Для одного изображения
"""prediction = ImagePrediction()
prediction.setModelTypeAsResNet()
prediction.setModelPath(os.path.join(execution_path,
                                     'P:\\Python Projects\\Learning_projects\\Nets'
                                     '\\resnet50_weights_tf_dim_ordering_tf_kernels.h5'))
prediction.loadModel()
"""

# Конфигурация нейронной сети для группы изображений
multiple_prediction = ImagePrediction()
multiple_prediction.setModelTypeAsResNet()
multiple_prediction.setModelPath(os.path.join(execution_path, "P:\\Python Projects\\Learning_projects\\Nets"
                                                              "\\resnet50_weights_tf_dim_ordering_tf_kernels.h5"))
multiple_prediction.loadModel()

# Для группы изображений
all_images_array = []  # Пустой массив с файлами рисунков

all_files = os.listdir(execution_path) # формирование массива со всеми файлами в папке проекта

for each_file in all_files:  # формирование массива с файлами только рисунков

    if each_file.endswith('.jpg') or each_file.endswith('.png'):
        all_images_array.append(each_file)

# Запуск модели на поиск объектов в файлах рисунков
results_array = multiple_prediction.predictMultipleImages(all_images_array, result_count_per_image=5)

# Для одного изображения
"""        
predictions, probabilities = prediction.predictImage(im_path, result_count=5)

for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction, ' : ', eachProbability)
    """
# Для группы изображений
# try:
for each_result in results_array:
    predictions, percentage_probabilities = each_result['predictions'], each_result['percentage_probabilities']

    for index in range(len(predictions)):
        print(predictions[index], ' : ', percentage_probabilities[index])
    print("________")
# except TypeError:
    # print('ОШИБКА\nOперация применена к объекту несоответсвтвующего типа!')