# Модуль:
# Поиск объектов на изображении

from imageai.Detection import ObjectDetection, VideoObjectDetection
import os
import cv2
from matplotlib import pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# execution_path = os.getcwd()

# Обычный поиск
"""
detector = ObjectDetection()

# Задаем тип модели экземпляра
detector.setModelTypeAsRetinaNet()

# Задаем путь до файла указанной модели
detector.setModelPath(os.path.join(execution_path,
                                   "P:\\Python Projects\\Learning_projects\\Nets\\"
                                   "ObjectDetection\\resnet50_coco_best_v2.1.0.h5"))
# Загружаем модель
detector.loadModel()

# Задаем функцию, выполняющую задачу обнаружения объекта на изображении после загрузки модели

detections = detector.detectObjectsFromImage(
    input_image=os.path.join(execution_path, 'P:\\Python Projects\\Learning_projects\\images\\scale_1200.jpg'),
    output_image_path=os.path.join(execution_path, 'P:\\Python Projects\\Learning_projects\\images'
                                                   '\\scale_1200_new.jpg'),
    minimum_percentage_probability=50)


# Вывод результатов
for eachObject in detections:
    print(eachObject['name'], ' : ', eachObject['percentage_probability'],
          " : ", eachObject['box_points'])
    print('______')"""

# Поик определенных объектов на изображении
"""
im_path = os.path.join('P:\\Python Projects\\Learning_projects\\images\\')  # Путь к папке с рисунками

detector_1 = ObjectDetection()

detector_1.setModelTypeAsRetinaNet()
detector_1.setModelPath(os.path.join(execution_path, 'P:\\Python Projects\\Learning_projects\\'
                                                     'Nets\\ObjectDetection\\resnet50_coco_best_v2.1.0.h5'))
# detector_1.setJsonPath(os.path.join(execution_path, 'P:\\Python Projects\\Learning_projects\\'
# 'Nets\\ObjectDetection\\detection_config.json'))
detector_1.loadModel()

# Задаем объекты которые необходимо найти
custom = detector_1.CustomObjects(person=True, dog=True)
detections = detector_1.detectCustomObjectsFromImage(custom_objects=custom,
                                                     input_image=os.path.join(im_path, 'dogkatchel.jpg'),
                                                     output_image_path=os.path.join(im_path, 'dogkatchel_new.jpg'),
                                                     minimum_percentage_probability=30)
for eachObject in detections:
    print(eachObject['name'], ' : ', eachObject['percentage_probability'], ' : ', eachObject['box_points'])
    print('_______')
"""

# Обнаружение объектов на видеопотоке
# camera = cv2.VideoCapture(0)
"""
detector_2 = VideoObjectDetection()
detector_2.setModelTypeAsRetinaNet()

detector_2.setModelPath(os.path.join(execution_path, 'P:\\Python Projects\\Learning_projects\\Nets\\ObjectDetection'
                                                     '\\resnet50_coco_best_v2.1.0.h5'))
detector_2.loadModel()
"""

# Для видео загруженного заранее
"""
vedeo_path = detector_2.detectCustomObjectsFromVideo(
    input_file_path=os.path.join(execution_path, 'P:\\Python Projects\\Learning_projects\\videos\\video_1.mp4'),
    output_file_path=os.path.join(execution_path, 'P:\\Python Projects\\Learning_projects\\videos\\video_1_new.mp4'),
    frames_per_second=20,
    log_progress=True)
print(vedeo_path)
"""
# Для прямого наблюдения с камеры
"""
video_path = detector_2.detectObjectsFromVideo(
    camera_input=camera,
    output_file_path=os.path.join(execution_path, 'P:\\Python Projects\\Learning_projects\\videos\\camera_new.mp4'),
    frames_per_second=20,
    log_progress=True,
    minimum_percentage_probability=30)
print(video_path)
"""

# Пользовательская функция, активирующаяся каждый раз, когда в кадре появляется новый объект
"""
def forFrame(frame_number, output_array, output_count):
    print("Frame nomber:", frame_number)
    print("Массив параметров найденных объектов:", output_array)
    print("Количество найденных объектов:", output_count)
    print('_______End_of_Frame_______')


execution_path = os.getcwd()
input_camera_0 = cv2.VideoCapture(0)

video_detector_4 = VideoObjectDetection()
video_detector_4.setModelTypeAsRetinaNet()
video_detector_4.setModelPath(os.path.join(execution_path, 'P:\\Python Projects\\Learning_projects\\Nets'
                                                           '\\ObjectDetection\\resnet50_coco_best_v2.1.0.h5'))
video_detector_4.loadModel()

video_detector_4.detectObjectsFromVideo(
    # Для загруженного видео
    input_file_path=os.path.join(execution_path,  'P:\\Python Projects\\Learning_projects\\videos\\video_1.mp4'),
    output_file_path=os.path.join(execution_path, 'P:\\Python Projects\\Learning_projects\\videos\\video_1_new_2.mp4'),
    # Для видео получаемого с камеры в данный момент
    # camera_input=input_camera_0,
    # output_file_path=os.path.join(execution_path, 'P:\\Python Projects\\Learning_projects\\videos\\camera_new_1.mp4'),
    # log_progress=True,
    frames_per_second=20,
    per_frame_function=forFrame,
    minimum_percentage_probability=30)
"""
# Пользовательская функция, обрабатывающая видеофайл и визуализирует в режиме реального времени
execution_path = os.getcwd()

# Библиотека соответсвующих предметов и их цвета
color_index = {'bus': 'red', 'handbag': 'steelblue', 'giraffe': 'orange', 'spoon': 'gray', 'cup': 'yellow',
               'chair': 'green', 'elephant': 'pink', 'truck': 'indigo', 'motorcycle': 'azure', 'refrigerator': 'gold',
               'keyboard': 'violet', 'cow': 'magenta', 'mouse': 'crimson', 'sports ball': 'raspberry',
               'horse': 'maroon', 'cat': 'orchid', 'boat': 'slateblue', 'hot dog': 'navy', 'apple': 'cobalt',
               'parking meter': 'aliceblue', 'sandwich': 'skyblue', 'skis': 'deepskyblue', 'microwave': 'peacock',
               'knife': 'cadetblue', 'baseball bat': 'cyan', 'oven': 'lightcyan', 'carrot': 'coldgrey',
               'scissors': 'seagreen', 'sheep': 'deepgreen', 'toothbrush': 'cobaltgreen', 'fire hydrant': 'limegreen',
               'remote': 'forestgreen', 'bicycle': 'olivedrab', 'toilet': 'ivory', 'tv': 'khaki',
               'skateboard': 'palegoldenrod', 'train': 'cornsilk', 'zebra': 'wheat', 'tie': 'burlywood',
               'orange': 'melon', 'bird': 'bisque', 'dining table': 'chocolate', 'hair drier': 'sandybrown',
               'cell phone': 'sienna', 'sink': 'coral', 'bench': 'salmon', 'bottle': 'brown', 'car': 'silver',
               'bowl': 'maroon', 'tennis racket': 'palevilotered', 'airplane': 'lavenderblush', 'pizza': 'hotpink',
               'umbrella': 'deeppink', 'bear': 'plum', 'fork': 'purple', 'laptop': 'indigo', 'vase': 'mediumpurple',
               'baseball glove': 'slateblue', 'traffic light': 'mediumblue', 'bed': 'navy', 'broccoli': 'royalblue',
               'backpack': 'slategray', 'snowboard': 'skyblue', 'kite': 'cadetblue', 'teddy bear': 'peacock',
               'clock': 'lightcyan', 'wine glass': 'teal', 'frisbee': 'aquamarine', 'donut': 'mincream',
               'suitcase': 'seagreen', 'dog': 'springgreen', 'banana': 'emeraldgreen', 'person': 'honeydew',
               'surfboard': 'palegreen', 'cake': 'sapgreen', 'book': 'lawngreen', 'potted plant': 'greenyellow',
               'toaster': 'ivory', 'stop sign': 'beige', 'couch': 'khaki'}

"""
resized = False


# Функция вывода изображения из обработанного видео файла с использованием
# Необязательного параметра per_second_function
def forSeconds(second_number, output_arras, count_arrays, average_output_count):
    print('Секунда:', second_number)
    print('Массив выходных данных каждого кадра:', output_arras)
    print('Массив подсчета уникальных объектов в каждом кадре:', count_arrays)
    print('Среднее количество уникальных объектов в последнюю секунду:', average_output_count)
    print('__________Data ending in that second__________')


# Функция вывода значения из обработанного видеофайла
def forFrame(frame_number, output_array, output_count, returned_frame):
    plt.clf()
    # Пустые массивы значений
    this_colors = []
    labels = []
    sizes = []

    # Счетчик
    counter = 0

    # Добавление найденных значений в массивы
    for eachItem in output_count:
        counter += 1
        labels.append(eachItem + ' = ' + str(output_count[eachItem]))
        sizes.append(output_count[eachItem])
        this_colors.append(color_index[eachItem])

    global resized

    #
    if resized == False:
        manager = plt.get_current_fig_manager()
        manager.resize(width=2000, height=1000)
        resized = True

    # Вывод результатов
    plt.subplot(1, 2, 1)
    plt.title("Frame :" + str(frame_number))
    plt.axis('off')
    plt.imshow(returned_frame, interpolation='none')

    plt.subplot(1, 2, 2)
    plt.title('Analysis: ' + str(frame_number))
    plt.pie(sizes, labels=labels, colors=this_colors, shadow=True, startangle=140, autopct='%1.1f%%')

    plt.pause(0.01)


# Функция, обрабатывабщая видеофайл и находящая в нем объекты в режиме реально времени и визуализирует результаты
# после кажой секнды обработки
resized = False


def forSecond(frame_number, output_arrays, count_arrays, average_count, returned_frame):
    plt.clf()

    this_colors = []
    labels = []
    sizes = []

    counter = 0

    for eachItem in average_count:
        counter += 1
        labels.append(eachItem + ' = ' + str(average_count[eachItem]))
        sizes.append(color_index[eachItem])

    global resized

    if resized == False:
        manager = plt.get_current_fig_manager()
        manager.resize(width=2000, height=1000)
        resized = True

    # Вывод результатов
    plt.subplot(1, 2, 1)
    plt.title("Frame :" + str(frame_number))
    plt.axis('off')
    plt.imshow(returned_frame, interpolation='none')

    plt.subplot(1, 2, 2)
    plt.title('Analysis: ' + str(frame_number))
    plt.pie(sizes, labels=labels, colors=this_colors, shadow=True, startangle=140, autopct='%1.1f%%')

    plt.pause(0.01)


def forMinute(minute_number, output_arrays, count_arrays, average_output_count):
    print('Минукта:', minute_number)
    print('Массив выходных данных каждого кадра:', output_arrays)
    print('Массив подсчета уникальных объектов в каждом кадре:', count_arrays)
    print('Среднее количество уникальных объектов за минуту:', average_output_count)
    print('__________Data ending in that minute__________')


# Запуск детектора

video_detector_4 = VideoObjectDetection()
video_detector_4.setModelTypeAsRetinaNet()
video_detector_4.setModelPath(os.path.join(execution_path, 'P:\\Python Projects\\Learning_projects\\Nets'
                                                           '\\ObjectDetection\\resnet50_coco_best_v2.1.0.h5'))
video_detector_4.loadModel()

plt.show()

# Обработка с загруженного видео
video_detector_4.detectObjectsFromVideo(
    input_file_path=os.path.join(execution_path, 'P:\\Python Projects\\Learning_projects\\videos\\video_1.mp4'),
    output_file_path=os.path.join(execution_path, 'P:\\Python Projects\\Learning_projects\\'
                                                  'videos\\video_1_new_4.avi'),
    # Для видео получаемого с камеры в данный момент
    # camera_input=input_camera_0,
    # output_file_path=os.path.join(execution_path, 'P:\\Python Projects\\Learning_projects\\videos\\camera_new_2.mp4'),
    # log_progress=True,
    frames_per_second=20,
    # 1) Использование функции для вывода данных в виде графика и просмотра обработки в режиме реально времени
    # per_frame_function=forFrame,
    # minimum_percentage_probability=35,
    # return_detected_frame=True
    # 2) Использование функции для вывода данных в виде текстовой информации посекундно
    # per_second_function=forSeconds,
    # minimum_percentage_probability=90
    # 3) Использование функции для вывода данных в режиме реального времени с использованием функции forSecond
    per_second_function=forSecond,
    minimum_percentage_probability=30,
    return_detected_frame=True,
    log_progress=True
    # 4) использование функции для вывода данных за минуту с использованием функции forMinute
    # per_second_function= forMinute,
    # minimum_percentage_probability=30,
    # returned_detected_frame=True,
    # log_progress=True
)
"""
# обрабатывание иформации с камеры

camera = cv2.VideoCapture(0)

detector_5 = VideoObjectDetection()
detector_5.setModelTypeAsRetinaNet()
detector_5.setModelPath(os.path.join(execution_path, 'P:\\Python Projects\\Learning_projects\\Nets\\ObjectDetection'
                                                     '\\resnet50_coco_best_v2.1.0.h5'))
detector_5.loadModel()

video_path = detector_5.detectObjectsFromVideo(camera_input=camera,
                                               output_file_path=os.path.join(execution_path, 'P:\\Python '
                                                                                             'Projects'
                                                                                             '\\Learning_projects'
                                                                                             '\\videos\\camera_new_2'),
                                               frames_per_second=20,
                                               log_progress=True,
                                               minimum_percentage_probability=40,
                                               detection_timeout=2)