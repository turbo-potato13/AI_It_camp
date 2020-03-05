#Добавляем библиотеки
from imageai.Detection import ObjectDetection#Библиотека для распознования объектов
import os#Библиотека для работы с операционной системой

exec_path=os.getcwd()#Указываем путь к этому проекту

detector=ObjectDetection()
detector.setModelTypeAsRetinaNet()# Использование RetinaNet, для распознавания
detector.setModelPath(os.path.join(
   exec_path, 'resnet50_coco_best_v2.0.1 (1).h5')#Добавление файла RetinaNet
)
detector.loadModel()#Загружаем модель

list=detector.detectObjectsFromImage(input_image=os.path.join(exec_path,
"obj.jpg"),
                                     output_image_path=os.path.join(exec_path,
"newobj.jpg")
                                     )