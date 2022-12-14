#main app file for running models
from imageai.Detection import ObjectDetection

detector = ObjectDetection()

model_path = "./models/yolo-tiny.h5"
# webp images do not work
# should add convertor
input_path = "./input/nyctest.jpeg"

output_path = "./output/newimage.jpg"

detector.setModelTypeAsTinyYOLOv3()

detector.setModelPath(model_path)

detector.loadModel()

detection = detector.detectObjectsFromImage(input_image=input_path, output_image_path=output_path)

for eachItem in detection:
    print(eachItem["name"] , " : ", eachItem["percentage_probability"])