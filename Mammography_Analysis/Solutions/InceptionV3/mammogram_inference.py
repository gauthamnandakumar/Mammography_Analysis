
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras.applications.imagenet_utils import preprocess_input
import skimage.io
import numpy as np
import os
json_file = open('mamo_model_trial.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("mamo_model_trial.h5")
print("Loaded model from disk")




data_path = '/home/ddh/Projects/DDH/Caries/Mask_RCNN/samples/Mammogram/Boston Meditech Group/Test_data/' #<-----------------edit here for updating the test file
test_image = os.listdir(data_path)


result = []
for img in test_image :
	x = skimage.io.imread(data_path + img)
	image1 = skimage.color.gray2rgb(x)
	img1 = image.img_to_array(image1)
	x = np.expand_dims(img1, axis=0) * 1. / 255
	prediction = loaded_model.predict(x)
	if prediction < 0.5:
		prediction = 0
		class_name = 'benign'
	else:
		prediction = 1
		class_name = 'malignant'
	result.append(prediction)
	print("The image " + str(img) + " is classified as " + str(class_name))



print(result)