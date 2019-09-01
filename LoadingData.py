
import  numpy as np
import matplotlib.pyplot as plt
import  os
import  cv2

# DATADIR = "traffic-signs-data/custom"
DATADIR = "PetImages"

CATEGORIES = ["Dog", "Cat"]

# CATEGORIES = ["Priority Road ", "No Entry"]

for category in CATEGORIES:  # do dogs and cats
    path = os.path.join(DATADIR,category)  # create path to dogs and cats
    for img in os.listdir(path):  # iterate over each image per dogs and cats
        img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
        plt.imshow(img_array, cmap='gray')  # graph it
        # plt.show()  # display!

        break  # we just want one for now so break
    break  #...and one more!

IMG_SIZE = 80

new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap='gray')
plt.show()

training_data = []
def create_training_data():
    for category in  CATEGORIES:
        path = os.path.join(DATADIR, category)  # create path to dogs and cats
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array,class_num])
            except Exception as e:
                pass

create_training_data()

print(len(training_data))

import  random
random.shuffle(training_data)
X = []
Y = []

for features,label in training_data:
    X.append(features)
    Y.append(features)
X = np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,1)



import pickle

pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(Y, pickle_out)
pickle_out.close()

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)