from sklearn import svm
from  sklearn import datasets
import matplotlib.pyplot as plt

digits = datasets.load_digits()

# clf = svm.SVC(gamma=0.001,C = 100
clf = svm.SVC(gamma=0.001, C=100.)
print(len(digits.data))

x,y = digits.data[:-10], digits.target[:-10]
clf.fit(x,y)

# print('Prediction', clf.predict(digits.data[-1]))

print('Prediction:',clf.predict(digits.data[-2:]))

plt.imshow(digits.images[-2], cmap=plt.cm.gray_r,interpolation="nearest")
plt.show()


# print(digits.data)
# print(digits.target)
# print(digits.images[0])