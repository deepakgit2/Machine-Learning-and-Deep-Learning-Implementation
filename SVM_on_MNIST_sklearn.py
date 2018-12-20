# It is the implementation of SVM on MNIST dataset
# using sklearn

from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn import svm
import time

# Loading dataset (we are using 60000 for training and testing)
(X, Y), (_, _) = mnist.load_data()
X = X.astype('float32')
# Normalizing
X /= 255

# As our input image is image so we have to resize our data
X.resize(X.shape[0],784)
# Splitting our data into training and testing
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.30, random_state=4)

# Training Model using Gaussian kernel
tic = time.time()
model = svm.SVC(kernel='rbf')
model.fit(x_train, y_train)
print 'training time:', time.time()-tic

y_pred_mod = model.predict(x_test)
accuracy = model.score(x_test, y_test)
print 'Accuracy is:', accuracy