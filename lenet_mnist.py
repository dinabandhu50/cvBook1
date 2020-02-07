# import the necessary packages
from pyimagesearch.nn.conv.lenet import Lenet
from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.datasets import mnist
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np

EPOCHS = 10
BATCH_SIZE = 128

print("[INFO] accessing MNIST...")
# scale the input data to the range [0,1]
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype("float") / 255.0
x_test = x_test.astype("float") / 255.0

# if we are using "channels first " ordering,
# then reshape the design matrix such that the matrix is:
# num_samples x depth x rows x columns
if K.image_data_format() == "channels_first":
    x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
    x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)
# otherwise, we are using "channels last" ordering.
# so the design matrix shape should be:
# num_samples x rows x columns x depth
else:
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# convert the labels form integers to vectors
le = LabelBinarizer()
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=0.01)
model = Lenet.build(width=28, height=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy",
              optimizer=opt, metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit(x_train, y_train, validation_data=(x_test, y_test),
              batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1)

# evaluate the network
print("[INFO] evaluating network...")
pred = model.predict(x_test, batch_size=BATCH_SIZE)
print(classification_report(y_test.argmax(axis=1), pred.argmax(
    axis=1), target_names=[str(x) for x in le.classes_]))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, EPOCHS), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, EPOCHS), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
