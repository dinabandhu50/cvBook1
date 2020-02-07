# import the necessary packages
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from pyimagesearch.nn.conv.minivggnet import MiniVGGNet
from keras.optimizers import SGD
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import argparse
# import the matplotlib backend
# so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")


EPOCHS = 100
# construch the argument parse and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
                help="path to the output loss/accuracy plot")
args = vars(ap.parse_args())

# load the training and testing data,
# then scale it into the range [0,1]
print("[INFO] loading CIFAR-10 data...")
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_test = x_test.astype("float")/255.0
x_train = x_train.astype("float")/255.0

# convert the labels from intehers to vectors
lb = LabelBinarizer()
y_test = lb.fit_transform(y_test)
y_train = lb.fit_transform(y_train)

# initialize the label names for the CIFAR-10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer",
              "dog", "frog", "horse", "ship", "truck"]

# initialize the optimizer and model
print("[INFO] compiling...")
opt = SGD(lr=0.01, decay=0.01/EPOCHS, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy",
              optimizer=opt, metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit(x_train, y_train, validation_data=(x_test, y_test),
              batch_size=64, epochs=EPOCHS, verbose=1)

# evaluate the network
print("[INFO] evaluating network...")
pred = model.predict(x_test, batch_size=64)
print(classification_report(y_test.argmax(axis=1),
                            pred.argmax(axis=1), target_names=labelNames))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, EPOCHS), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, EPOCHS), H.history["val_acc"], label="val_acc")
plt.title("Trainging loss and Accuracy on CIFAR1-0")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])
