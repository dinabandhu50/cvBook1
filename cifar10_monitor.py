# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from pyimagesearch.callbacks.trainingmonitor import TrainingMonitor
from pyimagesearch.nn.conv.minivggnet import MiniVGGNet
from keras.optimizers import SGD
from keras.datasets import cifar10
import argparse
import os
# set the matplotlib backend so figure can be saved in the background
import matplotlib
matplotlib.use("Agg")

EPOCHS = 2
BATCH_SIZE = 32

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
                help="path to the output dictionary")
args = vars(ap.parse_args())

# show information on the process ID
print("[INFO] process ID: {}".format(os.getpid()))
# load the training and testing data,
#  then scale it into the range[0,1]
print("[INFO] loading CIFAR-10 data...")
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype("float") / 255.0
x_test = x_test.astype("float") / 255.0

# convert the labels form integers to vectors
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.fit_transform(y_test)

# initialize the label names for the CIFAR-10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer",
              "dog", "frog", "horse", "ship", "truck"]

# initialize the SGD optimizer, but without any learning rate decay
print("[INFO] compiling model...")
opt = SGD(lr=0.01, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy",
              optimizer=opt, metrics=["accuracy"])

# construct the set of callbacks
figPath = os.path.sep.join([args["output"], "{}.png".format(os.getpid())])
jsonPath = os.path.sep.join([args["output"], "{}.json".format(os.getpid())])
callbacks = [TrainingMonitor(figPath, jsonPath=jsonPath)]

# train the network
print("[INFO] training network...")
model.fit(x_train, y_train, validation_data=(
    x_test, y_test), batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=callbacks, verbose=1)
