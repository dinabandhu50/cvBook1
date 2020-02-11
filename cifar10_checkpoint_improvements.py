# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from pyimagesearch.nn.conv.minivggnet import MiniVGGNet
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras.datasets import cifar10
import argparse
import os

EPOCHS = 40
BATCH_SIZE = 64
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weights", required=True,
                help="path to weights directory")
args = vars(ap.parse_args())

# load the training and testing data,
# then scale it into the range [0,1]
print("[INFO] loading CIFAR-10 data...")
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype("float") / 255.0
x_test = x_test.astype("float") / 255.0

# convert the labels from integers to vectors
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.fit_transform(y_test)

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=0.01, decay=0.01/EPOCHS, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy",
              optimizer=opt, metrics=["accuracy"])

# construct the callback to save only the best model
# to the disk based on the validation loss
fname = os.path.sep.join(
    [args["weights"], "weights-{epoch:03d}-{val_loss:.4f}.hdf5"])
checkpoint = ModelCheckpoint(
    fname, monitor="val_loss", mode="min", save_best_only=True, verbose=1)
callbacks = [checkpoint]

# train the network
print("[INFO] training network...")
H = model.fit(x_train, y_train, validation_data=(x_test, y_test),
              batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=callbacks, verbose=2)
