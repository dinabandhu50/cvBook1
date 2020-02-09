# import the necessary packages
import matplotlib
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from pyimagesearch.nn.conv.minivggnet import MiniVGGNet
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import argparse


BATCH_SIZE = 64
EPOCHS = 5
# set the matplotlib backend so figures can be saved
# in the background
matplotlib.use("Agg")


def step_decay(epoch):
    # initialize the base initial learning rate,
    # drop factor and epochs to drop every
    initAlpha = 0.01
    factor = 0.25
    dropEvery = 5

    # compute learning rate for the current epoch
    alpha = initAlpha * (factor ** np.floor((1 + epoch) / dropEvery))

    # return the learning rate
    return float(alpha)


# construct the argument parse and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
                help="path to the loss/accuracy plot")
args = vars(ap.parse_args())

# load the training and testing data,
# then scale it into the range [0,1]
print("[INFO] loading CIFAR-10 data...")
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype("float")/255.0
x_test = x_test.astype("float")/255.0

# convert the label from integer to vector
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.fit_transform(y_test)

# initialize the label names for the CIFAR-10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer",
              "dog", "frog", "horse", "ship", "truck"]

# define the set of callbacks
# to be passed to the model during training
callbacks = [LearningRateScheduler(step_decay)]

# initialize the optimizer and model
opt = SGD(lr=0.01, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy",
              optimizer=opt, metrics=["accuracy"])

# train the network
H = model.fit(x_train, y_train, validation_data=(x_test, y_test),
              batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=callbacks, verbose=1)

# evaluate the network
print("[INFO] evaluating network...")
pred = model.predict(x_test, batch_size=BATCH_SIZE)
print(classification_report(y_test.argmax(axis=1),
                            pred.argmax(axis=1), target_names=labelNames))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, EPOCHS), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, EPOCHS), H.history["val_acc"], label="val_acc")
plt.title("Training loss and Accuracy on CIFAR-10")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])
