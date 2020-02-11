# import the necessary packages
from pyimagesearch.nn.conv.lenet import Lenet
from keras.utils import plot_model

# initialize LeNet and then write he network
# architecture visualization graph to disk
model = Lenet.build(28, 28, 1, 10)
plot_model(model, to_file="./architecture/lenet.png", show_shapes=True)
