# repeats run05 with different seeds from 1 to 10
# random weights
# VGG16 from applications adjusted to 5 classes, no top_model stacked on top
# small batch size of 2
# Shuffle false



###############################################
# importing libraries and setting variables   #
###############################################
# import rest of libraries
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.layers import Input
from keras import optimizers
import itertools
import pickle

# path to the model weights files.
model_path = '/path/to/store/model' +str(seed)+'.h5'
weights_path = '/path/to/store/weights' +str(seed)+'.h5'
training_history_path = '/path/to/store/training_history' +str(seed)+'.pickle'
checkpoint_path = "/patch/to/store/checkpoint_weights" +str(seed)+"_weights-improvement-{epoch:02d}-{val_acc:.2f}.h5"
log_dir = "/patch/to/store/logs" +str(seed)+"_logs"
train_data_dir = 'path/to/training_ddataset'
validation_data_dir = 'path/to/validation_dataset'
nb_train_samples = 8000
nb_validation_samples = 2000
epochs = 10
batch_size = 2
img_width, img_height = 224, 224
title_cm = 'run19 seed' +str(seed)

print "dataset: name"
print "Number of training samples: " + str(nb_train_samples)
print "Number of validation samples: " + str(nb_validation_samples)
print "Number of epochs: " + str(epochs)
print "batch size: " + str(batch_size)
print "image: " + str(img_width) + " " + str(img_height)

###################################
# load, compile and train model   #
###################################
input_tensor = Input(shape=(224, 224,3))
model = VGG16(weights=None, include_top=True, input_tensor=input_tensor, classes=5)
print('Model loaded.')

# data preprocessing
train_gen = ImageDataGenerator(rescale=1. / 255)
val_gen = ImageDataGenerator(rescale=1. / 255)

train_batches = train_gen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

val_batches = val_gen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

# compile model
model.compile(optimizer=optimizers.Adam(lr=1e-4),
              loss='categorical_crossentropy', metrics=['accuracy'])
print ('Model compiled')

# use checkpoints during training to save weights and be able to check stats or quit training and go back
checkpointer = ModelCheckpoint(checkpoint_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early_stopping = EarlyStopping(patience=5)
tbCallBack = TensorBoard(log_dir, histogram_freq=0, write_graph=True, write_images=True)

print "checkpointer set, path: " + checkpoint_path
print "logdir: " + log_dir

# fine-tune the model
history = model.fit_generator(
    train_batches,
    steps_per_epoch=4000,
    epochs=epochs,
    callbacks=[early_stopping, checkpointer, tbCallBack],
    validation_data=val_batches,
    validation_steps=1000, 
    verbose = 0)

print "history dictionary: "
print history.history


###################################
# save history, weights, model    #
###################################

with open(training_history_path, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

file_pi.close()

#save weights
model.save_weights(weights_path)

#save model
model.save(model_path)

exit()

