from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import Model
from tensorflow.python.keras.applications import resnet as rn
from tensorflow.python.keras.layers import Dense, Input
from keras import backend as K
import tensorflow as tf
import keras
from keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.applications.resnet import preprocess_input

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
config = tf.ConfigProto( device_count = {'GPU': 1, 'CPU': 128} )
sess = tf.Session(config=config)
keras.backend.set_session(sess)

"""Parameter"""
img_width, img_height = 224, 224
train_data_dir = 'data/train'               #Training image directory
validation_data_dir = 'data/validation'     #Testing image directory
nb_train_samples = 80407                    #Number of training Samples
nb_validation_samples = 23535               #Number of testing Samples
epochs = 20                                 #Training Epochs
batch_size = 16                             #Batch Size
num_class = 7                               #Number of classes in the classification task

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
# Specify Model
new_input = Input(shape = (224, 224, 3))
model = rn.ResNet50(include_top=False, pooling='avg', weights=None, input_tensor=new_input)
x = model.output
fc = Dense(num_class,activation='softmax', name='Output')(x)
new_model = Model(inputs=model.inputs, outputs = fc)
new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    preprocessing_function = preprocess_input,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=True
)

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=True)

checkpoint = ModelCheckpoint("Weedling_Best.h5",
                             monitor='val_acc',
                             verbose = 2,
                             save_best_only = True,
                             mode='auto',
                             period = 1,
                             save_weights_only=False)
new_model.fit_generator(
    train_generator,
    steps_per_epoch = nb_train_samples // batch_size,
    epochs = epochs,
    validation_data = validation_generator,
    verbose = 1,
    callbacks=[checkpoint]
)

new_model.save('Weedling_Last.h5')