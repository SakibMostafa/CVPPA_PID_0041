from tensorflow.python.keras import models as kr
import numpy as np
from matplotlib import pyplot as plt
from keras.preprocessing import image
import tensorflow as tf
from tensorflow.python.framework import ops
import keras
from keras import backend as K

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.99
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))
from tensorflow.python.keras.applications.resnet import preprocess_input

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
config = tf.ConfigProto( device_count = {'GPU': 1, 'CPU': 128} )
sess = tf.Session(config=config)
keras.backend.set_session(sess)


# input image dimensions
img_rows, img_cols = 224, 224
H, W = img_rows, img_cols

def build_model():
    return kr.load_model('models/Weedling/2_Conv_ResNet_Full.h5')    #Update the path to the saved CNN Model

"""Load and preprocess image."""
def load_image(path, preprocess=True):
    x = image.load_img(path, target_size=(H, W))
    if preprocess:
        x = image.img_to_array(x)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)                                                                                 #CHANGE HERE
    return x


def deprocess_image(x):
    x = x.copy()
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.25
    x += 0.5
    x = np.clip(x, 0, 1)
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def build_guided_model():
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                   tf.cast(op.inputs[0] > 0., dtype)

    g = K.get_session().graph
    with g.gradient_override_map({'Relu': 'GuidedBackProp'}):
        new_model = build_model()
    return new_model


def guided_backprop(input_model, images, layer_name):
    input_imgs = input_model.input
    layer_output = input_model.get_layer(layer_name).output
    grads = K.gradients(layer_output, input_imgs)[0]
    backprop_fn = K.function([input_imgs, K.learning_phase()], [grads])
    grads_val = backprop_fn([images, 0])[0]
    return grads_val

def compute_saliency(guided_model, layer_name, image_Path):
    x = load_image(image_Path)
    gb = guided_backprop(guided_model, x, layer_name)
    return gb

model = build_model()
model.summary()
guided_model = build_guided_model()
image_Name = './images/Plant_Village/Image_0.jpg'         #Change the path to the image
image_Path_Name = [image_Name]

counter = 0
for layer in model.layers:
    name = layer.name
    if ('pad' not in name) and ('bn' not in name) and ('pool' not in name) and ('relu' not in name) and ('add' not in name) and ('conv' in name):
        counter += 1
        gb = compute_saliency(
            guided_model,
            layer_name=name,  # CHANGE HERE
            image_Path=image_Path_Name[0]
        )
        deprocessed_Image = np.array(np.flip(deprocess_image(gb[0]), -1))
        fileP = './results/Weedling/Small_Resnet_Full/Class_0_/' +str(counter)+ '.png'         #Change the path to save the images
        plt.imsave(fileP, deprocessed_Image)