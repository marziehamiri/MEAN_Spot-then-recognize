from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.compat.v1 as tf
from tensorflow.keras.regularizers import l2

def MEAN_Spot():
    inputs1 = layers.Input(shape=(42,42,1))
    conv1 = layers.Conv2D(3, (5,5), padding='same', activation='relu')(inputs1)
    pool1 = layers.MaxPooling2D(pool_size=(3, 3), strides=(3,3))(conv1)
    # channel 2
    inputs2 = layers.Input(shape=(42,42,1))
    conv2 = layers.Conv2D(5, (5,5), padding='same', activation='relu')(inputs2)
    pool2 = layers.MaxPooling2D(pool_size=(3, 3), strides=(3,3))(conv2)
    # channel 3
    inputs3 = layers.Input(shape=(42,42,1))
    conv3 = layers.Conv2D(8, (5,5), padding='same', activation='relu')(inputs3)
    pool3 = layers.MaxPooling2D(pool_size=(3, 3), strides=(3,3))(conv3)
    # merge
    merged = layers.Concatenate()([pool1, pool2, pool3])
    # interpretation
    merged_pool = layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2))(merged)
    flat = layers.Flatten()(merged_pool)
    dense = layers.Dense(400, activation='relu')(flat)
    outputs = layers.Dense(1, activation='linear', name='spot')(dense)
    #Takes input u,v,s
    model = keras.models.Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
    # compile
    sgd = keras.optimizers.SGD(lr=0.0005)
    model.compile(loss={'spot':'mse'}, 
        optimizer=sgd, 
        metrics={'spot':tf.keras.metrics.MeanAbsoluteError()}, )
    return model
# Transfer learning by removing and freeze layers
def MEAN_Recog_TL(model_spot, opt, emotion_class):
    for layer in model_spot.layers:
        layer.trainable = False
    # Until last convolutional later
    merged = model_spot.layers[-5].output 
    merged_conv = layers.Conv2D(8, (5,5), padding='same', activation='relu', kernel_regularizer=l2(0.1))(merged)
    merged_pool = layers.MaxPooling2D(pool_size=(2, 2), padding='same', strides=(2,2))(merged_conv)
    flat = layers.Flatten()(merged_pool)
    flat_do = layers.Dropout(0.2)(flat)
    outputs = layers.Dense(emotion_class, activation = "softmax", name='recog')(flat_do)
    
    model =  keras.models.Model(inputs = model_spot.input, outputs = outputs)
    model.compile(
        loss={'recog':'categorical_crossentropy'}, 
        optimizer=opt, 
        metrics={'recog':tf.keras.metrics.CategoricalAccuracy()}
    )
    return model

def MEAN_Spot_Recog_TL(model_spot, model_recog, opt):
    outputs1 = model_spot.layers[-1].output 
    outputs2 = model_recog.layers[-1].output 
    model = keras.models.Model(inputs = [model_spot.input], outputs = [outputs1, outputs2])
    model.compile(
        loss={'spot':'mse', 'recog':'categorical_crossentropy'}, 
        optimizer=opt, 
        metrics={'spot':tf.keras.metrics.MeanAbsoluteError(), 'recog':tf.keras.metrics.CategoricalAccuracy()}
    )
    return model