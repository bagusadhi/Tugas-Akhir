########################################################################
# import python-library
########################################################################
# from import
import tensorflow.keras.models
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Activation
from tensorflow.keras.models import Model


########################################################################
# keras model
########################################################################
def get_model(input_dim, lr):
    """
    define the keras model
    the model based on the simple dense auto encoder 
    (128*128*128*128*8*128*128*128*128)
    """

    x = Input(shape=(input_dim,))

    h = Dense(128)(x)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(128)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(128)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(128)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(8)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)
    
    h = Dense(128)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(128)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(128)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(128)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(input_dim)(h)

    model = Model(inputs=x, outputs=h)

    model.compile(optimizer=tensorflow.keras.optimizers.Adam(lr=lr), 
                  loss='mean_squared_error')

    return model

#########################################################################

def load_model(file_path):
    return tensorflow.keras.models.load_model(file_path, compile=False)

def clear_session():
    K.clear_session()
    