import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input, Conv3D, Dropout, Dense, Reshape, Embedding, Conv3DTranspose, Concatenate, Activation, MultiHeadAttention, LayerNormalization, GlobalMaxPooling1D, Layer
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Flatten, Dense, Input
from tensorflow.keras.layers import Conv3D, MaxPool3D, Flatten, Dense, ReLU, AveragePooling3D, LeakyReLU, Add
# -----------------------------------
#        MODEL ARCHITECTURE
# -----------------------------------

class ENCODER:
    @staticmethod
    def build(reg=l2(), shape=(193, 229, 193), init='he_normal'):
        # # Create the model
        i = Input(shape=(*shape, 1))

        # # The first two layers will learn a total of 64 filters with a 3x3x3 kernel size
        # o = Conv3D(16, (3, 3, 3), strides=(1, 1, 1), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg)(i)
        # d1 = Conv3D(16, (3, 3, 3), strides=(1, 1, 1), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg, name='d1')(o)
        # o = Conv3D(16, (2, 2, 2), strides=(2, 2, 2))(d1)  # Down-sampling (192, 128, 8)
        #
        # # Stack two more layers, keeping the size of each filter as 3x3x3 but increasing to 128 total learned filters
        # o = Conv3D(32, (3, 3, 3), strides=(1, 1, 1), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg)(o)
        # d2 = Conv3D(32, (3, 3, 3), strides=(1, 1, 1), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg, name='d2')(o)
        # o = Conv3D(32, (2, 2, 2), strides=(2, 2, 2))(d2)  # Down-sampling (96, 64, 4)
        #
        # # Stack two more layers, keeping the size of each filter as 3x3x3 but increasing to 256 total learned filters
        # o = Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg)(o)
        # d3 = Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg, name='d3')(o)
        # o = Conv3D(64, (2, 2, 2), strides=(2, 2, 2))(d3)  # Down-sampling  (48, 32, 2)
        #
        # # Stack two more layers, keeping the size of each filter as 3x3x3 but increasing to 256 total learned filters
        # o = Conv3D(128, (3, 3, 3), strides=(1, 1, 1), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg)(o)
        # d4 = Conv3D(128, (3, 3, 3), strides=(1, 1, 1), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg, name='d4')(o)
        # o = Conv3D(1, (2, 2, 2), strides=(2, 2, 2))(d4)  # Down-sampling  (24, 16, 1)

        d1 = Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same', name="d1",kernel_initializer=init,kernel_regularizer=reg)(i)
        x = BatchNormalization(name="norm21")(d1)
        x = MaxPool3D(pool_size=(2, 2, 2), strides=(3, 3, 3), padding='same', name="maxpool21")(x)
        x = ReLU()(x)

        # block 2
        d2 = Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same', name="d2",kernel_initializer=init,kernel_regularizer=reg)(x)
        x = BatchNormalization(name="norm22")(d2)
        x = MaxPool3D(pool_size=(2, 2, 2), strides=(3, 3, 3), padding='same', name="maxpool22")(x)
        x = ReLU()(x)
        x = Dropout(0.2)(x)  # Dropout added after ReLU

        # block 3
        d3 = Conv3D(filters=128, kernel_size=(3, 3, 3), padding='same', name="d3",kernel_initializer=init,kernel_regularizer=reg)(x)
        x = BatchNormalization(name="norm23")(d3)
        x = MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same', name="maxpool23")(x) # 2x2x2
        x = ReLU()(x)
        x = Dropout(0.2)(x)  # Dropout added after ReLU

        # # block 4
        # d4 = Conv3D(filters=128, kernel_size=(3, 3, 3), padding='same', name="d4")(x)
        # x = BatchNormalization(name="norm24")(d4)
        # x = MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same', name="maxpool24")(x)
        # x = ReLU()(x)
        #
        # # block 5
        # d5 = Conv3D(filters=256, kernel_size=(3, 3, 3), padding='same', name="d5")(x)  # 256
        # x = BatchNormalization(name="norm25")(d5)
        # x = MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same', name="maxpool25")(x)
        # x = ReLU()(x)

        # # block 6
        # d6 = Conv3D(filters=64, kernel_size=(1, 1, 1), padding='same', name="d6")(x)
        # x = BatchNormalization(name="norm26")(d6)
        # x = ReLU()(x)

        o = Conv3D(filters=1, kernel_size=(1, 1, 1), padding='same', name="d7")(x)

        # Encoder outputs
        size = o.shape[1] * o.shape[2] * o.shape[3]
        output = Reshape((1, size))(o)
        down = [d1, d2, d3] #, d4, d5, d6]
        return Model(inputs=i, outputs=[output, *down], name='Encoder')



class DECODER:
    @staticmethod
    def build(inputTensor,  reg=l2(), init='he_normal'): #down, n_classes,
        # Input bottleneck tensor and skip connections from the encoder
        o = inputTensor

        # Block 3 (Upsample from (11, 13, 11) to (22, 26, 22))
        u3 = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same', kernel_initializer=init)(o)
        #concat3 = Concatenate()([u3, down[2]])  # Add skip connection
        o = Conv3D(128, (3, 3, 3), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg)(
            u3)#concat3
        o = Conv3D(128, (3, 3, 3), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg)(
            o)

        # Block 2 (Upsample from (22, 26, 22) to (65, 77, 65))
        u2 = Conv3DTranspose(64, (3, 3, 3), strides=(3, 3, 3), padding='same', kernel_initializer=init)(o)
        #concat2 = Concatenate()([u2, down[1]])  # Add skip connection
        o = Conv3D(64, (3, 3, 3), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg)(
            u2)
        o = Conv3D(64, (3, 3, 3), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg)(o)

        # Block 1 (Upsample from (65, 77, 65) to (193, 229, 193))
        u1 = Conv3DTranspose(32, (3, 3, 3), strides=(3, 3, 3), padding='same', kernel_initializer=init)(o)
        #concat1 = Concatenate()([u1, down[0]])  # Add skip connection
        o = Conv3D(32, (3, 3, 3), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg)(
            u1)
        o = Conv3D(32, (3, 3, 3), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg)(o)

        # Final output layer to match original input shape
        output = Conv3D(1, (1, 1, 1), padding="same", activation='sigmoid', name='Output')(o)

        return output
        # # # Reshape to fit the decoder
        # # o = Reshape((24, 16, 1, 1))(inputTensor)  # TODO: automate reshape size
        #
        #
        # # Reshape to fit the decoder
        # o = Reshape((6, 7, 6, 1))(inputTensor)  # Adjusted reshape size
        #
        # # Up-sampling 1
        # u1 = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(o)  # Output: (12, 14, 12, 32)
        # concat1 = Concatenate()([u1, down[3]])
        # o = Conv3D(64, (3, 3, 3), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg)(
        #     concat1)
        # o = Conv3D(64, (3, 3, 3), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg)(o)
        #
        # # Up-sampling 2
        # u2 = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(o)  # Output: (24, 28, 24, 64)
        # concat2 = Concatenate()([u2, down[2]])
        # o = Conv3D(128, (3, 3, 3), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg)(
        #     concat2)
        # o = Conv3D(128, (3, 3, 3), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg)(
        #     o)
        #
        # # Up-sampling 3
        # u3 = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(o)  # Output: (48, 56, 48, 128)
        # concat3 = Concatenate()([u3, down[1]])
        # o = Conv3D(256, (3, 3, 3), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg)(
        #     concat3)
        # o = Conv3D(256, (3, 3, 3), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg)(
        #     o)
        #
        # # Up-sampling 4
        # u4 = Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), padding='same')(o)  # Output: (96, 115, 96, 256)
        # concat4 = Concatenate()([u4, down[0]])
        # o = Conv3D(256, (3, 3, 3), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg)(
        #     concat4)
        # o = Conv3D(256, (3, 3, 3), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg)(
        #     o)
        #
        # # Output layer
        # output = Conv3D(n_classes, (1, 1, 1), padding="same", name='Logit')(o)
        # output = Activation('None')(output) # softmax
        #
        # return output


class AddCLSToken(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cls_token = None

    def build(self, inputs_shape):
        initial_value = tf.zeros([1, 1, inputs_shape[2]])  # input.shape is [batch_size, num_patches, projection_dim]
        self.cls_token = tf.Variable(initial_value=initial_value, trainable=True, name="cls")

    def call(self, inputs, **kwargs):
        # Replace batch size with the appropriate value
        cls_token = tf.tile(self.cls_token, [tf.shape(inputs)[0], 1, 1])
        # Append learnable parameter [CLS] class token
        concat = tf.concat([cls_token, inputs], axis=1)  # cls token placed at the start of the sequence
        return concat
