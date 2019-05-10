from keras.layers import Input, Conv2D, Activation, BatchNormalization, GaussianNoise, add, UpSampling2D, Dropout, Concatenate, Merge
from keras.layers.merge import concatenate
from keras.models import Model
from keras.regularizers import l2
import tensorflow as tf
from keras.engine.topology import Layer
from keras.engine import InputSpec
from keras.utils import conv_utils
from keras.layers.advanced_activations import ELU

# ==================================================================================
def spatial_reflection_2d_padding(x, padding=((1, 1), (1, 1)), data_format=None):
    """Pads the 2nd and 3rd dimensions of a 4D tensor.
    """
    assert len(padding) == 2
    assert len(padding[0]) == 2
    assert len(padding[1]) == 2
    if data_format is None:
        data_format = image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format ' + str(data_format))

    if data_format == 'channels_first':
        pattern = [[0, 0],
                   [0, 0],
                   list(padding[0]),
                   list(padding[1])]
    else:
        pattern = [[0, 0],
                   list(padding[0]), list(padding[1]),
                   [0, 0]]
    return tf.pad(x, pattern, "REFLECT")

# ==================================================================================
class ReflectionPadding2D(Layer):
    """Reflection-padding layer for 2D input (e.g. picture).
    """
    
    def __init__(self,
                 padding=(1, 1),
                 data_format=None,
                 **kwargs):
        super(ReflectionPadding2D, self).__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)
        if isinstance(padding, int):
            self.padding = ((padding, padding), (padding, padding))
        elif hasattr(padding, '__len__'):
            if len(padding) != 2:
                raise ValueError('`padding` should have two elements. '
                                 'Found: ' + str(padding))
            height_padding = conv_utils.normalize_tuple(padding[0], 2,
                                                        '1st entry of padding')
            width_padding = conv_utils.normalize_tuple(padding[1], 2,
                                                       '2nd entry of padding')
            self.padding = (height_padding, width_padding)
        else:
            raise ValueError('`padding` should be either an int, '
                             'a tuple of 2 ints '
                             '(symmetric_height_pad, symmetric_width_pad), '
                             'or a tuple of 2 tuples of 2 ints '
                             '((top_pad, bottom_pad), (left_pad, right_pad)). '
                             'Found: ' + str(padding))
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            if input_shape[2] is not None:
                rows = input_shape[2] + self.padding[0][0] + self.padding[0][1]
            else:
                rows = None
            if input_shape[3] is not None:
                cols = input_shape[3] + self.padding[1][0] + self.padding[1][1]
            else:
                cols = None
            return (input_shape[0],
                    input_shape[1],
                    rows,
                    cols)
        elif self.data_format == 'channels_last':
            if input_shape[1] is not None:
                rows = input_shape[1] + self.padding[0][0] + self.padding[0][1]
            else:
                rows = None
            if input_shape[2] is not None:
                cols = input_shape[2] + self.padding[1][0] + self.padding[1][1]
            else:
                cols = None
            return (input_shape[0],
                    rows,
                    cols,
                    input_shape[3])

    def call(self, inputs):
        return spatial_reflection_2d_padding(inputs,
                                    padding=self.padding,
                                    data_format=self.data_format)

    def get_config(self):
        config = {'padding': self.padding,
                  'data_format': self.data_format}
        base_config = super(ReflectionPadding2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))




# ==================================================================================
def keepsize(nx, ny, nd, nq, noise, activation='relu', n_filters=32, l2_reg=1e-7):
    """ keepsize - Concatenate
    """

    def minires(inputs, n_filters):
        x = ReflectionPadding2D()(inputs)
        x = Conv2D(int(n_filters), (3, 3), padding='valid', 
            kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg))(x)
        x = ELU(alpha=1.0)(x)
        x = ReflectionPadding2D()(x)
        x = Conv2D(n_filters, (3, 3), padding='valid', 
            kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg))(x)
        return x

    def myblock(inputs, n_filters):
        x = ReflectionPadding2D()(inputs)
        x = Conv2D(n_filters, (3, 3), padding='valid', 
            kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg))(x)
        xo = ELU(alpha=1.0)(x)
        x = ReflectionPadding2D()(xo)
        x = Conv2D(n_filters, (3, 3), padding='valid', 
            kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg))(x)
        x = ELU(alpha=1.0)(x)
        x = add([x, xo])
        return x

    inputs = Input(shape=(nx, ny, nd)) # depth goes last in TensorFlow
    x = GaussianNoise(noise)(inputs)

# mode: concatenate
    x1 = myblock(x, n_filters)
    x1 = minires(x1, int(nq/7))

    x2 = myblock(x, n_filters)
    x2 = minires(x2, int(nq/7))

    x3 = myblock(x, n_filters)
    x3 = minires(x3, int(nq/7))

    x4 = myblock(x, n_filters)
    x4 = minires(x4, int(nq/7))

    x5 = myblock(x, n_filters)
    x5 = minires(x5, int(nq/7))

    x6 = myblock(x, n_filters)
    x6 = minires(x6, int(nq/7))

    x7 = myblock(x, n_filters)
    x7 = minires(x7, int(nq/7))

    final = concatenate([x1, x2, x3, x4, x5, x6, x7])

    return Model(inputs=inputs, outputs=final)
