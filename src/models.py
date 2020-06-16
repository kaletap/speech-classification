#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 17:43:51 2018

@author: chris
"""

from tensorflow.keras.layers import *
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Add, Reshape, LSTM, Bidirectional, Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras import layers


#%%
class ResNet():
    """
    Usage: 
        sr = ResNet([4,8,16], input_size=(50,50,1), output_size=12)
        sr.build()
        followed by sr.m.compile(loss='categorical_crossentropy', 
                                 optimizer='adadelta', metrics=["accuracy"])
        save plotted model with: 
            keras.utils.plot_model(sr.m, to_file = '<location>.png', 
                                   show_shapes=True)
    """
    def __init__(self,
                 filters_list=[], 
                 input_size=None, 
                 output_size=None,
                 initializer='glorot_uniform'):
        self.filters_list = filters_list
        self.input_size = input_size
        self.output_size = output_size
        self.initializer = initializer
        self.m = None        
    
    def _block(self, filters, inp):
        """ one residual block in a ResNet
        
        Args:
            filters (int): number of convolutional filters
            inp (tf.tensor): output from previous layer
            
        Returns:
            tf.tensor: output of residual block
        """
        layer_1 = BatchNormalization()(inp)
        act_1 = Activation('relu')(layer_1)
        conv_1 = Conv2D(filters, (3,3), 
                        padding = 'same', 
                        kernel_initializer = self.initializer)(act_1)
        layer_2 = BatchNormalization()(conv_1)
        act_2 = Activation('relu')(layer_2)
        conv_2 = Conv2D(filters, (3,3), 
                        padding = 'same', 
                        kernel_initializer = self.initializer)(act_2)
        return(conv_2)

    def build(self):
        """
        Returns:
            keras.engine.training.Model
        """
        i = Input(shape = self.input_size, name = 'input')
        x = Conv2D(self.filters_list[0], (3,3), 
                   padding = 'same', 
                   kernel_initializer = self.initializer)(i)
        x = MaxPooling2D(padding = 'same')(x)        
        x = Add()([self._block(self.filters_list[0], x),x])
        x = Add()([self._block(self.filters_list[0], x),x])
        x = Add()([self._block(self.filters_list[0], x),x])
        if len(self.filters_list) > 1:
            for filt in self.filters_list[1:]:
                x = Conv2D(filt, (3,3),
                           strides = (2,2),
                           padding = 'same',
                           activation = 'relu',
                           kernel_initializer = self.initializer)(x)
                x = Add()([self._block(filt, x),x])
                x = Add()([self._block(filt, x),x])
                x = Add()([self._block(filt, x),x])
        x = GlobalAveragePooling2D()(x)
        x = Dense(self.output_size, activation = 'softmax')(x)
        
        self.m = Model(i,x)
        return self.m
           
    
#%%
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args    
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)    
    
        
#%%
class CTC():
    """
    Usage:
        sr_ctc = CTC(enter input_size and output_size)
        sr_ctc.build()
        sr_ctc.m.compile()
        sr_ctc.tm.compile()
    """       
    def __init__(self,
                 input_size=None, 
                 output_size=None,
                 initializer='glorot_uniform'):
        self.input_size = input_size
        self.output_size = output_size
        self.initializer = initializer
        self.m = None
        self.tm = None
                   
    def build(self, 
              conv_filters = 196,
              conv_size = 13,
              conv_strides = 4,
              act = 'relu',
              rnn_layers = 2,
              LSTM_units = 128,
              drop_out = 0.2):
        """
        build CTC training model (self.m) and 
        prediction model without the ctc loss function (self.tm)
        
        Usage: 
            enter conv parameters for Cov1D layer
            specify number of rnn layers, LSTM units and dropout
        Args:
            
        Returns:
            self.m: keras.engine.training.Model
            self.tm: keras.engine.training.Model
        """        
        i = Input(shape = self.input_size, name = 'input')
        x = Conv1D(conv_filters, 
                   conv_size, 
                   strides = conv_strides, 
                   name = 'conv1d')(i)
        x = BatchNormalization()(x)
        x = Activation(act)(x)
        for _ in range(rnn_layers):          
            x = Bidirectional(LSTM(LSTM_units, 
                                   return_sequences = True))(x)
            x = Dropout(drop_out)(x)
            x = BatchNormalization()(x)
        y_pred = TimeDistributed(Dense(self.output_size, 
                                       activation = 'softmax'))(x)        
        # ctc inputs
        labels = Input(name='the_labels', shape=[None,], dtype='int32')
        input_length = Input(name='input_length', shape=[1], dtype='int32')
        label_length = Input(name='label_length', shape=[1], dtype='int32')    
        # Keras doesn't currently support loss funcs with extra parameters
        # so CTC loss is implemented in a lambda layer
        loss_out = Lambda(ctc_lambda_func, 
                          output_shape=(1,), 
                          name='ctc')([y_pred,
                                        labels,
                                        input_length,
                                        label_length])        
        self.tm = Model(inputs = i,
                        outputs = y_pred)
        self.m = Model(inputs = [i, 
                                 labels, 
                                 input_length, 
                                 label_length], 
                        outputs = loss_out)
        return self.m, self.tm
        
class ResNetLstm():
    """
    Usage: 
        sr = ResNet([4,8,16], input_size=(50,50,1), output_size=12)
        sr.build()
        followed by sr.m.compile(loss='categorical_crossentropy', 
                                 optimizer='adadelta', metrics=["accuracy"])
        save plotted model with: 
            keras.utils.plot_model(sr.m, to_file = '<location>.png', 
                                   show_shapes=True)
    """
    def __init__(self,
                 filters_list=[], 
                 input_size=None, 
                 output_size=None,
                 initializer='glorot_uniform'):
        self.filters_list = filters_list
        self.input_size = input_size
        self.output_size = output_size
        self.initializer = initializer
        self.m = None        
    
    def _block(self, filters, inp):
        """ one residual block in a ResNet
        
        Args:
            filters (int): number of convolutional filters
            inp (tf.tensor): output from previous layer
            
        Returns:
            tf.tensor: output of residual block
        """
        layer_1 = BatchNormalization()(inp)
        act_1 = Activation('relu')(layer_1)
        conv_1 = Conv2D(filters, (3,3), 
                        padding = 'same', 
                        kernel_initializer = self.initializer)(act_1)
        layer_2 = BatchNormalization()(conv_1)
        act_2 = Activation('relu')(layer_2)
        conv_2 = Conv2D(filters, (3,3), 
                        padding = 'same', 
                        kernel_initializer = self.initializer)(act_2)
        return(conv_2)

    def build(self):
        """
        Returns:
            keras.engine.training.Model
        """
        i = Input(shape = self.input_size, name = 'input')
        x = Conv2D(self.filters_list[0], (3,3), 
                   padding = 'same', 
                   kernel_initializer = self.initializer)(i)
        x = MaxPooling2D(padding = 'same')(x)        
        x = Add()([self._block(self.filters_list[0], x),x])
        x = Add()([self._block(self.filters_list[0], x),x])
        x = Add()([self._block(self.filters_list[0], x),x])
        if len(self.filters_list) > 1:
            for filt in self.filters_list[1:]:
                x = Conv2D(filt, (3,3),
                           strides = (2,2),
                           padding = 'same',
                           activation = 'relu',
                           kernel_initializer = self.initializer)(x)
                x = Add()([self._block(filt, x),x])
                x = Add()([self._block(filt, x),x])
                x = Add()([self._block(filt, x),x])
#         x = GlobalAveragePooling2D()(x)

        x = Reshape((x.shape[1], x.shape[2]*x.shape[3]))(x)
#         x = Reshape((16, 11*32))

        x = Bidirectional(LSTM(256, return_sequences=True))(x)
        x = Bidirectional(LSTM(256))(x)
        x = Flatten()(x)
        x = Dense(256)(x)
        x = Dense(self.output_size, activation = 'softmax')(x)
        
        self.m = Model(i,x)
        return self.m        
       
        
class DeepSpeech2Attention():
    """
    Usage: 
        
    """
    def __init__(self,
                 filters_list=[], 
                 input_size=None, 
                 output_size=None,
                 initializer='glorot_uniform'):
        self.filters_list = filters_list
        self.input_size = input_size
        self.output_size = output_size
        self.initializer = initializer
        self.m = None        
    
    def _block(self, filters, inp):
        """ one residual block in a ResNet
        
        Args:
            filters (int): number of convolutional filters
            inp (tf.tensor): output from previous layer
            
        Returns:
            tf.tensor: output of residual block
        """
        layer_1 = BatchNormalization()(inp)
        act_1 = Activation('relu')(layer_1)
        conv_1 = Conv2D(filters, (3,3), 
                        padding = 'same', 
                        kernel_initializer = self.initializer)(act_1)
        layer_2 = BatchNormalization()(conv_1)
        act_2 = Activation('relu')(layer_2)
        conv_2 = Conv2D(filters, (3,3), 
                        padding = 'same', 
                        kernel_initializer = self.initializer)(act_2)
        return(conv_2)

    def build(self):
        """
        Returns:
            keras.engine.training.Model
        """
        input_shape = self.input_size
        input_dim = input_shape[1]
        rnn_units = 128
        output_dim = 12
        
        input_tensor = Input(shape=(input_shape))

        x = layers.BatchNormalization(axis=2)(input_tensor)


        # Add 4th dimension [batch, time, frequency, channel]
        # x = layers.Lambda(keras.backend.expand_dims,
        #                   arguments=dict(axis=-1))(input_tensor)
        x = layers.Conv2D(filters=32,
                            kernel_size=[11, 41],
                            strides=[2, 2],
                            padding='same',
                            use_bias=False,
                            name='conv_1')(x)
        x = layers.BatchNormalization(name='conv_1_bn')(x)
        x = layers.ReLU(name='conv_1_relu')(x)

        x = layers.Conv2D(filters=32,
                            kernel_size=[11, 21],
                            strides=[1, 2],
                            padding='same',
                            use_bias=False,
                            name='conv_2')(x)
        x = layers.BatchNormalization(name='conv_2_bn')(x)
        x = layers.ReLU(name='conv_2_relu')(x)

        
        x = layers.Reshape((x.shape[1], x.shape[2]*x.shape[3]))(x)

        # for attention mechanism:

        recurrent = layers.GRU(units=rnn_units,
                            activation='tanh',
                            recurrent_activation='sigmoid',
                            use_bias=True,
                            return_sequences=True,
                            reset_after=True,
                            name=f'gru_{1}',
                            return_state=True)

        encoder_out, forward_h, backward_h = layers.Bidirectional(recurrent,
                                    name=f'bidirectional_{1}',
                                    merge_mode='concat')(x)
        encoder_hidden = layers.Concatenate([forward_h, backward_h])
        # encoder_hidden = [forward_h, backward_h]


        recurrent_d = layers.GRU(units=rnn_units,
                            activation='tanh',
                            recurrent_activation='sigmoid',
                            use_bias=True,
                            return_sequences=True,
                            reset_after=True,
                            name=f'gru_{2}',
                            return_state=True)

        decoder_out, forward_h_d, backward_h_d = layers.Bidirectional(recurrent_d,
                                    name=f'bidirectional_{2}',
                                    merge_mode='concat')(x, initial_state=[forward_h, backward_h])
        decoder_hidden = layers.Concatenate([forward_h, backward_h])


        attention = layers.dot([decoder_out, encoder_out], axes=[2, 2])
        attention = layers.Activation('softmax')(attention)

        context = layers.dot([attention, encoder_out], axes=[2,1])
        decoder_combined_context = layers.concatenate([context, decoder_out])

        x = decoder_combined_context

        x = layers.Flatten()(x)

        x = layers.Dense(units=rnn_units*2)(x)
        x = layers.Dropout(rate=0.5)(x)

        output_tensor = layers.Dense(output_dim, activation='softmax')(x)
        
        self.m = Model(input_tensor,output_tensor)
        return self.m               
        
        

        
class DeepSpeech2():
    """
    Usage: 
        
    """
    def __init__(self,
                 filters_list=[], 
                 input_size=None, 
                 output_size=None,
                 initializer='glorot_uniform'):
        self.filters_list = filters_list
        self.input_size = input_size
        self.output_size = output_size
        self.initializer = initializer
        self.m = None        
    
    def _block(self, filters, inp):
        """ one residual block in a ResNet
        
        Args:
            filters (int): number of convolutional filters
            inp (tf.tensor): output from previous layer
            
        Returns:
            tf.tensor: output of residual block
        """
        layer_1 = BatchNormalization()(inp)
        act_1 = Activation('relu')(layer_1)
        conv_1 = Conv2D(filters, (3,3), 
                        padding = 'same', 
                        kernel_initializer = self.initializer)(act_1)
        layer_2 = BatchNormalization()(conv_1)
        act_2 = Activation('relu')(layer_2)
        conv_2 = Conv2D(filters, (3,3), 
                        padding = 'same', 
                        kernel_initializer = self.initializer)(act_2)
        return(conv_2)

    def build(self):
        """
        Returns:
            keras.engine.training.Model
        """
        input_shape = self.input_size
        input_dim = input_shape[1]
        rnn_units = 512
        output_dim = 12
        
        input_tensor = Input(shape=(input_shape))

        x = layers.BatchNormalization(axis=2)(input_tensor)


        # Add 4th dimension [batch, time, frequency, channel]
        # x = layers.Lambda(keras.backend.expand_dims,
        #                   arguments=dict(axis=-1))(input_tensor)
        x = layers.Conv2D(filters=32,
                            kernel_size=[11, 41],
                            strides=[2, 2],
                            padding='same',
                            use_bias=False,
                            name='conv_1')(x)
        x = layers.BatchNormalization(name='conv_1_bn')(x)
        x = layers.ReLU(name='conv_1_relu')(x)

        x = layers.Conv2D(filters=32,
                            kernel_size=[11, 21],
                            strides=[1, 2],
                            padding='same',
                            use_bias=False,
                            name='conv_2')(x)
        x = layers.BatchNormalization(name='conv_2_bn')(x)
        x = layers.ReLU(name='conv_2_relu')(x)

        
        x = layers.Reshape((x.shape[1], x.shape[2]*x.shape[3]))(x)

        for i in [1, 2]:
#         for i in [1, 2, 3]:
#         for i in [1, 2, 3, 4, 5]:
            recurrent = layers.GRU(units=rnn_units,
                                   activation='tanh',
                                   recurrent_activation='sigmoid',
                                   use_bias=True,
                                   return_sequences=True,
                                   reset_after=True,
                                   name=f'gru_{i}')
            x = layers.Bidirectional(recurrent,
                                     name=f'bidirectional_{i}',
                                     merge_mode='concat')(x)
#             x = layers.Dropout(rate=0.5)(x) if i < 5 else x  # Only betwee
            x = layers.Dropout(rate=0.5)(x) if i < 2 else x  # Only betwee

            
        x = layers.Flatten()(x)

        x = layers.Dense(units=32)(x)
        x = layers.Dropout(rate=0.5)(x)

        output_tensor = layers.Dense(output_dim, activation='softmax')(x)
        
        self.m = Model(input_tensor,output_tensor)
        return self.m               
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
