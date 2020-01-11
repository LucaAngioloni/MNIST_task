import numpy as np
import json
import os

import argparse
from str2bool import str2bool

import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Concatenate, Flatten, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.datasets import mnist

def data_generator(x, y, batchsize=128, count=1024, fixed_random=False, task='sum'):
    while True:
        if fixed_random:
            np.random.rand(4)
        for i in range(count):
            i_A = np.random.randint(0,x.shape[0],batchsize)
            i_B = np.random.randint(0,x.shape[0],batchsize)
            x_A = x[i_A,:,:,:]
            x_B = x[i_B,:,:,:]
            if task == 'abs_diff':
                y_A = y[i_A]
                y_B = y[i_B]
                yy = np.abs(y_A - y_B)
            elif task == 'eucl_dist':
                yy = np.sum(np.square(x[i_A,:,:,:] - x[i_B,:,:,:]),axis=(1,2,3))
            else:
                y_A = y[i_A]
                y_B = y[i_B]
                yy = y_A + y_B
            yield [x_A,x_B], yy

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def get_model(input_size, filters=256, layers=4, conv_filters_dim=8,
            zdim=3, batch_normalization=True, learning_rate=0.0002, 
            merge_type='distance', compile_model=True,
            regularization=False, verbose=True, **kwargs):

    # Define the tensors for the two input images
    left_input = Input(tuple(input_size))
    right_input = Input(tuple(input_size))
    
    input_ = Input(input_size)
    scale = 2**(layers - 1)
    x = Conv2D(filters // scale, (conv_filters_dim,conv_filters_dim), strides=(2,2), padding='same')(input_)
    if batch_normalization:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    for i in range(1, layers):
        scale = 2**(layers - i - 1)
        if regularization:
            x = Conv2D(filters // scale, (conv_filters_dim,conv_filters_dim), strides=(2,2), padding='same',\
                        kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(x)
        else:
            x = Conv2D(filters // scale, (conv_filters_dim,conv_filters_dim), strides=(2,2), padding='same')(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)

    base_model = Model(input_, x)
    
    left_conv = base_model(left_input)
    right_conv = base_model(right_input)

    if merge_type == 'concat':
        merge_layer = Concatenate()([left_conv, right_conv])

        x = Conv2D(filters // scale, (conv_filters_dim,conv_filters_dim), strides=(2,2), padding='same')(merge_layer)
        if batch_normalization:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Flatten()(x)
        x = Dense(zdim)(x)
        x = Activation('relu')(x)
        distance = Dense(1)(x)
    elif merge_type == 'distance':
        left_flatten = Flatten()(left_conv)
        right_flatten = Flatten()(right_conv)
        left_z = Dense(zdim)(left_flatten)
        right_z = Dense(zdim)(right_flatten)
        distance = Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape)([left_z, right_z])
    else:
        print("Error, merge_type unknown")
        print(merge_type)

    model = Model(inputs=[left_input, right_input], outputs=distance)
    
    if verbose:
        model.summary()

    if compile_model:
        opt_ae = Adam(lr=learning_rate)
        model.compile(loss='mse', optimizer=opt_ae, metrics=['mae'])
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Learn a task like sum, abs_diff, eucl_dist with MNIST pairs')
    parser.add_argument(
        '--working_dir',
        type=str,
        default='save_dir',
        help='Path where to save files')
    parser.add_argument(
        '--batch_normalization',
        type=str2bool,
        default=True,
        help='Use BatchNormalization between conv layers')
    parser.add_argument(
        '--filters',
        type=int,
        default=256,
        help='Number of filters for each conv layer (scaled with depth)')
    parser.add_argument(
        '--layers',
        type=int,
        default=3,
        help='Number of layers')
    parser.add_argument(
        '--conv_filters_dim',
        type=int,
        default=5,
        help='Conv filters size')
    parser.add_argument(
        '--zdim',
        type=int,
        default=64,
        help='Size of the latent space')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64)
    parser.add_argument(
        '--epochs',
        type=int,
        default=25)
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001)
    parser.add_argument(
        '--regularization',
        type=str2bool,
        default=False)
    parser.add_argument(
        '--merge_type',
        type=str,
        default='distance',
        help='Type of merging layer: concat or distance')
    parser.add_argument(
        '--save_interval',
        type=int,
        default=1)
    parser.add_argument(
        '--id_gpu', 
        default=0, 
        type=int, 
        help='This argument specifies which gpu to use.')
    parser.add_argument(
        '--task',
        type=str,
        default='sum',
        help='Task to perform: [eucl_dist, sum, abs_diff]')
    parser.add_argument(
        '--test_size', 
        default=10000, 
        type=int, 
        help='The size of the validation set to use during training.')

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # The GPU id to use
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.id_gpu)
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    working_dir = args.working_dir
    working_dir = working_dir if working_dir[-1] != '/' else working_dir[:-1] # remove / at the end

    if not os.path.exists(working_dir):
        os.makedirs(working_dir, exist_ok=True)

    input_size = [28, 28, 1]

    # Create a parameters dictionary and save it
    params = {
        'batch_normalization': args.batch_normalization,
        'filters': args.filters,
        'layers': args.layers,
        'conv_filters_dim': args.conv_filters_dim,
        'zdim': args.zdim,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'regularization': args.regularization,
        'save_interval': args.save_interval,
        'merge_type': args.merge_type,
        'task': args.task,
        'test_size': args.test_size,
        'input_dim': input_size
    }

    np.save('{}/params.npy'.format(working_dir), params)

    with open(os.path.join(working_dir, 'params.json'), 'w') as f:
        f.write(json.dumps(params))

    # Load data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Normalize data
    x_train = x_train.reshape(-1,28,28,1)
    x_train = x_train/255.0
    x_test = x_test.reshape(-1,28,28,1)
    x_test = x_test/255.0

    # Prepare test/validation set and compute labels
    testsize=args.test_size
    i_A = np.random.randint(0,x_test.shape[0],testsize)
    i_B = np.random.randint(0,x_test.shape[0],testsize)
    x_test_A = x_test[i_A,:,:,:]
    x_test_B = x_test[i_B,:,:,:]
    if args.task == 'abs_diff':
        y_train = y_train.astype(float)
        y_test = y_test.astype(float)
        y_test_A = y_test[i_A]
        y_test_B = y_test[i_B]
        yy_test = np.abs(y_test_A - y_test_B)
    elif args.task == 'eucl_dist':
        yy_test = np.sum(np.square(x_test[i_A,:,:,:] - x_test[i_B,:,:,:]),axis=(1,2,3))
    else:
        y_train = y_train.astype(float)
        y_test = y_test.astype(float)
        y_test_A = y_test[i_A]
        y_test_B = y_test[i_B]
        yy_test = y_test_A + y_test_B


    model = get_model(input_size, **params)
    train_gen = data_generator(x_train, y_train, batchsize=args.batch_size, count=1024, fixed_random=False, task=args.task)

    filepath= working_dir + "/Ep.{epoch:02d}-{loss:.5f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True, period=args.save_interval)

    model.fit_generator(train_gen, steps_per_epoch=1024, epochs=args.epochs,\
        validation_data=([x_test_A,x_test_B], yy_test), validation_steps=1,\
        callbacks=[checkpoint])

    model.save_weights(filepath= working_dir + "/Ep.last.hdf5")
