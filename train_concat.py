import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  #To deactivate warnings

print(tf.__version__,'tensorflow version')
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True,allow_soft_placement=True))

import numpy as np
import platform
import time
import argparse
import json
import warnings
import h5py
import copy
import random
import glob

import keras.backend.tensorflow_backend as ktf
from keras.optimizers import Adam
import model_concat as nn_model
from keras.callbacks import Callback, ModelCheckpoint, LearningRateScheduler
from contextlib import redirect_stdout
from keras.utils.data_utils import Sequence


dire = '/scratch/carlos/DEEPL/CUBOS/'
simulist = glob.glob(dire+'*model*.h5')
stokeslist = glob.glob(dire+'*stokes*.h5')
simulist = sorted(simulist)
stokeslist = sorted(stokeslist)
stokelist, cubelist = [],[]
for simu in range(len(simulist)):
    ft = h5py.File(simulist[simu], 'r')
    f2 = h5py.File(stokeslist[simu], 'r')
    stokes = np.swapaxes(np.swapaxes(np.swapaxes(f2['stokes'][:],0,2),1,3),2,3)
    cube = np.swapaxes(ft['model'][:].T,0,1)
    cube[0,:,:,:] = cube[0,:,:,:] - np.mean(cube[0,0,:100,:100])
    ft.close()
    f2.close()
    stokelist.append(stokes)
    cubelist.append(cube)
    print('Simulation {0} added'.format(simulist[simu]))


def rotate(cubo, angle_index):
    # Rotate cubes
    nnqq = cubo.shape[2]
    ncubo = np.copy(cubo)
    for ii in range(nnqq):
        ncubo[:,:,ii] = np.rot90(cubo[:,:,ii],angle_index)
    return ncubo


class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, datasize,dx,batch_size):
        'Initialization'
        self.n_training_orig = datasize
        self.batch_size = batch_size
        self.dx = dx
        self.noise = 0. # The noise is added by the neural network model!

        self.batchs_per_epoch_training = int(self.n_training_orig / self.batch_size)
        self.n_training = self.batchs_per_epoch_training * self.batch_size
        print("Original set size: {0}".format(self.n_training_orig))
        print("   - Batches per epoch: {0}".format(self.batchs_per_epoch_training))

    def __getitem__(self, index):
        'Generate one batch of data'
        input_train_get, output_train_get = self.__data_generation(self)
        return input_train_get, output_train_get
   
    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.batchs_per_epoch_training

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        xlambda = np.loadtxt('/scratch/carlos/DEEPL/wavelength_Hinode.txt')

        simu = random.randint(0, 5)
        stokes = stokelist[simu]
        cube = cubelist[simu]
        ns, nl, ny, nx = stokes.shape
        nq, ntau, ny, nx = cube.shape
        Lx = nx
        Ly = ny
        dx = self.dx

        input_train = np.zeros((self.batch_size,dx,dx,int(nl*ns)))
        output_train = np.zeros((self.batch_size,dx,dx,int(nq*ntau)))
        for j in range(self.batch_size):
            xpos = random.randint(0, Lx-dx)
            ypos = random.randint(0, Ly-dx)
            rota = random.randint(0, 3)

            ministokes = stokes[:,:,ypos:ypos+dx,xpos:xpos+dx]
            minicubo = cube[:,:,ypos:ypos+dx,xpos:xpos+dx]
            nminicubo = np.zeros_like(minicubo)
            nministokes = np.zeros_like(ministokes)

            # Add some noise
            nministokes[0,:,:,:] = ministokes[0,:,:,:] + np.random.normal(0.,self.noise,(nl,dx,dx))
            nministokes[1,:,:,:] = ministokes[1,:,:,:] + np.random.normal(0.,self.noise,(nl,dx,dx))
            nministokes[2,:,:,:] = ministokes[2,:,:,:] + np.random.normal(0.,self.noise,(nl,dx,dx))
            nministokes[3,:,:,:] = ministokes[3,:,:,:] + np.random.normal(0.,self.noise,(nl,dx,dx))
           
            # Scale Q,U,V
            nministokes[1,:,:,:] = nministokes[1,:,:,:]*10.
            nministokes[2,:,:,:] = nministokes[2,:,:,:]*10.
            nministokes[3,:,:,:] = nministokes[3,:,:,:]*1.

            # Scale Z, Bz, Bx, By                    
            cbz = np.copy(minicubo[6,:,:,:])
            cpg = np.copy(minicubo[2,:,:,:])
            cvz = np.copy(minicubo[3,:,:,:])
            cb5 = np.copy(minicubo[5,:,:,:])
            cb4 = np.copy(minicubo[4,:,:,:])
            nminicubo[0,:,:,:] = minicubo[0,:,:,:]*10./1e8
            nminicubo[1,:,:,:] = minicubo[1,:,:,:]*1./1e3
            nminicubo[2,:,:,:] = cvz*1./1e5
            nminicubo[3,:,:,:] = cbz*10./1e3
            nminicubo[4,:,:,:] = np.sign(cb4*cb5)*(np.sqrt(np.abs(cb4*cb5))*10.)/1e3
            nminicubo[5,:,:,:] = np.sign(cb4**2. - cb5**2.)*(np.sqrt(np.abs(cb4**2. - cb5**2.))*10.)/1e3
            nminicubo[6,:,:,:] = np.log10(cpg)

            nministokes = np.reshape(nministokes, (ministokes.shape[0]*ministokes.shape[1],ministokes.shape[2],ministokes.shape[3]))
            nminicubo = np.reshape(nminicubo, (minicubo.shape[0]*minicubo.shape[1],minicubo.shape[2],minicubo.shape[3]))
            input_train[j,:,:,:] = rotate(np.swapaxes(nministokes,0,2),rota)
            output_train[j,:,:,:] = rotate(np.swapaxes(nminicubo,0,2),rota)
        return input_train, output_train


def flush_file(f):
    f.flush()
    os.fsync(f.fileno())    



class LossHistory(Callback):
    def __init__(self, root, losses, extra, **kwargs):        
        self.losses = losses
        self.losses_batch = copy.deepcopy(losses)
        self.extra = extra

        self.f_epoch_local = open("{0}_loss.json".format(root), 'w')
        self.f_epoch_local.write('['+json.dumps(self.extra))
        flush_file(self.f_epoch_local)
        

    def on_batch_end(self, batch, logs={}):
        pass

    def on_epoch_end(self, batch, logs={}):
        tmp = [time.asctime(),logs.get('loss').tolist(), logs.get('val_loss').tolist(), ktf.get_value(self.model.optimizer.lr).tolist()]
        self.f_epoch_local.write(','+json.dumps(tmp))
        flush_file(self.f_epoch_local)

    def on_train_end(self, logs):
        self.f_epoch_local.write(']')
        self.f_epoch_local.close()

    def finalize(self):
        pass

class deep_network(object):
    def __init__(self, root, noise, lr, lr_multiplier, batch_size,l2_regularization,datasize):

        # Only allocate needed memory
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        session = tf.Session(config=config)
        ktf.set_session(session)


        self.root = root
        self.noise = noise
        self.lr = lr
        self.lr_multiplier = lr_multiplier
        self.batch_size = batch_size
        self.l2_regularization = l2_regularization
        self.n_training_orig = datasize
        self.n_validation_orig = int(2000)

        self.input_file_images_training = dire+'hinode1.h5'
        f = h5py.File(self.input_file_images_training, 'r')
        self.ns, self.nl, self.nx, self.ny = f['stokes'].shape        
        self.nq, self.ntau, self.nx, self.ny = f['cube'].shape
        f.close()

        #Size of batch:
        self.dx = 32
        self.nx, self.ny = self.dx, self.dx
        self.batchs_per_epoch_training = int(self.n_training_orig / self.batch_size)
        self.batchs_per_epoch_validation = int(self.n_validation_orig / self.batch_size)
        self.n_training = self.batchs_per_epoch_training * self.batch_size
        self.n_validation = self.batchs_per_epoch_validation * self.batch_size


    def define_network(self):
        print("Setting up network...")
        self.model = nn_model.keepsize(self.nx, self.ny, int(self.nl*self.ns), int(self.ntau*self.nq), 
                self.noise,l2_reg=self.l2_regularization)
        
        json_string = self.model.to_json()
        f = open('{0}_model.json'.format(self.root), 'w')
        f.write(json_string)
        f.close()

        with open('{0}_summary.txt'.format(self.root), 'w') as f:
            with redirect_stdout(f):
                self.model.summary()
    
    def compile_network(self):        
        self.model.compile(loss='mse', optimizer=Adam(lr=self.lr))
    

    def learning_rate(self, epoch):
        value = self.lr
        if (epoch >= 10):
            value *= self.lr_multiplier
        return value

    def train(self, n_iterations):
        print("Training network...")        
        losses = []
        self.checkpointer = ModelCheckpoint(filepath="{0}_weights.hdf5".format(self.root), verbose=2, save_best_only=False)
        self.history = LossHistory(self.root, losses, {'name': '{0}'.format(self.root), 'init_t': time.asctime()})
        self.reduce_lr = LearningRateScheduler(self.learning_rate)
        
        # Generators
        training_generator_class = DataGenerator(self.n_training_orig,self.dx,self.batch_size)
        validation_generator_class = DataGenerator(self.n_validation_orig,self.dx,self.batch_size)

        self.metrics = self.model.fit_generator(training_generator_class, self.batchs_per_epoch_training, epochs=n_iterations, 
            callbacks=[self.checkpointer, self.history, self.reduce_lr], validation_data=validation_generator_class, validation_steps=self.batchs_per_epoch_validation,use_multiprocessing=True,workers=10)#
        self.history.finalize()

if (__name__ == '__main__'):

    name = 'concat/keepsize'
    nEpochs = 20
    noise = 1e-3
    lr = 1e-4
    lr_multiplier = 1.0
    batch_size = 1
    l2_regularization = 1e-5
    datasize = 10000


# Save parameters used
    with open("{0}_args.json".format(name), 'w') as f:
        string = " ".join(str(x) for x in [name,nEpochs,noise,lr,lr_multiplier,batch_size,l2_regularization,datasize])
        f.write(string)

    out = deep_network(name, noise, lr, lr_multiplier, batch_size, l2_regularization, datasize)
    out.define_network()
    out.compile_network()
    out.train(nEpochs)
