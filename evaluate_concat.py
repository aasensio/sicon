import numpy as np
import platform
import os
import time
import argparse
import warnings
import h5py
# To deactivate warnings: https://github.com/tensorflow/tensorflow/issues/7778
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
import model_concat as nn_model




class deep_network(object):

    def __init__(self):
        # Only allocate needed memory
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        session = tf.Session(config=config)
        ktf.set_session(session)
        self.nq = 49
        self.nlambda = 448


    def define_network(self):
        self.model = nn_model.keepsize(None, None, self.nlambda, self.nq, 0.0)
        
        print("Loading weights...")
        self.model.load_weights("concat/keepsize_weights.hdf5")

    
    def predict(self, image,save=False):
        self.image = self.prepare_data(image)
        self.nx = self.image.shape[0]
        self.ny = self.image.shape[1]
        

        input_validation = np.zeros((1,self.nx,self.ny,self.nlambda), dtype='float32')
        input_validation[0,:,:,:] = self.image
        start = time.time()
        out = self.model.predict(input_validation)
        end = time.time()
        print("Prediction took {0:3.2} seconds...".format(end-start))        
        
        # Inverse transformation
        out = np.reshape(out, (input_validation.shape[0],self.nx,self.ny,7,7))
        out[0,:,:,0,:] /= 10.
        out[0,:,:,3,:] /= 10.
        out[0,:,:,4,:] /= 10.
        out[0,:,:,5,:] /= 10.
        out = np.reshape(out, (input_validation.shape[0],self.nx,self.ny,49))
        self.out = out

        # Save in npy format
        if save:
            np.save('concatenate_prediction_ar10933.npy',out)   

    def prepare_data(self,image):
        ft = h5py.File(image, 'r')
        stokes = ft['stokes'][:,:,:,:]
        ft.close()

        # Concatenation of  wavelength points
        stokes = np.copy(np.swapaxes(stokes,1,3))
        # Continuum normalization
        sc = np.mean(stokes[0,0:3,:100,:100])
        stokes[0,:,:,:] = stokes[0,:,:,:]/sc
        stokes[1,:,:,:] = stokes[1,:,:,:]*10./sc
        stokes[2,:,:,:] = stokes[2,:,:,:]*10./sc
        stokes[3,:,:,:] = stokes[3,:,:,:]*1./sc  
        stokes2 = np.reshape(stokes, (stokes.shape[0]*stokes.shape[1],stokes.shape[2],stokes.shape[3]))
        return np.swapaxes(stokes2,0,2)


    def plot_results(self):

        import matplotlib as mpl
        mpl.use('Agg')
        mpl.rcParams['figure.dpi'] = 50
        mpl.rcParams['savefig.dpi'] = 50
        import matplotlib.pyplot as plt

        magTitle = ['h [km]', 'T [K]', r'v [km s$^{-1}$]', r'$B_z$ [kG]', r'$(B_xB_y)^{1/2}$ [kG]', r'$(B_x^2-B_y^2)^{1/2}$ [kG]', 'log P [dex]']
        nombre = ['z','T','vz','Bz','sqrtBxBy','sqrtBx2By2','logP']
        maplist = ['inferno','inferno','bwr','RdBu', 'inferno','inferno','inferno']
        mapscale = [1000.,1000.,1.0,1.0,1.0,1.0,1.0]
        maxilist = [None,None,+5,+2.5,None,None,None]
        minilist = [None,None,-5,-2.5,None,None,None]
        nq = 7
        name = 'figures/concatenate_ar10933_'
        extent = [0,500*0.16,0,500*0.16]
        colorzoom = 1.0

        for magnitud in range(7):
            extra = str(magnitud)
            plt.figure(figsize=(6*2,5*2))
            nlogtau = np.arange(-3.0,0.5,0.5)[::-1]
            print(nlogtau)
            zmira = 0
            # print(zz[zmira])
            for ii in range(4):
                plt.subplot(2,2,ii+1)
                plt.title(r'log$\tau$={0}'.format(nlogtau[int(ii*2)]))
                # ploti = histo_opt(salida[0,int(ii*2),:,:])
                # maxi = np.max([np.abs(ploti.min()),np.abs(ploti.max())])
                
                plt.imshow(np.flipud(self.out[0,:,:,magnitud*nq+int(ii*2)])*mapscale[magnitud],
                    interpolation='None',origin='lower',extent=extent,cmap=maplist[magnitud],vmax=maxilist[magnitud],vmin=minilist[magnitud])

                cb = plt.colorbar(shrink=1.0*colorzoom, pad=0.02)
                cb.set_label(r""+magTitle[magnitud], labelpad=8., y=0.5, fontsize=12.)
                if ii ==2 or ii==3:
                    plt.xlabel('Distance [arcsec]')
                if ii==0 or ii==2:
                    plt.ylabel('Distance [arcsec]')

            plt.tight_layout()
            # plt.subplots_adjust(top=0.85)
            plt.savefig(name+nombre[magnitud]+'.pdf',dpi=100)
            
if (__name__ == '__main__'):


    out = deep_network()
    out.define_network()
    out.predict(image='/scratch/carlos/DEEPL/CUBOS/ar10933_BIG.h5',save=True)
    out.plot_results()
    
    # To avoid the TF_DeleteStatus message:
    # https://github.com/tensorflow/tensorflow/issues/3388
    ktf.clear_session()




