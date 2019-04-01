import numpy as np
import matplotlib.pyplot as pl
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable
from torch.nn.modules.module import _addindent
import h5py
import time
import sys
import model_encdec as model


class deep_3d_inversor(object):
    def __init__(self, saveplots=True, checkpoint=None):

        # Check for the availability of a GPU
        self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda else "cpu")
               
        # Optical depth heights for the output
        self.ltau = np.array([0.0,-0.5,-1.0,-1.5,-2.0,-2.5,-3.0])

        # Some scales and labels for the plots
        self.variable = ["T", "v$_z$", "h", "log P", "$(B_x^2-B_y^2)^{1/2}$", "$(B_x B_y)^{1/2}$", "B$_z$"]
        self.variable_txt = ["T", "vz", "tau", "logP", "sqrtBx2By2", "sqrtBxBy", "Bz"]
        self.units = ["K", "km s$^{-1}$", "km", "cgs", "kG", "kG", "kG"]
        self.multiplier = [1.0, 1.e-5, 1.e-5, 1.0, 1.0e-3, 1.0e-3, 1.0e-3]


        self.z_tau1 = 1300.0

        # Instantiate the model
        self.model = model.block(in_planes=112*4+3, out_planes=7*7).to(self.device)
                
        # Load the weights
        self.checkpoint = '{0}.pth'.format(checkpoint)
        
        print("=> loading checkpoint '{}'".format(self.checkpoint))
        if (self.cuda):
            checkpoint = torch.load(self.checkpoint)
        else:
            checkpoint = torch.load(self.checkpoint, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(checkpoint['state_dict'])        
        print("=> loaded checkpoint '{}'".format(self.checkpoint))     

        # Load the normalizations for the output
        tmp = self.checkpoint.split('.')        
        f_normal = '{0}.normalization.npz'.format('.'.join(tmp[0:-1]))
        tmp = np.load(f_normal)
        self.phys_min, self.phys_max = tmp['minimum'], tmp['maximum']

        self.saveplots = saveplots
    
    def evaluate(self, save_output=False):

        # Open the file with the observations. Change this to
        # your own maps
        f = h5py.File('/scratch1/hinode/ar10933/ar10933_patch.h5', 'r')

        self.stokes = f['stokes'][:]

        # Compute the median in a quiet region
        stokes_median = np.median(self.stokes[0,0:10,0:10,0:3])
                
        f.close()

        # Transpose the Stokes array so that it is ordered as
        # (4,n_lambda,nx,ny)

        self.stokes = np.transpose(self.stokes, axes=(0,3,1,2))

        _, n_lambda, nx, ny = self.stokes.shape
        
        # Normalize by the median
        self.stokes /= stokes_median

        # Normalize QUV
        self.peak_val = np.max(np.abs(self.stokes / self.stokes[0:1,0:1,:,:]), axis=1)
        self.peak_val[0,:,:] = 1.0
        self.peak_val[self.peak_val < 0.03] = 0.03
        
        for i in range(3):
            self.stokes[1+i,:,:,:] /= self.peak_val[1+i,None,:,:]

        # Flatten the array by serializing all IQUV
        self.stokes = self.stokes.reshape((4*n_lambda,nx,ny))
        
        # Compute the scale in logarithmic units and substract the mean
        scale = np.log10(self.peak_val[1:,:,:])
        scale -= np.mean(scale, axis=(1,2))[:,None,None]

        # Concatenate the three scales
        self.stokes = np.vstack([self.stokes, scale])

        # Generate a singleton axis for batch sizes. This is mandatory in PyTorch
        self.stokes = np.expand_dims(self.stokes.reshape((4*n_lambda+3,nx,ny)), axis=0)

        # Put the model in evaulation mode                                                       
        self.model.eval()
        
        palettes = [pl.cm.inferno] * 7
        palettes[1] = pl.cm.bwr
        palettes[6] = pl.cm.RdBu

        pl.ioff()

        left = 0
        right = 200

        # Put PyTorch in evaluation mode
        with torch.no_grad():
            
            # Input tensor
            input = torch.from_numpy(self.stokes[:,:,left:right, left:right].astype('float32')).to(self.device)
            
            # Evluate the model and rescale the output
            start = time.time()
            output = np.squeeze(self.model(input).cpu().data.numpy())
            output = output * (self.phys_max[:,None,None] - self.phys_min[:,None,None]) + self.phys_min[:,None,None]
            print('Elapsed time : {0} s'.format(time.time()-start))            

            # Unflatten the results  and do some plots
            output = output.reshape((7,7,right-left,right-left))

            for j in range(7):
                f, ax = pl.subplots(nrows=2, ncols=2, figsize=(9.3,7.5), constrained_layout=True)

                ax[0,0].set_ylabel("Distance [arcsec]")
                ax[1,0].set_ylabel("Distance [arcsec]")
                
                ax[1,0].set_xlabel("Distance [arcsec]")
                ax[1,1].set_xlabel("Distance [arcsec]")
                ax = ax.flatten()
                
                for i in range(4):                    
                    if (j in [1,6]):                        
                        top = 0.65*np.max(np.abs(self.multiplier[j] * output[j,2*i,:,:]))
                        im = ax[i].imshow(self.multiplier[j] * output[j,2*i,:,:], cmap=palettes[j], extent=[0,0.16*(right-left),0,0.16*(right-left)], vmin=-top, vmax=top)
                    else:
                        im = ax[i].imshow(self.multiplier[j] * output[j,2*i,:,:], cmap=palettes[j], extent=[0,0.16*(right-left),0,0.16*(right-left)])

                    ax[i].set_title(r'log $\tau$={0}'.format(self.ltau[2*i]))
                    cbar = pl.colorbar(im, ax=ax[i])
                    cbar.set_label(r'{0} [{1}]'.format(self.variable[j], self.units[j]))                

                if (self.saveplots):
                    tmp = '.'.join(self.checkpoint.split('/')[-1].split('.')[0:2])                    
                    pl.savefig("{0}_{1}.pdf".format(tmp.replace('.','_'), self.variable_txt[j]))


        # Save the output as an HDF5 file
        if (save_output):
            print("Saving output")
            tmp = '.'.join(self.checkpoint.split('/')[-1].split('.')[0:2])
            f = h5py.File('{0}.h5'.format(tmp), 'w')
            db_logtau = f.create_dataset('tau_axis', self.ltau.shape)
            db_T = f.create_dataset('T', output[0,:,:,:].shape)
            db_vz = f.create_dataset('vz', output[1,:,:,:].shape)
            db_tau = f.create_dataset('tau', output[2,:,:,:].shape)
            db_logP = f.create_dataset('logP', output[3,:,:,:].shape)
            db_Bx2_By2 = f.create_dataset('sqrt_Bx2_By2', output[4,:,:,:].shape)
            db_BxBy = f.create_dataset('sqrt_BxBy', output[5,:,:,:].shape)
            db_Bz = f.create_dataset('Bz', output[6,:,:,:].shape)
            db_Bx = f.create_dataset('Bx', output[4,:,:,:].shape)
            db_By = f.create_dataset('By', output[5,:,:,:].shape)

            Bx = np.zeros_like(db_Bz[:])
            By = np.zeros_like(db_Bz[:])
                        

            db_logtau[:] = self.ltau
            db_T[:] = output[0,:,:,:] * self.multiplier[0]
            db_vz[:] = output[1,:,:,:] * self.multiplier[1]
            db_tau[:] = output[2,:,:,:] * self.multiplier[2]
            db_logP[:] = output[3,:,:,:] * self.multiplier[3]
            db_Bx2_By2[:] = output[4,:,:,:] * self.multiplier[4]
            db_BxBy[:] = output[5,:,:,:] * self.multiplier[5]
            db_Bz[:] = output[6,:,:,:] * self.multiplier[6]

            # Compute Bx and By from the combinations
            A = np.sign(db_Bx2_By2[:]) * db_Bx2_By2[:]**2    # We saved sign(Bx^2-By^2) * np.sqrt(Bx^2-By^2)
            B = np.sign(db_BxBy[:]) * db_BxBy[:]**2    # We saved sign(Bx*By) * np.sqrt(Bx*By)

            # This quantity is obviously always >=0
            D = np.sqrt(A**2 + 4.0*B**2)
            
            ind_pos = np.where(B >0)
            ind_neg = np.where(B < 0)
            ind_zero = np.where(B == 0)
            Bx[ind_pos] = np.sqrt(A[ind_pos] + D[ind_pos]) / np.sqrt(2.0)
            By[ind_pos] = np.sqrt(2.0) * B[ind_pos] / np.sqrt(1e-1 + A[ind_pos] + D[ind_pos])
            Bx[ind_neg] = -np.sqrt(A[ind_neg] + D[ind_neg]) / np.sqrt(2.0)
            By[ind_neg] = -np.sqrt(2.0) * B[ind_neg] / np.sqrt(1e-1 + A[ind_neg] + D[ind_neg])
            Bx[ind_zero] = 0.0
            By[ind_zero] = 0.0

            db_Bx[:] = Bx
            db_By[:] = By

            f.close()
        else:
            print("Not saving output")                

if (__name__ == '__main__'):
    pl.close('all')

    deep_network = deep_3d_inversor(checkpoint='encdec/2019-03-27-15:02_-lr_0.0003', saveplots=True)
    deep_network.evaluate(save_output=True)