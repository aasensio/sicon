import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import h5py
from torch.autograd import Variable
import torchvision
import scipy.io as io
import scipy.ndimage as nd
from tqdm import tqdm
import model_encdec as model
import time
import shutil
import database
import nvidia_smi
import os

class dataset_spot(torch.utils.data.Dataset):
    def __init__(self, n_training=10000, n_pixels=32, image_range=[0,10]):
        super(dataset_spot, self).__init__()

        noise_scale = 1e-3

        #--------------------
        # REMPEL Stokes
        #--------------------
        print("Reading REMPEL Stokes parameters")
        f = h5py.File('rempel_vzminus_stokes_spat_spec_degraded_sir.h5', 'r')
        stokes = f['stokes'][:]
        nx, ny, _, n_lambda = stokes.shape
        
        f.close()
        
        self.stokes = np.copy(stokes)
        self.stokes += np.random.normal(loc=0.0, scale=noise_scale, size=self.stokes.shape)
        self.peak_val = np.max(np.abs(self.stokes / self.stokes[:,:,0:1,0:1]), axis=3)
        self.peak_val[:,:,0] = 1.0
        self.peak_val[self.peak_val < 0.005] = 0.005
        
        for i in range(3):
            self.stokes[:,:,1+i,:] /= self.peak_val[:,:,1+i,None]

        self.stokes = self.stokes.reshape((nx,ny,4*n_lambda))

        scale = np.log10(self.peak_val[:,:,1:])

        scale -= np.mean(scale, axis=(0,1))[None,None,:]
        
        self.stokes = np.vstack([np.transpose(self.stokes, axes=(2,0,1)), np.transpose(scale, axes=(2,0,1))])

        #--------------------
        # REMPEL Stokes Inverted
        #--------------------
        print("Reading REMPEL Stokes parameters for inverted configuration")
        f = h5py.File('rempel_vzminus_stokes_invert_spat_spec_degraded_sir.h5', 'r')
        stokes = f['stokes'][:]
        nx, ny, _, n_lambda = stokes.shape

        f.close()
        
        self.stokes_inv = np.copy(stokes)
        self.stokes_inv += np.random.normal(loc=0.0, scale=noise_scale, size=self.stokes_inv.shape)
        self.peak_val_inv = np.max(np.abs(self.stokes_inv / self.stokes_inv[:,:,0:1,0:1]), axis=3)
        self.peak_val_inv[:,:,0] = 1.0
        self.peak_val_inv[self.peak_val_inv < 0.005] = 0.005

        for i in range(3):
            self.stokes_inv[:,:,1+i,:] /= self.peak_val_inv[:,:,1+i,None]

        self.stokes_inv = self.stokes_inv.reshape((nx,ny,4*n_lambda))

        scale = np.log10(self.peak_val_inv[:,:,1:])

        scale -= np.mean(scale, axis=(0,1))[None,None,:]        
        
        self.stokes_inv = np.vstack([np.transpose(self.stokes_inv, axes=(2,0,1)), np.transpose(scale, axes=(2,0,1))])

        #--------------------
        # CHEUNG Stokes
        #--------------------
        print("Reading CHEUNG Stokes parameters")
        f = h5py.File('cheung_vzminus_stokes_spat_spec_degraded_sir.h5', 'r')
        stokes = f['stokes'][:]
        nx, ny, _, n_lambda = stokes.shape

        f.close()
        
        self.stokes_cheung = np.copy(stokes)
        self.stokes_cheung += np.random.normal(loc=0.0, scale=noise_scale, size=self.stokes_cheung.shape)
        self.peak_val = np.max(np.abs(self.stokes_cheung / self.stokes_cheung[:,:,0:1,0:1]), axis=3)
        self.peak_val[:,:,0] = 1.0       
        self.peak_val[self.peak_val < 0.005] = 0.005
        
        for i in range(3):
            self.stokes_cheung[:,:,1+i,:] /= self.peak_val[:,:,1+i,None]

        self.stokes_cheung = self.stokes_cheung.reshape((nx,ny,4*n_lambda))

        scale = np.log10(self.peak_val[:,:,1:])

        scale -= np.mean(scale, axis=(1,2))[:,None,None]
        
        self.stokes_cheung = np.vstack([np.transpose(self.stokes_cheung, axes=(2,0,1)), np.transpose(scale, axes=(2,0,1))])

        #--------------------
        # REMPEL Model
        #--------------------        
        print("Reading REMPEL model parameters")
        f = h5py.File('rempel_vzminus_model_spat_degraded_sir.h5', 'r')
        T = np.transpose(f['model'][:,:,1,:], axes=(2,0,1))
        vz = np.transpose(f['model'][:,:,3,:], axes=(2,0,1))
        tau = np.transpose(f['model'][:,:,0,:], axes=(2,0,1))
        logP = np.log10(np.transpose(f['model'][:,:,2,:], axes=(2,0,1)))
        Bx = np.transpose(f['model'][:,:,4,:], axes=(2,0,1))
        By = np.transpose(f['model'][:,:,5,:], axes=(2,0,1))
        Bz = np.transpose(f['model'][:,:,6,:], axes=(2,0,1))

        tau = tau - np.median(tau[0,0:30,0:30])      # Substract the average height in the QS at tau=1

        f.close()
        
        self.phys = np.vstack([T, vz, tau, logP, np.sign(Bx**2-By**2)*np.sqrt(np.abs(Bx**2-By**2)), np.sign(Bx*By)*np.sqrt(np.abs(Bx*By)), Bz])
        self.max_phys = np.max(self.phys, axis=(1,2))
        self.min_phys = np.min(self.phys, axis=(1,2))

        #--------------------
        # REMPEL Model Inverted
        #--------------------        
        print("Reading REMPEL model parameters in inverted configuration")
        f = h5py.File('rempel_vzminus_model_invert_spat_degraded_sir.h5', 'r')
        T = np.transpose(f['model'][:,:,1,:], axes=(2,0,1))
        vz = np.transpose(f['model'][:,:,3,:], axes=(2,0,1))
        tau = np.transpose(f['model'][:,:,0,:], axes=(2,0,1))
        logP = np.log10(np.transpose(f['model'][:,:,2,:], axes=(2,0,1)))
        Bx = np.transpose(f['model'][:,:,4,:], axes=(2,0,1))
        By = np.transpose(f['model'][:,:,5,:], axes=(2,0,1))
        Bz = np.transpose(f['model'][:,:,6,:], axes=(2,0,1))

        tau = tau - np.median(tau[0,0:30,0:30])      # Substract the average height in the QS at tau=1

        f.close()
        
        self.phys_inv = np.vstack([T, vz, tau, logP, np.sign(Bx**2-By**2)*np.sqrt(np.abs(Bx**2-By**2)), np.sign(Bx*By)*np.sqrt(np.abs(Bx*By)), Bz])
        self.max_phys_inv = np.max(self.phys_inv, axis=(1,2))
        self.min_phys_inv = np.min(self.phys_inv, axis=(1,2))


        #--------------------
        # CHEUNG Model
        #--------------------        
        print("Reading CHEUNG model parameters")
        f = h5py.File('cheung_vzminus_model_spat_degraded_sir.h5', 'r')
        T = np.transpose(f['model'][:,:,1,:], axes=(2,0,1))
        vz = np.transpose(f['model'][:,:,3,:], axes=(2,0,1))
        tau = np.transpose(f['model'][:,:,0,:], axes=(2,0,1))
        logP = np.log10(np.transpose(f['model'][:,:,2,:], axes=(2,0,1)))
        Bx = np.transpose(f['model'][:,:,4,:], axes=(2,0,1))
        By = np.transpose(f['model'][:,:,5,:], axes=(2,0,1))
        Bz = np.transpose(f['model'][:,:,6,:], axes=(2,0,1))

        tau = tau - np.median(tau[0,0:30,20:50])      # Substract the average height in the QS at tau=1 (there is a patch in the corner that we avoid)

        f.close()
        
        self.phys_cheung = np.vstack([T, vz, tau, logP, np.sign(Bx**2-By**2)*np.sqrt(np.abs(Bx**2-By**2)), np.sign(Bx*By)*np.sqrt(np.abs(Bx*By)), Bz])
        self.max_phys_cheung = np.max(self.phys_cheung, axis=(1,2))
        self.min_phys_cheung = np.min(self.phys_cheung, axis=(1,2))

        #--------------------
        # Setup
        #--------------------        
        self.n_phys, self.nx_rempel, self.ny_rempel = self.phys.shape
        self.n_phys, self.nx_cheung, self.ny_cheung = self.phys_cheung.shape
        
        self.in_planes = 112 * 4 + 3
        self.out_planes = self.n_phys
        self.n_pixels = n_pixels
        self.n_training = n_training
        
        self.phys_max = np.max(np.vstack([self.max_phys,self.max_phys_inv,self.max_phys_cheung]), axis=0)
        self.phys_min = np.min(np.vstack([self.min_phys,self.min_phys_inv,self.min_phys_cheung]), axis=0)
            
        self.top = np.random.randint(0, self.nx_rempel - self.n_pixels, size=self.n_training)
        self.left = np.random.randint(0, self.ny_rempel - self.n_pixels, size=self.n_training)

        self.angle = np.random.randint(0, 4, size=n_training)
        self.flipx = np.random.randint(0, 2, size=n_training)
        self.flipy = np.random.randint(0, 2, size=n_training)

        self.v_shift = np.zeros(n_training, dtype='int') #np.random.randint(-0, 0, size=n_training)

        self.flip_snapshot = np.random.randint(0, 3, size=n_training)

        # Since Cheung simulation has a different size, modify accordingly the ranges for the subpatches
        ind = np.where(self.flip_snapshot == 2)[0]
        n = len(ind)
        self.top[ind] = np.random.randint(0, self.nx_cheung - self.n_pixels, size=n)
        self.left[ind] = np.random.randint(0, self.ny_cheung - self.n_pixels, size=n)
        
    def __getitem__(self, index):

        if (self.flip_snapshot[index] == 0):
            input = self.stokes[:,self.top[index]:self.top[index] + self.n_pixels, self.left[index]:self.left[index]+self.n_pixels]
            
            # Add jitter in velocity
            input[0:112,:,:] = np.roll(input[0:112,:,:], self.v_shift[index], axis=0)
            input[112:112*2,:,:] = np.roll(input[112:112*2,:,:], self.v_shift[index], axis=0)
            input[112*2:112*3,:,:] = np.roll(input[112*2:112*3,:,:], self.v_shift[index], axis=0)
            input[112*3:112*4,:,:] = np.roll(input[112*3:112*4,:,:], self.v_shift[index], axis=0)

            target = self.phys[:,self.top[index]:self.top[index] + self.n_pixels, self.left[index]:self.left[index]+self.n_pixels]
            
            # Add artificial velocity
            # target[7:14,:,:] += 1e0 * 0.0215 / 6302.0 * 3e5 * self.v_shift[index]

            target = (target - self.phys_min[:,None,None]) / (self.phys_max[:,None,None] - self.phys_min[:,None,None])            
        elif (self.flip_snapshot[index] == 1):
            input = self.stokes_inv[:,self.top[index]:self.top[index] + self.n_pixels, self.left[index]:self.left[index]+self.n_pixels]

            # Add jitter in velocity
            input[0:112,:,:] = np.roll(input[0:112,:,:], self.v_shift[index], axis=0)
            input[112:112*2,:,:] = np.roll(input[112:112*2,:,:], self.v_shift[index], axis=0)
            input[112*2:112*3,:,:] = np.roll(input[112*2:112*3,:,:], self.v_shift[index], axis=0)
            input[112*3:112*4,:,:] = np.roll(input[112*3:112*4,:,:], self.v_shift[index], axis=0)

            target = self.phys_inv[:,self.top[index]:self.top[index] + self.n_pixels, self.left[index]:self.left[index]+self.n_pixels]
            
            # Add artificial velocity
            # target[7:14,:,:] += 1e0 * 0.0215 / 6302.0 * 3e5 * self.v_shift[index]

            target = (target - self.phys_min[:,None,None]) / (self.phys_max[:,None,None] - self.phys_min[:,None,None])
        else:
            input = self.stokes_cheung[:,self.top[index]:self.top[index] + self.n_pixels, self.left[index]:self.left[index]+self.n_pixels]
            
            # Add jitter in velocity
            input[0:112,:,:] = np.roll(input[0:112,:,:], self.v_shift[index], axis=0)
            input[112:112*2,:,:] = np.roll(input[112:112*2,:,:], self.v_shift[index], axis=0)
            input[112*2:112*3,:,:] = np.roll(input[112*2:112*3,:,:], self.v_shift[index], axis=0)
            input[112*3:112*4,:,:] = np.roll(input[112*3:112*4,:,:], self.v_shift[index], axis=0)

            target = self.phys_cheung[:,self.top[index]:self.top[index] + self.n_pixels, self.left[index]:self.left[index]+self.n_pixels]
            
            # Add artificial velocity
            # target[7:14,:,:] += 1e0 * 0.0215 / 6302.0 * 3e5 * self.v_shift[index]

            target = (target - self.phys_min[:,None,None]) / (self.phys_max[:,None,None] - self.phys_min[:,None,None])


        input = np.rot90(input, self.angle[index], axes=(1,2)).copy()
        target = np.rot90(target, self.angle[index], axes=(1,2)).copy()
        
        if (self.flipx[index] == 1):
            input = np.flip(input, 1).copy()
            target = np.flip(target, 1).copy()            

        if (self.flipy[index] == 1):
            input = np.flip(input, 2).copy()
            target = np.flip(target, 2).copy()            

        return input.astype('float32'), target.astype('float32')

    def __len__(self):
        return self.n_training

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename+'.best')


class deep_3d_inversor(object):
    def __init__(self, batch_size, n_training=10000, n_validation=1000, n_pixels=32):
        self.cuda = torch.cuda.is_available()
        self.batch_size = batch_size
        self.device = torch.device("cuda" if self.cuda else "cpu")

        nvidia_smi.nvmlInit()
        self.handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0) 
        print("Computing in {0}".format(nvidia_smi.nvmlDeviceGetName(self.handle)))
               
        kwargs = {'num_workers': 4, 'pin_memory': True} if self.cuda else {}

        self.dataset_train = dataset_spot(n_training=n_training, n_pixels=n_pixels)
        self.train_loader = torch.utils.data.DataLoader(self.dataset_train, batch_size=self.batch_size, 
                                                        shuffle=True, **kwargs)

        self.dataset_test = dataset_spot(n_training=n_validation, n_pixels=n_pixels)
        self.test_loader = torch.utils.data.DataLoader(self.dataset_test, batch_size=self.batch_size, 
                                                        shuffle=True, **kwargs)  

        self.in_planes = self.dataset_train.in_planes   
        self.out_planes = self.dataset_train.out_planes

        self.model = model.block(in_planes=self.in_planes, out_planes=self.out_planes).to(self.device)        

    def optimize(self, epochs, lr=1e-4):

        self.lr = lr
        self.n_epochs = epochs

        root = 'weights_encdec'

        if not os.path.exists(root):
            os.makedirs(root)

        current_time = time.strftime("%Y-%m-%d-%H:%M")
        self.out_name = '{2}/{0}_-lr_{1}'.format(current_time, self.lr, root)

        print("Network name : {0}".format(self.out_name))

        # Copy model
        shutil.copyfile(model.__file__, '{0}.model.py'.format(self.out_name))        

        np.savez('{0}.normalization'.format(self.out_name), minimum=self.dataset_train.phys_min, maximum=self.dataset_train.phys_max)        

        # Open the index database
        self.db = database.neural_db(label=self.out_name, lr=self.lr, root=root)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.lossfn_L2 = nn.MSELoss().to(self.device)
        
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                            step_size=30,
                                            gamma=0.5)

        self.loss_L2 = []        
        self.loss_L2_val = []
        best_loss = -1e10

        trainF = open('{0}.loss.csv'.format(self.out_name, self.lr), 'w')

        for epoch in range(1, epochs + 1):
            self.scheduler.step()

            self.train(epoch)
            self.test()

            trainF.write('{},{},{}\n'.format(
                epoch, self.loss_L2[-1], self.loss_L2_val[-1]))
            trainF.flush()

            is_best = self.loss_L2_val[-1] > best_loss
            best_loss = max(self.loss_L2_val[-1], best_loss)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'best_loss': best_loss,
                'optimizer': self.optimizer.state_dict(),
            }, is_best, filename='{0}.pth'.format(self.out_name, self.lr))

            self.db.update(epoch, self.loss_L2[-1], self.loss_L2_val[-1])

        trainF.close()

    def train(self, epoch):
        self.model.train()
        print("Epoch {0}/{1} - {2}".format(epoch, self.n_epochs, time.strftime("%Y-%m-%d-%H:%M:%S")))
        t = tqdm(self.train_loader)
        
        loss_L2_avg = 0.0
        
        n = 1

        for param_group in self.optimizer.param_groups:
            current_lr = param_group['lr']

        for batch_idx, (data, target) in enumerate(t):

            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            
            loss_L2 = self.lossfn_L2(output, target)
                        
            loss_L2_avg += (loss_L2.item() - loss_L2_avg) / n            
            n += 1

            self.loss_L2.append(loss_L2_avg)

            loss_L2.backward()
            self.optimizer.step()

            tmp = nvidia_smi.nvmlDeviceGetUtilizationRates(self.handle) 

            t.set_postfix(loss=loss_L2_avg, lr=current_lr, gpu=tmp.gpu, mem=tmp.memory)

    def test(self):
        self.model.eval()
                
        loss_L2_avg = 0.0
        
        n = 1
        t = tqdm(self.test_loader)

        for param_group in self.optimizer.param_groups:
            current_lr = param_group['lr']
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(t):
                data, target = data.to(self.device), target.to(self.device)
            
                output = self.model(data)
            
            # sum up batch loss
                loss_L2 = self.lossfn_L2(output, target)
                                
                loss_L2_avg += (loss_L2.item() - loss_L2_avg) / n                
                n += 1

                self.loss_L2_val.append(loss_L2_avg)
        
                t.set_postfix(loss=loss_L2_avg, lr=current_lr)
            

deep_inversor = deep_3d_inversor(batch_size=128, n_training=50000, n_validation=2000, n_pixels=32)
deep_inversor.optimize(50, lr=3e-4)
