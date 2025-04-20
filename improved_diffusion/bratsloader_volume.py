import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import os
import os.path
import nibabel
from scipy import ndimage
import random

def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')

class Low_Resolution(object):
    def __init__(self, factor):
        self.factor=factor

    def __call__(self, x):
        X,Y,Z = x.shape
        if Z % self.factor != 0:
            x=x[...,0:Z//self.factor*self.factor]
        Z_new=Z//self.factor
        result=np.zeros((X,Y,Z_new))
        for i in range (self.factor):
            result=+x[...,i::self.factor]
        result=result.astype(float)
        result/=self.factor
        result=np.repeat(result,self.factor,-1)
        return result

class Resize_image(object):
    '''
      Returns:
        img: 3d array, (z,y,x) or (D, H, W)
    '''
    def __init__(self, size=(3,256,256), f16=False):
        if not _isArrayLike(size):
            raise ValueError('each dimension of size must be defined')
        if f16:
            self.size = np.array(size, dtype=np.float16)
        else:
            self.size = np.array(size, dtype=np.float32)
        self.f16 = f16

    def __call__(self, img):
        z, x, y = img.shape
        assert not self.f16, "Resize_image not supported for f16"
        ori_shape = np.array((z, x, y), dtype=np.float32)
        resize_factor = self.size / ori_shape
        return ndimage.zoom(img, resize_factor, order=1)

class BRATSVolumes(torch.utils.data.Dataset):
    def __init__(self, directory, test_flag=False, normalize=None, mode='train'):
        '''
        directory is expected to contain some folder structure:
                  if some subfolder contains only files, all of these
                  files are assumed to have a name like
                  brats_train_NNN_XXX_123_w.nii.gz
                  where XXX is one of t1n, t1c, t2w, t2f, seg
                  we assume these five files belong to the same image
                  seg is supposed to contain the segmentation
        '''
        super().__init__()
        self.mode = mode
        self.directory = os.path.expanduser(directory)
        self.normalize = normalize or (lambda x: x)
        self.test_flag = test_flag

        self.LR=Low_Resolution(factor=4)
        self.resize=Resize_image(size=(128,128,128))

        if test_flag:
            self.seqtypes = ['t1n', 't1c', 't2w', 't2f']
        else:
            self.seqtypes = ['t1n', 't1c', 't2w', 't2f', 'seg']
        self.seqtypes_set = set(self.seqtypes)
        self.database = []

        if not self.mode == 'fake': # Used during training and for evaluating real data
            for root, dirs, files in os.walk(self.directory):
                # if there are no subdirs, we have a datadir
                start_id=64
                end_id=100

                if not dirs:
                    files.sort()
                    datapoint = dict()
                    # extract all files as channels
                    for f in files:
                        seqtype = f.split('-')[4].split('.')[0]
                        datapoint[seqtype] = os.path.join(root, f)
                    self.database+=[(datapoint,slice_id) for slice_id in range(start_id,end_id)]
        else:   # Used for evaluating fake data
            for root, dirs, files in os.walk(self.directory):
                for f in files:
                    datapoint = dict()
                    datapoint['t1n'] = os.path.join(root, f)
                    self.database.append(datapoint)

    def __getitem__(self, x):
        filedict = self.database[x]
        name = filedict[0]['t1n']
        index_seg=filedict[1]
        nib_img = nibabel.load(name)  # We only use t1 weighted images
        out = nib_img.get_fdata()
        LR_img=self.LR(out)
        out=self.resize(np.asarray(out))
        LR_img=self.resize(np.asarray(LR_img))
        # CLip and normalize the images
        out_clipped = np.clip(out, np.quantile(out, 0.001), np.quantile(out, 0.999))
        out_normalized = (out_clipped - np.min(out_clipped)) / (np.max(out_clipped) - np.min(out_clipped))

        LR_out_clipped = np.clip(LR_img, np.quantile(LR_img, 0.001), np.quantile(LR_img, 0.999))
        LR_normalized = (LR_out_clipped - np.min(LR_out_clipped)) / (np.max(LR_out_clipped) - np.min(LR_out_clipped))

        out=torch.tensor(out_normalized).unsqueeze(0)
        out_LR = torch.tensor(LR_normalized).unsqueeze(0)

        image=out.permute(0,3,1,2)
        image_LR=out_LR.permute(0,3,1,2)

        return image.float(), image_LR.float()

    def __len__(self):
        return len(self.database)
