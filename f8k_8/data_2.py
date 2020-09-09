from torch.utils.data.dataset import Dataset
from scipy.io import loadmat, savemat
from torch.utils.data import DataLoader
import numpy as np
import sys
import importlib
importlib.reload(sys)
import torch
from options import TestOptions
from dataset import dataset_single
from model import DRIT
from saver import save_imgs
import os
import random
from recall_evaluate import rank_i2t,rank_t2i
from evaluation import i2t, t2i
from vocab import Vocabulary
import data
import model_2
import pickle
"""
f = open('./test_recall.log', 'a')
sys.stdout = f
sys.stderr = f
"""
parser = TestOptions()
opts = parser.parse()

vocab = pickle.load(open(os.path.join(opts.vocab_path, '%s_vocab.pkl' % opts.data_name), 'rb'))
opts.vocab_size = len(vocab)

test_loader = data.get_test_loader('test',opts.data_name, vocab, opts.crop_size, opts.batch_size, opts.workers, opts)

subspace = model_2.VSE(opts)
subspace.setgpu()
subspace.load_state_dict(torch.load(opts.resume2))
subspace.val_start()

# model
print('\n--- load model ---')
model = DRIT(opts)
model.setgpu(opts.gpu)
model.resume(opts.resume, train=False)
model.eval()
							
a = None
b = None	
c = None
d = None
for it, (images,captions,lengths,ids) in enumerate(test_loader):
    if it>=opts.test_iter:
        break
    images = images.cuda(opts.gpu).detach()
    captions = captions.cuda(opts.gpu).detach()
	
    img_emb, cap_emb = subspace.forward_emb(images, captions, lengths,volatile=True)
    
    img = img_emb.view(images.size(0),-1,32,32)
    cap = cap_emb.view(images.size(0),-1,32,32)
    image1,text1 = model.test_model2(img, cap)
    img2 = image1.view(images.size(0),-1)
    cap2 = text1.view(images.size(0),-1)
	
    if a is None:
        a = np.zeros((opts.batch_size*opts.test_iter, img_emb.size(1)))
        b = np.zeros((opts.batch_size*opts.test_iter, cap_emb.size(1))) 
        			
        c = np.zeros((opts.batch_size*opts.test_iter, img2.size(1)))
        d = np.zeros((opts.batch_size*opts.test_iter, cap2.size(1))) 	
		
    a[ids] = img_emb.data.cpu().numpy().copy()
    b[ids] = cap_emb.data.cpu().numpy().copy()
    
    c[ids] = img2.data.cpu().numpy().copy()
    d[ids] = cap2.data.cpu().numpy().copy()
	
aa=torch.from_numpy(a)
bb=torch.from_numpy(b)

cc=torch.from_numpy(c)
dd=torch.from_numpy(d)

#print(image1.size())
(r1, r5, r10, medr, meanr) = i2t(aa, bb)
print('subspace: Med:{}, r1:{}, r5:{}, r10:{}'.format(medr,r1, r5, r10))

(r1i, r5i, r10i, medri, meanri) = t2i(aa, bb,)
print('subspace: Med:{}, r1:{}, r5:{}, r10:{}'.format(medri,r1i, r5i, r10i))


(r2,r3,r4,m1,m2) = i2t(cc,dd)
print('encoder: Med:{}, r1:{}, r5:{}, r10:{}'.format(m1,r2,r3,r4))

(r2i,r3i,r4i,m1i,m2i) = t2i(cc,dd)
print('encoder: Med:{}, r1:{}, r5:{}, r10:{}'.format(m1i,r2i,r3i,r4i)) 



