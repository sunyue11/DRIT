import torch
from options import TrainOptions
from dataset import dataset_unpair
from torch.utils.data.dataset import Dataset
from scipy.io import loadmat, savemat
from torch.utils.data import DataLoader
from model import DRIT
from saver import Saver
from load_data import get_loader
import numpy as np
import model_2
from vocab import Vocabulary
import data 
import sys
import importlib
import os
import pickle
from evaluation import i2t, t2i
importlib.reload(sys)
from recall_evaluate import rank_i2t,rank_t2i
from torch.nn.utils.clip_grad import clip_grad_norm
import warnings
import numpy
warnings.filterwarnings("ignore")

f = open('./answer_2.log', 'a')
sys.stdout = f
sys.stderr = f

def adjust_learning_rate(opt, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every 30 epochs"""
    lr = opt.learning_rate * (0.1 ** (epoch // opt.lr_update))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

	
	
def main():
  # parse options
  parser = TrainOptions()
  opts = parser.parse()

  # daita loader
  print('\n--- load dataset ---')
  vocab = pickle.load(open(os.path.join(opts.vocab_path, '%s_vocab.pkl' % opts.data_name), 'rb'))
  vocab_size = len(vocab)
  opts.vocab_size=vocab_size
  torch.backends.cudnn.enabled = False
    # Load data loaders
  train_loader, val_loader = data.get_loaders(opts.data_name, vocab, opts.crop_size, opts.batch_size, opts.workers, opts)
  test_loader = data.get_test_loader('test',opts.data_name, vocab, opts.crop_size, opts.batch_size, opts.workers, opts)
  # model
  print('\n--- load subspace ---')
  subspace=model_2.VSE(opts)
  subspace.setgpu()
  print('\n--- load model ---')
  model = DRIT(opts)
  model.setgpu(opts.gpu)
  if opts.resume is None:#之前没有保存过模型
    model.initialize()
    ep0 = -1
    total_it = 0
  else:
    ep0, total_it = model.resume(opts.resume)
  model.set_scheduler(opts, last_ep=ep0)
  ep0 += 1
  print('start the training at epoch %d'%(ep0))

  # saver for display and output
  saver = Saver(opts)

  # train
  print('\n--- train ---')
  max_it = 500000
  score=0.0
  subspace.train_start()
  for ep in range(ep0, opts.pre_iter):
    print('-----ep:{} --------'.format(ep))
    for it, (images,captions,lengths,ids) in enumerate(train_loader):
      if it >= opts.train_iter:
        break
      # input data  
      images = images.cuda(opts.gpu).detach()
      captions = captions.cuda(opts.gpu).detach()
	  
      img,cap=subspace.train_emb(images,captions,lengths,ids,pre=True)#[b,1024]
	  
      subspace.pre_optimizer.zero_grad()	  
      img = img.view(images.size(0),-1,32,32)
      cap = cap.view(images.size(0),-1,32,32)
	  
      model.pretrain_ae(img,cap)
	  
      if opts.grad_clip > 0:
        clip_grad_norm(subspace.params, opts.grad_clip)
		
      subspace.pre_optimizer.step()
  
  
  for ep in range(ep0, opts.n_ep):
    subspace.train_start()
    adjust_learning_rate(opts, subspace.optimizer, ep)
    for it, (images,captions,lengths,ids) in enumerate(train_loader):
      if it >= opts.train_iter:
        break
      # input data	  
      images = images.cuda(opts.gpu).detach()
      captions = captions.cuda(opts.gpu).detach()
	  
      img,cap=subspace.train_emb(images,captions,lengths,ids)#[b,1024]
	  
      img = img.view(images.size(0),-1,32,32)
      cap = cap.view(images.size(0),-1,32,32)
	 
      subspace.optimizer.zero_grad()
	  
      for p in model.disA.parameters():
          p.requires_grad = True
      for p in model.disB.parameters():
          p.requires_grad = True	
      for p in model.disA_attr.parameters():
          p.requires_grad = True
      for p in model.disB_attr.parameters():
          p.requires_grad = True	
		  
      for i in range(opts.niters_gan_d):#5			
        model.update_D(img,cap)
        
      for p in model.disA.parameters():
          p.requires_grad = False
      for p in model.disB.parameters():
          p.requires_grad = False
      for p in model.disA_attr.parameters():
          p.requires_grad = False
      for p in model.disB_attr.parameters():
          p.requires_grad = False	
		  
      for i in range(opts.niters_gan_enc):
        model.update_E(img,cap)#利用新的content损失函数
		
      subspace.optimizer.step()

      print('total_it: %d (ep %d, it %d), lr %09f' % (total_it, ep, it, model.gen_opt.param_groups[0]['lr']))
      total_it += 1

    # decay learning rate
    if opts.n_ep_decay > -1:
      model.update_lr()

    # save result image
    #saver.write_img(ep, model)
    if (ep+1)%opts.n_ep == 0:
        print('save model')
        filename = os.path.join(opts.result_dir, opts.name)
        model.save('%s/final_model.pth' % (filename), ep, total_it)
        torch.save(subspace.state_dict(),'%s/final_subspace.pth' % (filename))  
    elif (ep+1)%10== 0:
        print('save model')
        filename = os.path.join(opts.result_dir, opts.name)
        model.save('%s/%s_model.pth' % (filename,str(ep+1)), ep, total_it)
        torch.save(subspace.state_dict(),'%s/%s_subspace.pth' % (filename,str(ep+1)))   

    if (ep+1)%opts.model_save_freq == 0:
        a = None
        b = None	
        c = None
        d = None
        subspace.val_start()		
        for it, (images,captions,lengths,ids) in enumerate(test_loader):
            if it >= opts.val_iter:
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
                a = np.zeros((opts.val_iter*opts.batch_size, img_emb.size(1)))
                b = np.zeros((opts.val_iter*opts.batch_size, cap_emb.size(1)))  
                				
                c = np.zeros((opts.val_iter*opts.batch_size, img2.size(1)))
                d = np.zeros((opts.val_iter*opts.batch_size, cap2.size(1))) 	
				
				
            a[ids] = img_emb.data.cpu().numpy().copy()
            b[ids] = cap_emb.data.cpu().numpy().copy()
            
            c[ids] = img2.data.cpu().numpy().copy()
            d[ids] = cap2.data.cpu().numpy().copy()
			

        aa=torch.from_numpy(a)
        bb=torch.from_numpy(b)
        
        cc=torch.from_numpy(c)
        dd=torch.from_numpy(d)
		

        (r1, r5, r10, medr, meanr) = i2t(aa, bb, measure=opts.measure)
        print('test640: subspace: Med:{}, r1:{}, r5:{}, r10:{}'.format(medr,r1, r5, r10))
		
        (r1i, r5i, r10i, medri, meanr) = t2i(aa, bb, measure=opts.measure)
        print('test640: subspace: Med:{}, r1:{}, r5:{}, r10:{}'.format(medri,r1i, r5i, r10i))
        
        (r2,r3,r4,m1,m2) = i2t(cc,dd,measure=opts.measure)
        print('test640: encoder: Med:{}, r1:{}, r5:{}, r10:{}'.format(m1,r2,r3,r4))
        
        (r2i,r3i,r4i,m1i,m2i) = t2i(cc,dd,measure=opts.measure)
        print('test640: encoder: Med:{}, r1:{}, r5:{}, r10:{}'.format(m1i,r2i,r3i,r4i)) 
		
        curr=r2+r3+r4+r2i+r3i+r4i
		
		
        if curr> score:
            score = curr
            print('save model')
            filename = os.path.join(opts.result_dir, opts.name)
            model.save('%s/best_model.pth' % (filename), ep, total_it)
            torch.save(subspace.state_dict(),'%s/subspace.pth' % (filename))
			
			
        a = None
        b = None	
        c = None
        d = None
		
        for it, (images,captions,lengths,ids) in enumerate(test_loader):
          
            images = images.cuda(opts.gpu).detach()
            captions = captions.cuda(opts.gpu).detach()
			
            img_emb, cap_emb = subspace.forward_emb(images, captions, lengths,volatile=True)
            
            img = img_emb.view(images.size(0),-1,32,32)
            cap = cap_emb.view(images.size(0),-1,32,32)
            image1,text1 = model.test_model2(img, cap)
            img2 = image1.view(images.size(0),-1)
            cap2 = text1.view(images.size(0),-1)
			
            
            if a is None:
                a = np.zeros((len(test_loader.dataset), img_emb.size(1)))
                b = np.zeros((len(test_loader.dataset), cap_emb.size(1)))  
                				
                c = np.zeros((len(test_loader.dataset), img2.size(1)))
                d = np.zeros((len(test_loader.dataset), cap2.size(1))) 	
				
				
            a[ids] = img_emb.data.cpu().numpy().copy()
            b[ids] = cap_emb.data.cpu().numpy().copy()
            
            c[ids] = img2.data.cpu().numpy().copy()
            d[ids] = cap2.data.cpu().numpy().copy()
			

        aa=torch.from_numpy(a)
        bb=torch.from_numpy(b)
        
        cc=torch.from_numpy(c)
        dd=torch.from_numpy(d)
		

        (r1, r5, r10, medr, meanr) = i2t(aa, bb, measure=opts.measure)
        print('test5000: subspace: Med:{}, r1:{}, r5:{}, r10:{}'.format(medr,r1, r5, r10))
		
        (r1i, r5i, r10i, medri, meanr) = t2i(aa, bb, measure=opts.measure)
        print('test5000: subspace: Med:{}, r1:{}, r5:{}, r10:{}'.format(medri,r1i, r5i, r10i))
        
        (r2,r3,r4,m1,m2) = i2t(cc,dd,measure=opts.measure)
        print('test5000: encoder: Med:{}, r1:{}, r5:{}, r10:{}'.format(m1,r2,r3,r4))
        
        (r2i,r3i,r4i,m1i,m2i) = t2i(cc,dd,measure=opts.measure)
        print('test5000: encoder: Med:{}, r1:{}, r5:{}, r10:{}'.format(m1i,r2i,r3i,r4i)) 
        
      	
  return

if __name__ == '__main__':
  main()
