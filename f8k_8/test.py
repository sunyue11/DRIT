import torch
from options import TestOptions
from dataset import dataset_single
from model import DRIT
from saver import save_imgs
import os
from load_data import get_loader
import sys
import importlib
importlib.reload(sys)
f = open('./test_pipei.log', 'a')
sys.stdout = f
sys.stderr = f
def main():
  # parse options
  parser = TestOptions()
  opts = parser.parse()

  # data loader
  train_loader, input_data_par = get_loader(1)

  # model
  print('\n--- load model ---')
  model = DRIT(opts)
  model.setgpu(opts.gpu)
  model.resume(opts.resume, train=False)
  model.eval()

  # directory
  result_dir = os.path.join(opts.result_dir, opts.name)
  if not os.path.exists(result_dir):
    os.mkdir(result_dir)

  # test
  print('\n--- testing ---')
  for it, (images_a, images_b,labels) in enumerate(train_loader['test']):
      images_a = images_a.cuda(opts.gpu).detach()
      images_b = images_b.cuda(opts.gpu).detach()
      with torch.no_grad():
        loss = model.test_model(images_a, images_b)
        print('it:{}, loss:{}'.format(it,loss))
  return

if __name__ == '__main__':
  main()
