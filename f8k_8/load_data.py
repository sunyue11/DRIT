from torch.utils.data.dataset import Dataset
from scipy.io import loadmat, savemat
from torch.utils.data import DataLoader
import numpy as np

class CustomDataSet(Dataset):
    def __init__(
            self,
            images,
            texts):
        self.images = images
        self.texts = texts

    def __getitem__(self, index):
        img = self.images[index]
        text = self.texts[index]
        return img, text

    def __len__(self):
        count = len(self.images)
        return count


def ind2vec(ind, N=None):
    ind = np.asarray(ind)
    if N is None:
        N = ind.max() + 1
    return np.arange(N) == np.repeat(ind, N, axis=1)

def get_loader(batch_size):
    """
	初始数据集
    img_train = loadmat(path+"train_img.mat")['train_img']
    img_test = loadmat(path + "test_img.mat")['test_img']
    text_train = loadmat(path+"train_txt.mat")['train_txt']
    text_test = loadmat(path + "test_txt.mat")['test_txt']
    label_train = loadmat(path+"train_img_lab.mat")['train_img_lab']
    label_test = loadmat(path + "test_img_lab.mat")['test_img_lab']
    label_train = ind2vec(label_train).astype(int)
    label_test = ind2vec(label_test).astype(int)
    """
	
	#自己生成的数据集
    img_train = np.load('/root/sy/SCAN-master/f8k/train_ims.npy')
    img_test = np.load('/root/sy/SCAN-master/f8k/test_ims.npy')
    img_dev = np.load('/root/sy/SCAN-master/f8k/dev_ims.npy')
    text_train = np.load('/root/sy/SCAN-master/f8k/train_caps.npy')
    text_test = np.load('/root/sy/SCAN-master/f8k/test_caps.npy')
    text_dev = np.load('/root/sy/SCAN-master/f8k/dev_caps.npy')
	

    imgs = {'train': img_train, 'test': img_test,'dev':img_dev}
    texts = {'train': text_train, 'test': text_test,'dev':text_dev }
    dataset = {x: CustomDataSet(images=imgs[x], texts=texts[x])
               for x in ['train', 'test', 'dev']}

    shuffle = {'train': False, 'test': False, 'dev':False}

    dataloader = {x: DataLoader(dataset[x], batch_size=batch_size,
                                shuffle=shuffle[x], num_workers=0) for x in ['train', 'test', 'dev']}

    img_dim = img_train.shape[1]
    text_dim = text_train.shape[1]

    input_data_par = {}
    input_data_par['img_test'] = img_test
    input_data_par['text_test'] = text_test
    input_data_par['img_train'] = img_train
    input_data_par['text_train'] = text_train
    input_data_par['img_dev'] = img_dev
    input_data_par['text_dev'] = text_dev
    input_data_par['img_dim'] = img_dim
    input_data_par['text_dim'] = text_dim
    return dataloader, input_data_par