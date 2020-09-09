import argparse

class TrainOptions():
  def __init__(self):
    self.parser = argparse.ArgumentParser()

    # data loader related
    self.parser.add_argument('--dataroot', type=str, help='path of data')
    self.parser.add_argument('--phase', type=str, default='train', help='phase for dataloading')
    self.parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    self.parser.add_argument('--resize_size', type=int, default=256, help='resized image size for training')
    self.parser.add_argument('--crop_size', type=int, default=224, help='cropped image size for training')
    self.parser.add_argument('--input_dim_a', type=int, default=1, help='# of input channels for domain A')
    self.parser.add_argument('--input_dim_b', type=int, default=1, help='# of input channels for domain B')
    self.parser.add_argument('--nThreads', type=int, default=8, help='# of threads for data loader')
    self.parser.add_argument('--no_flip', action='store_true', help='specified if no flipping')

    # ouptput related
    self.parser.add_argument('--name', type=str, default='trial', help='folder name to save outputs')
    self.parser.add_argument('--display_dir', type=str, default='../../logs', help='path for saving display results')
    self.parser.add_argument('--result_dir', type=str, default='../../results', help='path for saving result images and models')
    self.parser.add_argument('--display_freq', type=int, default=1, help='freq (iteration) of display')
    self.parser.add_argument('--img_save_freq', type=int, default=5, help='freq (epoch) of saving images')
    self.parser.add_argument('--model_save_freq', type=int, default=1, help='freq (epoch) of saving models')
    self.parser.add_argument('--no_display_img', action='store_true', help='specified if no dispaly')
    self.parser.add_argument('--train_iter', type=int, default=145*4, help='freq (epoch) of saving images')
    self.parser.add_argument('--val_iter', type=int, default=5*4, help='freq (epoch) of saving models')

    # training related
    self.parser.add_argument('--no_ms', action='store_true', help='disable mode seeking regularization')
    self.parser.add_argument('--concat', type=int, default=1, help='concatenate attribute features for translation, set 0 for using feature-wise transform')
    self.parser.add_argument('--dis_scale', type=int, default=2, help='scale of discriminator')
    self.parser.add_argument('--dis_norm', type=str, default='None', help='normalization layer in discriminator [None, Instance]')
    self.parser.add_argument('--dis_spectral_norm', action='store_true', help='use spectral normalization in discriminator')
    self.parser.add_argument('--lr_policy', type=str, default='lambda', help='type of learn rate decay')
    self.parser.add_argument('--n_ep', type=int, default=20, help='number of epochs') # 400 * d_iter
    self.parser.add_argument('--n_ep_decay', type=int, default=15, help='epoch start decay learning rate, set -1 if no decay') # 200 * d_iter
    self.parser.add_argument('--resume', type=str, default=None, help='specified the dir of saved models for resume the training')
    self.parser.add_argument('--d_iter', type=int, default=5, help='# of iterations for updating content discriminator')
    self.parser.add_argument('--gpu', type=int, default=0, help='gpu')
    self.parser.add_argument('--data_path', default='/root/wh/tmp/data/vseppdata/data/',help='path to datasets')
    self.parser.add_argument('--data_name', default='precomp',help='{coco,f8k,f30k,10crop}_precomp|coco|f8k|f30k')
    self.parser.add_argument('--vocab_path', default='/root/wh/tmp/data/vseppdata/vocab/',help='Path to saved vocabulary pickle files.')
    self.parser.add_argument('--workers', default=10, type=int,help='Number of data loader workers.')
    self.parser.add_argument('--use_restval', action='store_true',
                        help='Use the restval data for training on MSCOCO.')
    self.parser.add_argument('--margin', default=0.2, type=float,
                        help='Rank loss margin.')	
    self.parser.add_argument('--word_dim', default=300, type=int,
                        help='Dimensionality of the word embedding.')
    self.parser.add_argument('--embed_size', default=1024, type=int,
                        help='Dimensionality of the joint embedding.')
    self.parser.add_argument('--grad_clip', default=2., type=float,
                        help='Gradient clipping threshold.')
    self.parser.add_argument('--num_layers', default=1, type=int,
                        help='Number of GRU layers.')
    self.parser.add_argument('--learning_rate', default=.0002, type=float,
                        help='Initial learning rate.')
    self.parser.add_argument('--max_violation', action='store_true',
                        help='Use max instead of sum in the rank loss.')
    self.parser.add_argument('--img_dim', default=4096, type=int,
                        help='Dimensionality of the image embedding.')
    self.parser.add_argument('--finetune', action='store_true',
                        help='Fine-tune the image encoder.')
    self.parser.add_argument('--cnn_type', default='vgg19',
                        help="""The CNN used for image encoder
                        (e.g. vgg19, resnet152)""")
    self.parser.add_argument('--measure', default='cosine',
                        help='Similarity measure used (cosine|order)')
    self.parser.add_argument('--use_abs', action='store_true',
                        help='Take the absolute value of embedding vectors.')
    self.parser.add_argument('--no_imgnorm', action='store_true',
                        help='Do not normalize the image embeddings.')
    self.parser.add_argument('--lr_update', default=15, type=int,
                        help='Number of epochs to update the learning rate.')
    self.parser.add_argument('--niters_gan_enc', type=int, default=2,help='number of encoder iterations in training')
    self.parser.add_argument('--niters_gan_d', type=int, default=10,help='number of discriminator iterations in training')
    self.parser.add_argument('--lr_subspace', type=float, default=0.00002/4, help='subspace learning rate')
    self.parser.add_argument('--lr_MI', type=float, default=0.00002/4, help='MI estimator learning rate')
    #self.parser.add_argument('--lr_pre_enc', type=float, default=0.01,help='pretrain enc learning rate')
    self.parser.add_argument('--lr_dis', type=float, default=0.00002/4, help='discriminator learning rate')
    self.parser.add_argument('--lr_enc', type=float, default=0.00004/4,help='enc_a and enc_c learning rate')
    self.parser.add_argument('--lr_gen', type=float, default=0.00004/4, help='gen learning rate')
    self.parser.add_argument('--lr_gen_attr', type=float, default=0.00004/4, help='gen learning rate')
	
    self.parser.add_argument('--lr_pre_subspace', type=float, default=0.00008/4, help='subspace learning rate')
    self.parser.add_argument('--lr_pre_MI', type=float, default=0.00008/4, help='MI estimator learning rate')
    #self.parser.add_argument('--lr_pre_enc', type=float, default=0.01,help='pretrain enc learning rate')
    self.parser.add_argument('--lr_pre_enc', type=float, default=0.00008/4,help='enc_a and enc_c learning rate')
    self.parser.add_argument('--lr_pre_gen', type=float, default=0.00008/4, help='gen learning rate')
	
    self.parser.add_argument('--MI_w', type=int, default=1, help='weight of MI estimator loss')
    self.parser.add_argument('--semantic_w', type=int, default=1, help='weight of semantic loss')
    self.parser.add_argument('--recon_w', type=int, default=1, help='weight of recon loss')
    self.parser.add_argument('--gan_w', type=int, default=1, help='weight of gan loss')
    self.parser.add_argument('--content_w', type=int, default=0.01, help='weight of kl Gussian loss')
    self.parser.add_argument('--pre_iter', type=int, default=5, help='iteration of pretrain EG')


  def parse(self):
    self.opt = self.parser.parse_args()
    args = vars(self.opt)
    print('\n--- load options ---')
    for name, value in sorted(args.items()):
      print('%s: %s' % (str(name), str(value)))
    return self.opt

class TestOptions():
  def __init__(self):
    self.parser = argparse.ArgumentParser()

    # data loader related
    self.parser.add_argument('--dataroot', type=str, help='path of data')
    self.parser.add_argument('--phase', type=str, default='test', help='phase for dataloading')
    self.parser.add_argument('--resize_size', type=int, default=256, help='resized image size for training')
    self.parser.add_argument('--crop_size', type=int, default=216, help='cropped image size for training')
    self.parser.add_argument('--nThreads', type=int, default=4, help='for data loader')
    self.parser.add_argument('--input_dim_a', type=int, default=1, help='# of input channels for domain A')
    self.parser.add_argument('--input_dim_b', type=int, default=1, help='# of input channels for domain B')
    self.parser.add_argument('--a2b', type=int, default=1, help='translation direction, 1 for a2b, 0 for b2a')
    self.parser.add_argument('--data_path', default='/root/wh/tmp/data/vseppdata/data/',help='path to datasets')
    self.parser.add_argument('--data_name', default='precomp',help='{coco,f8k,f30k,10crop}_precomp|coco|f8k|f30k')
    self.parser.add_argument('--vocab_path', default='/root/wh/tmp/data/vseppdata/vocab/',help='Path to saved vocabulary pickle files.')
    self.parser.add_argument('--workers', default=10, type=int,help='Number of data loader workers.')
    self.parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    self.parser.add_argument('--test_iter', type=int, default=20, help='batch size')
	
    self.parser.add_argument('--use_restval', action='store_true',
                        help='Use the restval data for training on MSCOCO.')
    self.parser.add_argument('--margin', default=0.2, type=float,
                        help='Rank loss margin.')	
    self.parser.add_argument('--word_dim', default=300, type=int,
                        help='Dimensionality of the word embedding.')
    self.parser.add_argument('--embed_size', default=1024, type=int,
                        help='Dimensionality of the joint embedding.')
    self.parser.add_argument('--grad_clip', default=2., type=float,
                        help='Gradient clipping threshold.')
    self.parser.add_argument('--num_layers', default=1, type=int,
                        help='Number of GRU layers.')
    self.parser.add_argument('--learning_rate', default=.0002, type=float,
                        help='Initial learning rate.')
    self.parser.add_argument('--max_violation', action='store_true',
                        help='Use max instead of sum in the rank loss.')
    self.parser.add_argument('--img_dim', default=4096, type=int,
                        help='Dimensionality of the image embedding.')
    self.parser.add_argument('--finetune', action='store_true',
                        help='Fine-tune the image encoder.')
    self.parser.add_argument('--cnn_type', default='vgg19',
                        help="""The CNN used for image encoder
                        (e.g. vgg19, resnet152)""")
    self.parser.add_argument('--measure', default='cosine',
                        help='Similarity measure used (cosine|order)')
    self.parser.add_argument('--use_abs', action='store_true',
                        help='Take the absolute value of embedding vectors.')
    self.parser.add_argument('--no_imgnorm', action='store_true',
                        help='Do not normalize the image embeddings.')
    self.parser.add_argument('--lr_update', default=15, type=int,
                        help='Number of epochs to update the learning rate.')

    # ouptput related
    self.parser.add_argument('--num', type=int, default=5, help='number of outputs per image')
    self.parser.add_argument('--name', type=str, default='trial', help='folder name to save outputs')
    self.parser.add_argument('--result_dir', type=str, default='../../outputs', help='path for saving result images and models')

    # model related
    self.parser.add_argument('--concat', type=int, default=1, help='concatenate attribute features for translation, set 0 for using feature-wise transform')
    self.parser.add_argument('--no_ms', action='store_true', help='disable mode seeking regularization')
    self.parser.add_argument('--resume', type=str, required=True, help='specified the dir of saved models for resume the training')
    self.parser.add_argument('--resume2', type=str, required=True, help='specified the dir of saved models for resume the training')
    self.parser.add_argument('--gpu', type=int, default=0, help='gpu')
    self.parser.add_argument('--lr_subspace', type=float, default=1e-03, help='subspace learning rate')
    self.parser.add_argument('--lr_MI', type=float, default=1e-04, help='MI estimator learning rate')
    #self.parser.add_argument('--lr_pre_enc', type=float, default=0.01,help='pretrain enc learning rate')
    self.parser.add_argument('--lr_dis', type=float, default=1e-04/2, help='discriminator learning rate')
    self.parser.add_argument('--lr_enc', type=float, default=1e-04,help='enc_a and enc_c learning rate')
    self.parser.add_argument('--lr_gen', type=float, default=1e-04, help='gen learning rate')
    self.parser.add_argument('--lr_gen_attr', type=float, default=1e-04, help='gen learning rate')
	
    self.parser.add_argument('--lr_pre_subspace', type=float, default=1e-04, help='subspace learning rate')
    self.parser.add_argument('--lr_pre_MI', type=float, default=1e-04, help='MI estimator learning rate')
    #self.parser.add_argument('--lr_pre_enc', type=float, default=0.01,help='pretrain enc learning rate')
    self.parser.add_argument('--lr_pre_enc', type=float, default=1e-04,help='enc_a and enc_c learning rate')
    self.parser.add_argument('--lr_pre_gen', type=float, default=1e-04, help='gen learning rate')
	
    self.parser.add_argument('--MI_w', type=int, default=1, help='weight of MI estimator loss')
    self.parser.add_argument('--semantic_w', type=int, default=1, help='weight of semantic loss')
    self.parser.add_argument('--recon_w', type=int, default=1, help='weight of recon loss')
    self.parser.add_argument('--gan_w', type=int, default=1, help='weight of gan loss')
    self.parser.add_argument('--content_w', type=int, default=1, help='weight of kl Gussian loss')
    self.parser.add_argument('--pre_iter', type=int, default=2, help='iteration of pretrain EG')

  def parse(self):
    self.opt = self.parser.parse_args()
    args = vars(self.opt)
    print('\n--- load options ---')
    for name, value in sorted(args.items()):
      print('%s: %s' % (str(name), str(value)))
    # set irrelevant options
    self.opt.dis_scale = 3
    self.opt.dis_norm = 'None'
    self.opt.dis_spectral_norm = False
    return self.opt
