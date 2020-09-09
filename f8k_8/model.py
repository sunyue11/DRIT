import networks
import torch
import torch.nn as nn
import model_1
from MI_2methods import KLLoss,DIMLoss
from torch.autograd import Variable
from torch.nn.utils.clip_grad import clip_grad_norm
import torch.autograd as autograd
LAMBDA = 10 
def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())


def order_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score
	
	


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0, measure=False, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        if measure == 'order':
            self.sim = order_sim
        else:
            self.sim = cosine_sim

        self.max_violation = max_violation

    def forward(self, im, s):
        # compute image-sentence score matrix
        scores = self.sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()
class DRIT(nn.Module):
  def __init__(self, opts):
    super(DRIT, self).__init__()

    # parameters
    lr=0.0001
    self.nz = 8
    self.concat = opts.concat
    self.lr_subspace=opts.lr_subspace
    #self.lr_MI=opts.lr_MI
    self.lr_dis = opts.lr_dis
    self.lr_enc = opts.lr_enc
    self.lr_gen = opts.lr_gen
    self.lr_gen_attr = opts.lr_gen_attr
	
    self.lr_pre_subspace=opts.lr_pre_subspace
    #self.lr_pre_MI = opts.lr_pre_MI
    self.lr_pre_enc = opts.lr_pre_enc
    self.lr_pre_gen = opts.lr_pre_gen
	
    self.margin = opts.margin
    self.semantic_w = opts.semantic_w
    self.recon_w = opts.recon_w
    #self.MI_w = opts.MI_w
    self.gan_w = opts.gan_w
    self.content_w = opts.content_w
    self.no_ms = opts.no_ms
    self.loss_1=nn.BCEWithLogitsLoss()
    #映射到公共子空间的网络
    self.subspace=model_1.IDCM_NN(img_input_dim=4096, text_input_dim=300)
    #self.MI=KLLoss()
    self.criterion = ContrastiveLoss(margin=opts.margin,
                                     measure=opts.measure,
                                     max_violation=opts.max_violation)
    one = torch.tensor(1, dtype=torch.float).cuda(0)
    self.mone = one * -1
									 
    # discriminators
    if opts.dis_scale > 1:  #=3
      self.disA = networks.MultiScaleDis(opts.input_dim_a, n_scale=opts.dis_scale, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
      self.disB = networks.MultiScaleDis(opts.input_dim_b, n_scale=opts.dis_scale, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
      self.disA_attr = networks.MultiScaleDis(opts.input_dim_a, n_scale=opts.dis_scale, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
      self.disB_attr = networks.MultiScaleDis(opts.input_dim_b, n_scale=opts.dis_scale, norm=opts.dis_norm, sn=opts.dis_spectral_norm)

    else:
      self.disA = networks.Dis(opts.input_dim_a, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
      self.disB = networks.Dis(opts.input_dim_b, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
   
    # encoders
    self.enc_c = networks.E_content(opts.input_dim_a, opts.input_dim_b)
    if self.concat:
      self.enc_a = networks.E_attr_concat(opts.input_dim_a, opts.input_dim_b, self.nz, \
          norm_layer=None, nl_layer=networks.get_non_linearity(layer_type='lrelu'))
    else:
      self.enc_a = networks.E_attr(opts.input_dim_a, opts.input_dim_b, self.nz)

    # generator
    if self.concat:
      self.gen = networks.G_concat(opts.input_dim_a, opts.input_dim_b, nz=self.nz)
    else:
      self.gen = networks.G(opts.input_dim_a, opts.input_dim_b, nz=self.nz)
	
    self.gen_attr = networks.G_a(opts, opts.input_dim_a, opts.input_dim_b)

    # optimizers
    self.subspace_opt = torch.optim.Adam(self.subspace.parameters(), lr=self.lr_subspace, betas=(0.5, 0.999), weight_decay=0.0001)
    #self.MI_opt = torch.optim.Adam(self.MI.parameters(), lr=self.lr_MI, betas=(0.5, 0.999), weight_decay=0.0001)
    self.disA_opt = torch.optim.Adam(self.disA.parameters(), lr=self.lr_dis, betas=(0.5, 0.999), weight_decay=0.0001)
    self.disB_opt = torch.optim.Adam(self.disB.parameters(), lr=self.lr_dis, betas=(0.5, 0.999), weight_decay=0.0001)
    self.disA_attr_opt = torch.optim.Adam(self.disA_attr.parameters(), lr=self.lr_dis, betas=(0.5, 0.999), weight_decay=0.0001)
    self.disB_attr_opt = torch.optim.Adam(self.disB_attr.parameters(), lr=self.lr_dis, betas=(0.5, 0.999), weight_decay=0.0001)
    self.enc_c_opt = torch.optim.Adam(self.enc_c.parameters(), lr=self.lr_enc, betas=(0.5, 0.999), weight_decay=0.0001)
    self.enc_a_opt = torch.optim.Adam(self.enc_a.parameters(), lr=self.lr_enc, betas=(0.5, 0.999), weight_decay=0.0001)
    self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=self.lr_gen, betas=(0.5, 0.999), weight_decay=0.0001)
    self.gen_attr_opt = torch.optim.Adam(self.gen_attr.parameters(), lr=self.lr_gen_attr, betas=(0.5, 0.999), weight_decay=0.0001)
	
    self.subspace_pre_opt = torch.optim.Adam(self.subspace.parameters(), lr=self.lr_pre_subspace, betas=(0.5, 0.999), weight_decay=0.0001)
    #self.MI_pre_opt = torch.optim.Adam(self.MI.parameters(), lr=self.lr_pre_MI, betas=(0.5, 0.999), weight_decay=0.0001)
    self.enc_c_pre_opt = torch.optim.Adam(self.enc_c.parameters(), lr=self.lr_pre_enc, betas=(0.5, 0.999), weight_decay=0.0001)
    self.enc_a_pre_opt = torch.optim.Adam(self.enc_a.parameters(), lr=self.lr_pre_enc, betas=(0.5, 0.999), weight_decay=0.0001)
    self.gen_pre_opt = torch.optim.Adam(self.gen.parameters(), lr=self.lr_pre_gen, betas=(0.5, 0.999), weight_decay=0.0001)

    # Setup the loss function for training
    self.criterionL1 = torch.nn.L1Loss()
    self.MSE_loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)

  def initialize(self):
    self.subspace.apply(networks.gaussian_weights_init)
    #self.MI.apply(networks.gaussian_weights_init)
    self.disA.apply(networks.gaussian_weights_init)
    self.disB.apply(networks.gaussian_weights_init)
    self.disA_attr.apply(networks.gaussian_weights_init)
    self.disB_attr.apply(networks.gaussian_weights_init)
    self.gen.apply(networks.gaussian_weights_init)
    self.gen_attr.apply(networks.gaussian_weights_init)
    self.enc_c.apply(networks.gaussian_weights_init)
    self.enc_a.apply(networks.gaussian_weights_init)

  def set_scheduler(self, opts, last_ep=0):
    self.subspace_sch = networks.get_scheduler(self.subspace_opt, opts, last_ep)
    #self.MI_sch = networks.get_scheduler(self.MI_opt, opts, last_ep)
    self.disA_sch = networks.get_scheduler(self.disA_opt, opts, last_ep)
    self.disB_sch = networks.get_scheduler(self.disB_opt, opts, last_ep)
    self.disA_attr_sch = networks.get_scheduler(self.disA_attr_opt, opts, last_ep)
    self.disB_attr_sch = networks.get_scheduler(self.disB_attr_opt, opts, last_ep)
    self.enc_c_sch = networks.get_scheduler(self.enc_c_opt, opts, last_ep)
    self.enc_a_sch = networks.get_scheduler(self.enc_a_opt, opts, last_ep)
    self.gen_sch = networks.get_scheduler(self.gen_opt, opts, last_ep)
    self.gen_attr_sch = networks.get_scheduler(self.gen_attr_opt, opts, last_ep)
	
    self.subspace_pre_sch = networks.get_scheduler(self.subspace_pre_opt, opts, last_ep)
    #self.MI_pre_sch = networks.get_scheduler(self.MI_pre_opt, opts, last_ep)
    self.enc_c_pre_sch = networks.get_scheduler(self.enc_c_pre_opt, opts, last_ep)
    self.enc_a_pre_sch = networks.get_scheduler(self.enc_a_pre_opt, opts, last_ep)
    self.gen_pre_sch = networks.get_scheduler(self.gen_pre_opt, opts, last_ep)

  def setgpu(self, gpu):
    self.gpu = gpu
    self.subspace.cuda(self.gpu)
    #self.MI.cuda(self.gpu)
    self.disA.cuda(self.gpu)
    self.disB.cuda(self.gpu)
    self.disA_attr.cuda(self.gpu)
    self.disB_attr.cuda(self.gpu)
    self.enc_c.cuda(self.gpu)
    self.enc_a.cuda(self.gpu)
    self.gen.cuda(self.gpu)
    self.gen_attr.cuda(self.gpu)

  def get_z_random(self, batchSize, nz, random_type='gauss'):
    z = torch.randn(batchSize, nz).cuda(self.gpu)
    return z
	
  def test_model1(self,image_a,image_b):
    self.real_A_encoded_1,self.real_B_encoded_1=self.subspace(image_a,image_b)
    return self.real_A_encoded_1,self.real_B_encoded_1
 
  def test_model2(self,image_a,image_b):
    image_1,text_1= self.enc_c.forward(image_a,image_b)
    return image_1,text_1

	
  def pretrain_ae(self,image_a,image_b):
    #self.MI_pre_opt.zero_grad()
    self.enc_c_pre_opt.zero_grad()
    self.enc_a_pre_opt.zero_grad()
    self.gen_pre_opt.zero_grad()
	
    self.real_A_encoded_1 = image_a
    self.real_B_encoded_1 = image_b
    self.forward()
    loss_G_L1_AA = self.criterionL1(self.fake_AA_encoded, self.real_A_encoded_1) 
    loss_G_L1_BB = self.criterionL1(self.fake_BB_encoded, self.real_B_encoded_1)
    loss = loss_G_L1_AA + loss_G_L1_BB
    print('pretrain : loss_G_L1_AA:{}, loss_G_L1_BB:{}'.format(loss_G_L1_AA,loss_G_L1_BB))
    loss.backward(retain_graph=True)
    self.recon_loss_a = loss_G_L1_AA.item()
    self.recon_loss_b = loss_G_L1_BB.item()
    self.loss=loss.item()
	
    #self.MI_pre_opt.step()
    self.enc_c_pre_opt.step()
    self.enc_a_pre_opt.step()
    self.gen_pre_opt.step()
	
  def forward(self):
    # input images
    # get encoded z_c
    self.z_content_a, self.z_content_b = self.enc_c.forward(self.real_A_encoded_1, self.real_B_encoded_1)#[batch,256,8,8]
    
    # get encoded z_a
    if self.concat:
      self.mu_a, self.logvar_a, self.mu_b, self.logvar_b = self.enc_a.forward(self.real_A_encoded_1, self.real_B_encoded_1)
      std_a = self.logvar_a.mul(0.5).exp_()
      eps_a = self.get_z_random(std_a.size(0), std_a.size(1), 'gauss')
      self.z_attr_a = eps_a.mul(std_a).add_(self.mu_a)#确保服从高斯分布
      std_b = self.logvar_b.mul(0.5).exp_()
      eps_b = self.get_z_random(std_b.size(0), std_b.size(1), 'gauss')
      self.z_attr_b = eps_b.mul(std_b).add_(self.mu_b)
    else:
      self.z_attr_a, self.z_attr_b = self.enc_a.forward(self.real_A_encoded_1, self.real_B_encoded_1)
    #[20,8]
   
    if not self.no_ms:
      self.fake_A_encoded = self.gen.forward_a(self.z_content_b, self.z_attr_a)#[5,1,300,300]
      self.fake_B_encoded = self.gen.forward_b(self.z_content_a, self.z_attr_b)
      self.fake_AA_encoded = self.gen.forward_a(self.z_content_a, self.z_attr_a)#[5,1,64,64]
      self.fake_BB_encoded = self.gen.forward_b(self.z_content_b, self.z_attr_b)

    else:#进行拼接的目的就是一起放进去计算，其实也可以单独一个个放进去的，就是麻烦点
      #z_content_b和z_attr_a进行gen生成a域图像，然后在enc_c和enc_a生成特征，再进行gen，可以实现循环一致性损失
      #z_content_a和z_attr_a进行gen生成a域图像，然后与input a比较可以使gen进行训练
      #z_content_b和z_random进行gen生成图像，然后再通过enc_a生成属性特征，与z_random比较，可保证z_random不会被忽略
      input_content_forA = torch.cat((self.z_content_b, self.z_content_a),0)
      input_content_forB = torch.cat((self.z_content_a, self.z_content_b),0)
      input_attr_forA = torch.cat((self.z_attr_a, self.z_attr_a),0)
      input_attr_forB = torch.cat((self.z_attr_b, self.z_attr_b),0)
      output_fakeA = self.gen.forward_a(input_content_forA, input_attr_forA)
      output_fakeB = self.gen.forward_b(input_content_forB, input_attr_forB)
      self.fake_A_encoded, self.fake_AA_encoded = torch.split(output_fakeA, self.z_content_a.size(0), dim=0)
      self.fake_B_encoded, self.fake_BB_encoded = torch.split(output_fakeB, self.z_content_a.size(0), dim=0)
    # for display

    self.attr_a_encoded = self.gen_attr.forward_a(self.z_attr_b) 
    self.attr_b_encoded = self.gen_attr.forward_b(self.z_attr_a) 
		
  def calc_gradient_penalty(self,netD, real_data, fake_data):
    #print real_data.size()
    alpha = torch.rand(real_data.size(0), 1,1,1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda(0)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.cuda(0)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)
    gradients=torch.zeros(real_data.size()).cuda(0)
    for i in disc_interpolates:
        gradients += autograd.grad(outputs=i, inputs=interpolates,
                                  grad_outputs=torch.ones(i.size()).cuda(0),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty
		
	
  def update_D(self,image_a,image_b):
    self.real_A_encoded_1 = image_a
    self.real_B_encoded_1 = image_b
    self.forward()
  
    # update disA
    self.disA_opt.zero_grad()
    D_cost_A,Wasserstein_D_A = self.backward_D(self.disA, self.real_A_encoded_1, self.fake_A_encoded)
    print('D_cost_A:{},Wasserstein_D_A:{}'.format(D_cost_A,Wasserstein_D_A))
    self.disA_opt.step()
	
    # update disB
    self.disB_opt.zero_grad()
    D_cost_B,Wasserstein_D_B = self.backward_D(self.disB, self.real_B_encoded_1, self.fake_B_encoded)
    print('D_cost_B:{},Wasserstein_D_B:{}'.format(D_cost_B,Wasserstein_D_B))
    self.disB_opt.step()
	
    # update disA
    self.disA_attr_opt.zero_grad()
    D_cost_A_attr,Wasserstein_D_A_attr = self.backward_D(self.disA_attr, self.real_A_encoded_1, self.attr_a_encoded)
    print('D_cost_A_attr:{},Wasserstein_D_A_attr:{}'.format(D_cost_A_attr,Wasserstein_D_A_attr))
    self.disA_attr_opt.step()
	
    # update disB
    self.disB_attr_opt.zero_grad()
    D_cost_B_attr,Wasserstein_D_B_attr = self.backward_D(self.disB_attr, self.real_B_encoded_1, self.attr_b_encoded)
    print('D_cost_B_attr:{},Wasserstein_D_B_attr:{}'.format(D_cost_B_attr,Wasserstein_D_B_attr))
    self.disB_attr_opt.step()
	
  def backward_D(self, netD, real, fake):
    D_real = netD.forward(real.detach())
    loss1=0
    for i in D_real:
        loss1+=i.mean()
    loss1 = loss1.cuda(0)
    loss1.backward(self.mone)

    D_fake = netD.forward(fake.detach())#求出D(x)
    loss2=0
    for i in D_fake:
        loss2+=i.mean()
    loss2.backward(self.mone)

 
    gradient_penalty = self.calc_gradient_penalty(netD,real,fake)
    gradient_penalty.backward()
	
    D_cost = loss2 - loss1 + gradient_penalty
    Wasserstein_D = loss1 - loss2

    return D_cost,Wasserstein_D

  def update_E(self,image_a,image_b):
    self.real_A_encoded_1 = image_a
    self.real_B_encoded_1 = image_b
    self.forward()
	
    self.enc_c_opt.zero_grad()
    self.enc_a_opt.zero_grad()
    self.gen_opt.zero_grad()
    self.gen_attr_opt.zero_grad()
	
    loss_c = self.criterionL1(self.z_content_a,self.z_content_b)
    loss_a = self.criterionL1(self.z_attr_a,self.z_attr_b)
	
    self.z_content_a=self.z_content_a.view(self.real_A_encoded_1.size(0),-1)
    self.z_content_b = self.z_content_b.view(self.real_A_encoded_1.size(0),-1)	
    loss = self.criterion(self.z_content_a, self.z_content_b)*self.content_w
	
    # Ladv for generator
    loss_G_GAN_A = self.backward_G_GAN(self.fake_A_encoded, self.disA)
    loss_G_GAN_B = self.backward_G_GAN(self.fake_B_encoded, self.disB)

    loss_G_GAN_attr_A = self.backward_G_GAN(self.attr_a_encoded, self.disA_attr)
    loss_G_GAN_attr_B = self.backward_G_GAN(self.attr_b_encoded, self.disB_attr)

	
    loss.backward(retain_graph=True)

    print('loss_G_GAN_A:{}, loss_G_GAN_B:{},loss_G_GAN_attr_A:{},loss_G_GAN_attr_B:{},loss_content:{},loss_c:{},loss_a:{}'.format(loss_G_GAN_A,loss_G_GAN_B,loss_G_GAN_attr_A,loss_G_GAN_attr_B,loss,loss_c,loss_a))

    self.enc_c_opt.step()
    self.enc_a_opt.step()
    self.gen_opt.step()
    self.gen_attr_opt.step()
	

  def update_EG(self,image_a,image_b):
    # update subspace, G, Ec, Ea
    #self.MI_opt.zero_grad()
    self.real_A_encoded_1 = image_a
    self.real_B_encoded_1 = image_b
    self.forward()
	
    self.enc_c_opt.zero_grad()
    self.enc_a_opt.zero_grad()
    self.gen_opt.zero_grad()
    self.backward_EG()#
    #self.MI_opt.step()
    self.enc_c_opt.step()
    self.enc_a_opt.step()
    self.gen_opt.step()
    
    """
    # update G, Ec
    self.enc_c_opt.zero_grad()
    self.gen_opt.zero_grad()
    self.backward_G_alone()
    self.enc_c_opt.step()
    self.gen_opt.step()
    """
	
  def backward_EG(self):
    # content Ladv for generator
	#重构损失
    loss_G_L1_AA = self.criterionL1(self.fake_AA_encoded, self.real_A_encoded_1) *self.recon_w#高
    loss_G_L1_BB = self.criterionL1(self.fake_BB_encoded, self.real_B_encoded_1) *self.recon_w
	
    loss_G = loss_G_L1_AA + loss_G_L1_BB 
    print('loss_G_L1_AA:{}, loss_G_L1_BB:{}'.format(loss_G_L1_AA,loss_G_L1_BB))
    loss_G.backward(retain_graph=True)    
	
    self.l1_recon_AA_loss = loss_G_L1_AA.item()
    self.l1_recon_BB_loss = loss_G_L1_BB.item()
    #self.sub_loss = loss_sub.item()
    self.G_loss = loss_G.item()


  def backward_G_GAN(self, fake, netD=None):
    outs_fake = netD.forward(fake)
    loss=0
    for i in outs_fake:
        loss += i.mean()
    loss.backward(self.mone,retain_graph=True)
    G_cost = -loss
    return G_cost
    
  

  
  def update_lr(self):
    self.subspace_sch.step()
    #self.MI_sch.step()
    self.disA_sch.step()
    self.disB_sch.step()
    self.disA_attr_sch.step()
    self.disB_attr_sch.step()
    self.enc_c_sch.step()
    self.enc_a_sch.step()
    self.gen_sch.step()
    self.gen_attr_sch.step()
	
    self.subspace_pre_sch.step()
    #self.MI_pre_sch.step()
    self.enc_c_pre_sch.step()
    self.enc_a_pre_sch.step()
    self.gen_pre_sch.step()

  def _l2_regularize(self, mu):
    mu_2 = torch.pow(mu, 2)
    encoding_loss = torch.mean(mu_2)
    return encoding_loss

  def resume(self, model_dir, train=True):
    checkpoint = torch.load(model_dir)
    # weight
    self.subspace.load_state_dict(checkpoint['subspace'])
    if train:
      self.disA.load_state_dict(checkpoint['disA'])
      self.disB.load_state_dict(checkpoint['disB'])
      self.disA_attr.load_state_dict(checkpoint['disA_attr'])
      self.disB_attr.load_state_dict(checkpoint['disB_attr'])
    self.enc_c.load_state_dict(checkpoint['enc_c'])
    self.enc_a.load_state_dict(checkpoint['enc_a'])
    self.gen.load_state_dict(checkpoint['gen'])
    self.gen_attr.load_state_dict(checkpoint['gen_attr'])
    # optimizer
    if train:
      self.subspace_opt.load_state_dict(checkpoint['subspace_opt'])
      self.disA_opt.load_state_dict(checkpoint['disA_opt'])
      self.disB_opt.load_state_dict(checkpoint['disB_opt'])
      self.disA_attr_opt.load_state_dict(checkpoint['disA_attr_opt'])
      self.disB_attr_opt.load_state_dict(checkpoint['disB_attr_opt'])
      self.enc_c_opt.load_state_dict(checkpoint['enc_c_opt'])
      self.enc_a_opt.load_state_dict(checkpoint['enc_a_opt'])
      self.gen_opt.load_state_dict(checkpoint['gen_opt'])
      self.gen_attr_opt.load_state_dict(checkpoint['gen_attr_opt'])
    return checkpoint['ep'], checkpoint['total_it']

  def save(self, filename, ep, total_it):
    state = {
             'subspace': self.subspace.state_dict(),
             'disA': self.disA.state_dict(),
             'disB': self.disB.state_dict(),
             'disA_attr': self.disA_attr.state_dict(),
             'disB_attr': self.disB_attr.state_dict(),
             'enc_c': self.enc_c.state_dict(),
             'enc_a': self.enc_a.state_dict(),
             'gen': self.gen.state_dict(),
             'gen_attr': self.gen_attr.state_dict(),
             'disA_opt': self.disA_opt.state_dict(),
             'disB_opt': self.disB_opt.state_dict(),
             'disA_attr_opt': self.disA_attr_opt.state_dict(),
             'disB_attr_opt': self.disB_attr_opt.state_dict(),
             'subspace_opt': self.subspace_opt.state_dict(),
             'enc_c_opt': self.enc_c_opt.state_dict(),
             'enc_a_opt': self.enc_a_opt.state_dict(),
             'gen_opt': self.gen_opt.state_dict(),
             'gen_attr_opt': self.gen_attr_opt.state_dict(),
             'ep': ep,
             'total_it': total_it
              }
    torch.save(state, filename)
    return

  def assemble_outputs(self):
    images_a = self.normalize_image(self.real_A_encoded).detach()
    images_b = self.normalize_image(self.real_B_encoded).detach()
    images_a1 = self.normalize_image(self.fake_A_encoded).detach()
    """
    images_a2 = self.normalize_image(self.fake_A_random).detach()
    images_a3 = self.normalize_image(self.fake_A_recon).detach()
    """
    images_a4 = self.normalize_image(self.fake_AA_encoded).detach()
    images_b1 = self.normalize_image(self.fake_B_encoded).detach()
    """
    images_b2 = self.normalize_image(self.fake_B_random).detach()
    images_b3 = self.normalize_image(self.fake_B_recon).detach()
    """
    images_b4 = self.normalize_image(self.fake_BB_encoded).detach()
    #他这个就是两个一组，但是只有第一组是真实的。这个要改！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
    row1 = torch.cat((images_a, images_b1, images_a4),3)
    row2 = torch.cat((images_b, images_a1, images_b4),3)
    return torch.cat((row1,row2),2)

  def normalize_image(self, x):
    return x[:,0:3,:,:]
