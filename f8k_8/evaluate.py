###################双向检索评估效果######################################

def i2t(image, text, npts=None,return_ranks=False):
    """
    image->Text 
    image: (5N, K) matrix of image
    text: (5N, K) matrix of text
    """
		
    if npts is None:
        npts = image.shape[0] / 5
    index_list = []

    ranks = numpy.zeros(npts)
    top1 = numpy.zeros(npts)
    for index in range(npts):

        # Get query image
        im = image[5 * index].reshape(1, image.shape[1])

        # Compute scores
 
        d = numpy.dot(im, text.T).flatten()  #text转置
        inds = numpy.argsort(d)[::-1]#返回数组从小到大的索引值
        index_list.append(inds[0])

        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = numpy.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 =  len(numpy.where(ranks < 1)[0]) / len(ranks)  #R@1
    r5 =  len(numpy.where(ranks < 5)[0]) / len(ranks)  #R@5
    r10 =  len(numpy.where(ranks < 10)[0]) / len(ranks)  #R@10
    medr = numpy.floor(numpy.median(ranks)) + 1   #medr
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr), (ranks, top1)
    else:
        return (r1, r5, r10, medr)

    



def t2i(image, text, npts=None, return_ranks=False):
    """
    Text->image (Image Search)
    image: (5N, K) matrix of image
    text: (5N, K) matrix of texttions
    """
    if npts is None:
        npts = image.shape[0] / 5
    ims = numpy.array([image[i] for i in range(0, len(image), 5)])

    ranks = numpy.zeros(5 * npts)
    top1 = numpy.zeros(5 * npts)
    for index in range(npts):

        # Get query texttions
        queries = text[5 * index:5 * index + 5]

        # Compute scores
        
        d = numpy.dot(queries, ims.T)
        inds = numpy.zeros(d.shape)
        for i in range(len(inds)):
            inds[i] = numpy.argsort(d[i])[::-1]
            ranks[5 * index + i] = numpy.where(inds[i] == index)[0][0]
            top1[5 * index + i] = inds[i][0]

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)

	
	
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
     if return_ranks:
         return (r1, r5, r10, medr), (ranks, top1)
    else:
        return (r1, r5, r10, medr)


def evaluate(model_path, data_path=None, split='dev', fold5=False):
    """
    Evaluate a trained model on either dev or test. If `fold5=True`, 5 fold
    cross-validation is done (only for MSCOCO). Otherwise, the full data is
    used for evaluation.
    """
    # load model and options
    checkpoint = torch.load(model_path)
    opt = checkpoint['opt']
    if data_path is not None:
        opt.data_path = data_path

    # load vocabulary used by the model
    with open(os.path.join(opt.vocab_path,
                           '%s_vocab.pkl' % opt.data_name), 'rb') as f:
        vocab = pickle.load(f)
    opt.vocab_size = len(vocab)

    # construct model
    model = VSE(opt)

    # load model state
    model.load_state_dict(checkpoint['model'])

    print('Loading dataset')
    data_loader = get_test_loader(split, opt.data_name, vocab, opt.crop_size,
                                  opt.batch_size, opt.workers, opt)

    print('Computing results...')
    img_embs, text_embs = encode_data(model, data_loader)
    print('image: %d, text: %d' %
          (img_embs.shape[0] / 5, text_embs.shape[0]))

    # no cross-validation, full evaluation
    r, rt = i2t(img_embs, text_embs, return_ranks=True)  #image——>text
    ri, rti = t2i(img_embs, text_embs, return_ranks=True) #text———>image
    ar = (r[0] + r[1] + r[2]) / 3 
    ari = (ri[0] + ri[1] + ri[2]) / 3
    rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
    print("rsum: %.1f" % rsum)
    print("Average i2t Recall: %.1f" % ar)
    print("Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
    print("Average t2i Recall: %.1f" % ari)
    print("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)
    #else:
        # 5fold cross-validation, only for MSCOCO
        #results = []
        #for i in range(5):
        #    r, rt0 = i2t(img_embs[i * 5000:(i + 1) * 5000],
        #                 text_embs[i * 5000:(i + 1) *
        #                          5000], measure=opt.measure,
        #                 return_ranks=True)
        #    print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % r)
        #   ri, rti0 = t2i(img_embs[i * 5000:(i + 1) * 5000],
        #                   text_embs[i * 5000:(i + 1) *
        #                            5000], measure=opt.measure,
        #                   return_ranks=True)
        #    if i == 0:
        #        rt, rti = rt0, rti0
        #    print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % ri)
        #    ar = (r[0] + r[1] + r[2]) / 3
        #    ari = (ri[0] + ri[1] + ri[2]) / 3
        #    rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
        #    print("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))
        #    results += [list(r) + list(ri) + [ar, ari, rsum]]

    #    print("-----------------------------------")
    #    print("Mean metrics: ")
    #    mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
    #    print("rsum: %.1f" % (mean_metrics[10] * 6))
    #    print("Average i2t Recall: %.1f" % mean_metrics[11])
    #    print("Image to text: %.1f %.1f %.1f %.1f %.1f" %
    #          mean_metrics[:5])
    #    print("Average t2i Recall: %.1f" % mean_metrics[12])
    #    print("Text to image: %.1f %.1f %.1f %.1f %.1f" %
    #          mean_metrics[5:10])

    torch.save({'rt': rt, 'rti': rti}, 'ranks.pth.tar')


