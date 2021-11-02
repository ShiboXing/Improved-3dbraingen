import os
import torch
import pandas as pd
import nibabel as nib
from ipdb import set_trace
from time import time
from skimage.transform import resize
from nilearn import plotting
import nibabel as nib
import numpy as np
import cupy as cp
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable
from .tsne_torch import tsne
import pytorch_ssim

sp_size = 64
arr1 = [4,6,8,10,12,14,16,18,20,22,24,26,28,30,32]
arr2 = [34,36,38,40,42,44,46,48,50,52,54,56,58,60]


def inf_train_gen(data_loader):
    while True:
        for _,images in enumerate(data_loader):
            yield images
            
# calculate frechet inception distance  https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/
def calculate_fid(act1, act2, cuda_ind=0):
    with cp.cuda.Device(cuda_ind):
        act1, act2 = cp.array(act1), cp.array(act2)
        # calculate mean and covariance statistics
        mu1, sigma1 = act1.mean(axis=0), cp.cov(act1, rowvar=False)
        mu2, sigma2 = act2.mean(axis=0), cp.cov(act2, rowvar=False)
        # calculate sum squared difference between means
        ssdiff = cp.sum((mu1 - mu2)**2.0)
        # calculate sqrt of product between cov
        covmean = cp.array(sqrtm(sigma1.dot(sigma2).get()))
        # check and correct imaginary numbers from sqrt
        if cp.iscomplexobj(covmean):
            covmean = covmean.real
        # calculate score
        fid = ssdiff + cp.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid

def load_checkpoint(G, D, E, CD, fname, path='checkpoint'):
    # load the highest savepoints of all models
    iteration = 0
    checkpoint_pth = f'./{path}/'
    if os.path.exists(checkpoint_pth):
        files = set(os.listdir(checkpoint_pth))
        highest_pth = 0 
        for s in files:
            if 'iter' in s:
                curr_num = int(s.split('iter')[1].split('.')[0])
                highest_pth = max(highest_pth, curr_num)
        if files:
            if CD:
                CD.load_state_dict(torch.load(f'./{path}/CD{fname}{highest_pth}.pth'))
            if D:
                D.load_state_dict(torch.load(f'./{path}/D{fname}{highest_pth}.pth'))
            E.load_state_dict(torch.load(f'./{path}/E{fname}{highest_pth}.pth'))
            G.load_state_dict(torch.load(f'./{path}/G{fname}{highest_pth}.pth'))
            iteration = highest_pth
    else:
        os.mkdir(checkpoint_pth)
    
    return iteration
            
def w_egen(G, D, fname):
    # load the highest savepoints of all models
    iteration = 0
    checkpoint_pth = './vae_checkpoint/'
    if os.path.exists(checkpoint_pth):
        files = set(os.listdir(checkpoint_pth))
        highest_pth = 0 
        for s in files:
            if 'iter' in s:
                curr_num = int(s.split('iter')[1].split('.')[0])
                highest_pth = max(highest_pth, curr_num)
        if files:
            D.load_state_dict(torch.load(f'./{checkpoint_pth}/D{fname}{highest_pth}.pth'))
            G.load_state_dict(torch.load(f'./{checkpoint_pth}/G{fname}{highest_pth}.pth'))
            iteration = highest_pth
    else:
        os.mkdir(checkpoint_pth)
    
    return iteration

def vae_load_checkpoint(G, D, E, checkpoint_name='vae_checkpoint'):
    # load the highest savepoints of all models
    iteration = 0
    checkpoint_pth = f'./{checkpoint_name}/'
    if os.path.exists(checkpoint_pth):
        files = set(os.listdir(checkpoint_pth))
        highest_pth = 0 
        for s in files:
            if 'ep' in s:
                curr_num = int(s.split('ep_')[1].split('.')[0])
                highest_pth = max(highest_pth, curr_num)
        if files:
            D.load_state_dict(torch.load(f'./{checkpoint_name}/D_VG_ep_{highest_pth}.pth'))
            E.load_state_dict(torch.load(f'./{checkpoint_name}/E_VG_ep_{highest_pth}.pth'))
            G.load_state_dict(torch.load(f'./{checkpoint_name}/G_VG_ep_{highest_pth}.pth'))
            iteration = highest_pth + 1
    else:
        os.mkdir(checkpoint_pth)
    
    return iteration

def load_loss(path='checkpoint'):
    loss_pth = f'./{path}/loss.csv'
    if os.path.exists(loss_pth):
        return pd.read_csv(loss_pth)
    else:
        return pd.DataFrame()

def load_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        return pd.DataFrame()
    
def add_loss(df, loss_dict):
    return df.append(pd.DataFrame(loss_dict))

def write_loss(df, path='checkpoint'):
    df.to_csv(f'./{path}/loss.csv', index=False)

def sample_from_model(model, gen_load, gpu_ind, latent_size, batch_size, z_r, is_E, is_real=False):
    sample_df = pd.DataFrame()
    for s in range(512):
        noise = (torch.randn((batch_size, latent_size)) * z_r).cuda(gpu_ind) # get z_r * scale
        real = gen_load.__next__().cuda(gpu_ind) # get real_image sample
        if is_E: # sample for Z space
            if is_real: sample = torch.randn((batch_size, latent_size)).cuda(gpu_ind)
            else: 
                sample = model(real if real.shape[0] == 2 else gen_load.__next__().cuda(gpu_ind)) # guard for vae_gan (needs 2 batches)
                sample = sample if type(sample) != tuple else sample[2][0] # guard for vae_gan (3rd tensor, 2nd row vector)
        else: # sample for X space
            if is_real: sample = real
            else: sample = model(noise)
        sample_df = sample_df.append(pd.DataFrame(sample.view(1, -1).cpu().detach().numpy()))
    return sample_df

def pca_tsne(sample_df, is_tsne, is_pca, color, model_name=None):
    pca = PCA(n_components=2)
    if is_tsne and is_pca: # tsne -> 50 -> pca ->2
        pca = PCA(n_components=50)
        samples = pca.fit_transform(sample_df)
        samples = TSNE(n_components=2, method='exact').fit_transform(sample_df.values)
    elif is_tsne: # tsne only
        samples = TSNE(n_components=2).fit_transform(sample_df.values)
    else: # pca only
        samples = StandardScaler().fit_transform(sample_df)
        samples = pca.fit_transform(sample_df)
    pd.DataFrame(samples).to_csv(f'./{model_name}.csv', index=False)
    plt.scatter(samples[:, 0], samples[:, 1], color=color, linewidths=0.4)
        
def viz_pca_tsne(models:list, trainset, latent_size=1000, model_names=None, is_tsne=False, is_pca=True, is_cd=False, index=0, z_r=1, gpu_ind=0):
    if is_tsne and is_pca: title = 'TSNE-PCA'
    elif is_pca: title = 'PCA'
    else: title = 'TSNE'
    colors=['blue', 'red', 'pink', 'green', 'orange']
    model_names = model_names if model_names else [None for i in range(len(models))]
    plt.figure()
    for model, c, name in zip(models, colors, model_names):
        #init 
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=2 if model.__class__.__name__ == 'Encoder' else 1, shuffle=True, num_workers=4)
        gen_load = inf_train_gen(train_loader)
        try:
            if hasattr(model, 'set_gpu'): model.set_gpu(gpu_ind)
        except AttributeError: pass
        real_df = sample_from_model(model, gen_load, gpu_ind, latent_size, 1, z_r, is_cd, True)
        sample_df = sample_from_model(model, gen_load, gpu_ind, latent_size, 1, z_r, is_cd, False)

#         plot the variances of latent vectors
        if is_cd:
            sample_vars, real_vars = sample_df.transpose().var(axis=1).to_frame(), real_df.transpose().var(axis=1).to_frame()
            plt.figure()
            val_df = pd.concat([real_vars, sample_vars], axis=1)
            plt.show()

        blue_mean, yellow_mean = sample_df.mean(1).mean(0), real_df.mean(1).mean(0)
        blue_var, yellow_var = sample_df.var(1).mean(0), real_df.var(1).mean(0)
        print(f'index: {index}, sample_mean (blue): {blue_mean} sample_var:\
        {blue_var}, real_mean (yellow): {yellow_mean} real_var: {yellow_var}')
        r, = plt.plot(range(len(real_df.var())), real_df.var(), label='real prior')
        s, = plt.plot(range(len(sample_df.var())), sample_df.var(), label='encoder')
        plt.legend(handles=[r, s])
        plt.show()
        if is_cd: plt.title(f'latent vector {title} (blue is z_e)')
        else: plt.title(f'X {title} (blue is X_rand)')
        # execute PCA
        name_str = f'{name}_E_PCA' if is_cd else f'{name}_G_PCA'
        pca_tsne(sample_df, is_tsne, is_pca, c, model_name=name_str)
    # execute PCA
    name_str = f'real_E_PCA' if is_cd else f'real_G_PCA'
    pca_tsne(real_df, is_tsne, is_pca, colors[-1], model_name=name_str)
    plt.show()

def fetch_models(checkpoints, is_E=False, iteration=100000, latent_dim=1000, gpu=0):
    if is_E:
        from Model_VAEGAN import Encoder as VE
        from Model_alphaGAN import Discriminator as AE
    else:
        from Model_VAEGAN import Generator as VG
        from Model_alphaGAN import Generator as AG
        
    model_list = []
    for m in checkpoints:
        if 'vae' in m:
            if is_E:
                model = VE(out_class = latent_dim).cuda(gpu)
                model.load_state_dict(torch.load(f'./{m}/E_VG_ep_{int(iteration / 1000)}.pth')) 
            else:
                model = VG(noise=latent_dim).cuda(gpu)
                model.load_state_dict(torch.load(f'./{m}/G_VG_ep_{int(iteration / 1000)}.pth')) 
        else:
            if is_E:
                model = AE(out_class = latent_dim, is_dis=False).cuda(gpu)
                model.load_state_dict(torch.load(f'./{m}/E_iter{iteration}.pth')) 
            else:
                model = AG(noise = latent_dim).cuda(gpu)
                model.load_state_dict(torch.load(f'./{m}/G_iter{iteration}.pth')) 
        model_list.append(model)
    return model_list

def latent_mmd(model, gen_load, num=1, batch_size=4, gpu_ind=0):
    total, i = 0, 0
#     print('calculating mmd of latent vectors ...')
    while i < num:
        img = gen_load.__next__()
        if img.shape[0] != 4:
            continue
        z_e = model(img.cuda(gpu_ind)).view((batch_size, 1000))
        z_r = torch.randn((batch_size, 1000)).cuda(gpu_ind)
        xx, yy, zz = torch.mm(z_e,z_e.t()), torch.mm(z_r,z_r.t()), torch.mm(z_e,z_r.t())
        beta = (1./(batch_size*batch_size))
        gamma = (2./(batch_size*batch_size)) 
        score = beta * (torch.sum(xx)+torch.sum(yy)) - gamma * torch.sum(zz)
        total += score
        i += 1
    mmd = total/num
    
    return mmd
        
def viz_all_imgs(path, count):
    output_dir = './imgs_visualization'
    if not os.path.exists(output_dir):
        os.mkdir('./imgs_visualization')
    for f in os.listdir(path):
        if f.endswith('mgz'):
            print(f'{count[0]}: {path}/{f}')
            
            # further reprocessing
            img = nib.load(os.path.join(path,'brainmask.mgz'))
            img = np.swapaxes(img.get_data(),1,2)
            img = np.flip(img,1)
            img = np.flip(img,2)
            img = resize(img, (sp_size,sp_size,sp_size), mode='constant')
            img = torch.from_numpy(img).float().view(1,sp_size,sp_size,sp_size)
            img = img*2-1
            
            featmask = np.squeeze((0.5*img+0.5).data.cpu().numpy())
            featmask = nib.Nifti1Image(featmask,affine = np.eye(4))
            disp = plotting.plot_img(featmask,cut_coords=arr1,draw_cross=False,annotate=False,black_bg=True,display_mode='x')
            plotting.show()
            disp=plotting.plot_img(featmask,cut_coords=arr2,draw_cross=False,annotate=False,black_bg=True,display_mode='x')
            plotting.show()
            subject = path.split('/')[-2]
            plotting.plot_img(featmask,title=f'subject: {subject}_index: {count}')
            plotting.plot_img(featmask,title=f'subject: {subject}_index: {count}',\
                              output_file=f'{output_dir}/subject_{subject}_index_{count[0]}.png')
            plotting.show()
            count[0] += 1
        elif os.path.isdir(f'{path}/{f}'):
            viz_all_imgs(f'{path}/{f}', count)

def read_mmd(path='test_data', name='mmd.csv'):
    # return the last index
    if not os.path.exists(f'./{path}/{name}'):
        return 0
    df = load_csv(f'./{path}/{name}')
    return int(df.iloc[-1]['index'])

def read_ssim(path='test_data'):
    df = load_csv(f'./{path}/ssim.csv')
    last_ssim = int(df.iloc[-1]['index']) if len(df) else 0 
    return last_ssim

def calc_ssim(G, index, path, no_write=True, gpu=0, z_r=1):        
    sum_ssim = 0
    for i in range(1000):
        noise = (torch.randn((2, 1000)) * z_r).cuda(gpu)
        images = G(noise)
        img1 = images[0]
        img2 = images[1]

        msssim = pytorch_ssim.msssim_3d(img1,img2)
        sum_ssim = sum_ssim+msssim
#         if i % 100 == 0:
#             print(sum_ssim/1000)
    ssim = sum_ssim/1000
    print(f'{index} final ssim: {ssim}')
    df = load_csv(f'./{path}/ssim.csv')
    df = df.append(pd.DataFrame({
        'index': [index],
        'ssim': [float(ssim)]
    }))
    if not no_write:
        df.to_csv(f'./{path}/ssim.csv', index=False)
        
    return ssim
    
def calc_mmd(train_loader, model, iteration, count=1, latent_size=1000, no_write=False, mode='rbf', gpu_ind=1, path='test_data'):
    
    for p in model.parameters(): p.requires_grad = False
    if not os.path.exists(f'./{path}'): os.mkdir(f'./{path}')
    df = load_csv(f'./{path}/{mode}_mmd.csv')
    total_mean = []
    
    for s in range(0, count):
        distmean = 0
        start_time = time()
        for i,(y) in enumerate(train_loader):
            B = y.size(0)
            noise = torch.randn((B, latent_size)).cuda(gpu_ind)
            if model.__class__.__name__ == 'Generator': x, y = model(noise).view(B, -1), y.cuda(gpu_ind).view(B, -1)
            else: 
                tmp = model(y.cuda(gpu_ind))
                x = tmp[2].view(B, -1) if type(tmp) == tuple else tmp.view(B, -1) # guard for VAE-GAN
                y = torch.randn((B, latent_size)).cuda(gpu_ind)
            xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
            rx = (xx.diag().unsqueeze(0).expand_as(xx))
            ry = (yy.diag().unsqueeze(0).expand_as(yy))
            dxx = rx.t() + rx - 2. * xx 
            dyy = ry.t() + ry - 2. * yy 
            dxy = rx.t() + ry - 2. * zz 
            XX, YY, XY = torch.zeros(xx.shape).cuda(gpu_ind), \
            torch.zeros(xx.shape).cuda(gpu_ind), \
            torch.zeros(xx.shape).cuda(gpu_ind)
            
            if mode == 'rbf':
                bandwidth_range = [a * 1e+3 for a in [1, 1.5, 2, 5]]
                for a in bandwidth_range:
                    XX += torch.exp(-0.5*dxx/a)
                    YY += torch.exp(-0.5*dyy/a)
                    XY += torch.exp(-0.5*dxy/a)
            elif mode == 'multiscale':
                bandwidth_range = [a * 1e+3 for a in [0.2, 0.5, 0.9, 1.3]]
                for a in bandwidth_range:
                    XX += a**2 * (a**2 + dxx)**-1
                    YY += a**2 * (a**2 + dyy)**-1
                    XY += a**2 * (a**2 + dxy)**-1 
            elif mode == 'linear':
                beta = (1./(B*B))
                gamma = (2./(B*B)) 
                dist = beta * (torch.sum(xx)+torch.sum(yy)) - gamma * torch.sum(zz)
            else:
                print('unknown kernel')
                break
            
            distmean += dist if mode == 'linear' else torch.mean(XX + YY - 2. * XY) 
            
        mean = (distmean/(i+1))
        total_mean.append(mean.item())
        print(f'\niteration: {iteration}, count: {s}, Mean: {mean}, cost {time() - start_time} seconds')
    
    total_mean = np.array(total_mean)
    final_mean = np.mean(total_mean)
    final_std = np.std(total_mean)
    # write scores to csv
    df = df.append(pd.DataFrame({
        'index': [iteration],
        'mmd_score': [final_mean],
        'std': [final_std]
    }))
    if not no_write:
        df.to_csv(f'./{path}/{mode}_mmd.csv', index=False)
    print('Total_mean:'+str(final_mean)+' STD:'+str(final_std))

def eps_norm(x):
    _eps = 1e-15
    x = x.view(len(x), -1)
    return (x*x+_eps).sum(-1).sqrt()

def bi_penalty(x):
    return (x-1)**2

def calc_gradient_penalty(model, x, x_gen, w=10, cuda_ind=0):
    """WGAN-GP gradient penalty"""
    if not x.size()==x_gen.size():
        return 0
    alpha_size = tuple((len(x), *(1,)*(x.dim()-1)))
    alpha_t = torch.cuda.FloatTensor if x.is_cuda else torch.Tensor
    alpha = alpha_t(*alpha_size).uniform_().cuda(cuda_ind)
    #x_hat = x.data*alpha + x_gen.data*(1-alpha)
    x_hat = x*alpha + x_gen*(1-alpha)
    x_hat = Variable(x_hat, requires_grad=True)

    
    grad_xhat = torch.autograd.grad(model(x_hat).sum(), x_hat, create_graph=True, only_inputs=True)[0]

    penalty = w*bi_penalty(eps_norm(grad_xhat)).mean()
    return penalty