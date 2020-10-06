import os
import torch
import pandas as pd
import nibabel as nib
from ipdb import set_trace
from skimage.transform import resize
from nilearn import plotting
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



sp_size = 64
arr1 = [4,6,8,10,12,14,16,18,20,22,24,26,28,30,32]
arr2 = [34,36,38,40,42,44,46,48,50,52,54,56,58,60]

def load_checkpoint(G, D, E, CD, fname):
    # load the highest savepoints of all models
    iteration = 0
    checkpoint_pth = './checkpoint/'
    if os.path.exists(checkpoint_pth):
        files = set(os.listdir(checkpoint_pth))
        highest_pth = 0 
        for s in files:
            if 'iter' in s:
                curr_num = int(s.split('iter')[1].split('.')[0])
                highest_pth = max(highest_pth, curr_num)
        if files:
            D.load_state_dict(torch.load(f'./checkpoint/D{fname}{highest_pth}.pth'))
            CD.load_state_dict(torch.load(f'./checkpoint/CD{fname}{highest_pth}.pth'))
            E.load_state_dict(torch.load(f'./checkpoint/E{fname}{highest_pth}.pth'))
            G.load_state_dict(torch.load(f'./checkpoint/G{fname}{highest_pth}.pth'))
            iteration = highest_pth
    else:
        os.mkdir(checkpoint_pth)
    
    return iteration

def w_load_checkpoint(G, D, fname):
    # load the highest savepoints of all models
    iteration = 0
    checkpoint_pth = './checkpoint/'
    if os.path.exists(checkpoint_pth):
        files = set(os.listdir(checkpoint_pth))
        highest_pth = 0 
        for s in files:
            if 'iter' in s:
                curr_num = int(s.split('iter')[1].split('.')[0])
                highest_pth = max(highest_pth, curr_num)
        if files:
            D.load_state_dict(torch.load(f'./checkpoint/D{fname}{highest_pth}.pth'))
            G.load_state_dict(torch.load(f'./checkpoint/G{fname}{highest_pth}.pth'))
            iteration = highest_pth
    else:
        os.mkdir(checkpoint_pth)
    
    return iteration

def vae_load_checkpoint(G, D, E):
    # load the highest savepoints of all models
    iteration = 0
    checkpoint_pth = './checkpoint/'
    if os.path.exists(checkpoint_pth):
        files = set(os.listdir(checkpoint_pth))
        highest_pth = 0 
        for s in files:
            if 'ep' in s:
                curr_num = int(s.split('ep_')[1].split('_')[0])
                highest_pth = max(highest_pth, curr_num)
        if files:
            D.load_state_dict(torch.load(f'./checkpoint/D_VG_ep_{highest_pth}_247.pth'))
            E.load_state_dict(torch.load(f'./checkpoint/E_VG_ep_{highest_pth}_247.pth'))
            G.load_state_dict(torch.load(f'./checkpoint/G_VG_ep_{highest_pth}_247.pth'))
            iteration = highest_pth
    else:
        os.mkdir(checkpoint_pth)
    
    return iteration

def load_loss():
    if os.path.exists('./checkpoint/loss.csv'):
        return pd.read_csv('./checkpoint/loss.csv')
    else:
        return pd.DataFrame()
    
def add_loss(df, index, l):
    return df.append(pd.DataFrame({
        'index': [index],
        'loss:': [l]
    }))
    
def write_loss(df):
    df.to_csv('./checkpoint/loss.csv', index=False)
    
def viz_pca(G, trainset, latent_size=1000, is_fake=True, index=0):
    sample_df = pd.DataFrame()
    real_df = pd.DataFrame()

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=4)
    gen_load = inf_train_gen(train_loader)
    
    for s in range(512):
        noise = torch.randn((1, latent_size)).cuda()
        fake = np.squeeze(G(noise)).view(1, -1)
        sample_df = sample_df.append(pd.DataFrame(fake.cpu().detach().numpy()))

        real = np.squeeze(gen_load.__next__()).view(1, -1)
        real_df = real_df.append(pd.DataFrame(real.cpu().detach().numpy()))
    print(f'index: {index}')
    if is_fake:
        # PCA of fake images
        pca = PCA(n_components=2)
        samples = StandardScaler().fit_transform(sample_df)
        PCs = pca.fit_transform(sample_df)
        plt.scatter(PCs[:, 0], PCs[:, 1])
    else:
        # PCA of real images
        reals = StandardScaler().fit_transform(real_df)
        real_PCs = pca.fit_transform(real_df)
        plt.scatter(real_PCs[:, 0], real_PCs[:, 1])
    plt.show()
        
def inf_train_gen(data_loader):
    while True:
        for _,images in enumerate(data_loader):
            yield images

def viz_mmd():
    pass

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
            
    
