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
            D.load_state_dict(torch.load(f'./{path}/D{fname}{highest_pth}.pth'))
            CD.load_state_dict(torch.load(f'./{path}/CD{fname}{highest_pth}.pth'))
            E.load_state_dict(torch.load(f'./{path}/E{fname}{highest_pth}.pth'))
            G.load_state_dict(torch.load(f'./{path}/G{fname}{highest_pth}.pth'))
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

def load_loss(path='./checkpoint/loss.csv'):
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        return pd.DataFrame()

def load_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        return pd.DataFrame()
    
def add_loss(df, loss_dict):
    return df.append(pd.DataFrame(loss_dict))

def write_loss(df):
    df.to_csv('./checkpoint/loss.csv', index=False)
    
def viz_pca(model, trainset, batch_size=1, latent_size=1000, is_cd=False, viz_fake=True, viz_real=True, index=0, gpu_ind=0):
    sample_df = pd.DataFrame()
    real_df = pd.DataFrame()
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    gen_load = inf_train_gen(train_loader)
    
    try:
        if hasattr(model, 'set_gpu'):
            model.set_gpu(gpu_ind)
    except AttributeError:
        pass
    
    for s in range(512):
        noise = torch.randn((1, latent_size)).cuda(gpu_ind)
        if not is_cd:
            real = gen_load.__next__().cuda(gpu_ind) if batch_size == 1 else torch.unsqueeze(gen_load.__next__().cuda(gpu_ind)[0], 1)
            fake = np.squeeze(model(noise)).view(1, -1)
            sample_df = sample_df.append(pd.DataFrame(fake.cpu().detach().numpy()))
            real_df = real_df.append(pd.DataFrame(np.squeeze(real).view(1, -1).cpu().detach().numpy()))
            plt.title('Images PCA (blue is fake)')
            
        else:
            real = gen_load.__next__().cuda(gpu_ind)
            z_e = model(real)
            if type(z_e) != tuple:
                z_e = z_e.view(1, -1)
            else:
                z_e = z_e[2]
            sample_df = sample_df.append(pd.DataFrame(z_e.cpu().detach().numpy()))
            real_df = real_df.append(pd.DataFrame(noise.cpu().detach().numpy()))
    
#     if is_cd:
#         # calculate the variance of the vector's entries
#         sample_vars = sample_df.var(axis=0).transpose().to_frame()
#         real_vars = real_df.var(axis=0).transpose().to_frame()
#         plt.figure()
#         vars_plot = pd.concat([real_vars, sample_vars], axis=1)
#         vars_plot[[0,0]].plot()
#         plt.show()
        
        

    blue_mean = sample_df.mean(1).mean(0)
    yellow_mean = real_df.mean(1).mean(0)
    blue_var = sample_df.var(1).mean(0)
    yellow_var = real_df.var(1).mean(0)
    print(f'index: {index}, sample_mean: {blue_mean} sample_var: {blue_var}, yellow_mean: {yellow_mean} yellow_var: {yellow_var}')
    plt.title('latent vector PCA (blue is z_e)')
    if viz_fake:
        # PCA of fake images
        pca = PCA(n_components=2)
        samples = StandardScaler().fit_transform(sample_df)
        PCs = pca.fit_transform(sample_df)
        plt.scatter(PCs[:, 0], PCs[:, 1])
        
    if viz_real:
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


def calc_mmd(train_loader, G, df, iteration, count=1, no_write=False, gpu_ind=1, E=None, path='test_data_mmd', var=1):
    for p in G.parameters():
        p.requires_grad = False
    if not os.path.exists(f'./{path}'):
        os.mkdir(f'./{path}')
    df = load_csv(f'./{path}/mmd.csv')
    for s in range(0, count):
        distmean = 0
        for i,(y) in enumerate(train_loader):
            y = y.cuda(gpu_ind)
            if E:
                noise = E(y).cuda(gpu_ind) * var
            else:
                noise = torch.randn((y.size(0), 1000)).cuda(gpu_ind)
            x = G(noise)
            B = y.size(0)
            x = x.view(x.size(0), x.size(2) * x.size(3)*x.size(4))
            y = y.view(y.size(0), y.size(2) * y.size(3)*y.size(4))

            xx, yy, zz = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x,y.t())

            beta = (1./(B*B))
            gamma = (2./(B*B)) 

            Dist = beta * (torch.sum(xx)+torch.sum(yy)) - gamma * torch.sum(zz)
            distmean += Dist.item()
        mean = (distmean/(i+1))
        print(f'\nCount: {iteration} Mean: {mean}')
        
        if not no_write:
            # write scores to csv
            df = df.append(pd.DataFrame({
                'index': [iteration],
                'mmd_score': [mean]
            }))
            df.to_csv(f'./{path}/mmd.csv', index=False)
        torch.cuda.empty_cache()    
        
    
    meanarr = np.array(df.iloc[:]['mmd_score'].tolist())
#     print('Total_mean:'+str(np.mean(meanarr))+' STD:'+str(np.std(meanarr)))
    
