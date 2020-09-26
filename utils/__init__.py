import os
import torch
import pandas as pd
from ipdb import set_trace

trainset = ADNIdataset(augmentation=False)

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
    
def viz_pca(G):
    sample_df = pd.DataFrame()
    real_df = pd.DataFrame()

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=workers)
    gen_load = inf_train_gen(train_loader)

    for s in range(512):
        noise = torch.randn((1, 1000)).cuda()
        fake = np.squeeze(G(noise)).view(1, -1)
        sample_df = sample_df.append(pd.DataFrame(fake.cpu().detach().numpy()))

        real = np.squeeze(gen_load.__next__()).view(1, -1)
        real_df = real_df.append(pd.DataFrame(real.cpu().detach().numpy()))
        print(s, end=' ')

    # PCA of fake images
    pca = PCA(n_components=2)
    samples = StandardScaler().fit_transform(sample_df)
    PCs = pca.fit_transform(sample_df)
    plt.scatter(PCs[:, 0], PCs[:, 1])

    # PCA of real images
    reals = StandardScaler().fit_transform(real_df)
    real_PCs = pca.fit_transform(real_df)
    plt.scatter(real_PCs[:, 0], real_PCs[:, 1])
    plt.show()

def viz_mmd():
    pass

def viz_all_imgs(path):
    pass
    
