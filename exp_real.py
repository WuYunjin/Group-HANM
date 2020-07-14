import numpy as np
import torch
from model.clstm import cLSTM,  train_model_adam ,train_model_gista
import matplotlib.pyplot as plt
from sklearn import preprocessing # Normalization


def exp(A_p,data,seed=0,lam_nonsmooth=0.3):

    #print enviorment information 
    print("torch.__version__:",torch.__version__)
    if torch.cuda.is_available():
        print("torch.version.cuda:",torch.version.cuda)
        print("torch.backends.cudnn.version():",torch.backends.cudnn.version())
        print("torch.cuda.get_device_name(0):",torch.cuda.get_device_name(0))


    #set seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # For GPU acceleration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert into tensor
    A_p = torch.tensor(A_p,dtype=torch.int, device=device) 

    # Normalize data
    X_np = preprocessing.scale(data)

    CovX = torch.tensor(np.cov(X_np.T),dtype=torch.double,device=device)
    X = torch.tensor(X_np[np.newaxis], dtype=torch.float32, device=device)

    m = len(A_p) # Number of series

    # Pretrain (no regularization)

    # Set up model
    clstm = cLSTM(m, hidden=10).cuda(device=device) if torch.cuda.is_available() else cLSTM(m, hidden=10)
    check_every = 100
    train_loss_list,Y,A = train_model_adam(clstm, X, CovX, A_p, lr=1e-5, check_every=check_every)

    # Plot data Y
    Y_np = Y.view(-1,m).cpu().data.numpy()

    # Normalize data
    Y_np = preprocessing.scale(Y_np)
    fig, axarr = plt.subplots(1, 2, figsize=(16, 5))
    axarr[0].plot(Y_np)
    axarr[1].plot(Y_np[:100])
    plt.legend([i for i in range(A_p.sum().item())])
    plt.savefig("results/Data_Y_{}.png".format(A_p.cpu().data.numpy()))
    np.savetxt("data/CCA_timeseries_{}.txt".format(A_p.cpu().data.numpy()),Y_np) 

    
    Y = torch.tensor(Y_np[np.newaxis], dtype=torch.float32, device=device)


    # Train Y with GISTA using cLSTM
    check_every = 10
    train_loss_list, train_mse_list = train_model_gista(
        clstm, Y, lam=lam_nonsmooth, lam_ridge=1e-4, lr=0.005, max_iter=5000, check_every=check_every)#, truncation=5)

    # Loss function plot
    fig, axarr = plt.subplots(1, 2, figsize=(16, 5))

    axarr[0].plot(check_every * np.arange(len(train_loss_list)), train_loss_list)
    axarr[0].set_title('Train loss')

    axarr[1].plot(check_every * np.arange(len(train_mse_list)), train_mse_list)
    axarr[1].set_title('Train MSE')

    plt.savefig("results/Loss_{}.png".format(A_p.cpu().data.numpy()))


    # Verify learned Granger causality
    GC_est = clstm.GC().cpu().data.numpy()

    print('Estimated variable usage = %.2f%%' % (100 * np.mean(GC_est)))

    # Make figures
    roi_labels = ['PCC','LACC','LMTG','LAG','RACC','RMTG','RAG'] 
    fig, axarr = plt.subplots(1,1, figsize=(7, 7))
    axarr.imshow(GC_est, cmap='Blues', vmin=0, vmax=1, extent=(0, len(roi_labels), len(roi_labels), 0))
    axarr.set_title('λ =  {}'.format(lam_nonsmooth))
    axarr.set_ylabel('Affected series')
    axarr.set_xlabel('Causal series')
    tick_marks = np.arange(len(roi_labels)) +0.5
    plt.xticks(tick_marks, roi_labels, rotation=45 )
    plt.yticks(tick_marks, roi_labels)

    plt.savefig("results/Estimated_matrix_λ={}_{}.png".format(lam_nonsmooth ,A_p.cpu().data.numpy() ))


if __name__ == "__main__":

    # rhyming_timeseries = np.loadtxt("data/rhyming_concat_001.txt",skiprows=1)
    # print(rhyming_timeseries.shape) #(1440, 9)
    # A_p = [4,4,1]
    # exp(A_p,rhyming_timeseries,lam_nonsmooth=0.8)

    # rhyming_timeseries = np.loadtxt("data/rhyming_concat_001.txt",skiprows=1)
    # print(rhyming_timeseries.shape) #(1440, 9)
    # #switch the order of variables
    # reorder_timeseries = np.vstack((rhyming_timeseries[:,0],rhyming_timeseries[:,3], # LOCC and LIPL
    #                                 rhyming_timeseries[:,1],rhyming_timeseries[:,2], # LACC and LIFG
    #                                 rhyming_timeseries[:,4],rhyming_timeseries[:,7], # ROCC and RIPL
    #                                 rhyming_timeseries[:,5],rhyming_timeseries[:,6], # RACC and RIFG
    #                                 rhyming_timeseries[:,8])).T
    # A_p = [2,2,2,2,1]
    # exp(A_p,reorder_timeseries,lam_nonsmooth=0.5)

    nki_timeseries = np.loadtxt("data/NKI_dataset.txt",skiprows=0)
    print(nki_timeseries.shape) #(895, 1258)
    A_p = [116, 167, 183, 171, 191, 188, 242]
    exp(A_p,nki_timeseries,lam_nonsmooth=0.2)

    # nki_timeseries = np.loadtxt("data/CCA_timeseries_[116 167 183 171 191 188 242].txt",skiprows=0)
    # print(nki_timeseries.shape) #(895, 1258)
    # A_p = [1, 1, 1, 1, 1, 1, 1]
    # for i in np.arange(3,7,1):
    #     exp(A_p,nki_timeseries,lam_nonsmooth=i)
