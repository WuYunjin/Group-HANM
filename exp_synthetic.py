import numpy as np
import torch
from model.clstm import cLSTM,  train_model_adam ,train_model_gista
from data.synthetic import simulate_lorenz_96
import matplotlib.pyplot as plt
from sklearn import preprocessing # Normalization
from sklearn.metrics import confusion_matrix
import os


def multi_exp(A_p,T,seed,lam_nonsmooth=0.3):

    print("Now train under setting: A_p={}, T={}, seed={}".format(A_p,T,seed))
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

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    # For GPU acceleration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Simulate data
    A_p = torch.tensor(A_p,dtype=torch.int, device=device) #torch.tensor([3,5,4,3,2,1,3,2],dtype=torch.int, device=device)

    X_np, GC = simulate_lorenz_96( A_p,T=T,seed=seed)

    # Normalize data
    X_np = preprocessing.scale(X_np)

    CovX = torch.tensor(np.cov(X_np.T),dtype=torch.double,device=device)
    X = torch.tensor(X_np[np.newaxis], dtype=torch.float32, device=device)

    # # Plot data
    # fig, axarr = plt.subplots(1, 2, figsize=(16, 5))
    # axarr[0].plot(X_np)
    # axarr[1].plot(X_np[:100])
    # plt.legend([i for i in range(A_p.sum().item())])
    # plt.savefig("results/Data_X_{}.png".format(A_p.cpu().data.numpy()))

    m = len(A_p) # Number of series

    # Pretrain (no regularization)

    # Set up model
    clstm = cLSTM(m, hidden=10).cuda(device=device) if torch.cuda.is_available() else cLSTM(m, hidden=10)
    check_every = 100
    train_loss_list,Y,A = train_model_adam(clstm, X, CovX, A_p, lr=1e-5, check_every=check_every)


    # # Plot data Y
    # Y_np = Y.view(-1,m).cpu().data.numpy()
    # fig, axarr = plt.subplots(1, 2, figsize=(16, 5))
    # axarr[0].plot(Y_np)
    # axarr[1].plot(Y_np[:100])
    # plt.legend([i for i in range(A_p.sum().item())])
    # plt.savefig("results/Data_Y_{}.png".format(A_p.cpu().data.numpy()))


    # Train Y with GISTA using cLSTM
    check_every = 1
    train_loss_list, train_mse_list = train_model_gista(
        clstm, Y, lam=lam_nonsmooth, lam_ridge=1e-4, lr=0.005, max_iter=1, check_every=check_every)#, truncation=5)


    # Loss function plot
    fig, axarr = plt.subplots(1, 2, figsize=(16, 5))

    axarr[0].plot(check_every * np.arange(len(train_loss_list)), train_loss_list)
    axarr[0].set_title('Train loss')

    axarr[1].plot(check_every * np.arange(len(train_mse_list)), train_mse_list)
    axarr[1].set_title('Train MSE')

    plt.savefig("results/Ours/Loss_{}_{}_{}.png".format(A_p.cpu().data.numpy(),T,seed))


    # Verify learned Granger causality
    GC_est = clstm.GC().cpu().data.numpy()

    print('True variable usage = %.2f%%' % (100 * np.mean(GC)))
    print('Estimated variable usage = %.2f%%' % (100 * np.mean(GC_est)))
    # print('Accuracy = %.2f%%' % (100 * np.mean(GC == GC_est)))

    # # Make figures
    # fig, axarr = plt.subplots(1, 2, figsize=(10, 5))
    # axarr[0].imshow(GC, cmap='Blues')
    # axarr[0].set_title('GC actual')
    # axarr[0].set_ylabel('Affected series')
    # axarr[0].set_xlabel('Causal series')
    # axarr[0].set_xticks([])
    # axarr[0].set_yticks([])

    # axarr[1].imshow(GC_est, cmap='Blues', vmin=0, vmax=1, extent=(0, m, m, 0))
    # axarr[1].set_title('GC estimated, {}'.format(A_p.cpu().data.numpy()))
    # axarr[1].set_ylabel('Affected series')
    # axarr[1].set_xlabel('Causal series')
    # axarr[1].set_xticks([])
    # axarr[1].set_yticks([])

    # # Mark disagreements
    # for i in range(m):
    #     for j in range(m):
    #         if GC[i, j] != GC_est[i, j]:
    #             rect = plt.Rectangle((j, i-0.05), 1, 1, facecolor='none', edgecolor='red', linewidth=1)
    #             axarr[1].add_patch(rect)

    confusion = confusion_matrix(GC.reshape(1,-1)[0], GC_est.reshape(1,-1)[0])

    tp = confusion[1,1]
    fp = confusion[0,1]
    fn = confusion[1,0]
    f1 = 2*tp/(2*tp+fp+fn)
    # plt.savefig("results/Ours/Comparision_{}_{}_{}_{}%.png".format(A_p.cpu().data.numpy(),T,seed, f1) )
    context = np.vstack((GC,GC_est))
    np.savetxt("results/Ours/Comparision_{}_{}_{}_{}%.txt".format(A_p.cpu().data.numpy(), T, seed,100*f1), context) 



if __name__ == "__main__":

    for s in range(0,10):
        multi_exp(A_p = [3,4,5,3,4,5,3,5],  T = 1500, seed=s,lam_nonsmooth=0.4) 
        multi_exp(A_p = [3,4,5,3,4,5,3,5,4,4],  T = 1500, seed=s,lam_nonsmooth=0.3)
        multi_exp(A_p = [3,4,5,3,4,5,3,4,5,4,4,4],  T = 1500, seed=s,lam_nonsmooth=0.3)
        multi_exp(A_p = [3,4,5,3,4,5,3,4,5,3,4,5,4,4],  T = 1500, seed=s,lam_nonsmooth=0.3)
        multi_exp(A_p = [3,4,5,3,4,5,3,4,5,3,4,5,3,4,5,4],   T = 1500, seed=s,lam_nonsmooth=0.3)

        multi_exp(A_p = [1,2,3,1,2,3,1,2,3,2,2,2],  T = 1500, seed=s,lam_nonsmooth=0.2)
        multi_exp(A_p = [2,3,4,2,3,4,2,3,4,3,3,3],  T = 1500, seed=s,lam_nonsmooth=0.3)
        multi_exp(A_p = [4,5,6,4,5,6,4,5,6,5,5,5],  T = 1500, seed=s,lam_nonsmooth=0.4)
        multi_exp(A_p = [5,6,7,5,6,7,5,6,7,6,6,6],  T = 1500, seed=s,lam_nonsmooth=0.5)

        multi_exp(A_p = [3,4,5,3,4,5,3,4,5,4,4,4],  T = 500,  seed=s,lam_nonsmooth=0.3)
        multi_exp(A_p = [3,4,5,3,4,5,3,4,5,4,4,4],  T = 1000, seed=s,lam_nonsmooth=0.3)
        multi_exp(A_p = [3,4,5,3,4,5,3,4,5,4,4,4],  T = 2000, seed=s,lam_nonsmooth=0.3)
        multi_exp(A_p = [3,4,5,3,4,5,3,4,5,4,4,4],  T = 2500, seed=s,lam_nonsmooth=0.3)
        
        # multi_exp(A_p = [1,2,1,2,3,1,2,1,2,1,2,2], T = 1500, seed=s,lam_nonsmooth=0.2)
        # multi_exp(A_p = [1,3,2,4,3,3,2,2,3,1,4,2], T = 1500, seed=s,lam_nonsmooth=0.4)
        # multi_exp(A_p = [2,4,5,4,4,3,3,2,3,4,4,2], T = 1500, seed=s,lam_nonsmooth=0.5) # 0.55(87.3), 0.5(87.6 ), 0.45(87.6 ), 0.4(87 ), 0.35(85) ,0.3( 86%)
        # multi_exp(A_p = [4,4,5,5,4,6,3,5,3,4,4,3], T = 1500, seed=s,lam_nonsmooth=0.4)
        # multi_exp(A_p = [5,6,5,6,4,6,5,5,6,5,4,3], T = 1500, seed=s,lam_nonsmooth=0.5)
        # multi_exp(A_p = [3,3,3,3,4,4], T = 1500, seed=s,lam_nonsmooth=0.25) # 0.2(87),0.25(88.8),0.3(85.7),0.4(84)
        # multi_exp(A_p = [2,4,5,4,4,3,3,2,3], T = 1500, seed=s,lam_nonsmooth=0.3)
        # multi_exp(A_p = [2,4,5,4,4,3,3,2,3,5,3,2,3,4,3], T = 1500, seed=s,lam_nonsmooth=0.3)
        # multi_exp(A_p = [2,4,5,4,4,3,3,2,3,5,3,2,3,4,3,2,3,5], T = 1500, seed=s,lam_nonsmooth=0.3)
        # multi_exp(A_p = [2,4,5,4,4,3,3,2,3,4,4,2], T = 500, seed=s,lam_nonsmooth=0.3)
        # multi_exp(A_p = [2,4,5,4,4,3,3,2,3,4,4,2], T = 1000, seed=s,lam_nonsmooth=0.3)
        # multi_exp(A_p = [2,4,5,4,4,3,3,2,3,4,4,2], T = 2000, seed=s,lam_nonsmooth=0.3)
        # multi_exp(A_p = [2,4,5,4,4,3,3,2,3,4,4,2], T = 2500, seed=s,lam_nonsmooth=0.3)
