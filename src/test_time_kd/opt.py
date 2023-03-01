from datasets import *
from train import *
from torchvision import datasets, transforms, models
from matplotlib.pyplot import figure
import cvxpy as cp
import time

torch.manual_seed(int(time.time()))

n = 20
corruptions = ['gaussian_noise','impulse_noise','shot_noise','defocus_blur','glass_blur','motion_blur','zoom_blur','snow','frost','fog','brightness','contrast','elastic_transform','jpeg_compression','pixelate']

clean_confidences = torch.load("accs/clean_confs224.pt")
clean_results = torch.load("accs/clean_results224.pt")
clean_confidences64 = torch.load("accs/clean_confs64.pt")
clean_results64 = torch.load("accs/clean_results64.pt")

# corr_confidences = torch.load("accs/corr_confs224.pt")
# corr_results = torch.load("accs/corr_results224.pt")
# corr_confidences64 = torch.load("accs/corr_confs64.pt")
# corr_results64 = torch.load("accs/corr_results64.pt")
rand_order = torch.randperm(50000)[:n]
# print(rand_order)
# print(clean_results64[rand_order])
# print(sum(clean_results64[rand_order].numpy())/n)
# print(sum(clean_results[rand_order].numpy())/n)
# exit()
s_cost = 1
t_cost = 10
ops = np.linspace(s_cost,t_cost,21)
accs = []
effs = []
for o in tqdm(ops):
    x_sr = clean_results64[rand_order].numpy()
    x_br = clean_results[rand_order].numpy()

    x_cs = cp.Variable(n, boolean = True)
    x_cb = cp.Variable(n, boolean = True)
    ones_tri = np.triu((np.ones((n,n))))
    ones_vec = np.ones(n)
    budget = np.arange(1,n+1)*o

    # constraints = []
    # constraints = [x_cs + x_cb == ones_vec, s_cost*x_cs.T@ones_tri + t_cost*x_cb.T@ones_tri <= budget]
    constraints = [x_cs + x_cb == ones_vec, s_cost*x_cs.T@ones_vec + t_cost*x_cb.T@ones_vec <= o*n]
    acc = x_sr.T@x_cs + x_br.T@x_cb

    problem = cp.Problem(cp.Maximize(acc), constraints)

    problem.solve(verbose=False)
    accs.append(problem.value)
    effs.append(sum(x_cs.value)/n)
    #print("status:", problem.status)
    #print(effs)
    print("\np_s:",(s_cost+t_cost-o)/t_cost)
    print("sr:",sum(x_sr)/len(x_sr),x_sr)
    print("cs:",x_cs.value)
    print("tr:",sum(x_br)/len(x_br),x_br)
    print("ct:",x_cb.value)





accs2 = [acc/n for acc in accs]
# corr_accs2 = [acc/750000 for acc in corr_accs]
fig,ax1 = plt.subplots()
min_ops = 22*n
max_ops = (234)*n
ops = np.linspace(min_ops,max_ops,21)
ax1.scatter(ops/n,accs2,label="best accuracy for # of MACs (original dataset)")
# corr_min_ops = 22*750000
# corr_max_ops = (234)*750000
# corr_ops = np.linspace(corr_min_ops,corr_max_ops,21)
# ax1.scatter(corr_ops/750000,corr_accs2,label="best accuracy for # of MACs (shifted dataset)")
ax1.set_xlabel("available MACs (M) averaged over:"+str(n))
ax1.set_ylabel("Accuracy")
ax1.legend()
effs2 = [round(eff,2) for eff in effs]
# corr_effs2 = [round(corr_eff,2) for corr_eff in corr_effs]
for i, txt in enumerate(effs2):
    ax1.annotate(txt, ((ops)[i]/n, accs2[i]),rotation=25, size=7.5)
    # ax1.annotate(corr_effs2[i], ((corr_ops)[i]/750000, corr_accs2[i]),rotation=25, size=7.5)
plt.show()

print(np.sum(x_sr))
print(np.sum(x_br))