# ==================== import libraries ==================== #
import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, Button, RadioButtons



# ==================== set up global variables ==================== #
n_bins = 15
bin_boundaries = torch.linspace(0, 1, n_bins + 1)
bin_lowers = bin_boundaries[:-1]
bin_uppers = bin_boundaries[1:]
bin_centers = np.linspace(1/(2*n_bins),1-1/(2*n_bins),n_bins)

# expected height of each bin assuming uniform distribution, 
# e.g. for 10 bins bin 1 is [0.1,0.2] ~0.15
cal_bin_accs = bin_uppers-bin_uppers[0]/2 

# get density values of normal dist at points in x
def normal_density(mu,sigma,x):
    return 1/(np.sqrt(2*np.pi)*sigma)*np.exp(-np.square(x-mu)/(2*sigma**2))

# normalize to sum to one
def norm_sum(x):
    return np.array([x_i/sum(x) for x_i in x])

# align the bar graph labels
bar_ticks = np.arange(0,1+1*bin_uppers[0],1/n_bins)
bar_ticks = [str(round(bar,2)) for bar in bar_ticks]
bar_xpos = np.arange(0,n_bins+1)
bar_xtickpos = [bar_xpos[i]-0.5 for i in range(len(bar_xpos))]



# ==================== initialize figure layout ==================== #
plt.rc('axes', axisbelow=True)
fig, axs = plt.subplot_mosaic([['tradeoff_plot','stud_dist','stud_dist_mu','teach_dist','teach_dist_mu'],
                               ['tradeoff_plot','stud_dist','stud_dist_sigma','teach_dist','teach_dist_sigma'],
                               ['tradeoff_plot','stud_avg_acc','stud_avg_conf','teach_avg_acc','teach_avg_conf'],
                               ['tradeoff_plot','stud_rel','stud_rel_mu','teach_rel','teach_rel_mu'],
                               ['tradeoff_plot','stud_rel','stud_rel_sigma','teach_rel','teach_rel_sigma'],
                               ['tradeoff_plot','stud_rel','stud_rel_scale','teach_rel','teach_rel_scale'],
                               ['tradeoff_slider','stud_ECE','stud_ou','teach_ECE','teach_ou']],figsize=(20, 8),
                               gridspec_kw={'width_ratios': [1.5,0.75,0.25,0.75,0.25],'height_ratios': [0.65,0.65,0.5,0.5,0.5,0.5,0.5]})

# initialize widgets
stud_dist_mu = Slider(axs['stud_dist_mu'], 'μ', 0, 1.0, valinit=0.35,orientation="horizontal")
teach_dist_mu = Slider(axs['teach_dist_mu'], 'μ', 0, 1.0, valinit=0.75,orientation="horizontal")
stud_dist_sigma = Slider(axs['stud_dist_sigma'], 'σ', 0.05, 1.0, valinit=0.5,orientation="horizontal")
teach_dist_sigma = Slider(axs['teach_dist_sigma'], 'σ', 0.05, 1.0, valinit=0.5,orientation="horizontal")

stud_rel_mu = Slider(axs['stud_rel_mu'], 'μ', 0, 1.0, valinit=0.45,orientation="horizontal")
teach_rel_mu = Slider(axs['teach_rel_mu'], 'μ', 0, 1.0, valinit=0.45,orientation="horizontal")
stud_rel_sigma = Slider(axs['stud_rel_sigma'], 'σ', 0.05, 1.0, valinit=0.25,orientation="horizontal")
teach_rel_sigma = Slider(axs['teach_rel_sigma'], 'σ', 0.05, 1.0, valinit=0.5,orientation="horizontal")
stud_rel_scale = Slider(axs['stud_rel_scale'], 'scale', 0, 0.5, valinit=0.1,orientation="horizontal")
teach_rel_scale = Slider(axs['teach_rel_scale'], 'scale', 0, 0.5, valinit=0.1,orientation="horizontal")

policy = Slider(axs['tradeoff_slider'], 'policy', 0, 1.0, valinit=0.3,valstep=1/n_bins,orientation="horizontal")

stud_ou = RadioButtons(axs['stud_ou'],('overconfident','underconfident','calibrated'),active=2)
teach_ou = RadioButtons(axs['teach_ou'],('overconfident','underconfident','calibrated'),active=2)

axs['stud_avg_acc'].axis('off')
axs['stud_avg_conf'].axis('off')
axs['teach_avg_acc'].axis('off')
axs['teach_avg_conf'].axis('off')
axs['stud_ECE'].axis('off')
axs['teach_ECE'].axis('off')



# ==================== init stud dist ==================== #
# student conf distribution is truncated gaussian and normalized so adds to 1
b_S = norm_sum(
    normal_density(stud_dist_mu.val,stud_dist_sigma.val,bin_centers)
)
stud_dist_bars = axs['stud_dist'].bar(bar_xpos[:-1],b_S,width=1,edgecolor='black',linewidth=2)
axs['stud_dist'].set_xticks(bar_xtickpos,bar_ticks)
axs['stud_dist'].set_ylim([0,1])
axs['stud_dist'].set_xlim(-0.5,n_bins-0.5)
axs['stud_dist'].grid()
axs['stud_dist'].set_title("Student Prediction Distribution")
axs['stud_dist'].set_xlabel("Confidence")
axs['stud_dist'].set_ylabel("% of Samples")
axs['stud_dist'].set_yticks(np.linspace(0,1,11))

# update student confidence distribution
def stud_dist_update(val):
    # get the new distribution bins from latest parameters
    b_S = norm_sum(
            normal_density(stud_dist_mu.val,stud_dist_sigma.val,bin_centers)
        )
    # update the height of each bar, maxing out at 1
    for i,b in enumerate(stud_dist_bars):
        b_S[i] = 1 if (b_S[i] > 1) else b_S[i]
        b.set_height(b_S[i])
    # update the other panels
    stud_rel_update(1)
    tradeoff_update(1)

stud_dist_mu.on_changed(stud_dist_update)
stud_dist_sigma.on_changed(stud_dist_update)



# ==================== init teach dist ==================== #
# identical to student panel
b_T = norm_sum(normal_density(teach_dist_mu.val,teach_dist_sigma.val,bin_centers))
teach_dist_bars = axs['teach_dist'].bar(bar_xpos[:-1],b_T,width=1,edgecolor='black',linewidth=2)
axs['teach_dist'].set_xticks(bar_xtickpos,bar_ticks)
axs['teach_dist'].set_ylim([0,1])
axs['teach_dist'].set_xlim(-0.5,n_bins-0.5)
axs['teach_dist'].grid()
axs['teach_dist'].set_title("Teacher Prediction Distribution")
axs['teach_dist'].set_xlabel("Confidence")
axs['teach_dist'].set_ylabel("% of Samples")
axs['teach_dist'].set_yticks(np.linspace(0,1,11))

def teach_dist_update(val):
    b_T = norm_sum(normal_density(teach_dist_mu.val,teach_dist_sigma.val,bin_centers))
    for i,b in enumerate(teach_dist_bars):
        b_T[i] = 1 if (b_T[i] > 1) else b_T[i]
        b.set_height(b_T[i])
    teach_rel_update(1)
    tradeoff_update(1)

teach_dist_mu.on_changed(teach_dist_update)
teach_dist_sigma.on_changed(teach_dist_update)



# ==================== init stud rel ==================== #
stud_rel_bars = axs['stud_rel'].bar(bar_xpos[:-1],cal_bin_accs,width=1,edgecolor='black',linewidth=2,label="acc")
stud_rel_gap = axs['stud_rel'].bar(bar_xpos[:-1],0,label="gap",width=1,edgecolor='red',bottom=cal_bin_accs,color='red',alpha=0.25,linewidth=2)
axs['stud_rel'].set_xticks(bar_xtickpos,bar_ticks)
axs['stud_rel'].set_ylim([0,1])
axs['stud_rel'].set_xlim(-0.5,n_bins-0.5)
axs['stud_rel'].grid()
axs['stud_rel'].set_title("Student Reliability")
axs['stud_rel'].set_xlabel("Confidence")
axs['stud_rel'].set_yticks(np.linspace(0,1,11))
axs['stud_rel'].set_ylabel("Accuracy")

stud_ece = axs['stud_ECE'].text(0,0,"ECE: "+str(0),fontsize="large")
stud_avg_acc_line = axs['stud_dist'].axvline(sum([b.get_height()*a.get_height() for b,a in zip(stud_dist_bars,stud_rel_bars)])*n_bins-0.5,color="black",linestyle="-",linewidth=2)
stud_avg_conf_line = axs['stud_dist'].axvline(sum([b.get_height()*c for b,c in zip(stud_dist_bars,cal_bin_accs)])*n_bins-0.5,color="gray",linestyle="--",linewidth=2)

axs['stud_avg_acc'].set_xlim([0,0.5])
axs['stud_avg_acc'].set_ylim([-0.25,0.25])
axs['stud_avg_acc'].hlines(y=-0.05,xmin=0.0,xmax=0.05,color="black",linewidth=2)
axs['stud_avg_acc'].hlines(y=-0.05,xmin=0.3,xmax=0.35,color="gray",linewidth=2,linestyle="--")
stud_avg_acc_text = axs['stud_avg_acc'].text(0.06,-0.07,"acc: "+str(round(sum([b.get_height()*a.get_height() for b,a in zip(stud_dist_bars,stud_rel_bars)]),3)))
stud_avg_conf_text = axs['stud_avg_acc'].text(0.36,-0.07,"conf: "+str(round(sum([b.get_height()*c for b,c in zip(stud_dist_bars,cal_bin_accs)]).item(),3)))

# update student reliability diagram
def stud_rel_update(val):
    # get the new deviations from perfect calibration
    normal_weights = normal_density(stud_rel_mu.val,stud_rel_sigma.val,bin_centers)
    dev_factor = normal_density(stud_rel_mu.val,stud_rel_sigma.val,stud_rel_mu.val) # peak of gaussian
    devs = normal_weights*(stud_rel_scale.val/dev_factor) # scale the devs up or down
    
    ece = 0
    for i,b in enumerate(stud_rel_bars):
        if stud_ou.value_selected == "overconfident":
            # bar_val = 1 if ((cal_bin_accs[i]-devs[i]) > 1) else cal_bin_accs[i]-devs[i]
            bar_val = 0 if ((cal_bin_accs[i]-devs[i]) < 0) else cal_bin_accs[i]-devs[i]
            b.set_height(bar_val)
            stud_rel_gap[i].set(y=bar_val,height=cal_bin_accs[i]-bar_val)
            ece += (stud_dist_bars[i].get_height()*abs(bar_val-cal_bin_accs[i])).item()
        elif stud_ou.value_selected == "underconfident":
            bar_val = 1 if ((cal_bin_accs[i]+devs[i]) > 1) else cal_bin_accs[i]+devs[i]
            # bar_val = 0 if ((cal_bin_accs[i]+devs[i]) < 0) else cal_bin_accs[i]+devs[i]
            b.set_height(bar_val)
            stud_rel_gap[i].set(y=bar_val-devs[i],height=devs[i])
            ece += (stud_dist_bars[i].get_height()*abs(bar_val-cal_bin_accs[i])).item()
        elif stud_ou.value_selected == "calibrated":
            b.set_height(cal_bin_accs[i])
            stud_rel_gap[i].set(y=cal_bin_accs[i],height=0)
    # update ece value 
    stud_ece.set(text="ECE: "+str(round(ece,4)))

    # update avg acc and avg conf
    acc = sum([b.get_height()*a.get_height() for b,a in zip(stud_dist_bars,stud_rel_bars)]).item()
    conf = sum([b.get_height()*c for b,c in zip(stud_dist_bars,cal_bin_accs)]).item()
    stud_avg_acc_line.set(xdata=acc*n_bins-0.5)
    stud_avg_acc_text.set(text="acc: "+str(round(acc,3)))
    stud_avg_conf_line.set(xdata=conf*n_bins-0.5)
    stud_avg_conf_text.set(text="conf: "+str(round(conf,3)))
    tradeoff_update(1)
    plt.draw()

stud_ou.on_clicked(stud_rel_update)
stud_rel_mu.on_changed(stud_rel_update)
stud_rel_sigma.on_changed(stud_rel_update)
stud_rel_scale.on_changed(stud_rel_update)



# ==================== init teach rel ==================== #
teach_rel_bars = axs['teach_rel'].bar(bar_xpos[:-1],cal_bin_accs,width=1,edgecolor='black',linewidth=2,label="acc")
teach_rel_gap = axs['teach_rel'].bar(bar_xpos[:-1],0,label="gap",width=1,edgecolor='red',bottom=cal_bin_accs,color='red',alpha=0.25,linewidth=2)
axs['teach_rel'].set_xticks(bar_xtickpos,bar_ticks)
axs['teach_rel'].set_ylim([0,1])
axs['teach_rel'].set_xlim(-0.5,n_bins-0.5)
axs['teach_rel'].grid()
axs['teach_rel'].set_title("Teacher Reliability")
axs['teach_rel'].set_xlabel("Confidence")
axs['teach_rel'].set_yticks(np.linspace(0,1,11))
axs['teach_rel'].set_ylabel("Accuracy")

teach_ece = axs['teach_ECE'].text(0,0,"ECE: "+str(0),fontsize="large")
teach_avg_acc_line = axs['teach_dist'].axvline(sum([b.get_height()*a.get_height() for b,a in zip(teach_dist_bars,teach_rel_bars)])*n_bins-0.5,color="black",linestyle="-",linewidth=2)
teach_avg_conf_line = axs['teach_dist'].axvline(sum([b.get_height()*c for b,c in zip(teach_dist_bars,cal_bin_accs)])*n_bins-0.5,color="gray",linestyle="--",linewidth=2)

axs['teach_avg_acc'].set_xlim([0,0.5])
axs['teach_avg_acc'].set_ylim([-0.25,0.25])
axs['teach_avg_acc'].hlines(y=-0.05,xmin=0.0,xmax=0.05,color="black",linewidth=2)
axs['teach_avg_acc'].hlines(y=-0.05,xmin=0.3,xmax=0.35,color="gray",linewidth=2,linestyle="--")
teach_avg_acc_text = axs['teach_avg_acc'].text(0.06,-0.07,"acc: "+str(round(sum([b.get_height()*a.get_height() for b,a in zip(teach_dist_bars,teach_rel_bars)]),3)))
teach_avg_conf_text = axs['teach_avg_acc'].text(0.36,-0.07,"conf: "+str(round(sum([b.get_height()*c for b,c in zip(teach_dist_bars,cal_bin_accs)]).item(),3)))

def teach_rel_update(val):
    # get the new deviations from perfect calibration
    normal_weights = normal_density(teach_rel_mu.val,teach_rel_sigma.val,bin_centers)
    dev_factor = normal_density(teach_rel_mu.val,teach_rel_sigma.val,teach_rel_mu.val) # peak of gaussian
    devs = normal_weights*(teach_rel_scale.val/dev_factor) # scale the devs up or down
    # print(devs)
    ece = 0
    for i,b in enumerate(teach_rel_bars):
        if teach_ou.value_selected == "overconfident":
            # bar_val = 1 if ((cal_bin_accs[i]-devs[i]) > 1) else cal_bin_accs[i]-devs[i]
            bar_val = 0 if ((cal_bin_accs[i]-devs[i]) < 0) else cal_bin_accs[i]-devs[i]
            b.set_height(bar_val)
            teach_rel_gap[i].set(y=bar_val,height=cal_bin_accs[i]-bar_val)
            ece += (teach_dist_bars[i].get_height()*abs(bar_val-cal_bin_accs[i])).item()
        elif teach_ou.value_selected == "underconfident":
            bar_val = 1 if ((cal_bin_accs[i]+devs[i]) > 1) else cal_bin_accs[i]+devs[i]
            # bar_val = 0 if ((cal_bin_accs[i]+devs[i]) < 0) else cal_bin_accs[i]+devs[i]
            b.set_height(bar_val)
            teach_rel_gap[i].set(y=bar_val-devs[i],height=devs[i])
            ece += (teach_dist_bars[i].get_height()*abs(bar_val-cal_bin_accs[i])).item()
        elif teach_ou.value_selected == "calibrated":
            b.set_height(cal_bin_accs[i])
            teach_rel_gap[i].set(y=cal_bin_accs[i],height=0)
    # update ece value 
    teach_ece.set(text="ECE: "+str(round(ece,4)))

    # update avg acc and avg conf
    acc = sum([b.get_height()*a.get_height() for b,a in zip(teach_dist_bars,teach_rel_bars)]).item()
    conf = sum([b.get_height()*c for b,c in zip(teach_dist_bars,cal_bin_accs)]).item()
    teach_avg_acc_line.set(xdata=acc*n_bins-0.5)
    teach_avg_acc_text.set(text="acc: "+str(round(acc,3)))
    teach_avg_conf_line.set(xdata=conf*n_bins-0.5)
    teach_avg_conf_text.set(text="conf: "+str(round(conf,3)))
    tradeoff_update(1)
    plt.draw()

teach_ou.on_clicked(teach_rel_update)
teach_rel_mu.on_changed(teach_rel_update)
teach_rel_sigma.on_changed(teach_rel_update)
teach_rel_scale.on_changed(teach_rel_update)



# ==================== init tradeoff ==================== #
cost_T = 10
cost_S = 1
avg_S_acc = sum([b.get_height()*a.get_height() for b,a in zip(stud_dist_bars,stud_rel_bars)]).item()
avg_T_acc = sum([b.get_height()*a.get_height() for b,a in zip(teach_dist_bars,teach_rel_bars)]).item()
x = np.linspace(0,1,100)
cost = x*cost_S + (1-x)*cost_T
accuracy = x*avg_S_acc + (1-x)*avg_T_acc
rand_tradeoff_line, = axs['tradeoff_plot'].plot(cost,accuracy)
axs['tradeoff_plot'].set_xticks(np.arange(0,11))
axs['tradeoff_plot'].set_yticks(np.linspace(0,1,11))
axs['tradeoff_plot'].grid()
axs['tradeoff_plot'].set_xlabel("cost")
axs['tradeoff_plot'].set_ylabel("accuracy")
axs['tradeoff_plot'].set_ylim([0,1])
rand_tradeoff_point = axs['tradeoff_plot'].scatter((1-policy.val)*cost_S+(policy.val)*cost_T,(1-policy.val)*avg_S_acc+(policy.val)*avg_T_acc,color="black")
policies = np.linspace(0,1,n_bins+1) # to increase granularity, we need more bins
conf_tradeoffs = []

for p in policies:
    accs_S = np.array([a.get_height() for a in stud_rel_bars]) # accuracy per bins
    confs_S = np.array([b.get_height() for b in stud_dist_bars]) # confidence weight per bin

    # expected acc/cost is calculated using student bins higher/lower than conf threshold weighted by corresponding acc/cost
    if p < 1:
        exp_acc = sum(accs_S[cal_bin_accs > p]*confs_S[cal_bin_accs > p]) + sum(confs_S[cal_bin_accs <= p])*avg_T_acc
    else:
        exp_acc = sum(accs_S[cal_bin_accs > p]*confs_S[cal_bin_accs > p]) + sum(confs_S[cal_bin_accs <= p])*avg_T_acc
    exp_cost = sum(confs_S[cal_bin_accs > p])*cost_S + sum(confs_S[cal_bin_accs <= p]*(cost_T+cost_S))
    if p == policy.val:
        conf_tradeoffs.append(axs['tradeoff_plot'].scatter(exp_cost,exp_acc,color="red"))
    else:
        conf_tradeoffs.append(axs['tradeoff_plot'].scatter(exp_cost,exp_acc,color="blue"))

def tradeoff_update(val):
    # get the new avg accuracies
    avg_S_acc = sum([b.get_height()*a.get_height() for b,a in zip(stud_dist_bars,stud_rel_bars)]).item()
    avg_T_acc = sum([b.get_height()*a.get_height() for b,a in zip(teach_dist_bars,teach_rel_bars)]).item()
    x = np.linspace(0,1,100)
    cost = x*cost_S + (1-x)*cost_T
    accuracy = x*avg_S_acc + (1-x)*avg_T_acc
    rand_tradeoff_line.set_xdata(cost)
    rand_tradeoff_line.set_ydata(accuracy)
    rand_tradeoff_point.set_offsets(((1-policy.val)*cost_S+(policy.val)*cost_T,(1-policy.val)*avg_S_acc+(policy.val)*avg_T_acc))

    for i,p in enumerate(policies):
        accs_S = np.array([a.get_height() for a in stud_rel_bars])
        confs_S = np.array([b.get_height() for b in stud_dist_bars])
        if p < 1:
            exp_acc = sum(accs_S[cal_bin_accs > p]*confs_S[cal_bin_accs > p]) + sum(confs_S[cal_bin_accs <= p])*avg_T_acc
        else:
            exp_acc = sum(accs_S[cal_bin_accs > p]*confs_S[cal_bin_accs > p]) + sum(confs_S[cal_bin_accs <= p])*avg_T_acc
        exp_cost = sum(confs_S[cal_bin_accs > p])*cost_S + sum(confs_S[cal_bin_accs <= p]*(cost_T+cost_S))
        if p == policy.val:
            conf_tradeoffs[i].set_offsets((exp_cost,exp_acc))
            conf_tradeoffs[i].set(color="red")
        else:
            conf_tradeoffs[i].set_offsets((exp_cost,exp_acc))
            conf_tradeoffs[i].set(color="blue")
    plt.draw()

policy.on_changed(tradeoff_update)

plt.show()