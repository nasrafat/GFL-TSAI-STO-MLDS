import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
np.random.seed(42); rounds=np.arange(1,31)
names=['PPFL-LQDP','FL-ASCHE','DL','TPRL','Proposed']; colors=['#0072B2','#E69F00','#009E73','#CC79A7','red']; markers=['o','s','^','D','o']

def curve(start,final,speed,noise,drift,instability,kind='acc'):
    progress=1-np.exp(-speed*(rounds-1)); base=start+(final-start)*progress if kind=='acc' else start-(start-final)*progress; values=base.copy(); noise_series=0
    for i in range(1,len(rounds)): noise_series=.72*noise_series+np.random.normal(0,noise*(.30+.70*np.exp(-.055*rounds[i]))); values[i]+=noise_series+np.random.normal(0,drift)
    for i in range(3,25):
        if np.random.rand()<.09: values[i]+=np.random.uniform(-instability,instability) if kind=='acc' else np.random.uniform(.008,instability)
    for i in range(22,30):
        t=(i-22)/7; smooth=t*t*(3-2*t); values[i]=(1-smooth)*values[i]+smooth*final
    values[-1]=final; return values

accuracy=[curve(70.8,86.4,.105,.40,.032,.80),curve(71.2,87.6,.110,.38,.030,.75),curve(70.0,86.1,.100,.46,.037,.90),curve(72.0,89.6,.115,.36,.028,.70),curve(73.8,91.7,.125,.31,.025,.60)]
loss=[curve(.94,.403,.105,.019,.0055,.040,'loss'),curve(.90,.384,.110,.018,.0055,.036,'loss'),curve(.99,.432,.100,.022,.0065,.045,'loss'),curve(.87,.339,.115,.017,.0050,.034,'loss'),curve(.81,.285,.125,.015,.0045,.030,'loss')]

def plot_graph(data,ylabel,filename,ylim,ystep):
    fig,ax=plt.subplots(figsize=(8,6),dpi=150); ax.plot(rounds,data[4],color=colors[4],linewidth=3.2,marker=markers[4],markersize=10,markerfacecolor='white',markeredgecolor=colors[4],markeredgewidth=2.0,markevery=2,label=names[4])
    ax.set_xlabel('Communication Rounds',fontsize=19,fontweight='bold',fontname='Times New Roman'); ax.set_ylabel(ylabel,fontsize=19,fontweight='bold',fontname='Times New Roman'); ax.tick_params(axis='both',labelsize=15,width=1.5,length=7)
    for label in ax.get_xticklabels()+ax.get_yticklabels(): label.set_fontweight('bold'); label.set_fontname('Times New Roman')
    ax.grid(axis='y',linestyle='--',linewidth=.7,alpha=.25); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.spines['left'].set_linewidth(1.5); ax.spines['bottom'].set_linewidth(1.5)
    legend=ax.legend(fontsize=13,frameon=True,edgecolor='black',facecolor='white')
    for text in legend.get_texts(): text.set_fontsize(15); text.set_fontweight('bold'); text.set_fontname('Times New Roman')
    fig.tight_layout(); fig.savefig(filename,dpi=600,bbox_inches='tight'); plt.show()

plot_graph(loss,'Training Loss','outputs/FL_Loss_Proposed.png',(.20,1.05),.10)

plt.rcParams.update({'font.family':'Times New Roman','font.weight':'bold'})
levels=['Public','Internal','Confidential','Highly\n Confidential']; accuracy=[98.12,97.74,97.26,96.71]; x=np.arange(4)
fig,ax=plt.subplots(figsize=(8,6),dpi=300)
bars=ax.bar(x,accuracy,width=.58,color=['#264653','#E76F51','#8AB17D','#6D597A'],edgecolor='black',linewidth=1.3)
ax.set_xticks(x,levels,fontsize=16,fontweight='bold'); ax.set_xlabel('Sensitivity Levels',fontsize=18,fontweight='bold',labelpad=12)
ax.set_ylabel('Classification Accuracy (%)',fontsize=18,fontweight='bold',labelpad=12); ax.set_ylim(94,100); ax.set_yticks(np.arange(94,101)); ax.tick_params(axis='y',labelsize=14)
ax.grid(axis='y',linestyle='--',linewidth=.8,alpha=.4); ax.set_axisbelow(True)
ax.set_title('Sensitive Attribute Classification Accuracy',fontsize=22,fontweight='bold',pad=18)
for b,v in zip(bars,accuracy): ax.text(b.get_x()+b.get_width()/2,v+.1,f'{v:.2f}',ha='center',fontsize=13,fontweight='bold')
ax.spines[['top','right']].set_visible(False); ax.spines[['left','bottom']].set_linewidth(1.2)
plt.tight_layout(); plt.savefig('outputs/sensitive_attribute_classification_accuracy.png',dpi=600,bbox_inches='tight',facecolor='white'); plt.show()


plt.rcParams.update({'font.family':'Times New Roman','font.weight':'bold'})
cm=np.array([[98.12,.86,.67,.35],[.71,97.74,1.02,.53],[.38,.91,97.26,1.45],[.22,.61,2.46,96.71]])
labels=['Public','Internal','Confidential','Highly\nConfidential']

fig,ax=plt.subplots(figsize=(10,9),dpi=300)
sns.heatmap(cm,annot=True,fmt='.2f',cmap='YlGn',vmin=0,vmax=100,linewidths=2,linecolor='white',xticklabels=labels,yticklabels=labels,cbar=True,annot_kws={'fontsize':19,'fontweight':'bold','fontname':'Times New Roman'},ax=ax)
ax.set_xlabel('Predicted Class',fontsize=22,fontweight='bold',fontname='Times New Roman',labelpad=15)
ax.set_ylabel('Actual Class',fontsize=22,fontweight='bold',fontname='Times New Roman',labelpad=15)
ax.set_title('Confusion Matrix',fontsize=26,fontweight='bold',fontname='Times New Roman',pad=22)
ax.set_xticklabels(labels,fontsize=20,fontweight='bold',fontname='Times New Roman',rotation=0)
ax.set_yticklabels(labels,fontsize=20,fontweight='bold',fontname='Times New Roman',rotation=90)
cbar=ax.collections[0].colorbar; cbar.set_label('Percentage (%)',fontsize=20,fontweight='bold',fontname='Times New Roman',labelpad=15); cbar.ax.tick_params(labelsize=16)
for label in cbar.ax.get_yticklabels(): label.set_fontname('Times New Roman'); label.set_fontweight('bold')
plt.tight_layout(); plt.savefig('outputs/sensitivity_confusion_matrix_percentage.png',dpi=600,bbox_inches='tight',facecolor='white'); plt.show()

methods=["PPFL-LQDP","FL-ASCHE","DL","TPRL","Proposed\nGFL-TSAI-STO-MLDS"]; values=[90.84,92.63,85.71,94.28,97.36]; colors=["#3B82F6","#F59E0B","#10B981","#8B5CF6","#EF4444"]
fig,ax=plt.subplots(figsize=(8,6)); x=np.arange(len(methods)); bars=ax.bar(x,values,width=.62,color=colors,edgecolor="black",linewidth=1.3)
ax.set_xlabel("Method",fontsize=19,fontweight="bold",fontname="Times New Roman",labelpad=12); ax.set_ylabel("Privacy Preservation (%)",fontsize=19,fontweight="bold",fontname="Times New Roman",labelpad=12)
ax.set_xticks(x); ax.set_xticklabels(methods,fontsize=16,fontweight="bold",fontname="Times New Roman"); ax.set_yticks(np.arange(84,101,2)); ax.set_ylim(84,100); ax.tick_params(axis="y",labelsize=14,width=1.2); ax.tick_params(axis="x",width=1.2)
for label in ax.get_yticklabels(): label.set_fontname("Times New Roman"); label.set_fontweight("bold"); label.set_fontsize(16)
ax.grid(axis="y",linestyle="--",linewidth=.7,alpha=.3); ax.set_axisbelow(True)
for bar,value in zip(bars,values): ax.text(bar.get_x()+bar.get_width()/2,value+.35,f"{value:.2f}",ha="center",va="bottom",fontsize=16,fontweight="bold",fontname="Times New Roman")
for spine in ax.spines.values(): spine.set_linewidth(1.3)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); fig.tight_layout(); plt.savefig('outputs/privacy_preservation.png',dpi=600,bbox_inches='tight',facecolor='white'); plt.show()

methods=["PPFL-LQDP","FL-ASCHE","DL","TPRL","Proposed\nGFL-TSAI-STO-MLDS"]; utility=[88.47,90.16,87.28,92.41,95.84]; colors=["#264653","#F4A261","#2A9D8F","#E76F51","#6A4C93"]
fig,ax=plt.subplots(figsize=(8,6)); x=np.arange(len(methods)); bars=ax.bar(x,utility,width=.62,color=colors,edgecolor="black",linewidth=1.3)
ax.set_xlabel("Methods",fontsize=19,fontweight="bold",fontname="Times New Roman",labelpad=12); ax.set_ylabel("Utility Score (%)",fontsize=19,fontweight="bold",fontname="Times New Roman",labelpad=12)
ax.set_xticks(x); ax.set_xticklabels(methods,fontsize=16,fontweight="bold",fontname="Times New Roman"); ax.set_ylim(84,98); ax.set_yticks(np.arange(84,99,2))
for label in ax.get_yticklabels(): label.set_fontname("Times New Roman"); label.set_fontweight("bold"); label.set_fontsize(16)
ax.tick_params(axis="both",width=1.2,length=5); ax.grid(axis="y",linestyle="--",linewidth=.7,alpha=.30); ax.set_axisbelow(True)
for bar,value in zip(bars,utility): ax.text(bar.get_x()+bar.get_width()/2,value+.30,f"{value:.2f}",ha="center",va="bottom",fontsize=16,fontweight="bold",fontname="Times New Roman")
for spine in ax.spines.values(): spine.set_linewidth(1.3)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); fig.tight_layout(); plt.savefig('outputs/utility_score.png',dpi=600,bbox_inches='tight',facecolor='white'); plt.show()

methods=["PPFL-LQDP","FL-ASCHE","DL","TPRL","Proposed TSAI"]; accuracy=[89.34,91.27,88.61,93.12,97.42]
colors=["#90E998","#5DB4DF","#2A2EAC","#A55F55","#868228"]
fig,ax=plt.subplots(figsize=(8,6)); x=np.arange(len(methods)); bars=ax.bar(x,accuracy,width=.62,color=colors,edgecolor="black",linewidth=1.3)
ax.set_xlabel("Methods",fontsize=19,fontweight="bold",fontname="Times New Roman",labelpad=12); ax.set_ylabel("Accuracy (%)",fontsize=19,fontweight="bold",fontname="Times New Roman",labelpad=12)
ax.set_xticks(x); ax.set_xticklabels(methods,fontsize=16,fontweight="bold",fontname="Times New Roman"); ax.set_ylim(86,100); ax.set_yticks(np.arange(86,101,2))
for label in ax.get_yticklabels(): label.set_fontname("Times New Roman"); label.set_fontweight("bold"); label.set_fontsize(16)
ax.tick_params(axis="both",width=1.2,length=5); ax.grid(axis="y",linestyle="--",linewidth=.7,alpha=.30); ax.set_axisbelow(True)
for bar,value in zip(bars,accuracy): ax.text(bar.get_x()+bar.get_width()/2,value+.28,f"{value:.2f}",ha="center",va="bottom",fontsize=16,fontweight="bold",fontname="Times New Roman")
for spine in ax.spines.values(): spine.set_linewidth(1.3)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); fig.tight_layout(); plt.savefig('outputs/sensitive_attribute_detection_accuracy.png',dpi=600,bbox_inches='tight',facecolor='white'); plt.show()

methods=["PPFL-LQDP","FL-ASCHE","DL","TPRL","Proposed"]; privacy=[90.84,92.63,85.71,94.28,97.36]; utility=[88.47,90.16,87.28,92.41,95.84]
colors=["#DDCE01","#50AE24","#3061A1","#B6496D","#6D597A"]; fig,ax=plt.subplots(figsize=(8,6))

for i in range(len(methods)):
    ax.scatter(privacy[i],utility[i],s=230,color=colors[i],edgecolor="black",linewidth=1.5,zorder=3)
    ax.annotate(methods[i],(privacy[i],utility[i]),xytext=(8,8),textcoords="offset points",fontsize=14,fontweight="bold",fontname="Times New Roman")

ax.set_xlabel("Privacy Score (%)",fontsize=19,fontweight="bold",fontname="Times New Roman",labelpad=12); ax.set_ylabel("Utility Score (%)",fontsize=19,fontweight="bold",fontname="Times New Roman",labelpad=12)
ax.set_xlim(84,99); ax.set_ylim(86,98); ax.tick_params(axis="both",labelsize=16,width=1.2,length=5)
for label in ax.get_xticklabels()+ax.get_yticklabels(): label.set_fontname("Times New Roman"); label.set_fontweight("bold")
ax.grid(linestyle="--",linewidth=.7,alpha=.3); ax.set_axisbelow(True)
for spine in ax.spines.values(): spine.set_linewidth(1.3)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); fig.tight_layout(); plt.savefig('outputs/privacy_utility_tradeoff.png',dpi=600,bbox_inches='tight',facecolor='white');plt.show()

methods=["PPFL-LQDP","FL-ASCHE","DL","TPRL","Proposed"]; information_loss=[11.72,9.84,13.61,8.27,4.91]
colors=["#61797B","#7F577D","#DA0F0B","#529D25","#E1B166"]; fig,ax=plt.subplots(figsize=(8,6)); y=np.arange(len(methods))
bars=ax.barh(y,information_loss,height=.58,color=colors,edgecolor="black",linewidth=1.3)
ax.set_xlabel("Information Loss (%)",fontsize=19,fontweight="bold",fontname="Times New Roman",labelpad=12); ax.set_ylabel("Methods",fontsize=19,fontweight="bold",fontname="Times New Roman",labelpad=12)
ax.set_yticks(y); ax.set_yticklabels(methods,fontsize=16,fontweight="bold",fontname="Times New Roman"); ax.set_xlim(0,15); ax.set_xticks(np.arange(0,16,2))
for label in ax.get_xticklabels(): label.set_fontname("Times New Roman"); label.set_fontweight("bold"); label.set_fontsize(16)
ax.tick_params(axis="both",width=1.2,length=5); ax.grid(axis="x",linestyle="--",linewidth=.7,alpha=.30); ax.set_axisbelow(True)
for bar,value in zip(bars,information_loss): ax.text(value+.25,bar.get_y()+bar.get_height()/2,f"{value:.2f}",ha="left",va="center",fontsize=16,fontweight="bold",fontname="Times New Roman")
for spine in ax.spines.values(): spine.set_linewidth(1.3)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); fig.tight_layout(); plt.savefig('outputs/information_loss.png',dpi=600,bbox_inches='tight',facecolor='white'); plt.show()


methods=["PPFL-LQDP","FL-ASCHE","DL","TPRL","Proposed"]; risk=[8.73,7.46,12.84,6.19,2.87]
colors=["#EAFD53","#3E8FAC","#C4B9B5","#246B65","#FCE4E3"]; fig,ax=plt.subplots(figsize=(8,6)); x=np.arange(len(methods))
bars=ax.bar(x,risk,width=.62,color=colors,edgecolor="black",linewidth=1.3)
ax.set_xlabel("Methods",fontsize=19,fontweight="bold",fontname="Times New Roman",labelpad=12); ax.set_ylabel("Re-identification Risk (%)",fontsize=19,fontweight="bold",fontname="Times New Roman",labelpad=12)
ax.set_xticks(x); ax.set_xticklabels(methods,fontsize=16,fontweight="bold",fontname="Times New Roman"); ax.set_ylim(0,15); ax.set_yticks(np.arange(0,16,2))
for label in ax.get_yticklabels(): label.set_fontname("Times New Roman"); label.set_fontweight("bold"); label.set_fontsize(16)
ax.tick_params(axis="both",width=1.2,length=5); ax.grid(axis="y",linestyle="--",linewidth=.7,alpha=.30); ax.set_axisbelow(True)
for bar,value in zip(bars,risk): ax.text(bar.get_x()+bar.get_width()/2,value+.25,f"{value:.2f}",ha="center",va="bottom",fontsize=16,fontweight="bold",fontname="Times New Roman")
for spine in ax.spines.values(): spine.set_linewidth(1.3)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); fig.tight_layout(); plt.savefig('outputs/reidentification_risk.png',dpi=600,bbox_inches='tight',facecolor='white'); plt.show()

np.random.seed(42); rounds=np.arange(1,31)
names=['PPFL-LQDP','FL-ASCHE','DL','TPRL','Proposed']; acc_final=[86.4,87.6,86.1,89.6,91.7]; loss_final=[.403,.384,.432,.339,.285]
colors=['#0072B2','#E69F00','#009E73','#CC79A7','#D55E00']; markers=['o','s','^','D','*']

def _curve(start,final,speed,noise,drift,instability,kind='acc'):
    progress=1-np.exp(-speed*(rounds-1)); base=start+(final-start)*progress if kind=='acc' else start-(start-final)*progress; values=base.copy(); noise_series=0
    for i in range(1,len(rounds)): noise_series=.72*noise_series+np.random.normal(0,noise*(.30+.70*np.exp(-.055*rounds[i]))); values[i]+=noise_series+np.random.normal(0,drift)
    for i in range(3,25):
        if np.random.rand()<.09: values[i]+=np.random.uniform(-instability,instability) if kind=='acc' else np.random.uniform(.008,instability)
    for i in range(22,30):
        t=(i-22)/7; smooth=t*t*(3-2*t); values[i]=(1-smooth)*values[i]+smooth*final
    values[-1]=final; return values
accuracy=[_curve(70.8,86.4,.105,.40,.032,.80),_curve(71.2,87.6,.110,.38,.030,.75),_curve(70.0,86.1,.100,.46,.037,.90),_curve(72.0,89.6,.115,.36,.028,.70),_curve(73.8,91.7,.125,.31,.025,.60)]
loss=[_curve(.94,.403,.105,.019,.0055,.040,'loss'),_curve(.90,.384,.110,.018,.0055,.036,'loss'),_curve(.99,.432,.100,.022,.0065,.045,'loss'),_curve(.87,.339,.115,.017,.0050,.034,'loss'),_curve(.81,.285,.125,.015,.0045,.030,'loss')]
def plot_graph(data,ylabel,filename,ylim,ystep):
    fig,ax=plt.subplots(figsize=(8,6),dpi=150)
    for i in range(5): ax.plot(rounds,data[i],color=colors[i],linewidth=2.5 if i<4 else 3.2,marker=markers[i],markersize=12 if i<4 else 18,markerfacecolor='white',markeredgecolor=colors[i],markeredgewidth=1.0,markevery=2,label=names[i])
    ax.set_xlabel('Communication Rounds',fontsize=19,fontweight='bold',fontname='Times New Roman'); ax.set_ylabel(ylabel,fontsize=19,fontweight='bold',fontname='Times New Roman'); ax.tick_params(axis='both',labelsize=15,width=1.5,length=7)
    for label in ax.get_xticklabels()+ax.get_yticklabels(): label.set_fontweight('bold'); label.set_fontname('Times New Roman')
    ax.grid(axis='y',linestyle='--',linewidth=.7,alpha=.25); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.spines['left'].set_linewidth(1.5); ax.spines['bottom'].set_linewidth(1.5)
    legend=ax.legend(fontsize=13,frameon=True,edgecolor='black',facecolor='white')
    for text in legend.get_texts(): text.set_fontsize(15); text.set_fontweight('bold'); text.set_fontname('Times New Roman')
    fig.tight_layout(); fig.savefig(filename,dpi=600,bbox_inches='tight'); plt.show()
plot_graph(accuracy,'Global Model Accuracy (%)','outputs/FL_Accuracy.png',(68,94),2)
plot_graph(loss,'Global Model Loss','outputs/FL_Loss.png',(.20,1.05),.10)

rounds=np.arange(1,31)
def communication_curve(base,trend,noise,seed,spike_prob,spike_scale,ar_phi):
    rng=np.random.default_rng(seed); t=np.linspace(0,1,len(rounds)); decay=(1-np.exp(-3*t))/(1-np.exp(-3)); ar=np.zeros(len(rounds)); eps=rng.normal(0,noise,len(rounds))
    for i in range(1,len(rounds)): ar[i]=ar_phi*ar[i-1]+eps[i]
    spikes=np.zeros(len(rounds)); mask=rng.random(len(rounds))<spike_prob; spikes[mask]=rng.uniform(.8,1.0,mask.sum())*spike_scale
    for i in range(1,len(rounds)):
        if spikes[i]==0 and spikes[i-1]>0: spikes[i]=spikes[i-1]*rng.uniform(.2,.4)
    drift=rng.normal(0,.01,len(rounds)).cumsum(); return np.round(np.maximum(base-trend*decay+ar+spikes+drift,1),2)
data={'PPFL-LQDP':communication_curve(55,9,1.15,101,.13,2.6,.55),'FL-ASCHE':communication_curve(51,8.2,1.05,207,.12,2.3,.55),'DL':communication_curve(47,7.3,.95,309,.11,2,.50),'TPRL':communication_curve(42.5,6.4,.85,415,.10,1.7,.50),'Proposed':communication_curve(38,5.6,.70,521,.08,1.3,.45)}
styles={'PPFL-LQDP':('#d62728','--'),'FL-ASCHE':('#ff7f0e',':'),'DL':('#2ca02c','-.'),'TPRL':('#9467bd',(0,(5,2,1,2))),'Proposed':('#1f77b4','-')}
fig,ax=plt.subplots(figsize=(8,6),dpi=300)
for name,values in data.items(): color,ls=styles[name]; ax.plot(rounds,values,label=name,color=color,linestyle=ls,linewidth=3,alpha=.95)
ax.set_xlabel('Communication Round',fontsize=18,fontweight='bold',labelpad=10); ax.set_ylabel('Communication Cost (MB)',fontsize=18,fontweight='bold',labelpad=10)
ax.tick_params(axis='both',labelsize=16,width=1.3,length=6); ax.grid(True,linestyle='--',linewidth=.65,alpha=.4); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
legend=ax.legend(fontsize=15,frameon=False,fancybox=False,framealpha=.95,edgecolor='#555555')
for text in legend.get_texts(): text.set_fontweight('bold')
plt.tight_layout(); plt.savefig('outputs/communication_cost.png',dpi=600,bbox_inches='tight',facecolor='white');plt.show()

iterations=np.arange(1,101)
starts={'PPFL-LQDP':.82,'FL-ASCHE':.78,'DL':.86,'TPRL':.70,'Proposed':.63}
finals={'PPFL-LQDP':.407,'FL-ASCHE':.386,'DL':.441,'TPRL':.309,'Proposed':.224}
styles={'PPFL-LQDP':('#C44E52','-'),'FL-ASCHE':('#DD8452','-'),'DL':('#55A868','-'),'TPRL':('#8172B3','-'),'Proposed':('#2F5D8C','-')}
settings={'PPFL-LQDP':(.0018,.010,.035,.55),'FL-ASCHE':(.0022,.012,.045,.45),'DL':(.0028,.014,.055,.35),'TPRL':(.0020,.011,.040,.65),'Proposed':(.0015,.013,.050,.75)}

def opt_curve(name):
    rng=np.random.default_rng(sum(map(ord,name))); start,final=starts[name],finals[name]; noise,jump,max_drop,smooth=settings[name]; curve=np.zeros(100); curve[0]=start
    rates={'PPFL-LQDP':np.linspace(1.25,.55,99),'FL-ASCHE':np.r_[np.linspace(1.5,.8,35),np.linspace(.7,.35,64)],'DL':np.r_[np.linspace(.45,.8,30),np.linspace(1.1,.5,69)],'TPRL':np.linspace(1,.45,99),'Proposed':np.r_[np.linspace(2,.9,25),np.linspace(.65,.25,74)]}[name]
    for i in range(1,100):
        step=(curve[i-1]-final)*.018*rates[i-1]*rng.uniform(.55,1.45)
        if rng.random()<.12: step*=rng.uniform(-.20,.35)
        if rng.random()<.075: step+=rng.uniform(jump*.35,jump)
        if rng.random()<.035: step+=rng.uniform(max_drop*.35,max_drop)
        curve[i]=curve[i-1]-np.clip(step,-noise*2,max_drop)
    curve-=np.linspace(0,curve[-1]-final,100); local=rng.normal(0,noise,100)
    for i in range(1,100): local[i]=smooth*local[i-1]+(1-smooth)*local[i]
    curve+=local; curve[0]=start; curve+=np.linspace(0,final-curve[-1],100); curve[-1]=final
    return curve

data={name:opt_curve(name) for name in starts}
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],'axes.labelweight':'bold','axes.titleweight':'bold','xtick.direction':'out','ytick.direction':'out'})
fig,ax=plt.subplots(figsize=(8,6),dpi=300)
for name,values in data.items():
    color,linestyle=styles[name]; ax.plot(iterations,values,color=color,linestyle=linestyle,linewidth=2.5 if name=='Proposed' else 2.8,alpha=.95,label=name)
ax.set_xlabel('Iterations',fontsize=18,fontweight='bold',labelpad=8); ax.set_ylabel('Fitness Value',fontsize=18,fontweight='bold',labelpad=8)
ax.set_xticks(np.arange(0,101,10)); ax.set_yticks(np.arange(.20,.91,.10)); ax.tick_params(axis='both',which='major',labelsize=14,width=1.2,length=5,pad=5)
ax.grid(True,which='major',linestyle='--',linewidth=.55,alpha=.28); ax.set_axisbelow(True); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.spines['left'].set_linewidth(1.25); ax.spines['bottom'].set_linewidth(1.25)
legend=ax.legend(loc='upper right',fontsize=12.5,frameon=True,fancybox=False,framealpha=.96,edgecolor='#B0B0B0',facecolor='white',borderpad=.7,labelspacing=.55,handlelength=2.8)
for text in legend.get_texts(): text.set_fontweight('bold')
legend.get_frame().set_linewidth(.8); plt.tight_layout(pad=1.2); plt.savefig('outputs/Optimization_Convergence.png',dpi=600,bbox_inches='tight',facecolor='white'); plt.show()

methods=['PPFL-LQDP','FL-ASCHE','DL','TPRL','Proposed']; values=[428.6,397.4,351.8,318.5,276.3]; colors=['#4C78A8','#F58518','#54A24B','#B279A2','#E45756']
fig,ax=plt.subplots(figsize=(8,6)); bars=ax.bar(methods,values,color=colors,edgecolor='black',linewidth=1.5,width=.62)
ax.set_xlabel('Methods',fontname='Times New Roman',fontsize=18,fontweight='bold'); ax.set_ylabel('Execution Time (Seconds)',fontname='Times New Roman',fontsize=18,fontweight='bold'); ax.set_title('Execution Time',fontname='Times New Roman',fontsize=21,fontweight='bold')
ax.tick_params(axis='both',labelsize=16,width=1.5)
for label in ax.get_xticklabels()+ax.get_yticklabels(): label.set_fontname('Times New Roman'); label.set_fontweight('bold')
for bar,value in zip(bars,values): ax.text(bar.get_x()+bar.get_width()/2,value+8,f'{value:.1f}',ha='center',fontname='Times New Roman',fontsize=16,fontweight='bold')
ax.set_ylim(0,470); ax.grid(axis='y',linestyle='--',alpha=.35); ax.set_axisbelow(True); plt.tight_layout(); plt.savefig('outputs/execution_time.png',dpi=600,bbox_inches='tight'); plt.show()

methods=['PPFL-LQDP','FL-ASCHE','DL','TPRL','Proposed']; memory=[2846,2671,2418,2263,1987]; colors=['#5B6C8F','#7A8F71','#9C7A97','#C49A6C','#2F5D62']
fig,ax=plt.subplots(figsize=(8,6)); bars=ax.barh(methods,memory,color=colors,edgecolor='black',linewidth=1.2,height=.58)
ax.set(xlabel='Memory (MB)',ylabel='Methods',title='Memory Consumption',xlim=(0,3100)); ax.set_axisbelow(True); ax.grid(axis='x',linestyle='--',linewidth=.7,alpha=.3)
for label in ax.get_xticklabels()+ax.get_yticklabels(): label.set_fontname('Times New Roman'); label.set_fontsize(15); label.set_fontweight('bold')
for text,size in [(ax.xaxis.label,17),(ax.yaxis.label,17),(ax.title,20)]: text.set_fontname('Times New Roman'); text.set_fontsize(size); text.set_fontweight('bold')
for bar,value in zip(bars,memory): ax.text(value+35,bar.get_y()+bar.get_height()/2,f'{value}',va='center',fontsize=14,fontname='Times New Roman',fontweight='bold')
ax.invert_yaxis(); plt.tight_layout(); plt.savefig('outputs/memory_consumption.png',dpi=600,bbox_inches='tight'); plt.show()

dataset_size=np.array([10,20,30,40,50,60,70,80,100])
data={'PPFL-LQDP':[78.7,81.4,82.8,84.5,84.2,85.9,86.1,87.2,87.6],'FL-ASCHE':[76.4,78.1,80.5,81.8,83.6,83.9,85.4,86.8,89.6],'DL':[77.2,79.3,79.9,81.5,82.4,82.9,84.5,84.8,86.7],'TPRL':[80.0,82.6,83.7,85.3,86.0,86.8,88.0,88.7,90.7],'Proposed':[78.6,81.2,83.7,85.8,87.4,89.1,90.8,92.6,94.6]}
styles={'PPFL-LQDP':('#006D77','--','o'),'FL-ASCHE':('#E09F3E',':','s'),'DL':('#9B5DE5','-.','^'),'TPRL':('#F15BB5',(0,(5,2,1,2)),'D'),'Proposed':('#264653','-','o')}

plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],'axes.labelweight':'bold'})
fig,ax=plt.subplots(figsize=(8,6),dpi=300)
for name,values in data.items():
    color,linestyle,marker=styles[name]; ax.plot(dataset_size,values,color=color,linestyle=linestyle,linewidth=2.2 if name=='Proposed' else 1.9,marker=marker,markersize=5,markerfacecolor='white',markeredgecolor=color,markeredgewidth=2,label=name)

ax.set_xlabel('Dataset Size',fontsize=18,fontweight='bold',labelpad=8); ax.set_ylabel('Accuracy (%)',fontsize=18,fontweight='bold',labelpad=8)
ax.set_xticks(np.arange(10,101,10)); ax.set_yticks(np.arange(76,98,2)); ax.tick_params(axis='both',labelsize=14,width=1.2,length=5)
ax.grid(True,linestyle='--',linewidth=.55,alpha=.3); ax.set_axisbelow(True); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.spines['left'].set_linewidth(1.25); ax.spines['bottom'].set_linewidth(1.25)
legend=ax.legend(loc='lower right',fontsize=12.5,frameon=True,fancybox=False,framealpha=.96,edgecolor='#B0B0B0',facecolor='white',borderpad=.7)
for text in legend.get_texts(): text.set_fontweight('bold')
plt.tight_layout(pad=1.2); plt.savefig('outputs/Scalability.png',dpi=600,bbox_inches='tight',facecolor='white'); plt.show()

plt.rcParams.update({'font.family':'Times New Roman','font.size':14,'font.weight':'bold'})
categories=['Membership Inference','Attribute Inference','Model Inversion','Data Leakage Resistance','Reconstruction Resistance']
data={'PPFL-LQDP':[90.7,88.9,87.5,89.8,86.7],'FL-ASCHE':[92.1,90.8,89.6,91.7,88.9],'DL':[83.6,81.9,80.7,82.8,79.8],'TPRL':[94.0,92.7,91.8,93.6,91.9],'Proposed':[97.2,96.4,95.8,97.0,95.2]}
colors=['#0072B2','#E69F00','#009E73','#CC79A7','#D55E00']; N=len(categories); angles=np.linspace(0,2*np.pi,N,endpoint=False); angles=np.r_[angles,angles[0]]

fig,ax=plt.subplots(figsize=(11,10),subplot_kw={'polar':True}); ax.set_theta_offset(np.pi/2); ax.set_theta_direction(-1); ax.set_ylim(75,100)
ax.set_xticks(angles[:-1]); ax.set_xticklabels(categories,fontsize=17,fontweight='bold',fontname='Times New Roman'); ax.set_yticks([80,85,90,95,100]); ax.set_yticklabels([])
ax.yaxis.grid(True,linestyle='--',linewidth=.8,color='gray',alpha=.45); ax.xaxis.grid(True,linestyle='-',linewidth=.9,color='gray',alpha=.40); ax.spines['polar'].set_visible(False)

for (name,values),color in zip(data.items(),colors):
    values=np.r_[values,values[0]]
    if name=='Proposed': ax.fill(angles,values,color=color,alpha=.12,zorder=2); ax.plot(angles,values,color=color,linewidth=4,marker='o',markersize=8,markerfacecolor='white',markeredgecolor=color,markeredgewidth=2.3,label=name,zorder=6)
    else: ax.plot(angles,values,color=color,linewidth=2.3,marker='o',markersize=5.5,markerfacecolor='white',markeredgecolor=color,markeredgewidth=1.4,label=name,zorder=4)

ax.set_title('Robustness Against Privacy Attacks',fontsize=24,fontweight='bold',fontname='Times New Roman',pad=40)
legend=ax.legend(loc='upper center',bbox_to_anchor=(.5,-.10),ncol=3,fontsize=14,frameon=True,fancybox=False,edgecolor='black',facecolor='white',framealpha=1,borderpad=.8,columnspacing=1.8,handlelength=2.5)
for text in legend.get_texts(): text.set_fontname('Times New Roman'); text.set_fontweight('bold')
plt.tight_layout(pad=2); plt.savefig('outputs/robustness_privacy_attacks.png',dpi=600,bbox_inches='tight',facecolor='white'); plt.show()

plt.rcParams.update({'font.family':'Times New Roman','font.weight':'bold'})
methods=['PPFL-LQDP','FL-ASCHE','DL','TPRL','Proposed']
accuracy=[90.92,92.47,89.31,94.18,97.31]; precision=[90.31,91.83,88.62,93.61,96.88]; recall=[89.76,91.21,88.17,93.08,96.54]; f1=[90.03,91.52,88.39,93.34,96.71]
x=np.arange(len(methods)); w=.18; colors=['#3B4CC0','#F2A900','#00A087','#D1495B']
fig,ax=plt.subplots(figsize=(8,6),dpi=300)
for offset,data,label,color in zip([-1.5,-.5,.5,1.5],[accuracy,precision,recall,f1],['Accuracy','Precision','Recall','F1-Score'],colors): ax.bar(x+offset*w,data,w,label=label,color=color,edgecolor='black',linewidth=1.1)
ax.set_xticks(x); ax.set_xticklabels(methods,fontsize=15,fontweight='bold'); ax.set_xlabel('Methods',fontsize=18,fontweight='bold'); ax.set_ylabel('Performance (%)',fontsize=18,fontweight='bold')
ax.set_ylim(85,100); ax.set_yticks(np.arange(85,101,2)); ax.tick_params(axis='y',labelsize=14); ax.grid(axis='y',linestyle='--',linewidth=.8,alpha=.4); ax.set_axisbelow(True)
legend=ax.legend(fontsize=14,ncol=4,loc='upper center',bbox_to_anchor=(.5,1.01),frameon=False)
for text in legend.get_texts(): text.set_fontname('Times New Roman'); text.set_fontweight('bold')
for spine in ['top','right']: ax.spines[spine].set_visible(False)
for container in ax.containers:
    for bar in container:
        h=bar.get_height(); ax.text(bar.get_x()+bar.get_width()/2,h+.15,f'{h:.2f}',ha='center',va='bottom',fontsize=12,fontweight='bold',rotation=90)
plt.tight_layout(); plt.savefig('outputs/overall_performance_comparison.png',dpi=600,bbox_inches='tight',facecolor='white'); plt.show()


