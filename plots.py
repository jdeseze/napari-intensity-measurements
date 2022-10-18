# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 17:35:26 2022

@author: Jean
"""

import matplotlib.pyplot as plt
import pickle
import numpy as np

with open(r'.\results.pkl', 'rb') as output:
    results = pickle.load(output)


ax = plt.subplot(1, 1, 1)
result = results[-1]
ax.plot(np.array(result.act-result.background) /
        (result.act[0]-result.background))

ax.plot(result.whole_surf[0]*np.array(result.act_surf) /
        (result.act_surf[0]*np.array(result.whole_surf)))

result = results[-2]
ax.plot(np.array(result.act-result.background) /
        (result.act[0]-result.background))
ax.set_yscale('log')

ax1 = plt.subplot(1, 1, 1)
toplot = [result.plot(axes=ax1)
          for result in results if result.channel.name == 'TIRF 561' and not result.prot]
Result_array(toplot).plot_mean()

# %% remove one or two results

results.pop(9)
results.pop(8)
# %%
def write_on_text_file(results,output):
    output.write('Number of datas : '+str(len(results))+' \n')
    output.write('\n')
    for result in results:
        output.write('Experiment: '+str(result.exp.name)+' \n')
        output.write('Position: '+str(result.pos)+' \n')
        output.write('Channel: '+str(result.channel.name)+' \n')
        output.write('Background value: '+str(result.background)+' \n')
        output.write('Activated zone: '+str(result.act)+' \n')
        output.write('Not activated zone: '+str(result.notact)+' \n')
        output.write('Area of the cell: '+str(result.whole_surf))
        output.write('Area of the cell: '+str(result.act_surf))
        output.write('\n')

def write_on_pd(results,output):
    pdresult=pd.DataFrame(columns=['exp','time','prot','wl_ind','pos','startacq','act','notact','whole','whole_surf','act_surf','background','init_value'])
    for result in results:
        l=len(result.act)
        new_datapoints=pd.DataFrame({'exp':[result.exp.name]*l,
                       'time': (np.arange(int(-result.startacq),int(l-result.startacq)))*result.exp.wl[result.wl_ind].step,
                       'prot':[result.prot]*l,
                       'wl_ind':[result.exp.wl[result.wl_ind].name]*l,
                       'pos':[result.pos]*l,
                       'startacq':[result.startacq]*l,
                       'act':result.act,
                       'notact':result.notact,
                       'whole':result.whole,
                       'whole_surf':result.whole_surf,
                       'act_surf':result.act_surf,
                       'background':[result.background]*l,
                       'init_value':np.mean(result.act[0:result.startacq+1]),
            })
        
        pdresult=pd.concat([pdresult,new_datapoints],ignore_index = True,axis=0)    
    pickle.dump(pdresult,output)

from common import Result, Exp, WL
temps=[]
for res in temp:
    #print(res.whole_surf)
    new_res=Result(Exp(res.exp.name,[WL(wl.name,wl.step) for wl in res.exp.wl],res.exp.nbpos,res.exp.nbtime),res.prot,res.wl_ind,res.pos,res.startacq,res.act,res.notact,res.whole,whole_surf=res.whole_surf,act_surf=res.act_surf,background=res.background)
    temps.append(new_res)
    #print(new_res.whole_surf)
with open(r'./results.pkl', 'wb') as output:
    pickle.dump(temps, output, pickle.HIGHEST_PROTOCOL)
with open(r'./results.txt', 'w') as output:
    write_on_text_file(temps, output)
with open(r'./pdresults.pkl', 'wb') as output:
    write_on_pd(temps, output)

# %%


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N


with open(r'F:\optorhoa\220224_optorhoairfp_rbd\220224_rbd_singlecell.pkl', 'rb') as output:
    results = pickle.load(output)

plt.figure()
colors = [list(plt.rcParams['axes.prop_cycle'])[i]['color'] for i in range(10)]
ax1 = plt.subplot(1, 1, 1)
ax1.spines["top"].set_visible(False)
ax1.spines["bottom"].set_visible(True)
ax1.spines["right"].set_visible(False)
ax1.spines["left"].set_visible(True)
plt.style.use('dark_background')
ax1.get_xaxis().tick_bottom()
ax1.get_yaxis().tick_left()

for result in [results[10]]:
    y = running_mean((np.array(result.act)-result.background) /
                     (result.act[15]-result.background), 1)
    x = (np.arange(len(y))-15)*2
    ax1.plot(x, y, color=colors[0])
    y = running_mean((np.array(result.notact)-result.background) /
                     (result.notact[16]-result.background), 1)
    ax1.plot(x, y, color=colors[1])
    y = running_mean((np.array(result.act_surf))/(result.act_surf[15]), 1)
    #ax1.plot(x, y, color=colors[5])
plt.plot([0, 0], [0.3, 10], '--', color=colors[3])
plt.ylim(0.8, 1.2)
plt.xlim(-20, 100)
plt.xticks(np.arange(-20,100,10))
#ax1.set_yscale ('log')

# =============================================================================
# plt.figure()
# ax2=plt.subplot(1,1,1)
# [ax2.plot((np.array(result.act_surf)-result.act_surf[0]),color=colors[0]) for result in results if result.channel.name=='TIRF 561']
#
#
# =============================================================================
# %% MRLC


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N

# =============================================================================
# with open(r'./results.pkl','rb') as output:
#     res=pickle.load(output)
# =============================================================================
# =============================================================================
# with open(r'F:\optorhoa\220310_optorhoa_mrlc\220310_mrlc.pkl','rb') as output:
#     results+=pickle.load(output)
# =============================================================================

plt.figure()
colors = [list(plt.rcParams['axes.prop_cycle'])[i]['color'] for i in range(10)]
ax1 = plt.subplot(1, 1, 1)
ax1.spines["top"].set_visible(False)
ax1.spines["bottom"].set_visible(True)
ax1.spines["right"].set_visible(False)
ax1.spines["left"].set_visible(True)
plt.style.use('dark_background')
ax1.get_xaxis().tick_bottom()
ax1.get_yaxis().tick_left()

smooth_odd = 1
y1, y2, y3 = [], [], []
for i,result in enumerate(results):
    if (result.channel.name == 'TIRF 642' and result.prot) or (result.channel.name == 'TIRF 642' and i>15):# and 'cell10' in result.exp.name:
        y1 += [running_mean((np.array(result.act)-result.background) /
                            (np.mean(result.act[0:15])-result.background), smooth_odd)]
        x = (np.arange(len(y1[0]))-15+((smooth_odd-1)/2))*2
        # ax1.plot(x,y,color=colors[0])
        y2 += [running_mean((np.array(result.notact)-result.background) /
                            (np.mean(result.notact[0:15])-result.background), smooth_odd)]
        # ax1.plot(x,y,color=colors[1])
        # y=running_mean((np.array(result.act_surf))/(result.act_surf[15]),1)
        # ax1.plot(x,y,color=colors[5])
        y3 += [running_mean((np.array(result.act_surf)) /
                            (result.act_surf[15]), smooth_odd)]

[ax1.plot(x, y, color=colors[0], linewidth=0.5) for y in y1]
ax1.plot(x, np.mean(np.vstack(y1), axis=0), color=colors[0], linewidth=5)
[ax1.plot(x, y, color=colors[1], linewidth=0.5) for y in y2]
ax1.plot(x, np.mean(np.vstack(y2), axis=0), color=colors[1], linewidth=5)
#[ax1.plot(x, y, color=colors[2], linewidth=0.5) for y in y3]
#ax1.plot(x, np.mean(np.vstack(y3), axis=0), color=colors[2], linewidth=5)

plt.plot([0, 0], [0.3, 10], '--', color=colors[3])
plt.ylim(0.8,1.2)
plt.xlim(-30, 100)
plt.xticks(np.arange(-30,100,10))
plt.xlim(-30,100)
# %%

with open('./results.pkl', 'rb') as output:
    results = pickle.load(output)


plt.figure()
colors = [list(plt.rcParams['axes.prop_cycle'])[i]['color'] for i in range(10)]
ax1 = plt.subplot(1, 1, 1)
ax1.spines["top"].set_visible(False)
ax1.spines["bottom"].set_visible(True)
ax1.spines["right"].set_visible(False)
ax1.spines["left"].set_visible(True)
plt.style.use('dark_background')
ax1.get_xaxis().tick_bottom()
ax1.get_yaxis().tick_left()

smooth_odd = 1
c = 0

T561 = []
T642 = []
for result in results:
    if result.channel.name == 'TIRF 561':
        T561 += [np.mean(result.act[0:10])-result.background]

    if result.channel.name == 'TIRF 561':
        y = running_mean((np.array(result.act)-result.background) /
                         (result.act[0]-result.background), smooth_odd)
        x = (np.arange(len(result.act)-(smooth_odd-1))-0+((smooth_odd-1)/2))/2
        ax1.plot(x, y, label=str(result.pos), color=colors[int(c)])
        T642 += [y[13]]
        y = running_mean((np.array(result.act_surf))/(result.act_surf[7]), 1)
        ax1.plot(x, y, label=str(result.pos), color=colors[int(c)+5])
    if result.channel.name == 'TIRF 642':
        y = running_mean((np.array(result.act)-result.background) /
                         (result.act[0]-result.background), smooth_odd)
        x = (np.arange(len(result.act)-(smooth_odd-1))-0+((smooth_odd-1)/2))/1
        ax1.plot(x, y, label=str(result.pos), color=colors[int(c)])
        T642 += [y[13]]
        y = running_mean((np.array(result.act_surf))/(result.act_surf[7]), 1)

        # T642+=[y[5]]
    c += 0.5
# plt.scatter(T561,T642)
#ax1.set_xscale ('log')
# plt.legend()
plt.xlim(0, 35)
plt.ylim(0.8,)

# %% compare means results_cell3+cellgood.nd 210915

from scipy.signal import savgol_filter

def running_mean(x, N,cut):

    return x#np.append(x[0:cut],savgol_filter(x[cut:], 9, 3)) # window size 51, polynomial order 3

# =============================================================================
# with open(r'F:\optorhoa\210915_RPE_optorhoairfp+rbd\results cell3.pkl','rb') as output:
#     results = pickle.load(output)
# =============================================================================
with open(r'results_cell3+cellgood.pkl', 'rb') as output:
    results = pickle.load(output)
# =============================================================================
# with open(r'results.pkl', 'rb') as output:
#     results = pickle.load(output)
# =============================================================================

plt.figure()
colors = [list(plt.rcParams['axes.prop_cycle'])[i]['color'] for i in range(10)]
ax1 = plt.subplot(1, 1, 1)
ax1.spines["top"].set_visible(False)
ax1.spines["bottom"].set_visible(True)
ax1.spines["right"].set_visible(False)
ax1.spines["left"].set_visible(True)
plt.style.use('dark_background')
ax1.get_xaxis().tick_bottom()
ax1.get_yaxis().tick_left()

smooth_odd = 1
y1, y2, y3 , y4= [], [], [],[]
a,b=[],[]
for result in results:
    if result.channel.name == 'TIRF 642' and not result.prot:
        cut=3
        if 'cell3' in result.exp.name:
            y1+=[running_mean((np.array(result.act)-result.background)/(np.mean(result.act[0:4])-result.background), smooth_odd,cut)]
            y3+=[running_mean((np.array(result.notact)-result.background)/(np.mean(result.notact[0:4])-result.background), smooth_odd,cut)]

            #y1 += [running_mean((np.array(result.act)-np.mean(result.act[0:4]))/(
            #    np.mean(result.act[5:20])-np.mean(result.act[0:4])), smooth_odd,cut)]
            #y3 += [running_mean((np.array(result.act_surf)) /
            #                    (result.act_surf[3]), smooth_odd,cut)]
        else:
            y1+=[running_mean((np.array(result.act)-result.background)/(np.mean(result.act[0:4])-result.background), smooth_odd,cut)]
            y3+=[running_mean((np.array(result.notact)-result.background)/(np.mean(result.notact[0:4])-result.background), smooth_odd,cut)]

            #y1 += [running_mean((np.array(result.act[0:30])-np.mean(result.act[0:4]))/(
            #    np.mean(result.act[5:20])-np.mean(result.act[0:4])), smooth_odd,cut)]
            #y3 += [running_mean((np.array(result.act_surf[0:30])) /
            #                    (result.act_surf[3]), smooth_odd,cut)]
        a+=[np.mean(result.act[0:4])-result.background]
    if result.channel.name == 'TIRF 561' and result.prot:
        cut=6
        if 'cell3' in result.exp.name:
            #y2 += [running_mean((np.array(result.act)-np.mean(result.act[0:7]))/(np.mean(result.act[30:40])-np.mean(result.act[0:7])), smooth_odd,cut)]
            y2+=[running_mean((np.array(result.act)-result.background)/(np.mean(result.act[0:7])-result.background), smooth_odd,cut)]
            y4+=[running_mean((np.array(result.notact)-result.background)/(np.mean(result.notact[0:7])-result.background), smooth_odd,cut)]

        else:
            #y2 += [running_mean((np.array(result.act[0:61])-np.mean(result.act[0:7]))/(np.mean(result.act[30:40])-np.mean(result.act[0:7])), smooth_odd,cut)]
            y2+=[running_mean((np.array(result.act[0:61])-result.background)/(np.mean(result.act[0:7])-result.background), smooth_odd,cut)]
            ya+=[running_mean((np.array(result.notact[0:61])-result.background)/(np.mean(result.notact[0:7])-result.background), smooth_odd,cut)]

        b+=[y2[-1][10]-y2[-1][6]]

# =============================================================================
# x = (np.arange(len(y1[0]))-3)/1
# [ax1.plot(x, y, color=colors[0], linewidth=0.5) for y in y1]
# ax1.plot(x, np.mean(np.vstack(y1), axis=0), color=colors[0], linewidth=5)
# 
# =============================================================================
x = (np.arange((len(y2[0])-(smooth_odd-1)))-6)/2
[ax1.plot(x, y, color=colors[1], linewidth=0.5) for y in y2]
ax1.plot(x, np.mean(np.vstack(y2), axis=0), color=colors[1], linewidth=5)
[ax1.plot(x, y, color=colors[2], linewidth=0.5) for y in y4]
ax1.plot(x, np.mean(np.vstack(y4), axis=0), color=colors[2], linewidth=5)

#x = (np.arange((len(y3[0])-(smooth_odd-1)))-3)/1
#[ax1.plot(x,y,color=colors[2],linewidth=0.5) for y in y3]
# ax1.plot(x,np.mean(np.vstack(y3),axis=0),color=colors[2],linewidth=5)

plt.plot([0, 0], [-0.5, 1.5], '--', color=colors[3])
plt.ylim(-0.3, 4)
plt.xlim(-3, 3)
plt.xticks(np.arange(-2,3,1))

# =============================================================================
# plt.figure()
# ax2=plt.subplot(1,1,1)
# plt.figure()
# ax2.scatter(a,b)
# =============================================================================
#ax2.set_xscale('log')

# %%
# =============================================================================
# import altair as alt
#
# with open('./pdresults.pkl', 'rb') as output:
#     df=pickle.load(output)
#
# a=alt.Chart(df)
# a.mark_line().transform_calculate(
#     cat="datum.exp + '-' + datum.pos + '-' + datum.wl_ind"
# ).encode(
#     x='time:Q',
#     y='act:Q',
#     color='cat:N'
#     ).interactive().show()
# =============================================================================
import pandas as pd
import seaborn as sns
import altair as alt
def add_to_pd(typ,df,st):
    prot=[i for i in typ if i.prot and i.channel.name=='TIRF 561']
    cont=[i for i in typ if not i.prot and i.channel.name=='TIRF 561']
    prot642=[i for i in typ if i.prot and i.channel.name=='TIRF 642']
    cont642=[i for i in typ if not i.prot and i.channel.name=='TIRF 642']
    protint=[res.whole[0]-res.background+1 for res in prot]
    contint=[res.whole[0]-res.background+1 for res in cont]
    prot642int=[res.whole[0]-res.background+1 for res in prot642]
    cont642int=[res.whole[0]-res.background+1 for res in cont642]
    df1 = pd.DataFrame(list(zip([st]*(len(prot)+len(cont)),['protrusion']*len(prot)+['contraction']*len(cont), protint+contint,prot642int+cont642int)),columns =['Type','Phenotype', 'GEF intensity','642 intensity'])
    
    return pd.concat([df, df1], ignore_index = True, axis = 0)

plt.figure()
with open(r'F:\optorhoa\210708_RPE1_optoRhoa_ARHGEF11iRFP\210713optoRhoaarhgef11irfp_obj.pkl','rb') as output:
    arh=pickle.load(output)    

df=pd.DataFrame([],columns =['Type','Phenotype', '561 intensity','642 intensity'])
df=add_to_pd(arh,df,'arh')
df=add_to_pd(results,df,'arh2')

sw=sns.scatterplot(x="GEF intensity",y='642 intensity',hue='Phenotype',data=df[df['Type']=='arh'],palette=['blue','red'])
#sw.legend_.remove()
plt.rcParams["font.family"] = "Arial"
sw.set(xlabel='GEF')
sw.set(ylabel='arh')
plt.rcParams["font.size"]=18
sw.spines["top"].set_visible(False)    
sw.spines["bottom"].set_visible(True)    
sw.spines["right"].set_visible(False)    
sw.spines["left"].set_visible(True)  
alt.Chart(df[df['Type']=='arh']).mark_circle(size=60).encode(
    alt.X("GEF intensity",scale=alt.Scale(type='log')),
    alt.Y('642 intensity',scale=alt.Scale(type='log')),
    color='Phenotype'
    ).interactive().show()

#%%

results['toplot']=(results['act']-results['background'])/(results['act']-results['background'])

alt.Chart(results[results['wl_ind']=='TIRF 561']).mark_line().transform_calculate(
    cat1="datum.exp + '-' + datum.pos"
    ).encode(
    alt.X('time'),
    alt.Y('toplot'),
    color='cat1:N').interactive().show()

results['toplot']=results['act']-results['background']

# %% compare means results_cell3+cellgood.nd 210915

from scipy.signal import savgol_filter

def running_mean(x, N,cut):

    return x#np.append(x[0:cut],savgol_filter(x[cut:], 9, 3)) # window size 51, polynomial order 3

# =============================================================================
# with open(r'F:\optorhoa\210915_RPE_optorhoairfp+rbd\results cell3.pkl','rb') as output:
#     results = pickle.load(output)
# =============================================================================
with open(r'results_cell3+cellgood.pkl', 'rb') as output:
    results = pickle.load(output)
# =============================================================================
# with open(r'results.pkl', 'rb') as output:
#     results = pickle.load(output)
# =============================================================================

plt.figure()
colors = [list(plt.rcParams['axes.prop_cycle'])[i]['color'] for i in range(10)]
ax1 = plt.subplot(1, 1, 1)
ax1.spines["top"].set_visible(False)
ax1.spines["bottom"].set_visible(True)
ax1.spines["right"].set_visible(False)
ax1.spines["left"].set_visible(True)
plt.style.use('dark_background')
ax1.get_xaxis().tick_bottom()
ax1.get_yaxis().tick_left()

smooth_odd = 1
y1, y2, y3 , y4= [], [], [],[]
a,b=[],[]
for i,j in enumerate(results[results['time']==0]['act'].index):
    if j==1180:
        df=results.iloc[j:]
    else:
        df=results.iloc[j:results[results['time']==0]['act'].index[i+1]]
    act,notact,surf,background=list(df['act']),list(df['notact']),list(df['act_surf']),list(df['background'])[0]
    
    if list(df['wl_ind'])[0] == 'TIRF 642' and list(df['prot'])[0]==True:
        
        cut=3
        if 'cell3' in list(df['exp'])[0]:
            y1+=[running_mean((np.array(act)-background)/(np.mean(act[0:4])-background), smooth_odd,cut)]
            y3+=[running_mean((np.array(notact)-background)/(np.mean(notact[0:4])-background), smooth_odd,cut)]

            #y1 += [running_mean((np.array(act)-np.mean(act[0:4]))/(
            #    np.mean(act[5:20])-np.mean(act[0:4])), smooth_odd,cut)]
            #y3 += [running_mean((np.array(act_surf)) /
            #                    (act_surf[3]), smooth_odd,cut)]
        else:
            y1+=[running_mean((np.array(act[1:31])-background)/(np.mean(act[0:4])-background), smooth_odd,cut)]
            y3+=[running_mean((np.array(notact[1:31])-background)/(np.mean(notact[0:4])-background), smooth_odd,cut)]

            #y1 += [running_mean((np.array(act[0:30])-np.mean(act[0:4]))/(
            #    np.mean(act[5:20])-np.mean(act[0:4])), smooth_odd,cut)]
            #y3 += [running_mean((np.array(act_surf[0:30])) /
            #                    (act_surf[3]), smooth_odd,cut)]
        a+=[np.mean(act[0:4])-background]
    if list(df['wl_ind'])[0] == 'TIRF 561':
        cut=6
        if 'cell3' in list(df['exp'])[0]:
            if list(df['pos'])[0]==4:
                #y2 += [running_mean((np.array(act)-np.mean(act[0:7]))/(np.mean(act[30:40])-np.mean(act[0:7])), smooth_odd,cut)]
                y2+=[running_mean((np.array(act)-background)/(np.mean(act[0:7])-background), smooth_odd,cut)]
                y4+=[running_mean((np.array(notact)-background)/(np.mean(notact[0:7])-background), smooth_odd,cut)]

        elif list(df['pos'])[0]==4:
            #y2 += [running_mean((np.array(act[0:61])-np.mean(act[0:7]))/(np.mean(act[30:40])-np.mean(act[0:7])), smooth_odd,cut)]
            y2+=[running_mean((np.array(act[2:63])-background)/(np.mean(act[0:9])-background), smooth_odd,cut)]
            y4+=[running_mean((np.array(notact[2:63])-background)/(np.mean(notact[0:9])-background), smooth_odd,cut)]

        b+=[y2[-1][10]-y2[-1][6]]

# =============================================================================
# x = (np.arange(len(y1[0]))-3)/1
# [ax1.plot(x, y, color=colors[0], linewidth=0.5) for y in y1]
# ax1.plot(x, np.mean(np.vstack(y1), axis=0), color=colors[0], linewidth=5)
# 
# =============================================================================
x = (np.arange((len(y2[0])))-6)/2
[ax1.plot(x, y, color=colors[1], linewidth=0.5) for y in y2]
ax1.plot(x, np.mean(np.vstack(y2), axis=0), color=colors[1], linewidth=5)
[ax1.plot(x, y, color=colors[2], linewidth=0.5) for y in y4]
ax1.plot(x, np.mean(np.vstack(y4), axis=0), color=colors[2], linewidth=5)

#x = (np.arange((len(y3[0])-(smooth_odd-1)))-3)/1
#[ax1.plot(x,y,color=colors[2],linewidth=0.5) for y in y3]
# ax1.plot(x,np.mean(np.vstack(y3),axis=0),color=colors[2],linewidth=5)

plt.plot([0, 0], [-0.5, 1.5], '--', color=colors[3])
plt.ylim(0.8, 2)
plt.xlim(-2, 1)
plt.xticks(np.arange(-3,3,1))

# =============================================================================
# plt.figure()
# ax2=plt.subplot(1,1,1)
# plt.figure()
# ax2.scatter(a,b)
# =============================================================================
#ax2.set_xscale('log')
