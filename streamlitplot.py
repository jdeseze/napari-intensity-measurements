# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 14:56:00 2022

@author: Jean
"""

import pickle
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st 
import math
import matplotlib.pyplot as plt
from common import WL,Exp,get_exp,Result,Result_array

st.set_page_config(page_title="plot curves", page_icon=":microscope:",layout="wide")

with open('./results.pkl', 'rb') as output:
    results = pickle.load(output)
    
df=pd.DataFrame(columns=['exp','time','prot','wl_ind','pos','startacq','act','notact','whole','whole_surf','act_surf','init_value_actsurf','init_value_wholesurf','background','init_value'])
for result in results:
    l=len(result.act)
    new_datapoints=pd.DataFrame({'exp':[result.exp.name]*l,
                   'time': (np.arange(int(-result.startacq),int(l-result.startacq)))*result.exp.wl[result.wl_ind].step*0.5,
                   'prot':[result.prot]*l,
                   'wl_ind':[result.exp.wl[result.wl_ind].name]*l,
                   'pos':[result.pos]*l,
                   'startacq':[result.startacq]*l,
                   'act':result.act,
                   'notact':result.notact,
                   'whole':result.whole,
                   'whole_surf':result.whole_surf,
                   'act_surf':result.act_surf,
                   'init_value_actsurf':[np.mean(result.act_surf[0:result.startacq+1])]*l,
                   'init_value_wholesurf':[np.mean(result.whole_surf[0:result.startacq+1])]*l,
                   'background':[result.background]*l,
                   'init_value_act':[np.mean(result.act[0:result.startacq+1])]*l,
                   'init_value_notact':[np.mean(result.notact[0:result.startacq+1])]*l,
                   'init_value_whole':[np.mean(result.whole[0:result.startacq+1])]*l,
        })

    df=pd.concat([df,new_datapoints],ignore_index = True,axis=0)     

df['toplot_act']=(df['act']-df['background'])/(df['init_value_act']-df['background'])
df['toplot_notact']=(df['notact']-df['background'])/(df['init_value_notact']-df['background'])
df['toplot_actsurf']=df['act_surf']-df['init_value_actsurf']

selection = alt.selection_multi(fields=['pos','exp'], bind='legend')


act_561=alt.Chart(df[df['wl_ind']=='TIRF 561']).mark_line(color='red').transform_calculate(
    cat1="datum.exp + '-' + datum.pos"
    ).encode(
    alt.X('time'),
    alt.Y('toplot_act'),
    alt.Detail('cat1:N',title='cat1:N'),
    ).interactive()

notact_561=alt.Chart(df[df['wl_ind']=='TIRF 561']).mark_line().transform_calculate(
    cat1="datum.exp + '-' + datum.pos"
    ).encode(
    alt.X('time'),
    alt.Y('toplot_notact'),
    detail='cat1:N'
    ).interactive()

act_561_mean=alt.Chart(df[df['wl_ind']=='TIRF 561']).mark_line(color='red').transform_calculate(
    cat1="datum.exp + '-' + datum.pos"
    ).encode(
    alt.X('time'),
    alt.Y('mean(toplot_act)')).interactive()
act_561_sem=alt.Chart(df[df['wl_ind']=='TIRF 561']).mark_errorband(extent='ci',color='red').transform_calculate(
    cat1="datum.exp + '-' + datum.pos"
    ).encode(
    alt.X('time'),
    alt.Y('toplot_act')).interactive()
        
notact_561_mean=alt.Chart(df[df['wl_ind']=='TIRF 561']).mark_line().transform_calculate(
    cat1="datum.exp + '-' + datum.pos"
    ).encode(
    alt.X('time'),
    alt.Y('mean(toplot_notact)')).interactive()
notact_561_sem=alt.Chart(df[df['wl_ind']=='TIRF 561']).mark_errorband(extent='ci').transform_calculate(
    cat1="datum.exp + '-' + datum.pos"
    ).encode(
    alt.X('time'),
    alt.Y('toplot_notact')).interactive()
    
act_surf=alt.Chart(df).mark_line(color='red').transform_calculate(
    cat1="datum.exp + '-' + datum.pos"
    ).encode(
    alt.X('time'),
    alt.Y('toplot_actsurf'),
    detail='cat1:N').interactive()
        
mean_act_surf=alt.Chart(df).mark_line(color='red').transform_calculate(
    cat1="datum.exp + '-' + datum.pos"
    ).encode(
    alt.X('time'),
    alt.Y('mean(toplot_actsurf)')).interactive()
sem_act_surf=alt.Chart(df).mark_errorband(extent='ci',color='red').transform_calculate(
    cat1="datum.exp + '-' + datum.pos"
    ).encode(
    alt.X('time'),
    alt.Y('toplot_actsurf')).interactive()    

act_642=alt.Chart(df[df['wl_ind']=='TIRF 642']).mark_line(color='red').transform_calculate(
    cat1="datum.exp + '-' + datum.pos"
    ).encode(
    alt.X('time'),
    alt.Y('toplot_act'),
    detail='cat1:N'
    ).interactive()

notact_642=alt.Chart(df[df['wl_ind']=='TIRF 642']).mark_line().transform_calculate(
    cat1="datum.exp + '-' + datum.pos"
    ).encode(
    alt.X('time'),
    alt.Y('toplot_notact'),
    detail='cat1:N'
    ).interactive()
         

        
'Number of cells :'+str(len(results))
c1,c2,c3,c4=st.columns(4)
#(alt1+alt2).interactive().show()
c1.altair_chart(act_561+notact_561)
c1.altair_chart(act_561_mean+act_561_sem+notact_561_mean+notact_561_sem)
c2.altair_chart(act_642+notact_642)

c3.altair_chart(act_surf)
c3.altair_chart(mean_act_surf+sem_act_surf)

import pickle

[result.pos for result in results]

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
    
    
if st.button('Remove last data'):
    with open(r'.\results.pkl', 'rb') as output:
        results = pickle.load(output)
    temps=results[:-1]
    with open(r'./results.pkl', 'wb') as output:
        pickle.dump(temps, output, pickle.HIGHEST_PROTOCOL)
    with open(r'./results.txt', 'w') as output:
        write_on_text_file(temps, output)
    with open(r'./pdresults.pkl', 'wb') as output:
        write_on_pd(temps, output)
