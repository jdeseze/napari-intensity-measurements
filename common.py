# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 19:28:38 2022

@author: Jean
"""

from PIL import Image
import math
import os
from scipy.interpolate import interp1d
import exifread
import numpy as np
import matplotlib.pyplot as plt


class WL:
    def __init__(self,name,step=1):
        self.name=name
        self.step=step

class Exp:

    def __init__(self,expname,wl=[],nbpos=1,nbtime=1,comments=[]):
        self.name=expname
        self.nbpos=nbpos
        self.nbtime=nbtime
        self.wl=wl
        self.nbwl=len(wl)
        self.comments=comments
        if self.nbtime==1:
            self.timestep=0
        else:
            maxwl_ind=min(list(range(self.nbwl)), key=lambda ind:self.wl[ind].step)
            try:
                open(self.get_image_name(maxwl_ind,timepoint=1), 'rb')
                self.stacks=False
            except:
                self.stacks=True
            if self.stacks:
                #print(self.get_stack_name(maxwl_ind))
                self.timestep=2
            else:
                with open(self.get_image_name(maxwl_ind,timepoint=1), 'rb') as opened:
                    tags = exifread.process_file(opened)
                    time_str=tags['Image DateTime'].values
                    h, m, s = time_str.split(' ')[1].split(':')
                    time1=int(h) * 3600 + int(m) * 60 + float(s)
                with open(self.get_image_name(maxwl_ind,timepoint=int((nbtime-1)/self.wl[maxwl_ind].step+1)), 'rb') as opened:
                    tags = exifread.process_file(opened)
                    time_str=tags['Image DateTime'].values
                    h, m, s = time_str.split(' ')[1].split(':')
                    time2=int(h) * 3600 + int(m) * 60 + float(s)
                    self.timestep=(time2-time1)/self.nbtime
    
    #use this if the stack was not build 
    def get_image_name(self,wl_ind,pos=1,timepoint=1,sub_folder=''):
        if self.nbtime==1:
            tpstring=''
        else:
            tpstring='_t'+str(timepoint)
        if self.nbpos==1:
            posstring=''
        else:
            posstring='_s'+str(pos)
        return '\\'.join(self.name.split('/')[0:-1]+[self.name.split('/')[-1]])+'_w'+str(wl_ind+1)+self.wl[wl_ind].name+posstring+tpstring+'.tif'    
        #return self.name+'_w'+str(wl_ind+1)+self.wl[wl_ind].name+posstring+tpstring+'.tif'
   
    #use this if there is only the stack, in the "Stacks" folder
    def get_stack_name(self,wl_ind,pos=1,sub_folder='Stacks'):
        if self.nbpos==1:
            return '\\'.join(self.name.split('\\')[0:-1]+[sub_folder]+[self.name.split('\\')[-1]])+'_w'+str(wl_ind+1)+self.wl[wl_ind].name+'.tif'
        else:
            posstring='_s'+str(pos)
            return '\\'.join(self.name.split('\\')[0:-1]+[sub_folder]+[self.name.split('\\')[-1]])+'_w'+str(wl_ind+1)+self.wl[wl_ind].name+posstring+'.tif'
    
    def get_first_image(self,wl_ind,pos=1,timepoint=''):
        timepoint=1
        if self.stacks:
            I=Image.open(self.get_stack_name(wl_ind,pos))
            I.seek(timepoint)
            return I
        else:
            return Image.open(self.get_image_name(wl_ind,pos,timepoint))
    
    def get_last_image(self,wl_ind,pos=1,timepoint=1):
        last_ind=int(self.nbtime/self.wl[wl_ind].step-1)*self.wl[wl_ind].step+1
        if self.stacks:
            I=Image.open(self.get_stack_name(wl_ind,pos))
            I.seek(timepoint)
            return I
        else:        
            return Image.open(self.get_image_name(wl_ind,pos,last_ind))
    
    def get_sizeimg(self):
        return self.get_first_image(0).size
    
    def disp_message(self):
        return self.get_stack_name(0)
    
class Result:
    def __init__(self, exp,prot,wl_ind,pos,startacq=0,act=[],notact=[],whole=[],whole_surf=[],act_surf=[],background=0):
        self.exp=exp
        self.prot=prot
        self.wl_ind=wl_ind
        self.act=act
        self.notact=notact
        self.whole=whole
        self.channel=self.exp.wl[wl_ind]
        self.pos=pos
        self.startacq=startacq
        self.whole_surf=whole_surf
        self.act_surf=act_surf
        self.background=background
    
    def plot(self,axes,zone='act',plot_options=None):
        toplot=self.get_zone(zone)#running_mean(self.get_zone(zone),4)
        toplot[toplot==0]=math.nan
        
        toplot=(np.array(toplot)-self.background)/(toplot[0]-self.background)
        if not plot_options:
            plot_options={}            
        x=(np.arange(toplot.size))*self.channel.step*self.exp.timestep/60
        axes.plot(x,toplot,**plot_options)
        return x,toplot#go.Scatter(x=x,y=toplot,mode='lines')
    
    def get_abs_val(self,zone='act'):
        toplot=np.array(self.get_zone(zone))
        toplot[toplot==0]=math.nan
        abs_value=np.mean(toplot[0])-self.background      
        return abs_value
    
    def xy2plot(self,zone='act',plot_options=None):
        toplot=self.get_zone(zone)#running_mean(self.get_zone(zone),4)
        toplot[toplot==0]=math.nan
        toplot=(toplot)#-self.background)-(np.mean(toplot[0])-self.background)
        if not plot_options:
            plot_options={}            
        x=(np.arange(toplot.size))*self.channel.step*self.exp.timestep/60
        return x,toplot#go.Scatter(x=x,y=toplot,mode='lines')   
    
    def get_zone(self,zone):
        if zone=='act':
            return np.array(self.act)
        if zone=='notact':
            return np.array(self.notact)
        if zone=='whole':
            return np.array(self.whole)
    
    def name(self):
        return self.exp.name.split('\\')[-1]+' : pos '+str(self.pos)

def get_exp(filename):
    nb_pos=1
    nb_wl=1
    with open(filename,'r') as file:
        i=0
        line=file.readline()
        comments=[]
        iscomments=False
        while not line.rstrip().split(', ')[0]=='"NTimePoints"' and i<50:
            if line.rstrip().split(', ')[0]=='"StartTime1"':
                iscomments=False
            if iscomments:
                comments.append(line.rstrip())
            if line.rstrip().split(', ')[0]=='"Description"':
                iscomments=True
                comments.append(str(line.rstrip().split(', ')[1]))
            line=file.readline()
            i+=1
            
        #get number of timepoints
        nb_tp=int(line.rstrip().split(', ')[1])
        line=file.readline()
        
        #get positions if exist
        if line.split(', ')[1].rstrip('\n')=='TRUE':
            line=file.readline()
            nb_pos=int(line.split(', ')[1].rstrip('\n'))
            for i in range(nb_pos):
                file.readline()            
        file.readline()
        
        #get number of wavelengths
        line=file.readline()
        nb_wl=int(line.rstrip().split(', ')[1])
    
        #create all new wavelengths
        wl=[]
        for i in range (nb_wl):
            line=file.readline()
            wl.append(WL(line.rstrip().split(', ')[1].strip('\"')))
            file.readline()
    
        #change the time steps
        line=file.readline()
        while line.split(', ')[0].strip('\"')=='WavePointsCollected':
            sep=line.rstrip().split(', ')
            if len(sep)>3:
                wl[int(sep[1])-1].step=int(sep[3])-int(sep[2])
            line=file.readline()
        
        expname=filename.rstrip('d').rstrip('n').rstrip('.')
        
        print(str(nb_pos))
        
        return Exp(expname,wl,nb_pos,nb_tp,comments)

class Result_array(list):
    def __init__(self,data):
        list.__init__(self,data)
    
    def plot(self,axes,zone='act',wl_name="TIRF 561",prot=True,plot_options={}):
        [result.plot(axes,zone,plot_options) for result in self if result.channel.name==wl_name and result.prot==prot]    

    
    def xy2plot(self,zone='act',wl_name="TIRF 561",prot=True):
        toplot=[]
        zones=np.array(['act','notact','whole'])
        colors=np.array(['blue','red','green'])
        for res in self:
            if res.channel.name==wl_name and res.prot==prot:
                x,y=res.xy2plot(zone)            
                toplot.append((x,y))
        return toplot
    
    def plot_mean(self,zone='act',wl_name="TIRF 561",prot=True,plot_options={}):
        #time step should be in minutes

        t_start=0
        t_end=min((len(result.get_zone(zone))-1)*result.exp.timestep for result in self if result.channel.name==wl_name)
        nbsteps=min(len(result.get_zone(zone)) for result in self)
        interp=[]
        for result in self: 
            if result.channel.name==wl_name and (not math.isnan(np.sum(result.get_zone(zone)))) and result.prot==prot:
                values=result.get_zone(zone)
                tstep=result.exp.timestep
                normvals=(np.array(result.get_zone(zone))-result.background)/(np.mean(np.array(result.get_zone(zone))[0])-result.background)
                lasttime=len(normvals)*tstep-0.001
                times=np.arange(0,lasttime,tstep)
                
                if sum(np.array(values)==0)>0:
                    f_endtemp=list(values).index(next(filter(lambda x: x==0, values)))
                    normvals=normvals[0:f_endtemp]
                    times=np.arange(0,(f_endtemp)*result.exp.timestep,tstep)
                    if f_endtemp*result.exp.timestep<t_end:
                        t_end=times[-1]

                interp.append(interp1d(times,normvals))
                
        x=np.arange(t_start,t_end,int((t_end-t_start)/nbsteps))
        y=np.vstack([f(x) for f in interp])
        
        ym=np.average(y, axis=0)
        sigma=np.std(y,axis=0)
        
        yh=ym+sigma/(y.shape[0]**0.5)
        yb=ym-sigma/(y.shape[0]**0.5)
        
        #clear_plot(size)
        
        plt.plot(x/60,ym,linewidth=2,**plot_options)

        plt.plot(x/60,yh,linewidth=0.05,**plot_options)
        plt.plot(x/60,yb,linewidth=0.05,**plot_options)
        plt.fill_between(x/60,yh,yb,alpha=0.2,**plot_options)