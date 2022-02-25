# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 16:37:42 2021

@author: Jean
"""

# =============================================================================
# from skimage import data
# import napari
# from napari.settings import SETTINGS
# 
# SETTINGS.reset()
# viewer = napari.view_image(data.astronaut(), rgb=True)
# =============================================================================

#%%

import napari
#from dask_image.imread import imread
from glob import glob
from skimage.io import imread
from skimage.io.collection import alphanumeric_key
from dask import delayed
import dask.array as da
#import dask_image
import numpy as np
import exifread

def read_stack(filenames):
    sample = imread(filenames[0])
    
    lazy_imread = delayed(imread)  # lazy reader
    lazy_arrays = [lazy_imread(fn) for fn in filenames]
    dask_arrays = [
        da.from_delayed(delayed_reader, shape=sample.shape, dtype=sample.dtype)
        for delayed_reader in lazy_arrays
    ]
    # Stack into one large dask.array
    stack = da.stack(dask_arrays, axis=0)
    stack.shape  # (nfiles, nz, ny, nx)
    return stack

filenames = sorted(glob(r"D:\optorhoa\201208_RPE_optoRhoA_PAKiRFP\cell2s_50msact_1_w2TIRF 561_t*.tif"),key=alphanumeric_key)
stack=read_stack(filenames)
viewer=napari.view_image(stack, contrast_limits=[0,2000],name='561')  
viewer.layers[-1].reset_contrast_limits()

filenames = sorted(glob(r"D:\optorhoa\201208_RPE_optoRhoA_PAKiRFP\cell2s_50msact_1_w3TIRF 642_t*.tif"),key=alphanumeric_key)
stack=read_stack(filenames)
viewer.add_image(stack, contrast_limits=[0,2000],name='642')  
viewer.layers[-1].reset_contrast_limits()

filenames = sorted(glob(r"D:\optorhoa\201208_RPE_optoRhoA_PAKiRFP\cell2s_50msact_1_w1TIRF DIC_t*.tif"),key=alphanumeric_key)
stack=read_stack(filenames)
viewer.add_image(stack, contrast_limits=[199,200],name='DIC')  
viewer.layers[-1].reset_contrast_limits()

viewer.reset_view()
viewer.grid.enabled=True
#%% CLASSES
from PIL import Image
import math
import os
from scipy.interpolate import interp1d

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
        self.commments=comments
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
    def __init__(self, exp,prot,wl_ind,pos,startacq=0,act=[],notact=[],whole=[],background=0):
        self.exp=exp
        self.prot=prot
        self.wl_ind=wl_ind
        self.act=act
        self.notact=notact
        self.whole=whole
        self.channel=self.exp.wl[wl_ind]
        self.pos=pos
        self.startacq=startacq
        self.background=background
    
    def plot(self,axes,zone='act',plot_options=None):
        toplot=self.get_zone(zone)#running_mean(self.get_zone(zone),4)
        toplot[toplot==0]=math.nan
        toplot=(np.array(toplot)-self.background)-(toplot[0]-self.background)
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
#%% STACK VIEWER
from qtpy.QtWidgets import QPushButton,QLabel,QComboBox,QFileDialog,QWidget,QMessageBox,QMainWindow,QVBoxLayout
import magicgui

class StackViewer(QWidget):
    def __init__(self,viewer):
        super().__init__()
        
        self.viewer=viewer
        self.exp=None
        
        #set the vertical layout
        self.setLayout(QVBoxLayout())
        
        #add the 'Select Folder' button
        btn = QPushButton(self)
        btn.setText('Select Folder')
        self.layout().addWidget(btn)
        btn.clicked.connect(self.choose_exp)
        
        #add the dropdown list to select the experiment
        self.list_exp=QComboBox(self)
        self.layout().addWidget(self.list_exp)
        self.list_exp.currentTextChanged.connect(self.get_exp)
        
        #add the dropdown button to select the cell 
        self.cell_nb=QComboBox(self.list_exp)
        self.layout().addWidget(self.cell_nb)
        
        load=QPushButton(self)
        load.setText('Load experiment')
        self.layout().addWidget(load)
        load.clicked.connect(self.display_exp)
        
    #function to choose the directory and have the experiments in the folder displayed
    def choose_exp(self):
        dbox = QFileDialog(self)
        dbox.setDirectory('F:/optorhoa')   
        dbox.setFileMode(QFileDialog.Directory)          
        if dbox.exec_():
            self.folder = dbox.selectedFiles()
        for folder_path in self.folder:
             filenames=[f for f in os.listdir(folder_path) if f.endswith('.nd')]  
        self.list_exp.clear()
        self.list_exp.addItems(filenames)
        
    def get_exp(self):
        try:
            print(self.folder[0])
            self.exp=get_exp(os.path.join(self.folder[0],self.list_exp.currentText()))
        except:
            mess=QMessageBox(self)
            mess.setText('Unable to load experiment')
            self.layout().addWidget(mess)
        self.cell_nb.clear()
        if self.exp:
            self.cell_nb.addItems(list(map(str,range(1,self.exp.nbpos+1))))
            print(self.cell_nb)
            return self.exp
        else:
            print('no experiment was found')
            return False
        
    def display_exp(self):
        for layer in self.viewer.layers:
            if type(layer)==napari.layers.shapes.shapes.Shapes:
                shape=layer
        self.viewer.layers.clear()
        try:
            print('ok')
            self.viewer.add_layer(shape)
        except:
            pass
        
        #separate openings if it is only as a stack or not
        try:
            pos=int(self.cell_nb.currentText())
        except:
            pos=1
        #print(self.exp.nbpos)    
        for i in range(len(self.exp.wl)):
            if self.exp.stacks:
                filename=self.exp.get_stack_name(i,pos)
                #print(filename)
                lazy_imread = delayed(imread)
                self.viewer.add_image(imread(filename),name=self.exp.wl[i].name)

            else:
                filenames=sorted(glob(self.exp.get_image_name(i,pos,'*')),key=alphanumeric_key)
                stack=read_stack(filenames)
                self.viewer.add_image(stack,contrast_limits=[0,2000],name=self.exp.wl[i].name)
                #set contrast limits automatically
                self.viewer.layers[-1].reset_contrast_limits()
        
        
        self.viewer.reset_view()
        self.viewer.grid.enabled=True
        #for i in range(len(exp.wl)):

if __name__ == "__main__":
    w = StackViewer(viewer)
    
    w.resize(300,300)
    w.setWindowTitle('Select experiment')
    w.show()
    viewer.window.add_dock_widget(w)
    
#%% SEGMENTER

from magicgui import magicgui
from scipy import ndimage
from skimage import measure, filters
import matplotlib.pyplot as plt

def segment_threshold(img,thresh=1.0):
    #img=(img/2^8).astype(np.uint8)
    binary = img > thresh
    dil=ndimage.binary_dilation(binary,iterations=2)
    filled=ndimage.binary_fill_holes(dil).astype(int)
    label_img, cc_num = ndimage.label(filled)
    #CC = ndimage.find_objects(label_img)
    cc_areas = ndimage.sum(filled, label_img, range(cc_num+1))
    area_mask = (cc_areas < max(cc_areas))
    label_img[area_mask[label_img]] = 0
    try:
        contours = measure.find_contours(label_img, 0.8)
    except:
        contours=[]
    if len(contours)>0:
        contour=contours[0]
    else:
        contour=np.array([None])
    #return (label_img>0)*255, contour
# =============================================================================
#     fig, ax = plt.subplots()
#     ax.imshow(img, cmap=plt.cm.gray)
#     ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
#     ax.set_xticks([])
#     ax.set_yticks([])
# =============================================================================
    return (label_img>0)*255#np.array(fig)>0

@magicgui(call_button='Segment',
          coeff={"widget_type": "FloatSlider",'min':0.5, 'max': 1.5,'step':0.01},
          )
def segment(data:napari.types.ImageData,coeff=1.0):
    med=filters.median(data[0])
    pre_thresh=filters.threshold_otsu(med)
    #print(type(data))
    if type(data)==np.ndarray:
        print('np.ndarray, it cannot be segmented like this')
        data=da.from_array(data)
    segmented = data.map_blocks(segment_threshold,coeff*pre_thresh)
    #print(data[0])
    viewer.add_image(segmented,name='segmented',opacity=0.2)

@segment.coeff.changed.connect
def change_seg(new_coeff:int):
    viewer.layers.remove('segmented')
    segment()


if __name__ == "__main__":
    viewer.window.add_dock_widget(segment)
    
#%% Intensity calculator
import pandas as pd
import pickle
import time
from magicgui.widgets import Label

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
        output.write('\n')

def calculate_intandsurf(imgs,mask_seg,inds_mask_act):
    return []

try:
    exp_layers=[layer for layer in viewer.layers if layer.name in [wl.name for wl in w.get_exp().wl]]
except:
    exp_layers=viewer.layers
@magicgui(call_button='Calculate intensities',
          layer_meas={"widget_type": "RadioButtons", "choices":exp_layers,'orientation':'horizontal','allow_multiple':True},
          )
def calculate_intensities(
        layer_meas: napari.types.ImageData,
        mask_act : napari.types.ShapesData,
        prot=False,
        new_file=False):
    ta=time.time()
    exp=w.get_exp()
    if not exp:
        print('no experiment')
        return
    pos=int(w.cell_nb.currentText())
    print("position"+str(pos))
    wl_meas=next(i for i, v in enumerate(exp.wl) if v.name==layer_meas.name)
    wl_seg=next(i for i, v in enumerate(exp.wl) if v.name in segment.data.current_choice)
    whole, act, notact=[],[],[]
    result=Result(exp,prot,wl_meas,pos)
    stepmeas=exp.wl[wl_meas].step
    nb_img=int(exp.nbtime/stepmeas)
    stepseg=exp.wl[wl_seg].step      
    '''find indice of the area where the activation was done'''
    inds_mask_act=(tuple(slice(int(i[0]), int(i[1])) for i in np.array([mask_act[0][0][1:],mask_act[0][2][1:]]).T))
    tb=time.time()
    '''this is what takes the longest time because it needs to do the segmentation'''
    try:
        mask_np=np.array(viewer.layers['segmented'].data)
    except:
        print('no layer named "segmented", you should create one')
        return
    layer_meas_np=np.array(layer_meas.data)
    #print(nb_img)
    for i in range(nb_img):
        '''define the good frame to take for each segmentation or activation'''
        timg=i*stepmeas
        tseg=int(i*stepmeas/stepseg)*stepseg
        #print(tseg)
        '''take mask of segmentation'''
        mask_seg=mask_np[tseg,:,:]>0
        '''take values in current image'''
        #print(time.time()-tb)
        img=layer_meas_np[timg,:,:]
        #print(time.time()-tb)
        #img=np.array(Image.open(exp.get_image_name(wl_meas,pos,timg)))
        '''calculate whole intensity'''
        whole_int=img[mask_seg].sum()
        '''calculate whole surface'''
        whole_surf=mask_seg.sum()
        '''calculate the intensity in the area of activation'''
        act_int=(img[inds_mask_act][mask_seg[inds_mask_act]]).sum()
        '''calculate the area of activation'''
        act_surf=(mask_seg[inds_mask_act]).sum()
        if i==0:
            background=np.mean(img[0:20,0:20])
        whole.append(whole_int/whole_surf)
        notact.append((whole_int-act_int)/(whole_surf-act_surf))
        if act_surf==0:
            act.append(0)
        else:
            act.append(act_int/act_surf)
    print(time.time()-tb)
    result.whole, result.act, result.notact, result.background =whole,act,notact,background  
    #put into pd dataframe
# =============================================================================
#     current_resultpd=pd.DataFrame(np.array([npwhole,npact,npnotact]).transpose())
#     resultspd=pd.read_pickle('./pdresults.pkl')
#     new_resultspd=pd.concat([resultspd,current_resultpd], ignore_index=True, axis=1)
#     new_resultspd.to_pickle('./pdresults.pkl')
# =============================================================================
    #add to the list of results
    if new_file:
        results=[result]
    else:
        with open('./results.pkl', 'rb') as output:
            results=pickle.load(output)
        results.append(result)
    with open('./results.pkl', 'wb') as output:
        pickle.dump(results, output, pickle.HIGHEST_PROTOCOL)
    with open('./results.txt','w') as output:
        write_on_text_file(results,output)



    
if __name__ == "__main__":
    viewer.window.add_dock_widget(calculate_intensities)



#%% PLOT VALUES
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvas
import altair as alt
import pandas as pd

df=pd.DataFrame([[1,2],[4,5]],columns=['a','b'])

@magicgui(call_button='Plot values')
def plot_values():
    with open('./results.pkl', 'rb') as output:
        results=pickle.load(output)
    #Result_array(results).plot()
    fig=Figure()
    canvas=FigureCanvas(fig)
    ax=fig.add_subplot(111)
    #ax.plot(results[0].act)
    Result_array(results).plot(axes=ax,plot_options={'color':'blue'})
    Result_array(results).plot(axes=ax,zone='notact',plot_options={'color':'red'})
    viewer.window.add_dock_widget(canvas)
    
    alt.Chart(df).mark_circle(size=60).encode(
        alt.X('a',scale=alt.Scale(type='log')),
        alt.Y('b',scale=alt.Scale(type='log')),
        ).interactive().show()

if __name__ == "__main__":
    viewer.window.add_dock_widget(plot_values)
#%%
# =============================================================================
# 
# from magicgui import magicgui
# import os
# import pathlib
# from skimage.io import imread
# from skimage.io.collection import alphanumeric_key
# from dask import delayed
# from magicgui import widgets
# 
# @magicgui(foldername={"mode": "d"},
#           call_button='Load experiment',
#           )
# def folder_picker(foldername=pathlib.Path(r"F:/optorhoa")):
#     
#     print('Experiment loaded')
#     filepath=os.path.join(folder_picker.foldername.value,folder_picker.filename.value)
#     try:
#         exp=get_exp(filepath)
#     except:
#         print('unable to load experiment')
#     if expnumber==-1:
#         print('choose a valid experiment number')
#     elif exp:
#         #print("experiment loaded :"+str(filepath))
#         viewer.layers.clear()
#         for i in range(len(exp.wl)):
#             if exp.stacks:
#                 stackname=exp.get_stack_name(i,expnumber)
#                 viewer.add_image(imread(stackname),name=exp.wl[i].name)
#             else:
#                 files=sorted(glob(exp.get_image_name(i,expnumber,'*')),key=alphanumeric_key)
#                 #print(files)
#                 stack=read_stack(files)
#                 viewer.add_image(stack,contrast_limits=[0,2000],name=exp.wl[i].name)
#                 #set contrast limits automatically
#                 viewer.layers[-1].reset_contrast_limits()    
#         viewer.reset_view()
#         viewer.grid.enabled=True    
#     else:
#         print('errooor')
# 
# #when the folder name changes, we need to change the list of possible files
# @folder_picker.foldername.changed.connect
# def foldername_callback(new_foldername: pathlib.Path):
#     print('new folder is : '+str(new_foldername))
#     new_filenames=['']+[f for f in os.listdir(new_foldername) if f.endswith('.nd')]
#     folder_picker.filename.choices=new_filenames
#     
#     #folder_picker.show()
# 
# #when the filename changes, we need to change the experiment numbers
# @folder_picker.filename.changed.connect
# def filename_callback(new_filename: str):
#     print('new filename : '+str(os.path.join(folder_picker.foldername.value,new_filename)))
#     try:
#         folder=folder_picker.foldername.value
#         exp=get_exp(os.path.join(folder,new_filename))
#         print('new experiment loaded, '+str(exp.nbpos)+' different positions')
#         folder_picker.expnumber.choices=[-1]+list(map(str,range(1,exp.nbpos+1))) 
#     except:
#         print('error: not able to load experiment')
# 
# viewer.window.add_dock_widget(folder_picker)
# =============================================================================

#%%
# =============================================================================
# import pathlib 
# from magicgui import widgets,magicgui
# 
# methods=[]
# 
# class Container(widgets.Container):
#     def __init__(self):
#         self.folder=widgets.FileEdit(mode= "d",value='F:\optorhoa')
#         self.file=widgets.ComboBox(choices=[''])
#         self.num_exp=widgets.ComboBox(choices=[''])
#         self.widgets=[self.folder,self.file,self.num_exp]
#         self.native.layout().addStretch()
# 
# container_exp_choice=Container()
# folder=container_exp_choice.folder
# file=container_exp_choice.file
# num_exp=container_exp_choice.num_exp
# 
# # when the folder changes, populate the container with a widget showing the .nd files
# @folder.changed.connect
# def list_file(new_folder:pathlib.Path):
#     #while len(container_exp_choice) > 1:
#         #container_exp_choice.pop(-1).native.close()
#     list_file=[f for f in os.listdir(new_folder) if f.endswith('.nd')]
#     
#     #container_exp_choice.append(file)
#             
#     #this is after, to launch list_exp even if we just changed the folder
#     if len(list_file)>0:
#         file.choices=list_file
# 
#     #return container_exp_choice
# 
#     @file.changed.connect
#     def list_exp(new_file:str):
#         #if len(container_exp_choice) > 2:
#             #container_exp_choice.pop(-1).native.close()
#         try:
#             file_path=os.path.join(folder.value,new_file)
#             print('new filename : '+str(file_path))
#     
#             exp=get_exp(file_path)
#             nbpos=exp.nbpos
#             num_exp.choices=list(map(str,range(1,nbpos+1)))
#             container_exp_choice.append(num_exp)
#         except: 
#             print('no experiment numbers: problem')
#             
#         return container_exp_choice
#     
# 
# @magicgui(call_button='Load experiment')
# def load_exp():
#     try:
#         exp=get_exp(os.path.join(folder.value,file.value))
#         print('new experiment loading, '+str(exp.nbpos)+' different positions')
#         viewer.layers.clear()
#         for i in range(len(exp.wl)):
#             if exp.stacks:
#                 stackname=exp.get_stack_name(i,num_exp.value)
#                 viewer.add_image(imread(stackname),name=exp.wl[i].name)
#             else:
#                 files=sorted(glob(exp.get_image_name(i,num_exp.value,'*')),key=alphanumeric_key)
#                 #print(files)
#                 stack=read_stack(files)
#                 viewer.add_image(stack,contrast_limits=[0,2000],name=exp.wl[i].name)
#                 #set contrast limits automatically
#                 viewer.layers[-1].reset_contrast_limits()    
#         viewer.reset_view()
#         viewer.grid.enabled=True   
#     except Exception as inst:
#         print(type(inst))    # the exception instance
#         print(inst.args)     # arguments stored in .args
#         print(inst)   
#         print('error: not able to load experiment')    
# 
# container_load_exp=widgets.Container(widgets=[container_exp_choice,load_exp], labels=False)
# 
# viewer.window.add_dock_widget(container_load_exp)
# 
# 
# =============================================================================


#folder_picker.show()

# =============================================================================
# import time
# import warnings
# from qtpy.QtWidgets import QSpacerItem, QSizePolicy
# from napari_plugin_engine import napari_hook_implementation
# from qtpy.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLabel, QSpinBox, QCheckBox
# from qtpy.QtWidgets import QTableWidget, QTableWidgetItem, QWidget, QGridLayout, QPushButton, QFileDialog
# from qtpy.QtCore import Qt
# from magicgui.widgets import Table
# from napari._qt.qthreading import thread_worker
# from qtpy.QtCore import QTimer
# from magicgui import magicgui
# import pyqtgraph as pg
# import numpy as np
# import napari
# 
# 
# @magicgui(
#     call_button="Plot t profile",
# )
# def plot_t_profile(
#     data: napari.types.ImageData,
#     #zones: napari.types.,
#     thresh_coeff: float = 1.0,
#     thresh_min_size: int = 60,
#     rsize_factor: int = 2,
#     tophat_size: int = 10,
#     ridge_size: int = 5,
#     return_all: bool = False,
# ):
# 
#     
#     return
# 
# def extract_voxel_time_series(cpos, nlayer):
#     """Method to extract the array element values along the first axis of a napari viewer layer.
#     First the data array is extracted from a napari image layer and the cursor position is
#     translated into an array index. If the index points to an element inside of the array all values along the first
#     axis are returned as a list, otherwise None is returned.
#     :param cpos: Position of the cursor inside of a napari viewer widget.
#     :type cpos: numpy.ndarray
#     :param nlayer: Napari image layer to extract data from.
#     :type nlayer: napari.layers.image.Image
#     """
#     # get full data array from layer
#     data = nlayer.data
#     # convert cursor position to index
#     ind = tuple(map(int, np.round(nlayer.world_to_data(cpos))))
#     # return extracted data if index matches array
#     if all([0 <= i < max_i for i, max_i in zip(ind, data.shape)]):
#         return ind, data[(slice(None),) + ind[1:]]
#     return ind, None
# 
# from napari.layers.shapes import Shapes
# 
# data = viewer.layers[0].data
# for layer in viewer.layers:
#     if isinstance(layer,Shapes):
#         mask=layer.to_masks((61,1024,1024))[-1]
#         # convert cursor position to index
#         #ind = tuple(map(int, np.round(nlayer.world_to_data(cpos))))
#         toplot=data[mask]
# =============================================================================

# =============================================================================
#     @thread_worker
#     def do_segment():
#         if data is None:
#             return [([0, 0], {})]
# =============================================================================





# =============================================================================
# class PlotTProfile(QWidget):
#     def __init__(self, napari_viewer):
#         super().__init__()
#         self.viewer = napari_viewer
# 
#         self.data = None
#         self.former_line = None
# 
#         graph_container = QWidget(self)
# 
#         # histogram view
#         self.graphics_widget = pg.GraphicsLayoutWidget()
#         self.graphics_widget.setBackground(None)
# 
#         #graph_container.setMaximumHeight(100)
#         graph_container.setLayout(QHBoxLayout())
#         graph_container.layout().addWidget(self.graphics_widget)
# 
#         # individual layers: legend
#         self.labels = QWidget()
#         self.labels.setLayout(QVBoxLayout())
#         self.labels.layout().setSpacing(0)
# 
#         # setup layout
#         self.setLayout(QVBoxLayout())
# 
#         self.layout().addWidget(graph_container)
#         self.layout().addWidget(self.labels)
# 
#         num_points_container = QWidget()
#         num_points_container.setLayout(QHBoxLayout())
# 
#         lbl = QLabel("Number of points")
#         num_points_container.layout().addWidget(lbl)
#         self.sp_num_points = QSpinBox()
#         self.sp_num_points.setMinimum(2)
#         self.sp_num_points.setMaximum(10000000)
#         self.sp_num_points.setValue(100)
#         num_points_container.layout().addWidget(self.sp_num_points)
#         num_points_container.layout().setSpacing(0)
#         self.layout().addWidget(num_points_container)
# 
# 
#         btn_refresh = QPushButton("Refresh")
#         btn_refresh.clicked.connect(self.redraw)
#         self.layout().addWidget(btn_refresh)
# 
#         btn_list_values = QPushButton("List values")
#         btn_list_values.clicked.connect(self._list_values)
#         self.layout().addWidget(btn_list_values)
# 
#         verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
#         self.layout().addItem(verticalSpacer)
# 
#         self.redraw()
# 
#     def _list_values(self):
#         table = {}
#         for my_profile in self.data:
#             positions = np.asarray(my_profile['positions'])
#             for i, x in enumerate(positions[0]):
#                 table[my_profile['name'] + '_pos' + str(i)] = positions[:, i]
# 
#             table[my_profile['name'] + '_intensity'] = my_profile['intensities']
#             table[my_profile['name'] + '_distance'] = my_profile['distances']
# 
#         # turn table into a widget
#         dock_widget = table_to_widget(table)
# 
#         # add widget to napari
#         self.viewer.window.add_dock_widget(dock_widget, area='right')
# 
#     def _get_current_zone(self):
#         zone=None
#         for layer in self.viewer.layers:
#             if isinstance(layer, napari.layers.Shapes):
#                 selection = list(layer.selected_data)
#                 if len(selection) > 0:
#                     zone = layer.to_masks((1024,1024))[0]
#                     print(zone)
#                     break
#                 try:
#                     zone = layer.to_masks((1024,1024))[0]
#                     print(zone)
#                     break
#                 except IndexError:
#                     print('error')
#                     pass
#         return zone
# 
#     def redraw(self):
# 
#         zone = self._get_current_zone()
# 
#         if zone is None:
#             print('no zone found')
#             return
# 
#         self._reset_plot()
# 
#         # clean up
#         layout = self.labels.layout()
#         for i in reversed(range(layout.count())):
#             layout.itemAt(i).widget().setParent(None)
# 
#         # visualize plots
#         num_bins = self.sp_num_points.value()
#         colors = []
#         self.data = []
#         print('here i am')
#         for i, layer in enumerate(self.viewer.selection.active):
#             print(layer.name)
#             if isinstance(layer, napari.layers.Image):
#                 # plot profile
#                 my_profile = tprofile(layer, zone)
#                 print('profile calculated ...')
#                 my_profile['name'] = layer.name
#                 self.data.append(my_profile)
#                 colormap = layer.colormap.colors
#                 color = np.asarray(colormap[-1, 0:3]) * 255
#                 colors.append(color)
#     
#                 intensities = my_profile['intensities']
#                 if len(intensities) > 0:
#                     print('before plotting')
#                     self.p2.plot(np.arange(0,61), intensities, pen=color, name=layer.name)
#                     print('after plotting')
#                     text = '[%0.2f .. %0.2f], %0.2f +- %0.2f' % (np.min(intensities),np.max(intensities),np.mean(intensities),np.std(intensities))
#     
#                     row = LayerLabelWidget(layer, text, colors[i], self)
#                     layout.addWidget(row)
#                 else:
#                     print('could not get any intensity')
#             
#     def _reset_plot(self):
#         if not hasattr(self, "p2"):
#             self.p2 = self.graphics_widget.addPlot()
#             axis = self.p2.getAxis('bottom')
#             axis.setLabel("Distance")
#             axis = self.p2.getAxis('left')
#             axis.setLabel("Intensity")
#         else:
#             self.p2.clear()
# 
#     def selected_image_layers(self):
#         return [layer for layer in self.viewer.layers if (isinstance(layer, napari.layers.Image) and layer.visible)]
# 
# class LayerLabelWidget(QWidget):
#     def __init__(self, layer, text, color, gui):
#         super().__init__()
# 
#         self.setLayout(QHBoxLayout())
# 
#         lbl = QLabel(layer.name + text)
#         lbl.setStyleSheet('color: #%02x%02x%02x' % tuple(color.astype(int)))
#         self.layout().addWidget(lbl)
# 
# def tprofile(layer, zone):
#     intensities=[]
# 
#     for i in range(layer.data.shape[0]):
#         intensities.append(np.mean(layer.data[i][zone]))
#     return {
#         'intensities': intensities
#     }
# 
# # copied from napari-skimage-regionprops
# def table_to_widget(table: dict) -> QWidget:
#     """
#     Takes a table given as dictionary with strings as keys and numeric arrays as values and returns a QWidget which
#     contains a QTableWidget with that data.
#     """
#     view = Table(value=table)
# 
#     copy_button = QPushButton("Copy to clipboard")
# 
#     @copy_button.clicked.connect
#     def copy_trigger():
#         view.to_dataframe().to_clipboard()
# 
#     save_button = QPushButton("Save as csv...")
# 
#     @save_button.clicked.connect
#     def save_trigger():
#         filename, _ = QFileDialog.getSaveFileName(save_button, "Save as csv...", ".", "*.csv")
#         view.to_dataframe().to_csv(filename)
# 
#     widget = QWidget()
#     widget.setWindowTitle("region properties")
#     widget.setLayout(QGridLayout())
#     widget.layout().addWidget(copy_button)
#     widget.layout().addWidget(save_button)
#     widget.layout().addWidget(view.native)
# 
#     return widget
# 
# def min_max(data):
#     return data.min(), data.max()
# 
# 
# =============================================================================
