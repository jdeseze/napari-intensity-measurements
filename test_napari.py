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

#%% open default file

import napari
from glob import glob
from skimage.io import imread
from skimage.io.collection import alphanumeric_key
from dask import delayed
import dask.array as da
import numpy as np
import exifread
from common import get_exp,Result,Exp

inds_cropped_area=[]

#funciton to read the images from metamorph
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

#create different layers
filenames = sorted(glob(r"G:\optorhoa\201208_RPE_optoRhoA_PAKiRFP\cell2s_50msact_1_w2TIRF 561_t*.tif"),key=alphanumeric_key)
stack=read_stack(filenames)
viewer=napari.view_image(stack, contrast_limits=[0,2000],name='561')  
viewer.layers[-1].reset_contrast_limits()

filenames = sorted(glob(r"G:\optorhoa\201208_RPE_optoRhoA_PAKiRFP\cell2s_50msact_1_w3TIRF 642_t*.tif"),key=alphanumeric_key)
stack=read_stack(filenames)
viewer.add_image(stack, contrast_limits=[0,2000],name='642')  
viewer.layers[-1].reset_contrast_limits()

filenames = sorted(glob(r"G:\optorhoa\201208_RPE_optoRhoA_PAKiRFP\cell2s_50msact_1_w1TIRF DIC_t*.tif"),key=alphanumeric_key)
stack=read_stack(filenames)
viewer.add_image(stack, contrast_limits=[199,200],name='DIC')  
viewer.layers[-1].reset_contrast_limits()

viewer.reset_view()
viewer.grid.enabled=True

#%% STACK VIEWER
from qtpy.QtWidgets import QPushButton,QLabel,QComboBox,QFileDialog,QWidget,QMessageBox,QMainWindow,QVBoxLayout,QCheckBox
import os

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
        self.list_exp.currentTextChanged.connect(self.detail_exp)
        
        #add the dropdown button to select the cell 
        self.cell_nb=QComboBox(self.list_exp)
        self.layout().addWidget(self.cell_nb)
        
        #button to decide whether you keep the layers or not
        self.del_layers=QCheckBox()
        self.del_layers.setText('Keep layers')
        self.layout().addWidget(self.del_layers)
        
        #button to launch the loading of the experiment
        load=QPushButton(self)
        load.setText('Load experiment')
        self.layout().addWidget(load)
        load.clicked.connect(self.display_exp)
        
        
        
    '''function to choose the directory and have the experiments in the folder displayed'''
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
    
    def detail_exp(self):
        self.cell_nb.clear()
        self.get_exp()
        if self.exp:
            self.cell_nb.addItems(list(map(str,range(1,self.exp.nbpos+1))))        
        else:
            print('no experiment was found')
            return False
        
    def get_exp(self):
        try:
            print(self.folder[0])
            self.exp=get_exp(os.path.join(self.folder[0],self.list_exp.currentText()))
        except:
            mess=QMessageBox(self)
            mess.setText('Unable to load experiment')
            self.layout().addWidget(mess) 
        if self.exp:
            return self.exp

        
    def display_exp(self):
        #if he finds a shape layer, he keeps it
        for layer in self.viewer.layers:
            if type(layer)==napari.layers.shapes.shapes.Shapes:
                shape=layer
                
        #clear all layers
        if self.del_layers.checkState()==0:
            self.viewer.layers.clear()
        
        try:
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
    
    w.resize(100,50)
    w.setWindowTitle('Select experiment')
    w.show()
    viewer.window.add_dock_widget(w)

#%%CROPER 
''' you need to draw a rectangle, otherwise it dxoes something bizarre
you need also to draw it in the not grid mode'''

from magicgui import magicgui

@magicgui(call_button='Crop')
def crop(crop_zone:napari.layers.Shapes):
    global inds_cropped_area
    inds=[]
    for i in np.array([crop_zone.data[0][0][1:],crop_zone.data[0][2][1:]]).T:
        temp=[int(i[0]), int(i[1])]
        temp.sort()
        inds.append(slice(temp[0],temp[1]))
    nblayer=len(viewer.layers)
    inds_cropped_area=inds
    for i,layer in enumerate(viewer.layers):
        if layer._type_string=='image':
            layer_name=layer.name
            new_data=layer.data[:,inds[0],inds[1]]
            layer.data=new_data
    viewer.layers.remove(crop_zone)
    viewer.reset_view()
    viewer.grid.enabled=True   
        
if __name__ == "__main__":
    viewer.window.add_dock_widget(crop)
    
    
#%% SEGMENTER

from magicgui import magicgui
from scipy import ndimage
from skimage import measure, filters
import matplotlib.pyplot as plt

def segment_threshold(img,thresh):#bg,init_median):
    #img=(img/2^8).astype(np.uint8)
    ''' correct by the ratio betweent the mean fluorescence at the inital state and the one at the timepoint'''
    #bleach_correction=(np.median(np.array(img[img>bg])))/(init_median)
    binary = img > thresh
    dil=ndimage.binary_dilation(binary,iterations=1)
    filled=ndimage.binary_fill_holes(dil[0]).astype(int)
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
    return (label_img[np.newaxis,:,:]>0)*255

background_inds={'Top-left':tuple([slice(0,100),slice(0,100)]),
            'Top-right':tuple([slice(0,100),slice(-101,-1)]),
            'Bottom-left':tuple([slice(-101,-1),slice(0,100)]),
            'bottom-right':tuple([slice(-101,-1),slice(-101,-1)])}

@magicgui(call_button='Segment',
          coeff={"widget_type": "FloatSlider",'min':0.5, 'max': 1.5,'step':0.01},
          background={"choices":list(background_inds.keys())},
          )
def segment(data:napari.types.ImageData,
            coeff=1.0,
            background=list(background_inds.keys())[0],
            ):
    med=filters.median(data[0])
    pre_thresh=filters.threshold_otsu(med)
    '''' Find bacckground of first image and find the initial mean of the area that is fluorescent (superior than the background)'''
    bg=int(np.mean(data[0][background_inds[background]]))
    init_median=np.median(np.array(data[0][data[0]>bg]))
    end_median=np.median(np.array(data[-1][data[-1]>bg]))
    #print(type(data))
    if type(data)==np.ndarray:
        print('np.ndarray, it cannot be segmented like this')
        data=da.from_array(data)
    '''Make a simple bleach correction by a ratio for the segmentation: not used for the moment'''
    data_to_segment2=(data.T*(1+(end_median-init_median)/np.arange(1,len(data)+1))).T
    
    segmented = da.map_blocks(segment_threshold,data,thresh=coeff*pre_thresh)#bg=bg,init_median=init_median)
    #print(data[0])
    viewer.add_image(segmented,name='segmented',opacity=0.2)

@segment.coeff.changed.connect
def change_seg(new_coeff:int):
    viewer.layers.remove('segmented')
    segment()


if __name__ == "__main__":
    viewer.window.add_dock_widget(segment)

#%% Segmentor optical flow

from magicgui import magicgui
from scipy import ndimage
import matplotlib.pyplot as plt
import cv2 
from PIL import Image, ImageFilter


def segment_opt(data,iterations,threshold,mina,maxa,medmag,maxmag,size_erosion,size_close):
    
    img1=np.float32(data[1])/np.max(data[0])
    img2=np.float32(data[2])/np.max(data[1])

    gauss1=cv2.GaussianBlur(img1,(51,51),cv2.BORDER_DEFAULT )
    gauss2=cv2.GaussianBlur(img2,(51,51),cv2.BORDER_DEFAULT )
    
    img1=img1/gauss1
    img2=img2/gauss2
    #img1=(img1/gauss1-mina)/(maxa-mina)
    #img2=(img2/gauss2-mina)/(maxa-mina)
    #print('max(img1) is '+str(np.max(img1)))
    #print('max(img1/gauss1) is '+str(np.max(img1/gauss1)))
    #print('maxa is '+str(maxa))
    #print('min(img1/gauss1) is '+str(np.min(img1/gauss1)))
    #print('mina is '+str(mina))
    img1=(img1-mina)/(maxa-mina)
    img2=(img2-mina)/(maxa-mina)
    
    img1=(img1)*((img1)>0)*((img1)<1)
    img2=(img2)*((img2)>0)*((img2)<1)
    
    flow=cv2.calcOpticalFlowFarneback(img1,img2,None,0.5,1,iterations,5,5,1.2,0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    mag=((mag-medmag)/maxmag+1.0)
    
    #thresholding the optical flow
    th=((mag>threshold*0.1)*255).astype('uint8')
    ret, thresh = cv2.threshold(th.astype('uint8'),0.5, 255, 0)
    
    #erode stuff
    kernel = np.ones((3,3), np.uint8)
    
    contours,a=cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #sort contours by area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    mask=np.zeros(img1.shape)
    cv2.drawContours(mask,contours,0,(255,255,255),thickness=cv2.FILLED)  
    
    ret, thresh = cv2.threshold(mask.astype('uint8'),0.5, 255, 0)
    closed=cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,kernel,iterations=size_close) 
    er=cv2.erode(closed,kernel,iterations=size_erosion)
    smoothed=cv2.blur(er,(25,25))
    
    ret, thresh = cv2.threshold(smoothed.astype('uint8'),200, 255, 0)
    contours,a=cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    mask=np.zeros(img1.shape)
    cv2.drawContours(mask,contours,0,(10,10,10),thickness=cv2.FILLED)    
    #mask2=np.zeros(img1.shape)
    cv2.drawContours(mask,contours,0,(255,255,255),5) 
    
    if False:
        mask=segment_threshold(np.array([data[0]]),thresh=1)
    
    return np.array([mask]*3)#prev_img

@magicgui(call_button='Segment optical flow',
          )
def segment_opticalflow(data:napari.types.ImageData,
            threshold=2,
            iterations=1,
            size_erosion=0,
            size_close=0
            ):
    #calculate for the first frame to have the good normalization
    img=np.float32(data[0])/np.max(np.array(data[0]))
    gauss=cv2.GaussianBlur(img,(51,51),cv2.BORDER_DEFAULT)
    flat=np.array(img/gauss).flatten()
    flat.sort()
    mina=np.mean(flat[0:int(len(flat)/1000)])
    maxa=np.mean(flat[int(999*len(flat)/1000):])
    img=(img/gauss-mina)/(img-maxa)
    img1=np.float32(data[1])/np.max(np.array(data[0]))
    gauss1=cv2.GaussianBlur(img1,(51,51),cv2.BORDER_DEFAULT)
    img1=(img1/gauss1-mina)/(maxa-mina)
    
    flow=cv2.calcOpticalFlowFarneback(img,img1,None,0.5,1,iterations,5,5,1.2,0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    flat=mag.flatten()
    flat.sort()
    medmag=np.median(flat)
    maxmag=min(medmag-flat[0],flat[-1]-medmag)

    segmented = da.map_overlap(segment_opt,data,depth={0: 1, 1: 0,2:0},boundary='reflect',iterations=iterations,threshold=threshold,mina=mina,maxa=maxa,medmag=medmag,maxmag=maxmag,size_erosion=size_erosion,size_close=size_close,dtype=img.dtype)#bg=bg,init_median=init_median)
    #print(data[0])
    viewer.add_image(segmented,name='segmented optical flow',opacity=0.2)

@segment_opticalflow.threshold.changed.connect
def change_thresh(new_thresholE:int):
    viewer.layers.remove('segmented optical flow')
    segment_opticalflow()
@segment_opticalflow.iterations.changed.connect
def change_it(new_iterations:int):
    viewer.layers.remove('segmented optical flow')
    segment_opticalflow()
@segment_opticalflow.size_erosion.changed.connect
def change_er(new_size_erosion:int):
    viewer.layers.remove('segmented optical flow')
    segment_opticalflow()
@segment_opticalflow.size_close.changed.connect
def change_close(new_size_close:int):
    viewer.layers.remove('segmented optical flow')
    segment_opticalflow()
    
if __name__ == "__main__":
    viewer.window.add_dock_widget(segment_opticalflow)
#%% Intensity calculator
import pandas as pd
import pickle
import time

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


def calculate_intandsurf(imgs,mask_seg,inds_mask_act):
    return []

try:
    exp_layers=[layer for layer in viewer.layers if layer.name in [wl.name for wl in w.get_exp().wl]]
except:
    exp_layers=viewer.layers

background_inds={'Top-left':tuple([slice(0,100),slice(0,100)]),
            'Top-right':tuple([slice(0,100),slice(-101,-1)]),
            'Bottom-left':tuple([slice(-101,-1),slice(0,100)]),
            'bottom-right':tuple([slice(-101,-1),slice(-101,-1)])}

    
@magicgui(call_button='Calculate intensities',
          layers_meas={"choices":exp_layers,'allow_multiple':True},
          bg={"choices":list(background_inds.keys()),'name':'background_area'},
          )
def calculate_intensities(
        mask_act : napari.types.ShapesData,
        mask: napari.layers.Image,
        layers_meas=[exp_layers[0]],
        prot=False,
        new_file=False,
        time_of_activation=0,
        comments='',
        bg=list(background_inds.keys())[0]):
    global inds_cropped_area
    ta=time.time()
    exp=w.get_exp()
    if not exp:
        print('no experiment')
        return
    pos=int(w.cell_nb.currentText())
    print("position"+str(pos))
    print(layers_meas)
    "find the name of the layer that was used for the segmentation"
    wl_seg=next(i for i, v in enumerate(exp.wl) if v.name in segment_opticalflow.data.current_choice)
    stepseg=exp.wl[wl_seg].step      
    '''find indice of the area where the activation was done (it is a bit complicate, I didn't know how to simplify this) '''
    inds=[]
    for i in np.array([mask_act[0][0][1:],mask_act[0][2][1:]]).T:
        print(i)
        temp=[int(i[0]), int(i[1])]
        temp.sort()
        inds.append(slice(temp[0],temp[1]))
    inds_mask_act=tuple(inds)
    tb=time.time()
    
    '''this is what takes the longest time because it needs to do the segmentation'''
    try:
        mask_np=np.array(mask.data>0)#==10)
    except:
        print('no layer named "mask" selected')
        return
    for j,layer_meas in enumerate(layers_meas):
        whole, act, notact, wholesurfs, actsurfs=[],[],[],[],[]
        wl_meas=next(i for i, v in enumerate(exp.wl) if v.name==layer_meas.name)
        layer_meas_np=np.array(layer_meas.data)
        result=Result(exp,prot,wl_meas,pos)
    
        '''save areas of activation and areas of cropping inside the object result'''
        result.inds_cropped_area=inds_cropped_area
        result.inds_mask_act=inds_mask_act
        result.startacq=time_of_activation
        
        stepmeas=exp.wl[wl_meas].step
        
        nb_img=int(exp.nbtime/stepmeas)
        #print(nb_img)
        print(nb_img)
        i=0
        while i<nb_img:
            '''define the good frame to take for each segmentation or activation'''
            timg=i*stepmeas
            tseg=int(i*stepmeas/stepseg)*stepseg
            #print(tseg)
            '''take mask of segmentation'''
            mask_seg=mask_np[tseg,:,:]>0
            '''take values in current image'''
            img=layer_meas_np[i,:,:]
            #print(time.time()-tb)
            #img=np.array(Image.open(exp.get_image_name(wl_meas,pos,timg)))
            '''calculate whole intensity'''
            whole_int=img[mask_seg].sum()
            '''calculate whole surface'''
            whole_surf=mask_seg.sum()
            #print('whole surface of the cell in pixels : "+str(whole_surf))
            '''calculate the intensity in the area of activation'''
            act_int=(img[inds_mask_act][mask_seg[inds_mask_act]]).sum()
            '''calculate the area of activation'''
            act_surf=(mask_seg[inds_mask_act]).sum()
            if i==0:
                background=np.mean(img[background_inds[bg]])
                
            if act_surf==0:
                print('the cell is outside the area of activation : it stopped at timepoint '+str(i))
                #if the cell is outside the area of activation, I go out of the loop
                i=nb_img
            else:
                act.append(act_int/act_surf)
                wholesurfs.append(whole_surf)
                actsurfs.append(act_surf)
                whole.append(whole_int/whole_surf)
                notact.append((whole_int-act_int)/(whole_surf-act_surf))
                i+=1
        #print(time.time()-tb)
        result.whole, result.act, result.notact, result.background=whole,act,notact,background
        result.whole_surf,result.act_surf =wholesurfs,actsurfs
        #put into pd dataframe
    # =============================================================================
    #     current_resultpd=pd.DataFrame(np.array([npwhole,npact,npnotact]).transpose())
    #     resultspd=pd.read_pickle('./pdresults.pkl')
    #     new_resultspd=pd.concat([resultspd,current_resultpd], ignore_index=True, axis=1)
    #     new_resultspd.to_pickle('./pdresults.pkl')
    # =============================================================================
        #add to the list of results
        print(act)
        if new_file and j==0:
            results=[result]
        else:
            with open('./results.pkl', 'rb') as output:
                results=pickle.load(output)
            results.append(result)
        ta=time.time()    
        with open('./results.pkl', 'wb') as output:
            pickle.dump(results, output, pickle.HIGHEST_PROTOCOL)
        with open('./results.txt','w') as output:
            write_on_text_file(results,output)
        tb=print(time.time()-ta)
        with open('./pdresults.pkl','wb') as output:
            write_on_pd(results,output)
        tb=print(time.time()-ta)

    print('Done')
    
if __name__ == "__main__":
    viewer.window.add_dock_widget(calculate_intensities)

