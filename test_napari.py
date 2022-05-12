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
#from dask_image.imread import imread
from glob import glob
from skimage.io import imread
from skimage.io.collection import alphanumeric_key
from dask import delayed
import dask.array as da
#import dask_image
import numpy as np
import exifread
from common import WL,Exp,get_exp,Result,Result_array

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
filenames = sorted(glob(r"F:\optorhoa\201208_RPE_optoRhoA_PAKiRFP\cell2s_50msact_1_w2TIRF 561_t*.tif"),key=alphanumeric_key)
stack=read_stack(filenames)
viewer=napari.view_image(stack, contrast_limits=[0,2000],name='561')  
viewer.layers[-1].reset_contrast_limits()

filenames = sorted(glob(r"F:\optorhoa\201208_RPE_optoRhoA_PAKiRFP\cell2s_50msact_1_w3TIRF 642_t*.tif"),key=alphanumeric_key)
stack=read_stack(filenames)
viewer.add_image(stack, contrast_limits=[0,2000],name='642')  
viewer.layers[-1].reset_contrast_limits()

filenames = sorted(glob(r"F:\optorhoa\201208_RPE_optoRhoA_PAKiRFP\cell2s_50msact_1_w1TIRF DIC_t*.tif"),key=alphanumeric_key)
stack=read_stack(filenames)
viewer.add_image(stack, contrast_limits=[199,200],name='DIC')  
viewer.layers[-1].reset_contrast_limits()

viewer.reset_view()
viewer.grid.enabled=True

#%% STACK VIEWER
from qtpy.QtWidgets import QPushButton,QLabel,QComboBox,QFileDialog,QWidget,QMessageBox,QMainWindow,QVBoxLayout
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
    
    
# =============================================================================
# #%% SEGMENTER
# 
# from magicgui import magicgui
# from scipy import ndimage
# from skimage import measure, filters
# import matplotlib.pyplot as plt
# 
# def segment_threshold(img,thresh):#bg,init_median):
#     #img=(img/2^8).astype(np.uint8)
#     ''' correct by the ratio betweent the mean fluorescence at the inital state and the one at the timepoint'''
#     #bleach_correction=(np.median(np.array(img[img>bg])))/(init_median)
#     binary = img > thresh
#     dil=ndimage.binary_dilation(binary,iterations=1)
#     filled=ndimage.binary_fill_holes(dil[0]).astype(int)
#     label_img, cc_num = ndimage.label(filled)
#     #CC = ndimage.find_objects(label_img)
#     cc_areas = ndimage.sum(filled, label_img, range(cc_num+1))
#     area_mask = (cc_areas < max(cc_areas))
#     label_img[area_mask[label_img]] = 0
#     try:
#         contours = measure.find_contours(label_img, 0.8)
#     except:
#         contours=[]
#     if len(contours)>0:
#         contour=contours[0]
#     else:
#         contour=np.array([None])
#     #return (label_img>0)*255, contour
# # =============================================================================
# #     fig, ax = plt.subplots()
# #     ax.imshow(img, cmap=plt.cm.gray)
# #     ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
# #     ax.set_xticks([])
# #     ax.set_yticks([])
# # =============================================================================
#     return (label_img[np.newaxis,:,:]>0)*255
# 
# background_inds={'Top-left':tuple([slice(0,100),slice(0,100)]),
#             'Top-right':tuple([slice(0,100),slice(-101,-1)]),
#             'Bottom-left':tuple([slice(-101,-1),slice(0,100)]),
#             'bottom-right':tuple([slice(-101,-1),slice(-101,-1)])}
# 
# @magicgui(call_button='Segment',
#           coeff={"widget_type": "FloatSlider",'min':0.5, 'max': 1.5,'step':0.01},
#           background={"choices":list(background_inds.keys())},
#           )
# def segment(data:napari.types.ImageData,
#             coeff=1.0,
#             background=list(background_inds.keys())[0],
#             ):
#     med=filters.median(data[0])
#     pre_thresh=filters.threshold_otsu(med)
#     '''' Find bacckground of first image and find the initial mean of the area that is fluorescent (superior than the background)'''
#     bg=int(np.mean(data[0][background_inds[background]]))
#     init_median=np.median(np.array(data[0][data[0]>bg]))
#     end_median=np.median(np.array(data[-1][data[-1]>bg]))
#     #print(type(data))
#     if type(data)==np.ndarray:
#         print('np.ndarray, it cannot be segmented like this')
#         data=da.from_array(data)
#     '''Make a simple bleach correction by a ratio for the segmentation: not used for the moment'''
#     data_to_segment2=(data.T*(1+(end_median-init_median)/np.arange(1,len(data)+1))).T
#     
#     segmented = da.map_blocks(segment_threshold,data,thresh=coeff*pre_thresh)#bg=bg,init_median=init_median)
#     #print(data[0])
#     viewer.add_image(segmented,name='segmented',opacity=0.2)
# 
# @segment.coeff.changed.connect
# def change_seg(new_coeff:int):
#     viewer.layers.remove('segmented')
#     segment()
# 
# 
# if __name__ == "__main__":
#     viewer.window.add_dock_widget(segment)
# 
# =============================================================================
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
        mask_np=np.array(mask.data==10)
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


#%%

# =============================================================================
# def tm(img1):
#     #list_img=[img1,img2,img3]
#     #mean=np.mean([img[0] for img in list_img],axis=0)>0.5
#     #print(img1.shape)
#     #print(np.array(3*[np.mean(img1,axis=0)]).shape)
#     return np.array(3*[np.mean(img1,axis=0)])>np.max(img1)/2#mean>np.max(mean)/3
# 
# @magicgui(call_button='temporal mean',
#           )
# def temp_mean(data:napari.types.ImageData,
#             range_mean=3,):
#     #print(np.max(np.array(data[0])))
#     #print(data)
#     mean_test=da.map_overlap(tm,data,depth={0: 1, 1: 0,2:0},boundary=0,dtype=np.array(data[0]).dtype)
#     #print('shape of mean_test is '+str(mean_test.shape))
#     viewer.add_image(mean_test.compute(),name='temporal mean',opacity=0.2)    
#         
# if __name__ == "__main__":
#     viewer.window.add_dock_widget(temp_mean)
# =============================================================================
 #%%   
    
# =============================================================================
# #%% PLOT VALUES
# from matplotlib.figure import Figure
# from matplotlib.backends.backend_qt5agg import FigureCanvas
# import altair as alt
# import pandas as pd
# 
# df=pd.DataFrame([[1,2],[4,5]],columns=['a','b'])
# 
# @magicgui(call_button='Plot values')
# def plot_values():
#     with open('./results.pkl', 'rb') as output:
#         results=pickle.load(output)
#     #Result_array(results).plot()
#     fig1=Figure()
#     canvas1=FigureCanvas(fig1)
#     ax=fig1.add_subplot(111)
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#     #ax.plot(results[0].act)
#     for result in results:
#         #ax.plot(result.act,color='blue')
#         result.plot(axes=ax,zone='notact',plot_options={'color':'black'})
#         result.plot(axes=ax,zone='act',plot_options={'color':'blue'})
#     viewer.window.add_dock_widget(canvas1)   
#     
#     fig2=Figure()
#     canvas2=FigureCanvas(fig2)
#     ax2=fig2.add_subplot(111)
#     ax2.spines['right'].set_visible(False)
#     ax2.spines['top'].set_visible(False)
#     #ax.plot(results[0].act)
#     for result in results:
#         ax2.plot(np.array(result.whole_surf)/result.whole_surf[0],color='green')
#         ax2.plot(np.array(result.act_surf)/result.act_surf[0],color='blue')
#         ax2.plot((np.array(result.whole_surf)-np.array(result.act_surf))/(result.whole_surf[0]-result.act_surf[0]),color='black')
#     viewer.window.add_dock_widget(canvas2)    
#     
# # =============================================================================
# #     for prot in [True,False]:
# #         for wl_name in [layer.name for layer in calculate_intensities.layers_meas.value]:
# #             Result_array(results).plot(axes=ax,plot_options={'color':'blue'},prot=prot)
# #             Result_array(results).plot(axes=ax,zone='notact',plot_options={'color':'red'},prot=prot)
# # =============================================================================    
# # =============================================================================
# #     df=pd.DataFrame({'x':np.arange(len(result.act)),
# #                     'y':np.array(result.whole_surf)/result.whole_surf[0]})
# #     a=alt.Chart(df)
# #     a.mark_line().encode(
# #         x='x',
# #         y='y'
# #         )
# # =============================================================================
# 
# if __name__ == "__main__":
#     viewer.window.add_dock_widget(plot_values)
# =============================================================================
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
