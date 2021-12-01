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
import dask_image

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
#%%
from qtpy.QtWidgets import QPushButton,QLabel,QComboBox,QFileDialog,QWidget,QMessageBox,QMainWindow,QVBoxLayout
from common_functions import *
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
        self.list_exp.currentTextChanged.connect(self.get_exp)
        
        #add the dropdown button to select the cell 
        self.cell_nb=QComboBox(self.list_exp)
        self.layout().addWidget(self.cell_nb)
        self.cell_nb.currentTextChanged.connect(self.display_exp)
        
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
            self.exp=get_exp(os.path.join(self.folder[0],self.list_exp.currentText()))
        except:
            mess=QMessageBox(self)
            mess.setText('Unable to load experiment')
            self.layout().addWidget(mess)
        self.cell_nb.clear()
        if self.exp:
            self.cell_nb.addItems(list(map(str,range(1,self.exp.nbpos+1))))
            print(self.cell_nb)
        else:
            print('no experiment was found')
        
    def display_exp(self):
# =============================================================================
#         mess=QLabel(self)
#         mess.setText(str(is_exp))
#         self.layout().addWidget(mess)
# =============================================================================
        self.viewer.layers.clear()
        
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
# =============================================================================
#                 stack=read_stack(filename)
#                 self.viewer.add_image(stack,contrast_limits=[0,2000],name=self.exp.wl[i].name)
#                 self.viewer.layers[-1].reset_contrast_limits()
# =============================================================================
            else:
                filenames=sorted(glob(self.exp.get_image_name(i,pos,'*')),key=alphanumeric_key)
                stack=read_stack(filenames)
                self.viewer.add_image(stack,contrast_limits=[0,2000],name=self.exp.wl[i].name)
                #set contrast limits automatically
                self.viewer.layers[-1].reset_contrast_limits()
        
        self.viewer.reset_view()
        self.viewer.grid.enabled=True
        #for i in range(len(exp.wl)):

import time
import warnings
from qtpy.QtWidgets import QSpacerItem, QSizePolicy
from napari_plugin_engine import napari_hook_implementation
from qtpy.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLabel, QSpinBox, QCheckBox
from qtpy.QtWidgets import QTableWidget, QTableWidgetItem, QWidget, QGridLayout, QPushButton, QFileDialog
from qtpy.QtCore import Qt
from magicgui.widgets import Table
from napari._qt.qthreading import thread_worker
from qtpy.QtCore import QTimer
from magicgui import magicgui
import pyqtgraph as pg
import numpy as np
import napari


@magicgui(
    call_button="Plot t profile",
)
def plot_t_profile(
    data: napari.types.ImageData,
    #zones: napari.types.,
    thresh_coeff: float = 1.0,
    thresh_min_size: int = 60,
    rsize_factor: int = 2,
    tophat_size: int = 10,
    ridge_size: int = 5,
    return_all: bool = False,
):

    return
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
if __name__ == "__main__":
    w = StackViewer(viewer)
    
    #.resize(300,300)
    #w.setWindowTitle(‘Guru99’)
    
    w.show()
    plot_t_profile.show()
    viewer.window.add_dock_widget(w)
    viewer.window.add_dock_widget(plot_t_profile)

#viewer.layers[0].get_value(
