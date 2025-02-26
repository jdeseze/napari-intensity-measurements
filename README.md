# napari-intensity-measurements

This project has to be used for the data acquired through the Metamorph software. It is not well documented, but comments will be added. 

Launching the main 'napari_seg.py' after having installed all the libraries needed, you have different plugins available. 

First one is a Metamorph reader, it just reads the experiments in a folder created by metamorph: especially, it reads the stacks without creating them. 

Second is a cropper

Third is a segmentation for fluorescence: it requires images to be opened. 

Fourth is a segmentation for DIC of phase contrast, through optical flow algorithm.

Fifth is calculator that calculates intensities in the Shape area (that souldh be a rectangle) chooses, as well as inensity in the whole cell, for the decided wavelenghts. 

For working example, you can download data files, looking for 'Optogenetic control of a GEF of RhoA uncovers a signaling switch from retraction to protrusion' in https://www.ebi.ac.uk/biostudies database. 
