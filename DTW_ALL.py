# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 10:54:28 2016

@author: aniruddha
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import pdb
import pickle
from scipy.signal import hilbert


# clear workspace
def clear_all():
    """Clears all the variables from the workspace of the spyder application."""
    gl = globals().copy()
    for var in gl:
        if var[0] == '_': continue
        if 'func' in str(globals()[var]): continue
        if 'module' in str(globals()[var]): continue

        del globals()[var]

        
# Method to visualize  ofDistance Matrix
def visualize_Matrix (distances) :
    D=np.transpose(distances)
    im=plt.imshow(D, interpolation='none', cmap='seismic', aspect='auto')
    plt.clim(-50,50)
    #plt.gca().invert_yaxis()
    plt.xlabel('Distance in Surface')
    plt.ylabel('Time')
    plt.title('Normal vs ' + secondsig  + ' Point Wise Alighnment Distance (av)')
    plt.grid()
    plt.colorbar()
    plt.tight_layout()
    plt.show()

# Method to visualize  ofDistance Matrix
def visualize_errorMatrix (distances) :
    D=np.transpose(distances)
    im=plt.imshow(D)
    #plt.clim(0,2000)
    #plt.gca().invert_yaxis()
    plt.xlabel('Distance in Surface')
    plt.ylabel('Time')
    plt.title('Abs Error After Pointwise Alignment')
    plt.grid()
    plt.colorbar()
    plt.tight_layout()
    plt.show()     
    

# aligning signals based on normalized correlation coeff
# This is standard alignment with 

def aligned_av(x111,y111, HILBERT_FLAG, ABSOLUTE_FLAG, SIGNAL_DISTANCE_METRIC, ACC_COST_FLAG, METHOD_FLAG) :
    
            # Get the  signal     
            x1= x111[::-1]
            y1=y111[::-1]
          
            #### Hilber Transform of the signals
            if HILBERT_FLAG == 1 :
                
               analytic_signal1 = hilbert(x1)
               x = np.abs(analytic_signal1)
               
               analytic_signal2 = hilbert(y1)
               y = np.abs(analytic_signal2)
               
            elif ABSOLUTE_FLAG == 1 :
               x=np.abs(x1)
               y=np.abs(y1)
            else :
               x=x1
               y=y1
              
            #####
#########################################################
            # Find out distances between all pair of points
            distances =np.zeros((len(y), len(x)))
            
            # Find out the mean in case of normalized corr coeff
            if SIGNAL_DISTANCE_METRIC==1 :
                meanx=np.mean(x)
                meany=np.mean(y)
                
            # fill out the matrix using distance metric
            for i in range(len(y)) :
                for j in range(len(x)) :
                    if SIGNAL_DISTANCE_METRIC==0 :
                        distances[i,j]=np.absolute(x[j]-y[i])
                    if SIGNAL_DISTANCE_METRIC==1 :
                        distances[i,j]=(y[i]-meany)*(x[j]-meanx) 
                        
                        
                        
            # Accumulated Cost
            # Two accumulated  costs one is only diagonal (2) and another is best among three (1)
            accumulated_score1=np.zeros((len(y), len(x)))
            if ACC_COST_FLAG==1 :
                accumulated_score2=np.zeros((len(y), len(x)))
                
                
            backtrackDir=np.zeros((len(y), len(x)))
                        
   

    
            # initialize base rows
            for i in range(1, len(x)):
                accumulated_score1[0,i] = distances[0,i] + accumulated_score1[0, i-1] + PENALTY
                if ACC_COST_FLAG==1 :
                    accumulated_score2[0,i] = distances[0,i] + accumulated_score2[0, i-1] + PENALTY
                backtrackDir[0,i]=3
                
        
            for i in range(1, len(y)):
                accumulated_score1[i,0] = distances[i, 0] + accumulated_score1[i-1, 0] + PENALTY1
                if ACC_COST_FLAG==1 :
                    accumulated_score2[i,0] = distances[i, 0] + accumulated_score2[i-1, 0] + PENALTY1 
                backtrackDir[i,0]=1
                
            # compute rest of the accumulative distances
            for i in range(1, len(y)):
                for j in range(1, len(x)):
                    
                    # This one is for two types of costs : diagonal and overall
                    if ACC_COST_FLAG==1 :
                            
                            if (METHOD_FLAG == 1) or (METHOD_FLAG == 2) or (METHOD_FLAG == 3) :
                                
                                # This is when we reach end of the signal
                                if i==len(y)-1 :
                                   values=np.array([accumulated_score2[i-1, j] + distances[i, j], accumulated_score1[i-1, j-1]+ distances[i, j], accumulated_score2[i, j-1]+ np.mean(distances[i, j:])])
                                   values[2]=values[2]
                                   accumulated_score1[i,j]=np.max(values)
                                   accumulated_score2[i,j]=values[1]                       
                                   backtrackDir[i,j]=np.argmax(values) + 1
                                
                                # This is for all other situations                                      
                                else :
                                    
                                   values=np.array([accumulated_score2[i-1, j] + distances[i, j], accumulated_score1[i-1, j-1]+ distances[i, j], accumulated_score2[i, j-1]+ distances[i, j]])
                                   values[2]=values[2] + PENALTY
                                   values[0]=values[0] + PENALTY1
                                   accumulated_score1[i, j] = np.max(values)
                                   accumulated_score2[i,j]=values[1]
                                   backtrackDir[i,j]=np.argmax(values) + 1 
                            
                            # This is original DTW
                            if (METHOD_FLAG == 0) :
                                
                                # This is when we reach end of the signal No penalty
                                if i==len(y)-1 :
                                   values=np.array([accumulated_score1[i-1, j] + distances[i, j], accumulated_score1[i-1, j-1]+ distances[i, j], accumulated_score1[i, j-1]+ distances[i, j]])
                                   values[2]=values[2]
                                   accumulated_score1[i,j]=np.max(values)
                                   accumulated_score2[i,j]=values[1]                       
                                   backtrackDir[i,j]=np.argmax(values) + 1
                                
                                # This is for all other situations                                      
                                else :
                                    
                                   values=np.array([accumulated_score2[i-1, j] + distances[i, j], accumulated_score1[i-1, j-1]+ distances[i, j], accumulated_score2[i, j-1]+ distances[i, j]])
                                   values[2]=values[2] + PENALTY
                                   values[0]=values[0] + PENALTY1
                                   accumulated_score1[i, j] = np.max(values)
                                   accumulated_score2[i,j]=values[1]
                                   backtrackDir[i,j]=np.argmax(values) + 1 
                               
                  
                               
                    if ACC_COST_FLAG==0 :
                        
                        if (METHOD_FLAG == 1) or (METHOD_FLAG == 2) or (METHOD_FLAG == 3) :
                          
                            if i==len(y)-1 :
                                
                               values=np.array([accumulated_score1[i-1, j] + distances[i, j], accumulated_score1[i-1, j-1]+ distances[i, j], accumulated_score1[i, j-1]+ np.mean(distances[i, j:])])
                               values[2]=values[2]
                               accumulated_score1[i,j]=np.max(values)
                                            
                               backtrackDir[i,j]=np.argmax(values) + 1
                               
                               
                            else :
                               values=np.array([accumulated_score1[i-1, j] + distances[i, j], accumulated_score1[i-1, j-1]+ distances[i, j], accumulated_score1[i, j-1]+ distances[i, j]])
                               values[2]=values[2] + PENALTY
                               values[0]=values[0] + PENALTY1
                               accumulated_score1[i, j] = np.max(values)
                               backtrackDir[i,j]=np.argmax(values) + 1 

                        if (METHOD_FLAG == 0) :
                          
                            if i==len(y)-1 :
                                
                               values=np.array([accumulated_score1[i-1, j] + distances[i, j], accumulated_score1[i-1, j-1]+ distances[i, j], accumulated_score1[i, j-1]+ distances[i, j]])
                               values[2]=values[2]
                               accumulated_score1[i,j]=np.max(values)
                                            
                               backtrackDir[i,j]=np.argmax(values) + 1
                               
                               
                            else :
                               values=np.array([accumulated_score1[i-1, j] + distances[i, j], accumulated_score1[i-1, j-1]+ distances[i, j], accumulated_score1[i, j-1]+ distances[i, j]])
                               values[2]=values[2] + PENALTY
                               values[0]=values[0] + PENALTY1
                               accumulated_score1[i, j] = np.max(values)
                               backtrackDir[i,j]=np.argmax(values) + 1 
            
            
            
            # backtrack > dont allow consecutive hor/vert movement
            j = len(x) -1
            i = len(y)-1
            path = [[i,j]]
            
            flag=GAPPermit
            
            while i>0 or j>0 :
                if (i > len(y)-BEGIN_SKIP) and (j > len(x)-BEGIN_SKIP) :
                    i=i-1
                    j=j-1
                else :
                    
                    if j==0:
                        i=i-1
                    elif i==0:
                        j=j-1
                    elif ((backtrackDir[i,j]==1) and (flag==GAPPermit)) :
                        i=i-1
                        flag=0
                    elif backtrackDir[i,j]==2 :
                        i=i-1
                        j=j-1
                        if flag < GAPPermit :
                           flag=flag + 1
                    elif ((backtrackDir[i,j]==3) and (flag==GAPPermit)) :
                        j=j-1
                        flag=0
                    else :
                        i=i-1
                        j=j-1
                        if flag < GAPPermit :
                           flag=flag + 1
                
                path.append([i, j])
             
            path_y = [point[0] for point in path]
            path_x = [point[1] for point in path]
            
            aligned_y=y[path_y]
            aligned_x=x[path_x]             
              
###############################################################
            
            return (aligned_y + aligned_x)/2

    
#clear all past variables    
clear_all()  

# mat File names
File1='L296'
File2='L345'


# Load First MatFile
matL1 = scipy.io.loadmat('./MatFiles/'+ File1 + '.mat')
matL1 = matL1[File1]

# Load Second MatFile
matL2 = scipy.io.loadmat('./MatFiles/'+ File2 +'.mat')
matL2 = matL2[File2]


#Matrix required to plot the shift and error
fig2Plot=[]
error2Plot=[]

# Number of series
if len(matL1[0])==len(matL2[0]) :
    NumSeries=len(matL1[0])
else :
    print("Files does not contain same  number of signals")


# Flags
                  

FIGURE_FLAG=0  # execute fiure plot blocks : save the figures
FIGURE_SHOW_FLAG=0 # show the figures : FIGURE_FLAG should be 1 

# Signal Transformation Flags : default : no transform
HILBERT_FLAG=0
ABSOLUTE_FLAG=0

# Flag for computing distance metric
#0-absolute difference
#1- Normalized correlation coeff
SIGNAL_DISTANCE_METRIC=1

# Flag for computation of accumulative cost
#0-> Normal calculation
#1-> verical/horzontal movement preventive calculation
ACC_COST_FLAG=1

# Method Flags 
# 0-DTW Signal strenth difference is taken as absolute value
# 1-DTW with end adjustment
# 2-DTW with signal Smoothing : By averaging window
# 3- DTW with signal Smoothing : aligning two consecutive signal
METHOD_FLAG=0


# Setting up window size
# averaging window
if METHOD_FLAG==2 :
    WindowSize=3 # windowsize
# aligning two consecutive signal
if METHOD_FLAG==3 :
    WindowSize=2 # windowsize    


# Penalty for gaps
PENALTY= 0
PENALTY1=0

# How many consecutive gaps are not allowed
GAPPermit=0
# No skip at the beginning
BEGIN_SKIP=25



# figurestring for saved figures
figstring="Normal"
secondsig=""
if File2=='L297' :
    secondsig="normal"
    figstring=figstring + secondsig
if File2=='L345' :
    secondsig= "water"
    figstring=figstring + secondsig
if File2=='L388' :
    secondsig = "waterdye"
    figstring=figstring + secondsig  


if HILBERT_FLAG==1 :
    figstring=figstring + "_Hil" 
elif ABSOLUTE_FLAG==1 :
    figstring=figstring + "_Abs" 
else :
    figstring=figstring + "_NoTrans" 

if (METHOD_FLAG==0) or (METHOD_FLAG==1):
    Signal_Count=NumSeries
else :
    Signal_Count=NumSeries-WindowSize+1


    
# directory to save Figures
figfolder='../DTW_v12/Figs/'

for it in range(Signal_Count):
    
            # DTW Signal strenth difference is taken as absolute value
            # or DTW with end adjustment
            if (METHOD_FLAG==0) or (METHOD_FLAG==1):
                    x11=matL1[:,it]
                    y11=matL2[:,it]
                    
            ## 2-DTW with signal Smoothing : By averaging window
            if METHOD_FLAG==2:
                    x11=matL1[:,it]
                    y11=matL2[:,it]
                    
                    # Get the signals by taking average
                    for count in range(1, WindowSize) :                
                        x12=matL1[:,it+count]
                        y12=matL2[:,it+count]
                        x11=x11+x12
                        y11=y11+y12
                        
                    x11=x11/WindowSize
                    y11=y11/WindowSize
            
            # 3- DTW with signal Smoothing : aligning two consecutive signals
            if METHOD_FLAG==3 :
                    x12=matL1[:,it]
                    y12=matL2[:,it]
                    x13=matL1[:,it+1]
                    y13=matL2[:,it+1]
                   
                    # align two consecutive signals
                    x11=aligned_av(x12,x13 ,HILBERT_FLAG, ABSOLUTE_FLAG, SIGNAL_DISTANCE_METRIC, ACC_COST_FLAG, METHOD_FLAG)
                    y11=aligned_av(y12,y13 ,HILBERT_FLAG, ABSOLUTE_FLAG, SIGNAL_DISTANCE_METRIC, ACC_COST_FLAG, METHOD_FLAG)
                    
        
                  
            # reverse the signals
            x1= x11[::-1]
            y1=y11[::-1]
                  
            #### Transformation of the signals based on flags
            if HILBERT_FLAG == 1 :
                
               analytic_signal1 = hilbert(x1)
               x = np.abs(analytic_signal1)
               
               analytic_signal2 = hilbert(y1)
               y = np.abs(analytic_signal2)
               
            elif ABSOLUTE_FLAG == 1 :
               x=np.abs(x1)
               y=np.abs(y1)
            else :
               x=x1
               y=y1
               
            ### Figures
            
            if FIGURE_FLAG==1 :
                if HILBERT_FLAG==1:
                   fig = plt.figure()
                
                   ax0 = fig.add_subplot(211)
                   ax0.plot(x11, 'c', label ='normal')
                   ax0.plot(x, 'r', label='x Hilbert Env1 ') 
                   ax0.legend(ncol=2)
                
                   ax1 = fig.add_subplot(212)
                   ax1.plot(y11, 'c', label= secondsig)
                   ax1.plot(y, 'b', label='y Hilbert Env2')
                   ax1.legend(ncol=2)
                   ax1.set_xlabel("time")
                   
                   plt.savefig(figfolder +figstring +'_{0}_1.eps'.format(it), format='eps')
                   if FIGURE_SHOW_FLAG :
                       plt.show()
                   plt.close(fig)
                   
                else :
                    
                   fig = plt.figure()
                
                   ax0 = fig.add_subplot(211)
                   ax0.plot(x11, 'r', label ='x-normal')
                   
                   ax0.legend(ncol=2)
                
                   ax1 = fig.add_subplot(212)
                   ax1.plot(y11, 'b', label='y-' + secondsig)
                   
                   ax1.legend(ncol=2)
                   ax1.set_xlabel("time")
                   plt.savefig(figfolder +figstring +'_{0}_1.eps'.format(it), format='eps')
                   if FIGURE_SHOW_FLAG :
                       plt.show()
                   plt.close(fig)
                   
            # Find out distances between all pair of points
            distances =np.zeros((len(y), len(x)))
            
            # Find out the mean in case of normalized corr coeff
            if SIGNAL_DISTANCE_METRIC==1 :
                meanx=np.mean(x)
                meany=np.mean(y)
                
            # fill out the matrix using distance metric
            for i in range(len(y)) :
                for j in range(len(x)) :
                    if SIGNAL_DISTANCE_METRIC==0 :
                        distances[i,j]=np.absolute(x[j]-y[i])
                    if SIGNAL_DISTANCE_METRIC==1 :
                        distances[i,j]=(y[i]-meany)*(x[j]-meanx) 
                        
                        
                        
            # Accumulated Cost
            # Two accumulated  costs one is only diagonal (2) and another is best among three (1)
            accumulated_score1=np.zeros((len(y), len(x)))
            if ACC_COST_FLAG==1 :
                accumulated_score2=np.zeros((len(y), len(x)))
                
                
            backtrackDir=np.zeros((len(y), len(x)))
                        
   

    
            # initialize base rows
            for i in range(1, len(x)):
                accumulated_score1[0,i] = distances[0,i] + accumulated_score1[0, i-1] + PENALTY
                if ACC_COST_FLAG==1 :
                    accumulated_score2[0,i] = distances[0,i] + accumulated_score2[0, i-1] + PENALTY
                backtrackDir[0,i]=3
                
        
            for i in range(1, len(y)):
                accumulated_score1[i,0] = distances[i, 0] + accumulated_score1[i-1, 0] + PENALTY1
                if ACC_COST_FLAG==1 :
                    accumulated_score2[i,0] = distances[i, 0] + accumulated_score2[i-1, 0] + PENALTY1 
                backtrackDir[i,0]=1
                
            # compute rest of the accumulative distances
            for i in range(1, len(y)):
                for j in range(1, len(x)):
                    
                    # This one is for two types of costs : diagonal and overall
                    if ACC_COST_FLAG==1 :
                            
                            if (METHOD_FLAG == 1) or (METHOD_FLAG == 2) or (METHOD_FLAG == 3) :
                                
                                # This is when we reach end of the signal
                                if i==len(y)-1 :
                                   values=np.array([accumulated_score2[i-1, j] + distances[i, j], accumulated_score1[i-1, j-1]+ distances[i, j], accumulated_score2[i, j-1]+ np.mean(distances[i, j:])])
                                   values[2]=values[2]
                                   accumulated_score1[i,j]=np.max(values)
                                   accumulated_score2[i,j]=values[1]                       
                                   backtrackDir[i,j]=np.argmax(values) + 1
                                
                                # This is for all other situations                                      
                                else :
                                    
                                   values=np.array([accumulated_score2[i-1, j] + distances[i, j], accumulated_score1[i-1, j-1]+ distances[i, j], accumulated_score2[i, j-1]+ distances[i, j]])
                                   values[2]=values[2] + PENALTY
                                   values[0]=values[0] + PENALTY1
                                   accumulated_score1[i, j] = np.max(values)
                                   accumulated_score2[i,j]=values[1]
                                   backtrackDir[i,j]=np.argmax(values) + 1 
                            
                            # This is original DTW
                            if (METHOD_FLAG == 0) :
                                
                                # This is when we reach end of the signal No penalty
                                if i==len(y)-1 :
                                   values=np.array([accumulated_score1[i-1, j] + distances[i, j], accumulated_score1[i-1, j-1]+ distances[i, j], accumulated_score1[i, j-1]+ distances[i, j]])
                                   values[2]=values[2]
                                   accumulated_score1[i,j]=np.max(values)
                                   accumulated_score2[i,j]=values[1]                       
                                   backtrackDir[i,j]=np.argmax(values) + 1
                                
                                # This is for all other situations                                      
                                else :
                                    
                                   values=np.array([accumulated_score2[i-1, j] + distances[i, j], accumulated_score1[i-1, j-1]+ distances[i, j], accumulated_score2[i, j-1]+ distances[i, j]])
                                   values[2]=values[2] + PENALTY
                                   values[0]=values[0] + PENALTY1
                                   accumulated_score1[i, j] = np.max(values)
                                   accumulated_score2[i,j]=values[1]
                                   backtrackDir[i,j]=np.argmax(values) + 1 
                               
                  
                               
                    if ACC_COST_FLAG==0 :
                        
                        if (METHOD_FLAG == 1) or (METHOD_FLAG == 2) or (METHOD_FLAG == 3) :
                          
                            if i==len(y)-1 :
                                
                               values=np.array([accumulated_score1[i-1, j] + distances[i, j], accumulated_score1[i-1, j-1]+ distances[i, j], accumulated_score1[i, j-1]+ np.mean(distances[i, j:])])
                               values[2]=values[2]
                               accumulated_score1[i,j]=np.max(values)
                                            
                               backtrackDir[i,j]=np.argmax(values) + 1
                               
                               
                            else :
                               values=np.array([accumulated_score1[i-1, j] + distances[i, j], accumulated_score1[i-1, j-1]+ distances[i, j], accumulated_score1[i, j-1]+ distances[i, j]])
                               values[2]=values[2] + PENALTY
                               values[0]=values[0] + PENALTY1
                               accumulated_score1[i, j] = np.max(values)
                               backtrackDir[i,j]=np.argmax(values) + 1 

                        if (METHOD_FLAG == 0) :
                          
                            if i==len(y)-1 :
                                
                               values=np.array([accumulated_score1[i-1, j] + distances[i, j], accumulated_score1[i-1, j-1]+ distances[i, j], accumulated_score1[i, j-1]+ distances[i, j]])
                               values[2]=values[2]
                               accumulated_score1[i,j]=np.max(values)
                                            
                               backtrackDir[i,j]=np.argmax(values) + 1
                               
                               
                            else :
                               values=np.array([accumulated_score1[i-1, j] + distances[i, j], accumulated_score1[i-1, j-1]+ distances[i, j], accumulated_score1[i, j-1]+ distances[i, j]])
                               values[2]=values[2] + PENALTY
                               values[0]=values[0] + PENALTY1
                               accumulated_score1[i, j] = np.max(values)
                               backtrackDir[i,j]=np.argmax(values) + 1 
            
            
            
            # backtrack > dont allow consecutive hor/vert movement
            j = len(x) -1
            i = len(y)-1
            path = [[i,j]]
            
            flag=GAPPermit
            
            while i>0 or j>0 :
                if (i > len(y)-BEGIN_SKIP) and (j > len(x)-BEGIN_SKIP) :
                    i=i-1
                    j=j-1
                else :
                    
                    if j==0:
                        i=i-1
                    elif i==0:
                        j=j-1
                    elif ((backtrackDir[i,j]==1) and (flag==GAPPermit)) :
                        i=i-1
                        flag=0
                    elif backtrackDir[i,j]==2 :
                        i=i-1
                        j=j-1
                        if flag < GAPPermit :
                           flag=flag + 1
                    elif ((backtrackDir[i,j]==3) and (flag==GAPPermit)) :
                        j=j-1
                        flag=0
                    else :
                        i=i-1
                        j=j-1
                        if flag < GAPPermit :
                           flag=flag + 1
                
                path.append([i, j])
             
            path_y = [point[0] for point in path]
            path_x = [point[1] for point in path] 
            
            path_y1 = [len(y)-point[0]-1 for point in path]
            path_x1 = [len(x)-point[1]-1 for point in path] 
            
            aligned_y=y[path_y]
            aligned_x=x[path_x]
            
   
            
            # Plot Aligned Signal
            
            if FIGURE_FLAG==1 :
               fig = plt.figure()
               plt.suptitle('Signals Before and After Alignment')
               ax0 = fig.add_subplot(211)
               ax0.plot(x11, 'r', label ='x-normal')
               ax0.plot(y11, 'b', label='y-' + secondsig) 
               ax0.legend(ncol=2)
            
               ax1 = fig.add_subplot(212)
               ax1.plot(aligned_x, 'r', label ='x align')
               ax1.plot(aligned_y, 'b', label='y align') 
               ax1.legend(ncol=2)
               ax1.set_xlabel("time")
               plt.savefig(figfolder +figstring +'_before_and_after_pos{0}.eps'.format(it), format='eps')
               if FIGURE_SHOW_FLAG :
                  plt.show() 
               plt.close(fig)
            
            
            acc_cost1=  accumulated_score1[path_y1,path_x1] 
               
            LinX=[]
            LinY=[]
            for a1 in range(len(x)):
                LinX.append(a1) 
                LinY.append(a1)
                
            if FIGURE_FLAG==1 :  
               fig = plt.figure()
               plt.plot(path_x1, path_y1, label ='alignment path');
               plt.plot(LinX,LinY, 'r', label ='No alignment')
               plt.legend(ncol=2, loc=2)
               plt.savefig(figfolder +figstring + 'Alignmentpath_pos{0}.eps'.format(it), format='eps')
               if FIGURE_SHOW_FLAG :
                   plt.show()
               plt.close(fig)
            
            ### debug
            
            y_m=[]
            x_m=[]
            
        
            
            for count in range(len(x)) :
                indices=[]
                x_m.append(count)
                indices=[i1 for i1, j1 in enumerate(path_x1) if j1 == count]
                if len(indices) > 0 :
                   #y_min=min([path_y[i2] for i2 in indices ])
                   y_mean=int(np.mean([path_y1[i2] for i2 in indices ]))
                else :
                   y_mean = count
                y_m.append(y_mean)
                
                
            diff= np.subtract(y_m, x_m)
            error2Plot.append( np.abs(y11[y_m] - x11[x_m]))   
            fig2Plot.append(diff)
            #pdb.set_trace() 

               
visualize_Matrix(fig2Plot)
visualize_errorMatrix(error2Plot) 
filename1=figstring + '_adist.pckl'
f = open(filename1, 'wb')
pickle.dump(fig2Plot,  f)
f.close()

filename2=figstring + '_err.pckl'
f = open(filename2, 'wb')
pickle.dump(error2Plot,  f)
f.close()               
               
     
    
