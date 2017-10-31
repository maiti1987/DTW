# -*- coding: utf-8 -*-
"""
Created on Thu May 26 14:30:28 2016

@author: aniruddha
"""

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

def clear_all():
    """Clears all the variables from the workspace of the spyder application."""
    gl = globals().copy()
    for var in gl:
        if var[0] == '_': continue
        if 'func' in str(globals()[var]): continue
        if 'module' in str(globals()[var]): continue

        del globals()[var]

# Method to visualize Distance Matrix
def visualize_Distance_Matrix (distances) :
    im=plt.imshow(distances, interpolation='nearest', cmap='BrBG')
    plt.gca().invert_yaxis()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid()
    plt.colorbar()
    plt.show()
#-----------------------------------------

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


def aligned_av(x111,y111, HILBERT_FLAG, ABSOLUTE_FLAG) :
    
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
            


            # Find out distances between all pair of points
            distances =np.zeros((len(y), len(x)))
            meanx=np.mean(x)
            meany=np.mean(y)
            # fill out the matrix using euclidean distances
            for i in range(len(y)) :
                for j in range(len(x)) :
                    distances[i,j]=(y[i]-meany)*(x[j]-meanx) # + 5*np.sqrt(i+j+1)/abs(i-j)

            # Accumulated Cost
            # Now two accumulated  costs one is only diagonal (2) and another is best among three (1)
            accumulated_score1=np.zeros((len(y), len(x)))
            accumulated_score2=np.zeros((len(y), len(x)))
            
            backtrackDir=np.zeros((len(y), len(x)))
            

            # start with 
            accumulated_score1[0,0] = 0
            accumulated_score2[0,0] = 0

            # initialize base rows
            for i in range(1, len(x)):
                accumulated_score1[0,i] = distances[0,i] + accumulated_score1[0, i-1] + PENALTY
                accumulated_score2[0,i] = distances[0,i] + accumulated_score2[0, i-1] + PENALTY
                backtrackDir[0,i]=3
                

            for i in range(1, len(y)):
                accumulated_score1[i,0] = distances[i, 0] + accumulated_score1[i-1, 0] + PENALTY1
                accumulated_score2[i,0] = distances[i, 0] + accumulated_score2[i-1, 0] + PENALTY1 
                backtrackDir[i,0]=1
                
                
                
                
            # compute rest of the accumulative distances
            # No consecutive deletion/streching allowed now
                
            for i in range(1, len(y)):
                for j in range(1, len(x)):

                    
                    if i==len(y)-1 :
                        
                       values=np.array([accumulated_score2[i-1, j] + distances[i, j], accumulated_score1[i-1, j-1]+ distances[i, j], accumulated_score2[i, j-1]+ np.mean(distances[i, j:])])
                       values[2]=values[2]
                       accumulated_score1[i,j]=np.max(values)
                       accumulated_score2[i,j]=values[1]                       
                       backtrackDir[i,j]=np.argmax(values) + 1
                       
#                       # change backtracking histry in case of vertical or horizontal movement
#                       if ((backtrackDir[i,j]==1) and (i !=1)) :
#                          backtrackDir[i-1,j]=2
#                       if ((backtrackDir[i,j]==3) and (j != 1)) :
#                          backtrackDir[i,j-1]=2
                    
                          
                       
                    else :
                       values=np.array([accumulated_score2[i-1, j] + distances[i, j], accumulated_score1[i-1, j-1]+ distances[i, j], accumulated_score2[i, j-1]+ distances[i, j]])
                       values[2]=values[2] + PENALTY
                       values[0]=values[0] + PENALTY1
                       accumulated_score1[i, j] = np.max(values)
                       accumulated_score2[i,j]=values[1]
                       backtrackDir[i,j]=np.argmax(values) + 1
                       
                       
                       
            j = len(x) -1
            i = len(y)-1
            path = [[i,j]]
            
            # backtrack > dont allow consecutive hor/vert movement
            flag=1
            
            while i>0 or j>0 :
                if j==0:
                    i=i-1
                elif i==0:
                    j=j-1
                elif ((backtrackDir[i,j]==1) and (flag==1)) :
                    i=i-1
                    flag=0
                elif backtrackDir[i,j]==2 :
                    i=i-1
                    j=j-1
                    flag=1
                elif ((backtrackDir[i,j]==3) and (flag==1)) :
                    j=j-1
                    flag=0
                else :
                    i=i-1
                    j=j-1
                    flag=1
                
                path.append([i, j])
             
            path_y = [point[0] for point in path]
            path_x = [point[1] for point in path] 
            
            
            aligned_y=y[path_y]
            aligned_x=x[path_x]
            
            return (aligned_y + aligned_x)/2


#clear all past variables    
clear_all()    
# mat File names
File1='L296'
File2='L345'

# Load First MatFile
matL1 = scipy.io.loadmat('./MatFiles/'+ File1 + '.mat')
matL1 = matL1[File1]

# Load First MatFile
matL2 = scipy.io.loadmat('./MatFiles/'+ File2 +'.mat')
matL2 = matL2[File2]


TotalAccCost=[]

fig2Plot=[]
error2Plot=[]
NumSeries=len(matL1[0])
# initialize counter

FIGURE_FLAG=0
FIGURE_SHOW_FLAG=0

PENALTY=0
PENALTY1=0

HILBERT_FLAG=0
ABSOLUTE_FLAG=0

WindowSize=2

# figurestring
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
    figstring=figstring + "NoTrans_al" 

# directory
figfolder='../DTW_v12/Align_NW_1g_NH/'

for it in range(NumSeries-WindowSize+1):

            
            # pdb.set_trace()
            #it=190
            # Print counter
            
            print(it)
            x12=matL1[:,it]
            y12=matL2[:,it]
            x13=matL1[:,it+1]
            y13=matL2[:,it+1]
            
         
                
            x11=aligned_av(x12,x13 ,HILBERT_FLAG, ABSOLUTE_FLAG)
            y11=aligned_av(y12,y13 ,HILBERT_FLAG, ABSOLUTE_FLAG)
            
            
            
            x1= x11[::-1]
            y1=y11[::-1]
          
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
            meanx=np.mean(x)
            meany=np.mean(y)
            # fill out the matrix using euclidean distances
            for i in range(len(y)) :
                for j in range(len(x)) :
                    distances[i,j]=(y[i]-meany)*(x[j]-meanx) # + 5*np.sqrt(i+j+1)/abs(i-j)

            # Accumulated Cost
            # Now two accumulated  costs one is only diagonal (2) and another is best among three (1)
            accumulated_score1=np.zeros((len(y), len(x)))
            accumulated_score2=np.zeros((len(y), len(x)))
            
            backtrackDir=np.zeros((len(y), len(x)))
            

            # start with 
            accumulated_score1[0,0] = 0
            accumulated_score2[0,0] = 0

            # initialize base rows
            for i in range(1, len(x)):
                accumulated_score1[0,i] = distances[0,i] + accumulated_score1[0, i-1] + PENALTY
                accumulated_score2[0,i] = distances[0,i] + accumulated_score2[0, i-1] + PENALTY
                backtrackDir[0,i]=3
                

            for i in range(1, len(y)):
                accumulated_score1[i,0] = distances[i, 0] + accumulated_score1[i-1, 0] + PENALTY1
                accumulated_score2[i,0] = distances[i, 0] + accumulated_score2[i-1, 0] + PENALTY1 
                backtrackDir[i,0]=1
                
                
                
                
            # compute rest of the accumulative distances
            # No consecutive deletion/streching allowed now
                
            for i in range(1, len(y)):
                for j in range(1, len(x)):

                    
                    if i==len(y)-1 :
                        
                       values=np.array([accumulated_score2[i-1, j] + distances[i, j], accumulated_score1[i-1, j-1]+ distances[i, j], accumulated_score2[i, j-1]+ np.mean(distances[i, j:])])
                       values[2]=values[2]
                       accumulated_score1[i,j]=np.max(values)
                       accumulated_score2[i,j]=values[1]                       
                       backtrackDir[i,j]=np.argmax(values) + 1
                       
#                       # change backtracking histry in case of vertical or horizontal movement
#                       if ((backtrackDir[i,j]==1) and (i !=1)) :
#                          backtrackDir[i-1,j]=2
#                       if ((backtrackDir[i,j]==3) and (j != 1)) :
#                          backtrackDir[i,j-1]=2
                    
                          
                       
                    else :
                       values=np.array([accumulated_score2[i-1, j] + distances[i, j], accumulated_score1[i-1, j-1]+ distances[i, j], accumulated_score2[i, j-1]+ distances[i, j]])
                       values[2]=values[2] + PENALTY
                       values[0]=values[0] + PENALTY1
                       accumulated_score1[i, j] = np.max(values)
                       accumulated_score2[i,j]=values[1]
                       backtrackDir[i,j]=np.argmax(values) + 1
                       
#                       # change backtracking history in case of vertical or horizontal movement
#                       if ((backtrackDir[i,j]==1) and (i !=1)) :
#                          backtrackDir[i-1,j]=2
#                          # accumulated_score1[i-1, j] =accumulated_score2[i-1,j]
#                          
#                       if ((backtrackDir[i,j]==3) and (j != 1)) :
#                          backtrackDir[i,j-1]=2
                       
                       
            j = len(x) -1
            i = len(y)-1
            path = [[i,j]]
            
            # backtrack > dont allow consecutive hor/vert movement
            flag=1
            
            while i>0 or j>0 :
                if j==0:
                    i=i-1
                elif i==0:
                    j=j-1
                elif ((backtrackDir[i,j]==1) and (flag==1)) :
                    i=i-1
                    flag=0
                elif backtrackDir[i,j]==2 :
                    i=i-1
                    j=j-1
                    flag=1
                elif ((backtrackDir[i,j]==3) and (flag==1)) :
                    j=j-1
                    flag=0
                else :
                    i=i-1
                    j=j-1
                    flag=1
                
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


filename1=figstring + '_adist.pckl'
f = open(filename1, 'wb')
pickle.dump(fig2Plot,  f)
f.close()

filename2=figstring + '_err.pckl'
f = open(filename2, 'wb')
pickle.dump(error2Plot,  f)
f.close()
           
K1=[]
M1=[]
for sig in fig2Plot :
       K1.append(sig[0:300])

for sig in error2Plot :
       M1.append(sig[0:300])

            
visualize_Matrix(K1)
visualize_errorMatrix(M1) 
