#!/usr/bin/python
'''
Created on July 11, 2014
Testing the H matrices across the various depth planes.

Functional but not user-friendly yet!

@author: carlos torres <carlitos408@gmail.com>
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
import imtools


pathH = '/home/carlos/Documents/PYTHON/ComputerVision/wound/npOut5/'
pathH1 = '/home/carlos/Documents/PYTHON/ComputerVision/wound/npOut/'
pathH2 = '/home/carlos/Documents/PYTHON/ComputerVision/wound/npOut2/'

def load_H_paths(pathHs = '/home/carlos/Documents/PYTHON/ComputerVision/wound/npOut/',name='Hmatrix',ext='out'):
    """Uses imtools to load the paths to precomputed H matrices. The sorted idx
    list repensents the distance to the device (605 = 605mm).
    (str) -> (list, list)"""
    Hs, idx = imtools.get_nat_list(pathHs,name,ext)
    return Hs,idx
#load_H_paths

def evalHs(pathH1):
    ''' Homography decomposition
    H = |f 0 0| |r11 r12 tx| = |fr11 fr12 ftx| = |h11 h12 h13|
        |0 f 0| |r21 r22 ty|   |fr21 fr22 fty|   |h21 h22 h23|
        |0 0 1| |r31 r32 tz|   |r31  r32  tz |   |h31 h32 h33|
    
    H_translation = |1, 0, tx|
                    |0, 1, ty|
                    |0, 0,  1| '''
                    
    I = np.eye(3)
    # load all the homographies and pick the one closes to pixel value
    Hs,idx = load_H_paths(pathH1) # use the default values to load all the H matrices
    #print 'Hs & idx:', len(Hs), len(idx)
    x_offset=[]
    y_offset=[]
    for i in xrange(len(idx)):
        #print i
        Hname = Hs[i]
        #print '\t',Hname
        H = np.loadtxt(Hname)
        S = H* H.T-I
        T = np.array([[1,0,1], [0,1,1], [0,0,1]])
        TH = T*H # purely translation homography 
        x_offset.append(TH[0,2])
        y_offset.append(TH[1,2])
        
        # extracting the H elements
    
    x_offset = np.asarray(x_offset, dtype = np.float)
    y_offset = np.asarray(y_offset, dtype = np.float)
    return idx, x_offset, y_offset
#evalHs


def decomposeH(H_L=None):#(H):
    ''' Using vision UCLA's book ch-5 section 5.3.3
    ref: http://vision.ucla.edu//MASKS/MASKS-ch5.pdf
    Numerical example 5.20 is particularly useful for Testing implementation.

    H_L = lambda*(R+(1/d)*T*N.T), = H*lambda, so the componnets are:

    H = {R, T/d, N} 
    S = H.T * H = V*Sigma*V.T
    Sigma = diag{sig_1**2,sig_2**2, sig_3**2 }, singular values of H
    [v_1, v_2, v_3] = column vectors of V, aka singular vectors of H
    
    R = |cos()  0  sin()|     dT = | Tx/d |
        |0      1      0|          | Ty/d |
        |-sin() 0  cos()|          | Tz/d |
    ''' 

    if H_L == None: # test matrix from ref example 5.20
        H_L = np.array([[5.404, 0, 4.436],[0, 4, 0],[-1.236, 0, 3.804] ])
    #print 'Processing input H_L:\n\t', H_L
    _, k , _ = np.linalg.svd(H_L)
    k = k[1]
    # nomalize by the scale factor = middle singular value
    H = H_L / k

    ## Start the decomposition of H
    #NOTE: numpy method: U,s,V = np.linalg.svd()
    S = np.dot(H.T,H)
    V, s, VT = np.linalg.svd(S, full_matrices=1, compute_uv=1)
    if np.linalg.det(VT) == -1:
        #print 'Determinant test: switch (V) -> (-V)' 
        V = -V
        VT= -VT
    VT= VT.T
    # column vectors of V
    v1 = V[:,0]
    v2 = V[:,1]
    v3 = V[:,2]
    #compute vectors u1 & u2:
    a = np.sqrt( 1 -s[2] )
    b = np.sqrt( s[0] -1 )
    c = np.sqrt( s[0] - s[2] )
    u1= (a*v1 + b*v3) / c
    u2= (a*v1 - b*v3) / c    
    #Computes unit-normal vectors using the crossproduct
    N1 = np.cross(v2,u1)
    N2 = np.cross(v2,u2)
    v2_hat_u1 = N1 / np.linalg.norm(N1)
    v2_hat_u2 = N2 / np.linalg.norm(N2)

    def composeMatrices():
        '''
        U1 = [v2,   u1, ^(v2)u1  ]
        U2 = [v2,   u2, ^(v2)u2  ]
        W1 = [Hv2, Hu1, ^(Hv2)Hu1]
        W2 = [Hv2, Hu2, ^(Hv2)Hu2]
        '''
        # Build the matrices from the vectors. Order is tricky in numpy
        U1 = np.array([v2, u1, v2_hat_u1]).T
        U2 = np.array([v2, u2, v2_hat_u2]).T
        
        Hv2 = np.dot(H,v2)
        
        Hu1 = np.dot(H,u1)
        Hu2 = np.dot(H,u2)
        Hv2_hat_Hu1 = np.cross(Hv2, Hu1)
        Hv2_hat_Hu2 = np.cross(Hv2, Hu2)
        # normalize
        Hv2_hat_Hu1 = Hv2_hat_Hu1/np.linalg.norm( Hv2_hat_Hu1,2 )
        Hv2_hat_Hu2 = Hv2_hat_Hu2/np.linalg.norm( Hv2_hat_Hu2,2 )
        
        W1 = np.array([Hv2, Hu1, Hv2_hat_Hu1]).T
        W2 = np.array([Hv2, Hu2, Hv2_hat_Hu2]).T
        
        # Lists to store the components
        R_list     = []
        N_list     = []
        dT_list    = []
        scale_list = []
        theta_list = []
        tx_list    = []
        ty_list    = []
        tz_list    = []
        
        ## Decomposition solutions
        # sol1:
        R1  = np.round(np.dot(W1,U1.T),decimals=3)
        N1  = np.round(v2_hat_u1,3)
        dT1 = np.round(np.dot((H-R1),N1),3)
        # sol2:
        R2  = np.round(np.dot(W2,U2.T),3)
        N2  = np.round(v2_hat_u2,3)
        dT2 = np.round(np.dot((H-R2),N2),3)
        # sol3:
        R3  = R1
        N3  = -N1
        dT3 = -dT1
        # sol 4: 
        R4  = R2
        N4  = -N2
        dT4 = -dT2

        # Impose positive depth constraint -> in front of the camera: N.T > 0
        # This constraint yields 2 possible solutions.        
        if (np.sum(N1)>0) & (dT1[2]>0):
            R_list.append(R1)
            N_list.append(N1)
            dT_list.append(dT1)
            #scale_list.append(k)
            #theta_list.append( np.arccos(R1[0,0]) ) # radians
            #tx_list.append(dT1[0])
            #ty_lixt.append(dT1[1])
            #tz_list.append(dT1[2])
            #print '\nR1:\n\t',  R1
            #print '\nN1:\n\t',  N1
            #print '\ndT1:\n\t', dT1
        if (np.sum(N2)>0) & (dT2[2]>0):
            R_list.append(R2)
            N_list.append(N2)
            dT_list.append(dT2)
        if (np.sum(N3)>0) & (dT3[2]>0):
            R_list.append(R3)
            N_list.append(N3)
            dT_list.append(dT3)
        if (np.sum(N4)>0) & (dT4[2]>0):
            R_list.append(R4)
            N_list.append(N4)
            dT_list.append(dT4)
        return R_list, N_list, dT_list, k
    #composeMatrices()

    Rl, Nl, dTl,k = composeMatrices()
    #print '\n\n=== The Decomposed Viable Solution Elements === '
    #print 'Normalized H:\n\t', H
    #print 'Found %d solution(s) meeting the depth constraints.'%(len(Rl))
    if len(Rl)>0:
        # pick one of the solutions only
        # k = scale factor
        R = Rl [0]
        N = Nl [0]
        dT= dTl[0]
        theta = np.arccos(R[0,0])
        tx = dT[0]
        ty = dT[1]
        tz = dT[2]
    else:
        R=0
        N=0
        dT=0
        theta = 0
        tx = 0
        ty = 0
        tz = 0
    return k, theta, tx, ty, tz
# decomposeH


def main():
    d, x, y = evalHs(pathH)
    #print 'Displacements for h at %d:  %d-x, %d-y:'%(d[0],x[0],y[0])
    # plto the results:
    plt.plot(d,x,'r', d,y,'b')
    plt.title('H displacements as function of depth')
    plt.ylabel('H translation offset')
    plt.xlabel('Distance to sensor [mm]')
    #plt.axis([300, 700, -5, 50])
    plt.grid(True)
    plt.legend(['x','y'])
    plt.show()
#main()



### test matrix
##H = np.array([[1.037, 2.034e-02, 2.534e+01],
##              [-2.948e-05, 1.045, 2.925],
##              [-2.948e-05, -9.874e-06, 1.00]])
##
### The parameters from the Homography
##k, theta, tx,ty,tz = decomposeH(H)
##print 'Scale factor(k): ' , k
##print 'Rotation angle = %d rads ' %(np.round(theta,3)) # three decimals
##print 'Translations [%d-x, %d-y, %d-z]'%(tx,ty,tz)


def run1():


    Hs,idx = load_H_paths(pathH) # use the default values to load all the H matrices
    k_list = []
    theta_list = []
    tx_list = []
    ty_list = []
    tz_list = []

    for i in xrange(len(idx)):
        #print i
        Hname = Hs[i]
        #print '\t',Hname
        H = np.loadtxt(Hname)
        k, theta, tx,ty,tz = decomposeH(H)
        k_list.append(k)
        theta_list.append(theta)
        tx_list.append(tx)
        ty_list.append(ty)
        tz_list.append(tz)
    k = np.asarray(k_list)
    theta = (np.asarray(theta_list))
    

    
    tx = np.asarray(tx_list)
    ty = np.asarray(ty_list)
    tz = np.asarray(tz_list)
    # moving averages
    theta_ave = imtools.movingAverage(theta,10)
    txa = imtools.movingAverage(tx)
    tya = imtools.movingAverage(ty)
    tza = imtools.movingAverage(tz)
    
    print 'Generating the plots'
    ## The plots
    #plt.figure(2)
    plt.subplot(211)
    plt.plot(idx,txa,'r', idx,tya,'b', idx, tza,'g')
    plt.title('Translation displacements as function of depth')
    plt.ylabel('H translation offset')
    plt.xlabel('Distance to sensor [mm]')
##    plt.axis([450, 1000, -20, 50])
    plt.legend(['tx','ty', 'tz'])
    plt.grid(True)
    
    #plt.figure(3)
    plt.subplot(212)
    plt.plot(idx,theta,'ro', idx,theta_ave,'r', idx,k,'b')
    plt.title('Rot Angle and Scale as function of Depth')
    plt.ylabel('Magnitude')
    plt.xlabel('Distance to sensor [mm]')
##    plt.axis([450, 1000, -5, 5])
    plt.legend(['Theta','ThetaAve','Scale'])
    plt.show()
    
##
##    Hs2,idx2 = load_H_paths(pathH2) # use the default values to load all the H matrices
##    k_list = []
##    theta_list = []
##    tx_list = []
##    ty_list = []
##    tz_list = []
##
##    for i in xrange(len(idx2)):
##        #print i
##        Hname = Hs2[i]
##        #print '\t',Hname
##        H = np.loadtxt(Hname)
##        k, theta, tx,ty,tz = decomposeH(H)
##        k_list.append(k)
##        theta_list.append(theta)
##        tx_list.append(tx)
##        ty_list.append(ty)
##        tz_list.append(tz)
##    k2 = np.asarray(k_list)
##    theta2 = np.asarray(theta_list)
##    tx2 = np.asarray(tx_list)
##    ty2 = np.asarray(ty_list)
##    tz2 = np.asarray(tz_list)
##




##    ## The plots
##    plt.figure(4)
##    #plt.subplot(411)
##    plt.plot(idx2,tx2,'r', idx2,ty2,'b', idx2,tz2,'g')
##    plt.title('Translation displacements as function of depth')
##    plt.ylabel('H translation offset')
##    plt.xlabel('Distance to sensor [mm]')
##    plt.axis([300, 700, -5, 50])
##    plt.legend(['x','y', 'z'])
##    plt.figure(5)
##    #plt.subplot(412)
##    plt.plot(idx2,theta2,'r', idx2,k2,'b')
##    plt.title('Rot Angle and Scale as function of Depth')
##    plt.ylabel('Magnitude')
##    plt.xlabel('Distance to sensor [mm]')
##    plt.axis([300, 700, 0, 5])
##    plt.legend(['Angle','Scale'])
##    plt.show()
#run1()

run1()

##main()
