#!/usr/bin/python
'''
Version Aug 11, 2014
CURRENT FEATURES:
1. List ofImages under a path
    a. All images
    b. Natural order
    c. Histogram Equalization
    d. Moving average
    e. Median array
    f. Accurate system clock timed-event
    g. Rotation: Euler-Rodriguez Formula

@author: Carlos Torres <carlitos408@gmail.com>
'''

import os
import numpy as np
import math


def get_nat_list(path, name="file", ext = ".txt"):
    """ 
    Returns a list of PATHS for all fils w the given sub-strings
    in NATURAL ORDER located in the given path.
    usage:
    for folder with files named: [filename0.txt, filename1.txt,... ,filenameN.txt]
    files = get_nat_list('../root/data/', 'filename','.txt') 
    (str,str,str) -> (list)
    """
    # list of paths:
    names = [os.path.join(path,f) for f in os.listdir(path) if (( f.endswith(ext) or f.endswith(ext.upper())) 
           and (f.count(name)>0 ) )]
    names = sorted(names, key=lambda x:int(x.split(name)[1].split(".")[0]))
    # images idx numbers:
    idx = sorted([ n.split(name)[1].split(".")[0] for n in names ])
    if len(names)== 0:
        print "No images with the given conditions were found... please check that they exist!"
        print "The given search conditions were: ", name
    idx = np.asarray(idx,dtype=int)
    return names, idx
#get_nat_imlist


# generate a list of image_filenames of all the images in a folder
def get_imlist(path):
    """ 
    Returns a list of filenames for all jpg images in a directory.
    """
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]
#get_imlist


def eqImg(im):
    """
    Histogram equalization.
    """
    hist, bins = np.histogram(im.flatten(), 256,[0,256])
    
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/ cdf.max()
    
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    im = cdf[im]
    return im
#eqImg


def movingAverage(values,window=3):
    """
    Moving average of an array (values) within a given window
    (array)(scalar) -> (array)
    Example:
    dataset = [1,5,7,2,6,7,8,2,2,7,8,3,7,3,7,3,15,6]
    #Will print out a 3MA for our dataset
    print movingAverage(dataset,3)
    """
    window = np.ones(int(window))/float(window)
    ##smas = np.convolve(values, weigths, 'valid')
    smas = np.convolve(values, window, 'same')
    return smas # as a numpy array
#movingAverage()


def medianArray(array_list):
    """
    Computes the median array for list of arrays [a, b, c, ...,n]
    d[i,j] = median(a[1,j],b[1,j],c[1,j])
    (list[np.array, np.array,..., np.array]) -> np.array
    """
    for ii in xrange(len(array_list)):
        a = array_list[ii]
        # check number of channels
        if len(a.shape)>2: # 3channel images
            a = np.mean(a,axis=2)
        array_list[ii] = a
    med = np.asarray(array_list)
    med = np.uint8(np.median(med,axis=0))
    return med
#medianArray

def timeEvent(t=2,d=2):
    """
    Creates a variable pause based on the time at which the function was 
    called. For example, function called at 1:13:50 with minute_tick: t=2 and 
    minute_delay: d=2 will pause until 1:16:00. Which is the next minute_tick 
    that meets the factors of t=2 condition and the minute_delay > 2 (d=2)
    """
    from time import localtime, gmtime, strftime
    x = strftime("%a, %d %b %Y %H:%M:%S:%s", gmtime())
    c = localtime() # struct
    print 'First executed at time: %d:%d:%.2f\n' %(c.tm_hour, c.tm_min, c.tm_sec)
    ctrl = False
    done = False
    v = np.asarray(xrange(0,60,t)) # 1D array of  ticks

    while not done:
        c = localtime()
        m,s = c.tm_min, c.tm_sec/60.0 #min & seconds(converted to minutes
        dst = np.abs(v-(m+s+d))
        idx = np.argmin(dst)

        if (not ctrl):
            tick = v[idx]
            mm = tick-m-s
            print 'Time left: %d mins & %d secs. Looking for tick %d\n'%(np.floor(mm),(mm-np.floor(mm))*60 , tick)
            ctrl = True
        if m == tick:
            c = localtime()
            print 'Reached tick %dhr:%dmm:%dss\n'%(c.tm_hour, c.tm_min, c.tm_sec)
            r = 1
            k = m
            ctrl = False
            done = True
    print 'Event Timed!'
    return r,k
#timeEvent()



def rotate(s,theta=0,axis='x'):
    """
    Counter Clock wise rotation of a vector s, along the axis by angle theta
    s:= array/list of scalars. Contains the vector coordinates [x,y,z]
    theta:= scalar, <degree> rotation angle for counterclockwise rotation
    axis:= str, rotation axis <x,y,z>
    """
    theta = np.radians(theta) # degree -> radians
    r = 0
    if axis.lower() == 'x':
        r = [s[0],
             s[1]*np.cos(theta) - s[2]*np.sin(theta),
             s[1]*np.sin(theta) + s[2]*np.cos(theta)]
    elif axis.lower() == 'y':
        r = [s[0]*np.cos(theta) + s[2]*np.sin(theta),
             s[1],
             -s[0]*np.sin(theta) + s[2]*np.cos(theta)]
    elif axis.lower() == 'z':
        r = [s[0] * np.cos(theta) - s[1]*np.sin(theta),
             s[0] * np.sin(theta) + s[1]*np.cos(theta),
             s[2]]
    else:
        print "Error! Invalid axis rotation"
    return r
#rot_vector
