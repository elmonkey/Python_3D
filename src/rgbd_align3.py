#!/usr/bin/python

'''
Created on July 15, 2014
ref: 
    http://euanfreeman.co.uk/tag/openni/
    http://docs.opencv.org/trunk/doc/py_tutorials/py_gui/py_mouse_handling/py_mouse_handling.html

CURRENT FEATURES:
=> Display
1. openni depthmap --> numpy/opencv array for display
2. openni rgb --> numpy/opencv array for display
=> Mouse Clicking Events
3. Prints on the prompt the distance (in milimiters) to the object's pixel
4. Draws circles where the user has clicked
    blue circle: valid point
    red cirlcle: invalid point (no data can be read ofrm tha position/material)
    etc
5. RGB and Depth streams are aligned. 
6. Images are saved w the index of the current frame by pressing 'spacebar'
   Distance map is also saved
    
DONE: 
1. Save images by pressing spacebar -- DONE
2. Overlay and align(tricky) rgb on the pointcloud (depth image) -- DONE
3. Save the pixel's information (col=y,row=x,depth=z, and intensity=l) -- DONE
4. Saves distance map as distance_<frame number>.out
    NOTE: depth_generator.map has 480x640 shape. the distArray is
          created with 640x480 shape to follow the image shape.
ToDo:
1. Compute a median DistanceMap

Status = Working

@author: carlos
'''

from openni import *
import numpy as np
import cv2
import cv
import time
import imtools

# Averaging/Median parameters
n=0
N=5
compute_median = True

# Initialize
context = Context()
context.init()

# Create the depth genrator to access the depth stream
depth_generator = DepthGenerator()
depth_generator.create(context)
depth_generator.set_resolution_preset(RES_VGA)
depth_generator.fps = 30
depth_map = None


# Create the rgb image generator
image_generator = ImageGenerator()
image_generator.create(context)
image_generator.set_resolution_preset(RES_VGA)
image_generator.fps = 30

# Apply a view point transformation to overlay the streams
depth_generator.alternative_view_point_cap.set_view_point(image_generator)


# Circle drawing parameters
row = 480/2 # height
col = 640/2 # width
radius = 2
filled = -1
color  = (0,0,0)
red    = (0,0,255)
blue   = (255,0,0)
green  = (0,255,0)
yellow = (0,255,255)
pink   = (255,0,255)
teal   = (255,255,0)
purple = (102,0,102)
w = 640
h = 480

# Mouse call back variables
xg,yg = 0,0
printing = False

def capture_depth():
    """ Create np.array from Carmine raw depthmap string using 16 or 8 bits
    depth = np.fromstring(depth_generator.get_raw_depth_map_8(), "uint8").reshape(480, 640)
    max = 255 #=(2**8)-1"""
    depth = np.fromstring(depth_generator.get_raw_depth_map(), "uint16").reshape(480, 640)
    max = 8191 # = (2**13)-1
    depth_norm=(depth.astype(float) * 255/max).astype(np.uint8)
    d4d = cv2.cvtColor(depth_norm, cv2.COLOR_GRAY2RGB) # depth4Display
    return depth, d4d
#capture_depth

def mask_depth(d4d,x,y):
    """Returns a grayscale image uint16 masked by the distance-to-sensor."""
    #mask = np.zeros((480,640,3),dtype=np.uint8)
    mask = np.zeros((480,640,3),dtype=np.uint8)
    val = d4d[y,x,0]
    idx = (d4d<=val)
    mask[idx] = d4d[idx]
    # apply hist equalization
    imeq = imtools.eqImg(mask) # equalized image
    return mask,imeq
#mask_depth

def mask_rgbd(d4d,rgb):
    """Overlays images and uses some blur to slightly smooth the mask"""
    mask = d4d.copy()
    #mask = cv2.GaussianBlur(mask, (5,5),0)
    idx =(mask>0)
    mask[idx] = rgb[idx]
    return mask
#mask_rgbd

def capture_rgb():
    '''Get rgb stream from primesense and convert it to an rgb numpy array'''
    rgb = np.fromstring(image_generator.get_raw_image_map_bgr(), 
                        dtype=np.uint8).reshape(480, 640, 3)
    return rgb
# capture_rgb

# creating a callback function
def click_point(event, x,y,flags, param):
    """Mouse call back function. cv2.EVENT_< >
    Continuous promts: < > = MOUSEMOVE
    Selective prompts: < > = LBUTTONDOWN
    """
    global xg,yg, printing
    #if event == cv2.EVENT_MOUSEMOVE:
    if event == cv2.EVENT_LBUTTONDOWN:#cv2.EVENT_MOUSEMOVE
        xg = x
        yg = y
        printing = True
#draw_circle

def harriscorners(im):
    """Compute harris coners and overlay them on the input image. Pixel accuracy
    http://docs.opencv.org/trunk/doc/py_tutorials/py_feature2d/py_features_harris/py_features_harris.html
    """
    rgb = im.copy()
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).astype(np.float32)
    corners = cv2.cornerHarris(gray,2,3,0.04)
    #result is dilated for marking the corners, not important
    corners = cv2.dilate(corners,None)
    # thresholded corners will appear red
    rgb[corners>0.01*corners.max()] = [0,0,255]
    return rgb, corners
# harris corners


def computeMedianArray(array_list):
    """Computes the median array for list of arrays [a, b, c, ...,n]
    d[i,j] = median(a[1,j],b[1,j],c[1,j])
    (list[np.array, np.array, np.array]) -> np.array"""
    for ii in xrange(len(array_list)):
        a = array_list[ii]
        # check number of channels
        if len(a.shape)>2: # 3channel images
            a = np.mean(a,axis=2)
        array_list[ii] = a
    med = np.asarray(array_list)
    med = np.uint8(np.median(med,axis=0))
    return med
#computeMedianArray



##=============================================================================
## MAIN LOOP 
## ----------------------------------------------------------------------------
# start the device carmine
context.start_generating_all()

winName = 'CANVAS'
cv2.namedWindow(winName)
cv2.setMouseCallback(winName,click_point)

pts  = [] # rgb points
dpts = [] # depth points

# Run metrics
globalframe = 0
run_time    = 0
accu_depth  = np.ones((480,640,N),dtype=np.float64)
med_depth   = np.ones((480,640,3),dtype=np.uint8)
depth_list  = [accu_depth]*N
ready = False

cos,ros= depth_generator.map.size
print 'Dimensions of depth map (%d,%d)'%(cos,ros)

## --- MAIN LOOP ---
mask2 = np.zeros((480,640,3), dtype=np.uint8)
imeq  = mask2.copy()
i = 1
done = False
while not done:
    tic = time.time()
    # keyboard commands: terminate || save 
    key =cv2.waitKey(1)
    if key == 27:
        print type(depth), depth.dtype
        done = True
    elif key == ord(' '): #spacebar -> save images
        print 'Saving images and distance map with index %d' %i
        p = '../data/testcaptures/' #datapath
        #p = ''
        cv2.imwrite(p+'rgb_'+str(i)+'.png',rgb)
        cv2.imwrite(p+'depth_'+str(i)+'.png', depth)
        cv2.imwrite(p+'d4d_'+str(i)+'.png',d4d)
        cv2.imwrite(p+'d4deq_'+str(i)+'.png',d4deq)
        cv2.imwrite(p+'masked_'+str(i)+'.png',mask)
        # Extract and save the distances map
        distArray = np.ones((ros,cos), dtype = int)
        for xx in range(0,cos-1):
            for yy in range(0,ros-1):
                # NOTE: depth_generator.map has 480x640 shape. the distArray is
                #       created with 640x480 shape to follow the image shape.
                distArray[yy,xx] = depth_generator.map[xx,yy] #size
        print 'distArray info:', type(distArray), distArray.dtype, distArray.shape
        np.savetxt(p+'distance_'+str(i)+'.out',distArray)

    # Read depth & rgb streams
    depth, d4d = capture_depth()
    d4deq = imtools.eqImg(d4d)
    rgb = capture_rgb()
    mask = mask_rgbd(d4d,rgb)
    
    if compute_median:
        # The accumulator for median depth image
        if n == N:
            n=0
        else:
            depth_list[n]= d4d
            med_d4d = cv2.cvtColor(computeMedianArray(depth_list), cv2.COLOR_GRAY2RGB)
            maskmean = mask_rgbd(med_d4d,rgb)  # mask_depth(d4d,xg,yg)
            n+=1

    # Captions and cross-eye
    d4dtxt = d4d.copy()
    cv2.putText(d4dtxt,"depth",(10,470), cv2.FONT_HERSHEY_PLAIN, 2.0, yellow,
                thickness=2, lineType=cv2.CV_AA)
    cv2.line(d4dtxt,(col,row-5), (col,row+5), yellow, thickness=1)
    cv2.line(d4dtxt,(col-5,row), (col+5,row), yellow, thickness=1)

    rgbtxt = rgb.copy()
    cv2.putText(rgbtxt,"rgb",(10,470), cv2.FONT_HERSHEY_PLAIN, 2.0, blue,
                thickness=2, lineType=cv2.CV_AA)
    cv2.line(rgbtxt,(col,row-5), (col,row+5), yellow, thickness=1)
    cv2.line(rgbtxt,(col-5,row), (col+5,row), yellow, thickness=1)

    masktxt = mask.copy()
    cv2.putText(masktxt,"mask",(10,470), cv2.FONT_HERSHEY_PLAIN, 2.0, green,
                thickness=2, lineType=cv2.CV_AA)
    cv2.line(masktxt,(col,row-5), (col,row+5), yellow, thickness=1)
    cv2.line(masktxt,(col-5,row), (col+5,row), yellow, thickness=1)
    
    maskmeantxt = maskmean.copy()
    cv2.putText(maskmeantxt,"mean("+str(N)+")",(10,470), cv2.FONT_HERSHEY_PLAIN, 2.0, purple,
                thickness=2, lineType=cv2.CV_AA)
    cv2.line(maskmeantxt,(col,row-5), (col,row+5), yellow, thickness=1)
    cv2.line(maskmeantxt,(col-5,row), (col+5,row), yellow, thickness=1)
    
    d4deqtxt = d4deq.copy()
    cv2.putText(d4deqtxt,"eq_depth",(10,470), cv2.FONT_HERSHEY_PLAIN, 2.0, teal,
                thickness=2, lineType=cv2.CV_AA)
    cv2.line(d4deqtxt,(col,row-5), (col,row+5), yellow, thickness=1)
    cv2.line(d4deqtxt,(col-5,row), (col+5,row), yellow, thickness=1)
    
    if printing:
        if xg>=0 and xg <640: # rgb
            color = blue
            ck = 'rgb'
        elif xg >= 640 and xg <1279: # depth
            xg -= 639
            color = yellow
            ck = 'depth'
        elif xg >= 1279 and xg<1919: # masked
            xg -= 1279
            color = green
            ck = 'mask'
        elif xg >= 1920 and xg < 2559: # mean image
            xg -= 1920
            color = purple
            ck = 'slab'
        pts.append([xg,yg]) # append (xg,yg) points
        pixel = depth_generator.map[int(xg),int(yg)]
        intensity = d4d[yg,xg,0]

        if not pixel  == 0:
            print 'Cursor at (%d,%d): %dmm ' %(xg,yg,pixel)
        else: 
##            print 'Clicked on (%d, %d): Unknown!' %(xg,yg)
            color = red
##        mask2,imeq = mask_depth(d4d,xg,yg)
        printing = False
    #if printing

    # draw the points
    cv2.circle(rgbtxt,(xg,yg),radius,color,filled)
    cv2.circle(d4dtxt,(xg,yg),radius,color,filled)
    cv2.circle(masktxt,(xg,yg),radius,color,filled)
    cv2.circle(maskmeantxt,(xg,yg),radius,color,filled)

    #corners,detected = harriscorners(rgb)
    # display image
##    canvas = np.hstack((rgb,d4d,mask,mask2,med_d4d))#,imeq))#, corners, med_depth))
    canvas = np.hstack((rgbtxt,d4dtxt,masktxt, d4deqtxt,maskmeantxt))#,imeq))#, corners, med_depth))

    cv2.imshow(winName, canvas)
    # uptdate the streams
    context.wait_any_update_all()
    #n +=1
    globalframe += 1
    #print "Frame number %d" %globalframe
    toc = time.time()
    run_time += toc-tic
    i+=1

#while

cv2.destroyAllWindows()
# close carmine context and stop device
context.stop_generating_all()
