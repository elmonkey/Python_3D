#!/usr/bin/python

'''
Created on 8 September 2014

@author: Carlos Torres <carlitos408@gmail.com>
'''

import cv2
import numpy as np
import imtools

# Mouse call back variables
xg,yg = 0,0
printing = False
ix,iy = 0,0
ox,oy = 0,0
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

def load_imgs(n=54):
    p = '../data/screenshots/' #datapath
    d4d   = cv2.imread(p+'d4d_'+str(n)+'.png')
    d4deq = cv2.imread(p+'d4deq_'+str(n)+'.png')
    depth = cv2.imread(p+'depth_'+str(n)+'.png')
    rgb   = cv2.imread(p+'rgb_'+str(n)+'.png')
    mask  = cv2.imread(p+'masked_'+str(n)+'.png')
    distance = np.loadtxt(p+'distance_'+str(n)+'.out')
    return rgb, d4d, depth, mask, d4deq, distance
#load_imgs()


# creating a callback function
def click_point(event, x,y,flags, param):
    """Mouse call back function. cv2.EVENT_< >
    Continuous promts: < > = MOUSEMOVE
    Selective prompts: < > = LBUTTONDOWN
    """
    global xg,yg, printing
##    if event == cv2.EVENT_MOUSEMOVE:
    if event == cv2.EVENT_LBUTTONDOWN:
        xg = x
        yg = y
        printing = True
#draw_circle

def getBoxCoords(pts, distMap):
    """Given the opposite corners of a rectangle, returns (x,y) coordinates
    for the elements contained in the rectangle.
                      a(x1,y1) ------ b(x2,y1)
                         |              |
      [pt1,pt2] ->       |              |
                         |              |
                      c(x1,y2) ------ d(x2,y2)
    """
    # test:
    #pts = ([2,1],[6,4])
    #pts=([319,305],[351,331])
    coords = []
    p1,p2 = pts[-2:]
    if p1[0] == p2[0] or p1[1]==p2[1]:
        print 'Warning: Points are on the same line'
    # extract point coordinates
    x1, x2 = min(pts[-1][0], pts[-2][0]), max(pts[-1][0], pts[-2][0])
    y1, y2 = min(pts[-1][1], pts[-2][1]), max(pts[-1][1], pts[-2][1])
    xs = range(x1,x2)
    ys = range(y1,y2)
    for xx in xs:
        for yy in ys:
            #print xx,yy
            if (distMap[xx,yy] != 0) and (distMap[xx,yy]<750):
                coords.append([xx,yy,distMap[xx,yy]])
##            coords.append([xx,yy,distMap[xx,yy]])
    coords = np.asarray(coords)
    #square = [[x1,y1], [x2,y1], [x1,y2], [x2,y2]]
    square = [x1,x2,y1,y2]
    return coords, square
#getBoxCoords()


def genCube(a,b,c,d,f,g,h, ax):
    """Given coordinates of a square's corners and the depth map, it estimates 
    the coordinates of the faces of a cube each coord is given as x,y,z triplet
    a: (x,y,z)
    Frontal face:       Rear face:
          e --- f             e === f
         /     /|            /||   ||
        a === b |           a ||   ||
        ||   || h           | g === h
        ||   ||/            |/     /
        c === d             c --- d """
    ax.plot3D(*zip(a, b),color="b")
    ax.plot3D(*zip(a, e),color="b")
    ax.plot3D(*zip(a, c),color="b")
    
    ax.plot3D(*zip(d, b),color="b")
    ax.plot3D(*zip(d, c),color="b")
    ax.plot3D(*zip(d, h),color="b")
    
    ax.plot3D(*zip(f, b),color="b")
    ax.plot3D(*zip(f, e),color="b")
    ax.plot3D(*zip(f, h),color="b")
    
    ax.plot3D(*zip(g, c),color="b")
    ax.plot3D(*zip(g, e),color="b")
    ax.plot3D(*zip(g, h),color="b")
    return ax
#genCube



def plotDepth(coords,square):
    """Plots the depth Map on a 3D form
    a: (x,y,z)
    Frontal face:       Rear face:
          e --- f             e === f
         /     /|            /||   ||
        a === b |           a ||   ||
        ||   || h           | g === h
        ||   ||/            |/     /
        c === d             c --- d 
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    X = coords[:,0]
    Y = coords[:,1]
    Z = coords[:,2]
##    print type(Y), Y.shape
    Y = np.abs(Y - Y.max())

    # ranges of the data
    x_min, x_max = X.min(), X.max()
    y_min, y_max = Y.min(), Y.max()
    z_min, z_max = Z.min(), Z.max()
    
    # CUBE 
    #corners front face
    x1,x2,y1,y2 = square
    a =np.array([x1,y1,z_min])
    b =np.array([x2,y1,z_min])
    d =np.array([x2,y2,z_min])
    c =np.array([x1,y2,z_min])
    # corners rear face
    e = a.copy(); e[2] = z_max
    f = b.copy(); f[2] = z_max
    g = c.copy(); g[2] = z_max
    h = d.copy(); h[2] = z_max
    
    xs = np.array([x1,x2,x2,x1,x1]*2+[x2,x2,x2,x2, x1, x1])#,x2,x2,x2])
    ys = np.array([y1,y1,y2,y2,y1]*2+[y1,y1,y2,y2, y2, y2])#,y2,y1,y1])
    ys = np.abs(ys - ys.max())
    zs = np.array([z_min,]*5 + [z_max]*5+[z_min,z_max,z_min,z_max,z_max, z_min])#,z_max,z_max, z_max])
    
    # plot the points
    fig = plt.figure(1, figsize=(8,6))
    ax = Axes3D(fig)#,elev=30, azim=-30)#, elev=-150, azim=110)
    ax.scatter(X,Z,Y, cmap=plt.cm.Paired)

##    # plot the cube
##    ax.plot3D(*zip(a, b),color="b")
##    ax.plot3D(*zip(a, e),color="b")
##    ax.plot3D(*zip(a, c),color="b")
##
##    ax.plot3D(*zip(d, b),color="b")
##    ax.plot3D(*zip(d, c),color="b")
##    ax.plot3D(*zip(d, h),color="b")
##
##    ax.plot3D(*zip(f, b),color="b")
##    ax.plot3D(*zip(f, e),color="b")
##    ax.plot3D(*zip(f, h),color="b")
##
##    ax.plot3D(*zip(g, c),color="b")
##    ax.plot3D(*zip(g, e),color="b")
##    ax.plot3D(*zip(g, h),color="b")

    ax.plot(xs,zs,ys,color='r')
    
    # close the corners (3 more lines)
    #ax.plot([x2,x2],[z_min,z_max], [y1,y1], 'g')
    
    ax.set_title("3D Wound: Volumetric View")
    ax.set_xlabel("X axis[pixel]")
    #ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("Depth[mm]")
    #ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("Y axis[pixel]")
    #ax.w_zaxis.set_ticklabels([])
    
    # draw cube lines
    
    plt.show()
#plotDepth()



if __name__ == '__main__':
    winName = 'CANVAS'
    cv2.namedWindow(winName)
    cv2.setMouseCallback(winName,click_point)
    ### display
    rgb, d4d, depth, rgbd, d4deq, dist54 = load_imgs(54)
    ##c1 = np.hstack((rgb,d4d,rgbd,d4deq))#,imeq))#, corners, med_depth))
    ##
    rgb, d4d, depth, rgbd2, d4deq, dist2 = load_imgs(81)
    ##c2 = np.hstack((rgb,d4d,rgbd2,d4deq))#,imeq))#, corners, med_depth))
    ##
    ##cv2.imshow( winName, np.vstack((c1,c2)) )
    ##cv2.waitKey(0)
    ##print type(dist54), dist54.dtype, dist54.shape
    pts = [] # list to store the clicked-points
##    cv2.imshow( 'depth', np.hstack((d4d,d4deq)) )
    done = False
    while not done:
        rgbdtxt = rgbd2.copy()
        key = cv2.waitKey(5)
        if key == 27:
            printing = False
            done = True
        if printing == True:
            print 'Distance to (%.0d, %.0d) = %0.1f mm' %(xg,yg,dist2[xg,yg])
            pts.append([xg,yg])
            printing = False
        if len(pts)>=2:
            ix,iy = pts[-2]
            ox,oy = pts[-1]
            coords,square = getBoxCoords(([ix,iy],[ox,oy]), dist2)
            print "square:", square
            pts = [] # restart
##            for xx,yy,zz in coords:
##                cv2.circle(rgbdtxt,(int(xx),int(yy)),radius,red,filled)
            cv2.rectangle(rgbdtxt,(ix,iy),(ox,oy),(0,255,0),1)
            im_crop = rgbd[square[2]:square[3], square[0]:square[1], :]
            cv2.imshow( winName, rgbdtxt)
            #cv2.imshow('cropped section', im_crop)
            #cv2.waitKey(0)
            plotDepth(coords,square)

        cv2.circle(rgbdtxt,(xg,yg),radius,teal,filled)
        cv2.rectangle(rgbdtxt,(ix,iy),(ox,oy),(0,255,0),1)
        cv2.imshow( winName, rgbdtxt )
    #while
    cv2.destroyAllWindows()
    
##    print 'Distance to center 1 = %0.1f mm' %dist54[320,240]
##    print 'Distance to center 1 = %0.1f mm' %dist519[320,240]
#__main__
