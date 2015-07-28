#!/usr/bin/python

'''
Created on 23 September 2014

@author: Carlos Torres <carlitos408@gmail.com>
'''

import cv2
import numpy as np
import imtools as imt
import pylab as pl

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

def load_imgs(n=19):
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
            if (distMap[xx,yy] != 0) and (distMap[xx,yy]<1000):
                coords.append([xx,yy,distMap[xx,yy]])
    coords = np.asarray(coords)
    square = [x1,x2,y1,y2]
    return coords, square
#getBoxCoords()


def getEdges(image):
    """Computes sobel and robers edges on uint8 single channel grayscale image
    input: image, single channel numpy array, uint8, and 0-255 range
    outputs: 
        edge_roberts, single channel numpy array, uint64, and 0-1 range
        edge_sobel,   single channel numpy array, uint64, and 0-1 range"""
    from skimage.filter import roberts, sobel
    edge_roberts = roberts(image)
    edge_sobel = sobel(image)

    #print 'Image info', type(image),           image.shape,      image.dtype,      image.min(),      image.max()
    #print 'Sobel info', type(edge_sobel), edge_sobel.shape, edge_sobel.dtype, edge_sobel.min(), edge_sobel.max()

    # Change the range to 0-255 and the type to uint8 
    edge_sobel = np.uint8(edge_sobel * 255)
    edge_roberts = np.uint8(edge_roberts * 255)
    cv2.imshow('input || sobel || roberts', np.hstack((image, edge_sobel, edge_roberts)))
    # cv2.waitKey(0)
    return edge_roberts, edge_sobel


def getCubeWireframe(coods, square):
    """
    a: (x,y,z)
    Frontal face:       Rear face:
        y         z         y         z
        ^        /          ^        /
        | e --- f           | e === f
        |/     /|           |/||   ||
        a === b |           a ||   ||
        ||   || h           | g === h
        ||   ||/            |/     /
        c === d--> x        c --- d --> x
    
    """
    X = coords[:,0]
    Y = coords[:,1]
    Z = coords[:,2]

    # ranges of the data
    x_min, x_max = X.min(), X.max()
    y_min, y_max = Y.min(), Y.max()
    z_min, z_max = Z.min(), Z.max()
    print 'Z:', z_min, z_max
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
    # lines that connect the cube
##    cube = np.vstack((a,b,
##                      a,c,
##                      #a,e,
##                      d,b,
##                      d,c,
##                      #d,h,
##                      #f,b,
##                      #f,e,
##                      #f,h,
##                      #g,c,
##                      #g,e,
##                      #g,h
##                    ))
    cube = np.vstack((c,a,
                      c,g,
                      a,e,
                      g,e,
##                        d,f,
##                        d,h,
                        #connections
##                        a-c,b-d,
##                        c-c,d-d,
##                        e-d,f-d,
##                        g-c,h-d,
                        ))

    #rotate cube
    theta = 60
    rot_cube = cube.copy()
    for i in cube.shape[0]-1:
        rot_cube[i] = imt.rotate(cube[i],theta=theta,axis='x')

    # plotting the cube
    ax.plot3D(cube[:,0],cube[:,1],cube[:,2],color="k")

#getCubeWireframe


def plotDepth(coords,square):
    """Plots the depth Map on a 3D form 
    a: (x,y,z)
    Frontal face:       Rear face:
        y         z         y         z
        ^        /          ^        /
        | e --- f           | e === f
        |/     /|           |/||   ||
        a === b |           a ||   ||
        ||   || h           | g === h
        ||   ||/            |/     /
        c === d--> x        c --- d --> x

    """

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    X = coords[:,0]
    Y = coords[:,1]
    Z = coords[:,2]
    Y_max = Y.max()
    Y = np.abs(Y - Y.max())
    Z = np.abs(Z - Z.max())
    # ranges of the data
    x_min, x_max = X.min(), X.max()
    y_min, y_max = Y.min(), Y.max()
    z_min, z_max = Z.min(), Z.max()
    print 'Z:', z_min, z_max
    
    # CUBE 
    #corners front face
    x1,x2,y1,y2 = square
    y1 = np.abs(y1-Y_max)
    y2 = np.abs(y2-Y_max)
    
    a =np.array([x1,y1,z_min])
    b =np.array([x2,y1,z_min])
    d =np.array([x2,y2,z_min])
    c =np.array([x1,y2,z_min])
    # corners rear face
    e = a.copy(); e[2] = z_max
    f = b.copy(); f[2] = z_max
    g = c.copy(); g[2] = z_max
    h = d.copy(); h[2] = z_max

##    cube={'a':a, 'b':b, 'c':c, 'd':d, 'e':e, 'f':f, 'g':g, 'h':h, }
##    corners = ['a', 'b',
##               'a', 'c',
##               'd', 'b',
##               'd', 'c']

##    cube = np.vstack((c,a,
##                      c,g,
##                      a,e,
##                      g,e,
##                        ))

    cube = np.vstack((a,b,
                      a,c,
                      a,e,
                      d,b,
                      d,c,
                      d,h,
                      f,b,
                      f,e,
                      f,h,
                      g,c,
                      g,e,
                      g,h
                    ))
    # one face

    print 'cube volumen',
    # CubeRotation
    theta = -.5
    rot_cube = cube.copy()
    for i in range(cube.shape[0]):
        rot_cube[i]=imt.rotate(rot_cube[i],theta=theta,axis='x')

    # Correcting cube displacement
    delta_x,delta_y, delta_z = rot_cube[3] - c
    print 'prior to displacement: ', rot_cube[3]
    #rot_cube[3]-=c
    print 'displacements: ', rot_cube[3]

    # plot the depth points: 3D
    fig = plt.figure(1, figsize=(10,8))
    ax = Axes3D(fig,elev=80, azim=-90)#, elev=-150, azim=110)
    ax.scatter(X,Y,Z, color='g')# cmap=plt.cm.Paired)
    ax.plot3D(cube[:,0],cube[:,1],cube[:,2],color="k")
##    ax.plot3D(rot_cube[:,0],rot_cube[:,1],rot_cube[:,2],color="r")
    ax.set_title("3D Wound: Volumetric View")
    ax.set_xlabel("X axis[pixel]")
    #ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("Y axis[pixel]")
    #ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("Depth[mm]")
    #ax.w_zaxis.set_ticklabels([])
    
    
    # plot the box front face: 2D
    pl.figure(2,figsize=(10,8))
    pl.scatter(Z,Y, color='g')# cmap=plt.cm.Paired)
    pl.axis('equal')
    pl.plot(cube[:,2],cube[:,1],color="k")    
    pl.plot(rot_cube[:,2],rot_cube[:,1],color="r")
    pl.xlabel("Y-axis [pixel]")
    pl.ylabel("Depth [mm]")
    pl.grid('on')
    pl.show()
    
    # draw cube lines
    
    plt.show()
#plotDepth()



if __name__ == '__main__':
    winName = 'CANVAS'
    cv2.namedWindow(winName)
    cv2.setMouseCallback(winName,click_point)
    ### display
    rgb, d4d, depth, rgbd, d4deq, dist54 = load_imgs(20)
    ##c1 = np.hstack((rgb,d4d,rgbd,d4deq))#,imeq))#, corners, med_depth))
    #robert_edges, sobel_edges = getEdges(d4d[:,:,0])
    rgb, d4d, depth, rgbd2, d4deq, dist2 = load_imgs(19)
    #robert_edges, sobel_edges = getEdges(d4d[:,:,0])

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
            print 'Box (%d tall x %d wide)' %(np.abs(iy-oy),np.abs(ix-ox))
            rgbd_crop = rgbd[square[2]:square[3], square[0]:square[1], :]
            d4d_crop = d4d[square[2]:square[3], square[0]:square[1], :]
            eq_d4d_crop = imt.eqImg(d4d_crop)
            robert_edges, sobel_edges = getEdges(eq_d4d_crop[:,:,0])

            cv2.imshow( winName, rgbdtxt)
            #cv2.imshow('cropped section', rgbd_crop)
            #cv2.imshow('equalized cropped section', eq_d4d_crop)
            #cv2.imshow('Sobel edges of the qualized cropped section',sobel_edges)

            cv2.waitKey(0)
            plotDepth(coords,square)

        cv2.circle(rgbdtxt,(xg,yg),radius,teal,filled)
        cv2.rectangle(rgbdtxt,(ix,iy),(ox,oy),(0,255,0),1)
        cv2.imshow( winName, rgbdtxt )
    #while
    cv2.destroyAllWindows()
    
##    print 'Distance to center 1 = %0.1f mm' %dist54[320,240]
##    print 'Distance to center 1 = %0.1f mm' %dist519[320,240]
#__main__
