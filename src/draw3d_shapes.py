#!/usr/bin/python
'''
Created on Sep 18, 2014

Uses mtplaot lib to draw 3 dimensional shapes

refs: 
    http://stackoverflow.com/questions/11140163/python-matplotlib-plotting-a-3d-cube-a-sphere-and-a-vector
    http://mathworld.wolfram.com/RotationMatrix.html


@author: Carlos Torres <carlitos408@gmail.com>
'''

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from itertools import product, combinations
import cv2
import imtools as imt

fig = plt.figure()
ax = fig.gca(projection="3d")
ax.set_aspect("equal")

# draw cube
#def drawCube(r=[-1,1]):
r=[-5,5]
axis  = [1,0,0]
p =0

# large cube: (s)tart  & (e)nd
for s,e in combinations(np.array(list(product(r,r,r))),2):
    ax.scatter(0,0,0, "r") # origin (0,0,0)
    if np.sum(np.abs(s-e)) == r[1]-r[0]:
        coords =  (zip(s,e))
        ax.plot3D(*zip(s,e),color="b")



#smaller cube & rotated
##theta = np.radians(45)
theta = 45
r = [-2,2]
for s,e in combinations(np.array(list(product(r,r,r))),2):
    if np.sum(np.abs(s-e)) == r[1]-r[0]:

        s_Rx = imt.rotate(s,'x',theta)
        e_Rx = imt.rotate(e,'x',theta)

        s_Ry = imt.rotate(s,'y',theta)
        e_Ry = imt.rotate(e,'y',theta)
        
        s_Rz = imt.rotate(s,'z',theta)
        e_Rz = imt.rotate(e,'z',theta)

        ax.plot3D(*zip(s,e),color="g")
        ax.plot3D(*zip(s_Ry, e_Ry),color="b")



ax.set_title("3D Shapes")
ax.set_xlabel("X axis")
#ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("Y axis")
#ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("Z axis")
    
plt.show()
# draw sphere
##u,v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
##x = np.cos(u)*np.sin(v)
##y = np.sin(u)*np.sin(v)
##z = np.cos(v)
##ax.plot_wireframe(x,y,z,color='r')

### draw vector
##from matplotlib.patches import FancyArrowPatch
##from mpl_toolkits.mplot3d import proj3d
##class Arrow3D(FancyArrowPatch):
##    def __init__(self, xs, ys, zs, *args, **kwargs):
##        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
##        self._verts3d = xs, ys, zs
##    
##    def draw(self, renderer):
##        sx3d, ys3d, zs3d = self._verts3d
##        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
##        self.set_positions((xs[0], ys[0]),(xs[1], ys[1]))
##        FancyArrowPatch.draw(self,renderer)
##
##a = Arrow3D([0,1],[0,1],[0,1], mutation_Scale = 20, lw=1, arrowstyle = "-|>", 
##            color= "k")
##
##ax.add_artist(a)
##
###plt.show()
