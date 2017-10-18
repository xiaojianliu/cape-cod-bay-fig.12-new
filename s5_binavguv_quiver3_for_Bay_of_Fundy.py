#http://www.ngdc.noaa.gov/mgg/coast/
# coast line data extractor

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 15:27:09 2012

@author: vsheremet
"""
import numpy as np
#from pydap.client import open_url
import matplotlib.pyplot as plt
#from SeaHorseLib import *
#from datetime import *
#from scipy import interpolate
#import sys
#from SeaHorseTide import *
#import shutil
import matplotlib.mlab as mlab
import matplotlib.cm as cm
def sh_bindata(x, y, z, xbins, ybins):
    """
    Bin irregularly spaced data on a rectangular grid.

    """
    ix=np.digitize(x,xbins)
    iy=np.digitize(y,ybins)
    xb=0.5*(xbins[:-1]+xbins[1:]) # bin x centers
    yb=0.5*(ybins[:-1]+ybins[1:]) # bin y centers
    zb_mean=np.empty((len(xbins)-1,len(ybins)-1),dtype=z.dtype)
    zb_median=np.empty((len(xbins)-1,len(ybins)-1),dtype=z.dtype)
    zb_std=np.empty((len(xbins)-1,len(ybins)-1),dtype=z.dtype)
    zb_num=np.zeros((len(xbins)-1,len(ybins)-1),dtype=int)    
    for iix in range(1,len(xbins)):
        for iiy in range(1,len(ybins)):
#            k=np.where((ix==iix) and (iy==iiy)) # wrong syntax
            k,=np.where((ix==iix) & (iy==iiy))
            zb_mean[iix-1,iiy-1]=np.mean(z[k])
            zb_median[iix-1,iiy-1]=np.median(z[k])
            zb_std[iix-1,iiy-1]=np.std(z[k])
            zb_num[iix-1,iiy-1]=len(z[k])
            
    return xb,yb,zb_mean,zb_median,zb_std,zb_num
"""
from netCDF4 import Dataset

# read in etopo5 topography/bathymetry.
url = 'http://ferret.pmel.noaa.gov/thredds/dodsC/data/PMEL/etopo5.nc'
etopodata = Dataset(url)

topoin = etopodata.variables['ROSE'][:]
lons = etopodata.variables['ETOPO05_X'][:]
lats = etopodata.variables['ETOPO05_Y'][:]
# shift data so lons go from -180 to 180 instead of 20 to 380.
topoin,lons = shiftgrid(180.,topoin,lons,start=False)
"""



"""
BATHY=np.genfromtxt('necscoast_noaa.dat',dtype=None,names=['coast_lon', 'coast_lat'])
coast_lon=BATHY['coast_lon']
coast_lat=BATHY['coast_lat']
"""

#BATHY=np.genfromtxt('coastlineNE.dat',names=['coast_lon', 'coast_lat'],dtype=None,comments='>')
#coast_lon=BATHY['coast_lon']
#coast_lat=BATHY['coast_lat']


# www.ngdc.noaa.gov
# world vector shoreline ascii
FNCL='necscoast_worldvec.dat'
# lon lat pairs
# segments separated by nans
"""
nan nan
-77.953942	34.000067
-77.953949	34.000000
nan nan
-77.941035	34.000067
-77.939568	34.001241
-77.939275	34.002121
-77.938688	34.003001
-77.938688	34.003881
"""
CL=np.genfromtxt(FNCL,names=['lon','lat'])


FN='binned_drifter120460.npz'
#FN='binned_model.npz'
Z=np.load(FN) 
xb=Z['xb']
yb=Z['yb']
ub_mean=Z['ub_mean']
ub_median=Z['ub_median']
ub_std=Z['ub_std']
ub_num=Z['ub_num']
vb_mean=Z['vb_mean']
vb_median=Z['vb_median']
vb_std=Z['vb_std']
vb_num=Z['vb_num']
Z.close()

#cmap = matplotlib.cm.jet
#cmap.set_bad('w',1.)
xxb,yyb = np.meshgrid(xb, yb)
cc=np.arange(-1.5,1.500001,0.03)
#cc=np.array([-1., -.75, -.5, -.25, -0.2, -.15, -.1, -0.05, 0., 0.05, .1, .15, .2, .25, .5, .75, 1.])
fig,axes=plt.subplots(2,2,figsize=(15,10))
#plt.figure()
ub = np.ma.array(ub_mean, mask=np.isnan(ub_mean))
vb = np.ma.array(vb_mean, mask=np.isnan(vb_mean))
Q=axes[0,0].quiver(xxb,yyb,ub.T,vb.T,scale=5.)
qk=axes[0,0].quiverkey(Q,0.9,0.6,0.5, r'$0.1m/s$', fontproperties={'weight': 'bold'})

#plt.xlabel('''Mean current derived from historical drifter data (1-20m)''')

#plt.plot(coast_lon,coast_lat,'b.')
axes[0,0].plot(CL['lon'],CL['lat'])
axes[0,0].set_xlabel('a')
axes[0,0].text(-65.8,44.5,'Bay of Fundy',fontsize=12)
#axes[1].text(-67.5,41.5,'Georges Bank',fontsize=7)
axes[0,0].plot([-68.8,-68.5],[44.4,44.7],'y-')
axes[0,0].plot([-65.8,-66.3],[44.5,44.7],'y-')
axes[0,0].plot([-67,-68],[43.4,43.5],'y-')
axes[0,0].plot([-70.8,-70],[44,42.8],'y-')
axes[0,0].plot([-70.2,-70],[43.85,43.75],'y-')
axes[0,0].plot([-69.8,-69.8],[44.25,44.1],'y-')
axes[0,0].plot([-67.5,-67.6],[45.25,45.15],'y-')
axes[0,0].plot([-66.1,-66.2],[45.25,45.25],'y-')
axes[0,0].plot([-68.85,-70],[44.6,44.7],'y-')
#axes[1].text(-70.5,44.7,'Penobscot River',fontsize=7)
axes[0,0].text(-66.9,45.24,'St. John River',fontsize=12)
axes[0,0].text(-67.85,45.1,'St. Croix River',fontsize=12)
axes[0,0].text(-67.7,44.8,'Eastern',fontsize=12)
axes[0,0].text(-67.7,44.72,'Maine',fontsize=12)
#axes[1].text(-69.4,44.2,'Western',fontsize=7)
#axes[1].text(-69.4,44.1,'Maine',fontsize=7)
#axes[1].text(-70.4,44.3,'Kennebec River',fontsize=7)
#axes[1].text(-70.7,45,'Maine',fontsize=7)
#axes[0].text(-70.6,43.9,'Casco Bay',fontsize=7)
#axes[1].text(-67,43.4,'Jordan Basin',fontsize=7)
axes[0,0].text(-67.8,45,'Grand Manan',fontsize=12)
axes[0,0].text(-67.65,44.92,'Island',fontsize=12)
axes[0,0].plot([-67.5,-66.8],[44.99,44.7],'y-')
#axes[1].text(-69,44.7,'Penobscot Bay',fontsize=7)
axes[0,0].text(-66,44.0,'Nova Scotia',fontsize=12)
#axes[1].text(-70.9,44,'Wikkson Basin',fontsize=7)
axes[0,0].axis([-67.875,-64.75,43.915,45.33])#axes[0].axis([-71,-64.75,42.5,45.33])-67.875,-64.75,43.915,45.33
axes[0,0].xaxis.tick_top() 

#plt.show()
###################################################################################33
FN='binned_drifter120780.npz'
#FN='binned_model.npz'
Z1=np.load(FN) 
xb1=Z1['xb']
yb1=Z1['yb']
ub_mean1=Z1['ub_mean']
ub_median1=Z1['ub_median']
ub_std1=Z1['ub_std']
ub_num1=Z1['ub_num']
vb_mean1=Z1['vb_mean']
vb_median1=Z1['vb_median']
vb_std1=Z1['vb_std']
vb_num1=Z1['vb_num']
Z1.close()

#cmap = matplotlib.cm.jet
#cmap.set_bad('w',1.)
xxb,yyb = np.meshgrid(xb, yb)
cc=np.arange(-1.5,1.500001,0.03)
#cc=np.array([-1., -.75, -.5, -.25, -0.2, -.15, -.1, -0.05, 0., 0.05, .1, .15, .2, .25, .5, .75, 1.])
#fig,axes=plt.subplots(3,2,figsize=(7,5))
#plt.figure()
ub1 = np.ma.array(ub_mean1, mask=np.isnan(ub_mean1))
vb1 = np.ma.array(vb_mean1, mask=np.isnan(vb_mean1))
Q=axes[1,0].quiver(xxb,yyb,ub1.T,vb1.T,scale=5.)
qk=axes[1,0].quiverkey(Q,0.9,0.6,0.5, r'$0.1m/s$', fontproperties={'weight': 'bold'})

#plt.xlabel('''Mean current derived from historical drifter data (1-20m)''')
axes[1,0].set_xticklabels([])
axes[1,0].set_xlabel('c')
axes[0,1].set_xlabel('b')
#plt.plot(coast_lon,coast_lat,'b.')
axes[1,0].plot(CL['lon'],CL['lat'])
axes[1,0].axis([-67.875,-64.75,43.915,45.33])#axes[0].axis([-71,-64.75,42.5,45.33])-67.875,-64.75,43.915,45.33
#axes[1,0].xaxis.tick_top() 

for a in np.arange(len(xxb[0])):
    for b in np.arange(len(yyb)):
        if -67.5<xxb[0][a]<-66.38 and 44.4<yyb[b][0]<44.9 and ub_num[a][b]!=0:
            #plt.text(xxb[0][a],yyb[b][0],ubn[a][b],fontsize='smaller')
            axes[0,1].text(xxb[0][a],yyb[b][0],ub_num[a][b],fontsize=12)
            #axes[1,1].scatter(xxb[0][a],yyb[b][0],s=ubn[a][b]/float(100),color='red',marker='o')
axes[0,1].plot(CL['lon'],CL['lat'])
axes[0,1].axis([-67.5,-66.37,44.4,44.9])
axes[0,1].xaxis.tick_top() 

for a in np.arange(len(xxb[0])):
    for b in np.arange(len(yyb)):
        if -67.5<xxb[0][a]<-66.38 and 44.4<yyb[b][0]<44.9 and ub_num[a][b]!=0:
            #plt.text(xxb[0][a],yyb[b][0],ubn[a][b],fontsize='smaller')
            axes[1,1].text(xxb[0][a],yyb[b][0],ub_num1[a][b],fontsize=12)
            #axes[1,1].scatter(xxb[0][a],yyb[b][0],s=ubn[a][b]/float(100),color='red',marker='o')
axes[1,1].plot(CL['lon'],CL['lat'])
axes[1,1].axis([-67.5,-66.37,44.4,44.9])
#axes.xaxis.tick_top() 
axes[1,1].set_xticklabels([])
axes[1,1].set_xlabel('d')
#plt.title('binned_drifter_num')
plt.savefig('drifter4_6to7_8xinhaha',dpi=400)
