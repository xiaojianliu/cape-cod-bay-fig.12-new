# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 09:13:45 2017

@author: bling
"""
from mpl_toolkits.basemap import Basemap  
import sys
import datetime as dt
from matplotlib.path import Path
import netCDF4
from dateutil.parser import parse
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from datetime import datetime, timedelta
from math import radians, cos, sin, atan, sqrt  
import numpy as np
import sys
import datetime as dt
from matplotlib.path import Path
import netCDF4
from dateutil.parser import parse
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from datetime import datetime, timedelta
from math import radians, cos, sin, atan, sqrt  
from matplotlib.dates import date2num,num2date
def haversine(lon1, lat1, lon2, lat2): 
    """ 
    Calculate the great circle distance between two points  
    on the earth (specified in decimal degrees) 
    """   
    #print 'lon1, lat1, lon2, lat21',lon1, lat1, lon2, lat2
    #print 'lon1, lat1, lon2, lat22',lon1, lat1, lon2, lat2
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])  
    #print 34
    dlon = lon2 - lon1   
    dlat = lat2 - lat1   
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2  
    c = 2 * atan(sqrt(a)/sqrt(1-a))   
    r = 6371 
    d=c * r
    #print 'd',d
    return d
def calculate_SD(dmlon,dmlat):
    '''compare the model_points and drifter point(time same as model point)
    (only can pompare one day)!!!'''
    #print modelpoints,dmlon,dmlat,drtime
    #print len(dmlon)
    
    dd=0
    
    for a in range(len(dmlon)-1):
        #print 12
        #dla=(dmlat[a+1]-dmlat[a])*111
        #dlo=(dmlon[a+1]-dmlon[a])*(111*np.cos(dmlat[a]*np.pi/180))
        #d=sqrt(dla**2+dlo**2)#Calculate the distance between two points 
        #print model_points['lon'][a][j],model_points['lat'][a][j],dmlon[a][j],dmlat[a][j],d           
        #print 'd',d
        d=haversine(dmlon[a+1],dmlat[a+1],dmlon[a],dmlat[a])
        dd=dd+d
    #print 'dd',dd
    return dd

FN='necscoast_worldvec.dat'
CL=np.genfromtxt(FN,names=['lon','lat'])

length=60*0.009009009
latc=np.linspace(44.65,45.02,10)
lonc=np.linspace(-66.6,-65.93,10)
    
p1 = Path.circle(((lonc[5]+lonc[4])/2,(latc[5]+latc[4])/2),radius=length)
    
fig,axes=plt.subplots(1,1,figsize=(10,10))#figure()
cl=plt.Circle(((lonc[5]+lonc[4])/2,(latc[5]+latc[4])/2),length,alpha=0.6,color='yellow')
axes.add_patch(cl)
latc=np.linspace(44.65,45.02,10)
lonc=np.linspace(-66.6,-65.93,10)
lon1=[]
latc1=np.linspace(44.65,45.02,6)

#lat1=[]
for a in np.arange(len(lonc)-2):
    lon1.append((lonc[a]+lonc[a+1])/2)
lat1=latc[0]-(latc1[1]-latc1[0])
########################################
lon2=[]
#lat1=[]
#lon2.append(lon1[0]-(lon1[1]-lon1[0]))
for a in np.arange(len(lon1)-2):
    lon2.append((lon1[a]+lon1[a+1])/2)
lat2=latc[0]-(latc1[1]-latc1[0])*2
######################################333
lon3=[]
#lat1=[]
for a in np.arange(len(lon2)-2):
    lon3.append((lon2[a]+lon2[a+1])/2)
lat3=latc[0]-(latc1[1]-latc1[0])*3
#######################################3333
lon4=[]
#lat1=[]
for a in np.arange(1,len(lon3)-1,1):
    lon4.append((lon3[a]+lon3[a+1])/2)
lat4=latc[0]-(latc1[1]-latc1[0])*4
#####################################333
###################################333333
lon5=[]
#lat1=[]
for a in np.arange(len(lonc)):
    lon5.append(lonc[a]-(lonc[1]-lonc[0])/2)
lon5.append(lon5[-1]+(lonc[-1]-lonc[-2]))
lat5=latc[0]+(latc1[1]-latc1[0])*1
########################################3
lon6=[]
#lat1=[]
for a in np.arange(len(lon5)):
    lon6.append(lon5[a]-(lon5[1]-lon5[0])/2)
lon6.append(lon6[-1]+(lon5[-1]-lon5[-2]))
lat6=latc[0]+(latc1[1]-latc1[0])*2
##############################################

lon7=[]
for a in np.arange(len(lon6)):
    lon7.append(lon6[a]-(lon6[1]-lon6[0])/2)
lon7.append(lon7[-1]+(lon6[1]-lon6[0]))
lat7=latc[0]+(latc1[1]-latc1[0])*3
##############################################
lon8=lon6
lat8=latc[0]+(latc1[1]-latc1[0])*4
######################################
lon9=[]
for a in np.arange(len(lon8)-1):
    lon9.append((lon8[a]+lon8[a+1])/2)
lat9=latc[0]+(latc1[1]-latc1[0])*5
###################################333333
lon10=[]
for a in np.arange(3,len(lon9)-1,1):
    lon10.append((lon9[a]+lon9[a+1])/2)
lat10=latc[0]+(latc1[1]-latc1[0])*6
##########################################3333
lon11=[]
for a in np.arange(3,len(lon10)-1,1):
    lon11.append((lon10[a]+lon10[a+1])/2)
lat11=latc[0]+(latc1[1]-latc1[0])*7
##########################################3333

##########################################3333
st_lat=[]
st_lon=[]

for aa in np.arange(len(lonc)):
    st_lat.append(latc[0])
    st_lon.append(lonc[aa])
##########################################
for a in np.arange(len(lon1)):
    st_lon.append(lon1[a])
    st_lat.append(lat1)
#######################################3
for a in np.arange(len(lon2)):
    st_lon.append(lon2[a])
    st_lat.append(lat2)
#################################33
for a in np.arange(len(lon3)):
    st_lon.append(lon3[a])
    st_lat.append(lat3)
####################################
for a in np.arange(len(lon4)):
    st_lon.append(lon4[a])
    st_lat.append(lat4)
###################################333
for a in np.arange(len(lon5)):
    st_lon.append(lon5[a])
    st_lat.append(lat5)
###################################333
for a in np.arange(len(lon6)):
    st_lon.append(lon6[a])
    st_lat.append(lat6)
###################################333
for a in np.arange(len(lon7)):
    st_lon.append(lon7[a])
    st_lat.append(lat7)
###############################3
for a in np.arange(len(lon8)):
    st_lon.append(lon8[a])
    st_lat.append(lat8)
###############################3
for a in np.arange(len(lon9)):
    st_lon.append(lon9[a])
    st_lat.append(lat9)
###############################3
for a in np.arange(len(lon10)):
    st_lon.append(lon10[a])
    st_lat.append(lat10)
#####################################
for a in np.arange(len(lon11)):
    st_lon.append(lon11[a])
    st_lat.append(lat11)
#####################################
st_lat.append(lat2)
st_lon.append(lon2[0]-(lon2[1]-lon2[0]))
axes.scatter(st_lon,st_lat,s=5,color='red')
m = Basemap(projection='cyl',llcrnrlat=43,urcrnrlat=46,\
                llcrnrlon=-69,urcrnrlon=-64,resolution='h')#,fix_aspect=False)
        #  draw coastlines
m.drawcoastlines()
m.ax=axes
m.fillcontinents(color='white',alpha=1,zorder=2)
    
    #draw major rivers
m.drawmapboundary()
    #draw major rivers
m.drawrivers()
parallels = np.arange(43,46,1.)
m.drawparallels(parallels,labels=[1,0,0,0],dashes=[1,1000],fontsize=10,zorder=0)
meridians = np.arange(-70.,-64.,1.)
m.drawmeridians(meridians,labels=[0,0,1,0],dashes=[1,1000],fontsize=10,zorder=0)
    
plt.plot([-66.3,-65.9],[44.9,44.45],zorder=2)
plt.text(-65.8,44.4,'Bay of Fundy',fontsize=12)
plt.plot([-66,-65.5],[45.5,45],'b-',zorder=2)
plt.text(-67,45.55,'North of Bay of Fundy',fontsize=12)
plt.plot([-67,-67.5],[44.3,45],'b-',zorder=2)
plt.text(-68.5,45.2,'South of Bay of Fundy',fontsize=12)
plt.plot(CL['lon'],CL['lat'],'b-')
#plt.axis([-67.875,-64.75,43.915,45.33])
plt.savefig('xxxin',dpi=300)