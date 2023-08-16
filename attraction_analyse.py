import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import geopy.distance
from exif import Image
from geopy.geocoders import Nominatim
from tkinter.filedialog import askopenfilenames, Tk
from sklearn.metrics.pairwise import haversine_distances
from math import radians
import seaborn

seaborn.set()

def day_trip(matched,allVA):
    matchedCoord = []
    matchedCoord.append(matched.iloc[0]['lat'])
    matchedCoord.append(matched.iloc[0]['lon'])
    cate = matched.iloc[0]['amenity']
    

    nextA = allVA.drop(allVA[allVA.amenity.values == cate].index)

    dis = []
    matchedCoord_radians = rad(matchedCoord)
    dl = len(nextA['name'])
    for i in range(dl):
        x = []
        x.append(nextA.iloc[i]['lat'])
        x.append(nextA.iloc[i]['lon'])
        x_radians = rad(x)
        y = haversine_distances([matchedCoord_radians, x_radians]) * (6371000/1000)
        dis.append(y[0][1])
    nextA['distance'] = dis
    nextA = nextA.sort_values(by=['distance'])
    return nextA

def rad(x):
    rad = [radians(_) for _ in x]
    return rad

def decimal_coords(coords, ref):    #code extracted from https://medium.com/spatial-data-science/how-to-extract-gps-coordinates-from-images-in-python-e66e542af354
    decimal_degrees = coords[0] + coords[1] / 60 + coords[2] / 3600
    if ref == 'S' or ref == 'W':
        decimal_degrees = -decimal_degrees
    return decimal_degrees

def image_coordinates():  #code inspired from https://medium.com/spatial-data-science/how-to-extract-gps-coordinates-from-images-in-python-e66e542af354
    coords = []
    imagepath = askopenfilenames()
    for i in imagepath:
        with open(i, 'rb') as src:
            img = Image(src)
            if img.has_exif:
                try:
                    img.gps_longitude
                    coords.append(decimal_coords(img.gps_latitude,img.gps_latitude_ref))
                    coords.append(decimal_coords(img.gps_longitude,img.gps_longitude_ref)) 
                except AttributeError:
                    print ('No Coordinates')
            else:
                print ('The Image has no EXIF information')
    return coords

def KNNtest(x,data):
    X = data.drop(columns=['amenity','name']).values
    y = data['name'].values
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)
    KN = KNeighborsClassifier(n_neighbors = 8)
    model = KN.fit(X_train, y_train)
    y_predictedK = model.predict(x)
    return y_predictedK

def mergedata(lists):
    a = len(lists) - 1
    result = lists[0]
    for i in range(a):
        result = pd.concat([result,lists[i + 1]])
    return result

def main():
    coord = []
    coord = image_coordinates()
    print(coord)
    
    dataList = []
    data = pd.read_csv('marketplace.csv')
    dataList.append(data)
    data = pd.read_csv('education.csv')
    dataList.append(data)
    data = pd.read_csv('bar.csv')
    dataList.append(data)
    data = pd.read_csv('activity.csv')
    dataList.append(data)
    data = pd.read_csv('casino.csv')
    dataList.append(data)
    data = pd.read_csv('film.csv')
    dataList.append(data)
    data = pd.read_csv('rental.csv')
    dataList.append(data)
    
    
    
    
    dataLength = len(dataList)


    dataName = [
        'marketplace',
        'education',
        'bar',
        'activity',
        'casino',
        'film',
        'rental',
        'scenery'
        ]
    colorList = np.array([
        [255,0,0],
        [255, 112, 0],
        [255, 255, 0],
        [0, 231, 0],
        [0, 0, 255],
        [185, 0, 185],
        [117, 60, 0],
        [255, 184, 184],
        ])
    colortrans = colorList / 255

    mapvan = plt.imread('map.png')
    Boundary = [-123.5,-122,49,49.5]
    f = plt.figure(figsize = (12,10)) 
    allv = plt.subplot(1,1,1)
    
    allv.imshow(mapvan, zorder=0, extent = Boundary, aspect= 'equal')
    
    for i in range(dataLength):
        dataList[i]=dataList[i].dropna()
        datai = dataList[i]
        datai = datai.astype({'lon':'float','lat':'float'})
        datai = datai.loc[(datai['lon'] > -123.5) & (datai['lon'] < -122) & (datai['lat'] > 49) & (datai['lat'] < 49.5)]
    
        allv.scatter(datai['lon'],datai['lat'], zorder=1, alpha= 1, c=colortrans[i], s=1, label = dataName[i])
    allv.set_title('All activity in Vancouver')
    plt.legend(loc='upper right')
    plt.savefig('All activities in Vancouver.png')
    
    allVA = mergedata(dataList)
    print(allVA)
    
    if(coord != []):
        coord = [coord]
        print('coord successful\n',coord)
        y_predictedK = KNNtest(coord,allVA)
        print(y_predictedK)
    else:
        x = pd.read_csv('location.txt',sep = ':', header=None)
        loc = x.iloc[0][1]
        print(loc)
        geolocator = Nominatim(user_agent="catuserbot")
        geoloc = geolocator.geocode(loc)
        print('Nominatim successful\n',geoloc)
        index = []
        index.append(geoloc.latitude)
        index.append(geoloc.longitude)
        index = [index]
        print(index)
        y_predictedK = KNNtest(index,allVA)
        print(y_predictedK)
    
    


    matched = allVA[allVA['name'].values == y_predictedK]
    print(matched)
    nextA = day_trip(matched,allVA)
    print(nextA)
    nextA = nextA.drop(['distance'],axis = 1)

    first_place = nextA.iloc[[0]]
    
    nextB = day_trip(first_place,nextA)
    print(nextB)
    nextB = nextB.drop(['distance'],axis = 1)

    second_place = nextB.iloc[[0]]
    plan = pd.concat([matched,first_place,second_place])
    plan.to_csv('closest_route.csv',index=False)
    allv.scatter(plan['lon'],plan['lat'],zorder=2, alpha= 1, c=[0,0,0], s=1,label = 'closest activities', linestyle='-')
    allv.set_title('closest route for today')
    plt.legend(loc='upper right')
    plt.savefig('Closest route for today.png')
    
if __name__ == '__main__':
    main()
