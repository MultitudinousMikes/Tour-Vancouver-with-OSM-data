import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.stats.weightstats as sw
import sys

filename = sys.argv[1]

data = pd.read_json(filename, compression='gzip', lines=True)
data['tags'] = data.tags.astype('str')

# Clean data
data = data.dropna(how='any')
data = data.drop_duplicates()

food = data[(data.amenity == 'fast_food') | (data.amenity == 'restaurant')]
food = food.groupby('name').count()
food['NewChain'] = (food.tags >= 2)
food = food.reset_index()
data = pd.merge(data, food[["name", "NewChain"]], on="name")
food = data[["lat", "lon", "name", "NewChain"]]
print('restaurant number: ', len(food))

Chain = food[food.NewChain == True]
NonChain = food[food.NewChain == False]

# Data description of chain restaurants and non-chain restaurants.
print('Chain Count: ', Chain.name.count(), 'Non-Chain Count: ', NonChain.name.count())
print('Chain Latitude Mean: ', Chain.lat.mean(), 'Non-Chain Latitude Mean: ', NonChain.lat.mean())
print('Chain Longitude Mean: ', Chain.lon.mean(), 'Non-Chain Longitude Mean: ', NonChain.lon.mean())
print('Chain Latitude Standard Deviation: ', Chain.lat.std(), 'Non-Chain Latitude Standard Deviation: ',
      NonChain.lat.std())
print('Chain Longitude Standard Deviation: ', Chain.lon.std(), 'Non-Chain Longitude Standard Deviation: ',
      NonChain.lon.std())

# visualize chains' density relative to non-chains
plt.figure()
chain_density, x, y, patches = plt.hist2d(Chain.lon, Chain.lat)
plt.colorbar(patches)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Chains Density')

plt.figure()
not_chain_density, x, y, patches = plt.hist2d(NonChain.lon, NonChain.lat)
plt.colorbar(patches)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Non-Chains Density')

plt.figure()
h = plt.imshow(chain_density - not_chain_density)
plt.xticks(np.arange(0, 10, 2), np.around(x[:10:2], 1))
plt.yticks(np.arange(0, 10, 2), np.around(y[:10:2], 2))
plt.colorbar()
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Density difference between Chains and Non-Chains')

plt.show()

# Find outliers
print('=======Find outliers========')
plt.figure()

print(Chain.lat.mean(), Chain.lat.std())
print(Chain.lat.min(), Chain.lat.max())
plt.subplot(2, 2, 1)
plt.boxplot(Chain.lat, notch=True, vert=False)
plt.title('Chain Restaurants Latitude')

print(Chain.lon.mean(), Chain.lon.std())
print(Chain.lon.min(), Chain.lon.max())
plt.subplot(2, 2, 2)
plt.boxplot(Chain.lon, notch=True, vert=False)
plt.title('Chain Restaurants Longitude')

print(NonChain.lat.mean(), NonChain.lat.std())
print(NonChain.lat.min(), NonChain.lat.max())
plt.subplot(2, 2, 3)
plt.boxplot(NonChain.lat, notch=True, vert=False)
plt.xlabel('Non-Chain Restaurants Latitude')

print(NonChain.lon.mean(), NonChain.lon.std())
print(NonChain.lon.min(), NonChain.lon.max())
plt.subplot(2, 2, 4)
plt.boxplot(NonChain.lon, notch=True, vert=False)
plt.xlabel('Non-Chain Restaurants Longitude')
plt.show()

# Plot density for Restaurants
plt.figure(figsize=(6, 5))
plt.plot(NonChain['lon'], NonChain['lat'], 'g.', alpha=0.3)
plt.title('Density for NonChain Restaurant')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

plt.figure(figsize=(6, 5))
plt.plot(Chain['lon'], Chain['lat'], 'r.', alpha=0.3)
plt.title('Density for Chain Restaurant')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

plt.figure(figsize=(6, 5))
plt.plot(NonChain['lon'], NonChain['lat'], 'g.', alpha=0.6)
plt.plot(Chain['lon'], Chain['lat'], 'r.', alpha=0.6)
plt.title('Density for NonChain and Chain Restaurant (put together)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(['Non-Chain', 'Chain'])
plt.show()

# Clustering
from sklearn.cluster import DBSCAN

plt.figure()
plt.subplot(2, 1, 1)
chain_labels = DBSCAN(eps=0.03, min_samples=3).fit_predict(Chain[['lon', 'lat']])
plt.scatter(Chain['lon'], Chain['lat'], c=chain_labels)
plt.title('Chain Restaurant Clustering')

plt.subplot(2, 1, 2)
chain_labels = DBSCAN(eps=0.03, min_samples=3).fit_predict(NonChain[['lon', 'lat']])
plt.scatter(NonChain['lon'], NonChain['lat'], c=chain_labels)
plt.title('Non-Chain Restaurant Clustering')
plt.tight_layout()
plt.show()

print('Latitude Z-test: ', sw.ztest(Chain['lat'], NonChain['lat']))
print('Longitude Z-test: ', sw.ztest(Chain['lon'], NonChain['lon']))
