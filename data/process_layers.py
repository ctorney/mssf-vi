
import rioxarray as rxr
import rasterio
from rasterio.plot import plotting_extent
import xarray as xr
import geopandas as gpd
import numpy as np
np.set_printoptions(suppress=True)

##
# import the data
landuse=rxr.open_rasterio("landuse.tif",masked=True)
population=rxr.open_rasterio("population.tif",masked=True)
elevation=rxr.open_rasterio("elevation.tif",masked=True)


##

# plot the data on different figures
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

landuse.plot(ax=ax)

fig, ax = plt.subplots()

population.plot(ax=ax)

fig, ax = plt.subplots()

elevation.plot(ax=ax)

plt.show()


##

merged = xr.concat([landuse, population, elevation],dim='band')

print(merged)

##

# print the dimensions of the individual datasets
print(landuse.rio.shape)
print(population.rio.shape)
print(elevation.rio.shape)

# print the extent of the individual datasets
print(landuse.rio.bounds())
print(population.rio.bounds())
print(elevation.rio.bounds())

##
# resample the population to match the elevation
population_resampled = population.rio.reproject_match(elevation)
population_resampled = population_resampled.rio.interpolate_na(method="nearest")
# resample the landuse to match the elevation
landuse_resampled = landuse.rio.reproject_match(elevation)


print(population_resampled.rio.shape)
print(population_resampled.rio.bounds())

print(landuse_resampled.rio.shape)
print(landuse_resampled.rio.bounds())

population_resampled.plot()
plt.show()

landuse_resampled.plot()
plt.show()

elevation.plot()
plt.show()

##

# create a new xarray dataset that is 0 if landuse_resampled is less than 160 and 1 if it is greater than 160
landuse_wet = xr.where(landuse_resampled >= 160, 1, 0)

# plot the new dataset
landuse_wet.plot()
plt.show()

##
# create a new xarray dataset that is 1 if landuse_resampled is between 120 and 160 otherwise 0
landuse_grass = xr.where((landuse_resampled >= 120) & (landuse_resampled < 160), 1, 0)

# plot the new dataset
landuse_grass.plot()
plt.show()

##

merged = xr.concat([population_resampled, elevation, landuse_grass, landuse_wet],dim='band')

print(merged)
# save as tif
merged.rio.to_raster("amt_covars_merged.tif")

