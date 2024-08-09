import matplotlib.pyplot as plt
import scipy.stats as stats

import numpy as np
import pandas as pd

import rioxarray as rxr
import rasterio
from rasterio.plot import plotting_extent
import xarray as xr
import geopandas as gpd
np.set_printoptions(suppress=True)

import matplotlib.ticker as tick
plt.style.use('ggplot') 
plt.style.use('seaborn-v0_8-paper') 
plt.style.use('seaborn-v0_8-whitegrid') 

##
# Import data
FH=pd.read_csv("../data/amt_fisher.csv",low_memory=False)
covars = rxr.open_rasterio("../data/amt_covars_merged.tif",masked=True)
FH=FH[FH["name"]=="Lupe"]


##

# Visualisation
fig, ax = plt.subplots(2, 2, figsize=(22,20))
ax[0, 0].set_aspect('equal')
# plot with a colorbar with labelsize 20
vals = covars.sel(band=1).values[::-1]

ax00 = ax[0,0].contourf(vals, cmap="viridis",extent=plotting_extent(covars.sel(band=1),covars.rio.transform()),levels=100)
# ax00 = covars.sel(band=1).plot(robust=True,ax=ax[0,0],cmap="viridis",
        # cbar_kwargs={"label":"Population density","pad":0.015,"shrink":1, "extend":"neither", 'labelsize': 20})
# ax00 = covars.sel(band=1).plot(robust=True,ax=ax[0,0],cmap="viridis",
        # cbar_kwargs={"label":"Population density","pad":0.015,"shrink":1, "extend":"neither", 'labelsize': 20})

#axs[0,0].set_title('A',loc='left',size=30,pad=100)
ax[0,0].text(-0.1,0.96,'A', size=30, transform=ax[0, 0].transAxes)
ax[0,0].tick_params( labelleft=False, labelbottom=False) 








cbar = fig.colorbar(ax00, ax=ax[0, 0],fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=20) 
cbar.set_label(label=r'population density',size=20)





ax[0,0].plot(FH["X"],FH["Y"],color="black")

ax[0,1].set_aspect('equal')
vals = covars.sel(band=2).values[::-1]
ax01 = ax[0,1].contourf(vals, cmap="viridis",extent=plotting_extent(covars.sel(band=2),covars.rio.transform()),levels=100)

ax[0,1].text(-0.1,0.96,'B', size=30, transform=ax[0, 1].transAxes)
ax[0,1].tick_params( labelleft=False, labelbottom=False)

cb = fig.colorbar(ax01, ax=ax[0, 1],fraction=0.046, pad=0.04)
cb.ax.tick_params(labelsize=20)
cb.set_label(label=r'elevation',size=20)

ax[0,1].plot(FH["X"],FH["Y"],color="black")

ax[1,0].set_aspect('equal')
vals = covars.sel(band=3).values[::-1]
ax10 = ax[1,0].contourf(vals, cmap="viridis",extent=plotting_extent(covars.sel(band=3),covars.rio.transform()),levels=100)

ax[1,0].text(-0.1,0.96,'C', size=30, transform=ax[1, 0].transAxes)
ax[1,0].tick_params( labelleft=False, labelbottom=False)

cbar = fig.colorbar(ax10, ax=ax[1, 0],fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=20)
cbar.set_label(label=r'grass',size=20)

tick_locator = tick.MaxNLocator(nbins=1)
cbar.locator = tick_locator
cbar.update_ticks()

ax[1,0].plot(FH["X"],FH["Y"],color="black")

ax[1,1].set_aspect('equal')
vals = covars.sel(band=4).values[::-1]
ax11 = ax[1,1].contourf(vals, cmap="viridis",extent=plotting_extent(covars.sel(band=4),covars.rio.transform()),levels=100)

ax[1,1].text(-0.1,0.96,'D', size=30, transform=ax[1, 1].transAxes)
ax[1,1].tick_params( labelleft=False, labelbottom=False)

cbar = fig.colorbar(ax11, ax=ax[1, 1],fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=20)
cbar.set_label(label=r'wet',size=20)

tick_locator = tick.MaxNLocator(nbins=1)
cbar.locator = tick_locator
cbar.update_ticks()

ax[1,1].plot(FH["X"],FH["Y"],color="black")
# ax[1,1].plot(sp[:,0], sp[:,1],".",color="black",alpha=0.5)

ax[0,0].set_axis_off()
ax[0,1].set_axis_off()
ax[1,0].set_axis_off()
ax[1,1].set_axis_off()

ax[0,0].set_title("")
ax[0,1].set_title("")
ax[1,0].set_title("")
ax[1,1].set_title("")
plt.subplots_adjust(wspace=0.1, hspace=0.1)


plt.savefig("case_study_covars.png",bbox_inches='tight',dpi=300)
plt.close()
##


beta_mean = np.load("../case_study/beta_mean.npy")
beta_std = np.load("../case_study/beta_std.npy")

fig,ax=plt.subplots(1,4,figsize=(24,10))#, constrained_layout=True)

labels = ['A', 'B', 'C', 'D']
colors = ['C0', 'C1', 'C2', 'C3']
xlabels = [r"$\beta_1$", r"$\beta_2$", r"$\beta_3$", r"$\beta_4$"]
for i in range(4):
    mean = beta_mean[i]
    std = beta_std[0][i]
    x = np.linspace(mean-4*std,mean+4*std,100)
    y = stats.norm.pdf(x,mean,std)
    ax[i].plot(x,y,color=colors[i])
    ax[i].fill_between(x, y, 0, color=colors[i], alpha=0.3)
    ax[i].set_xlabel(xlabels[i],fontsize=15)
    ax[i].set_ylim(0,None)
    ax[i].text(-0.15,1.01,labels[i], size=30, transform=ax[i].transAxes)
    ax[i].tick_params(axis='x', which='major', length=0,labelsize=15)
    ax[i].tick_params(axis='y', direction='out', length=10,labelsize=15)
    
# plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.savefig("case_study_distributions.png",bbox_inches='tight',dpi=300)
plt.close()

