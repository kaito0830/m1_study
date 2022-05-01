# This code was written by Shinsuke Takasao and Kengo Shibata.
#%%
import numpy as np
import glob
from os.path import join
import athena_read
import imp
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import plotlib # user-defined library

## for plot
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import subprocess

## for MPI 
from mpi4py import MPI
import time

# MPI process
comm_mpi = MPI.COMM_WORLD
rank_mpi = comm_mpi.Get_rank()
size_mpi = comm_mpi.Get_size()

def CalculateAz(Bx,By,xv,yv,dxf,dyf,ng):
    ncells1 = xv.size
    ncells2 = yv.size
    IntegBx = np.zeros(Bx.shape)  ## int^y_ymin Bx(x,y') dy'
    IntegBy = np.zeros(Bx.shape)  ## int^x_xmin By(x',y) dx'

    ## Second order integration (Trapezoidal integration)

    ## By is integrated along x-axis at y=ymin
    IntegBy[:,ng] = By[ng,ng] * (0.5*dxf[ng])
    IntegBy[:,ng-1] = IntegBy[:,ng] - By[ng,ng-1] * dxf[ng-1] ## one step outside the boundary
    for i in range(ng+1,ncells1):
        IntegBy[:,i] = IntegBy[:,i-1] + 0.5*(By[:,i-1]+By[:,i]) * dxf[i-1]

    IntegBx[ng,:] = Bx[ng,:] * (0.5*dyf[ng])
    IntegBx[ng-1,:] = IntegBx[ng,:] - Bx[ng-1,:] * dyf[ng-1] ## one step outside the boundary
    for j in range(ng+1,ncells2):
        IntegBx[j,:] = IntegBx[j-1,:] + 0.5*(Bx[j-1,:]+Bx[j,:]) * dyf[j-1]

    ## Get Az
    #Az = IntegBx[:,:] - IntegBy[ng,:]
    Az = IntegBx[:,:] - 0.5*(IntegBy[ng-1,:]+IntegBy[ng,:])
    return Az


#%%

InputForMakeMovieWithBlines \
    = imp.load_source('InputForMakeMovieWithBlines','./InputForMakeMovieWithBlines.py')

path_data, mainfilename_var, mainfilename_Bcc, name_az, flag_createAz, ng, \
    varname, movname, extent, vmin, vmax, colormap, norm, nlines, linecolor, figsize, zoom \
    = InputForMakeMovieWithBlines.InputParameters()

## Get the list of files used

extfname  = '.athdf'
files_var = [x for x in glob.glob(join(path_data,mainfilename_var+'*'+extfname))]
files_var.sort() # sorting in a natural order
files_Bcc = [x for x in glob.glob(join(path_data,mainfilename_Bcc+'*'+extfname))]
files_Bcc.sort() # sorting in a natural order

#############= MPI TASK DISTRIBUTION =#############
N_end = len(files_var)                           # number of data files
num_per_rank = N_end // size_mpi        # number of files to treat per cpu
# (floor division //) rounds N_end to the nearest integer number (smaller one).

# Define lower and upper band for each rank
lower_bound = 0 + rank_mpi * num_per_rank
upper_bound = 0 + (rank_mpi + 1) * num_per_rank
N_end_mpi = upper_bound - lower_bound           # variable for log
N_missing = N_end - num_per_rank*size_mpi
# Add to each missing files (if N_end / size_mpi != integer)
if (N_missing != 0.0):
    if(rank_mpi <  N_missing):
        i_missing = int((size_mpi) * num_per_rank + rank_mpi)      # new lower_bound
        print("i_missing ",i_missing+1)
        files_Bcc_rank = np.concatenate((files_Bcc[lower_bound:upper_bound],[files_Bcc[i_missing]]))
        files_var_rank = np.concatenate((files_var[lower_bound:upper_bound],[files_var[i_missing]]))
        print("This is processor ", rank_mpi, "and I am treating files numbers from", lower_bound," to ", upper_bound-1, " and file ", i_missing, flush=True)
    else:
        files_Bcc_rank = files_Bcc[lower_bound:upper_bound]
        files_var_rank = files_var[lower_bound:upper_bound]
        print("This is processor ", rank_mpi, "and I am treating files numbers from", lower_bound," to ", upper_bound-1, flush=True)
else:
    files_Bcc_rank = files_Bcc[lower_bound:upper_bound]
    files_var_rank = files_var[lower_bound:upper_bound]
    print("This is processor ", rank_mpi, "and I am treating files numbers from", lower_bound," to ", upper_bound-1, flush=True)


comm_mpi.Barrier()
start_time = time.time()

if not (len(files_var) == len(files_Bcc)):
    raise athena_read.AthenaError('File number mismatch')

##################= MAIN LOOP AZ=##################
# Start loop over all data files
# Each rank is assigned a different range of data file to treat
print('*=== Starting main AZ loop for cpu-', rank_mpi,' ===*')
if (flag_createAz):
    data = athena_read.athdf(filename=files_Bcc[0],num_ghost=ng)
    nx1 = data['x1v'].size # including ghost cells
    nx2 = data['x2v'].size # including ghost cells
    ntime = len(files_Bcc)
    Az = np.zeros((ntime,nx2,nx1))

    for step_rank,filename in enumerate(files_Bcc_rank):
        if ((step_rank==len(files_var_rank)-1) and (rank_mpi <  N_missing) and (N_missing != 0.0)):
            step=i_missing
        else:
            step=step_rank+lower_bound
        print('filename: ',filename, " step= ",step," rank step= ",step_rank)
        ## Note: Specify ghost cell number correctly.
        data = athena_read.athdf(filename=filename,num_ghost=ng)

        Bx = data['Bcc1'][0,:,:]
        By = data['Bcc2'][0,:,:]
    
        ## cell-centered coordinates
        xv = data['x1v'].copy()
        yv = data['x2v'].copy()
    
        ## cell-face-centered coordinates
        xf = data['x1f'].copy()
        yf = data['x2f'].copy()
    
        dxf = xf[1:] - xf[0:-1]
        dyf = yf[1:] - yf[0:-1]
    
        Az[step,:,:] = CalculateAz(Bx,By,xv,yv,dxf,dyf,ng)

    ## Save Az as a numpy binary file
    print('Creating '+path_data+'/'+name_az+str(rank_mpi)+'.npz')
    np.savez(path_data+'/'+name_az+str(rank_mpi), Az=Az, xv_g=xv, yv_g=yv)


comm_mpi.Barrier()
#%%
## Plot results
print('*=== Starting main plot loop for cpu-', rank_mpi,' ===*')
data_Az = np.load(path_data+'/'+name_az+str(rank_mpi)+'.npz')

Az_min = data_Az['Az'].min()
Az_max = data_Az['Az'].max()
levels = np.linspace(Az_min,Az_max,nlines)

fs = 20 ## font size
fs_old = plt.rcParams["font.size"]
plt.rcParams["font.size"] = fs
plt.close('all')
plt.ioff()
for step_rank,filename in enumerate(files_var_rank):
    if ((step_rank==len(files_var_rank)-1) and (rank_mpi <  N_missing) and (N_missing != 0.0)):
        step=i_missing
    else:
        step=step_rank+lower_bound
        
    data = athena_read.athdf(filename=filename)
    if varname == 'temperature':
        var = 5.0/3.0*data['press'][0,:,:]/data['rho'][0,:,:]
    elif varname == 'rho':
        var  = np.log10(data[varname][0,:,:]) # var is assumed to include NO ghost cells
    else:
        var  = data[varname][0,:,:] # var is assumed to include NO ghost cells
     
    time = data['Time']
    
    plt.clf()
    fig = plt.figure(1,figsize=figsize)
    fig.subplots_adjust(bottom=0.2,top=0.85,left=0.2,right=0.9)
    ax = fig.add_subplot(111)
    
    im = ax.imshow(var,vmin=vmin,vmax=vmax,
                   extent=extent,interpolation='nearest',origin='lower',aspect=1,
                   cmap=colormap)
    
    X, Y = np.meshgrid(data_Az['xv_g'],data_Az['yv_g'])
    cn = ax.contour(X[ng-1:-ng+1,ng-1:-ng+1],Y[ng-1:-ng+1,ng-1:-ng+1],
                    data_Az["Az"][step,ng-1:-ng+1,ng-1:-ng+1],
                    levels=levels,colors=linecolor,linestyles='-')

    ## Set color bar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="8%", pad=0.1)
    cbar = plt.colorbar(im,ax=ax,cax=cax)

    ax.set_xlabel(r'$x$') ## in tex format
    ax.set_ylabel(r'$y$')
    ax.text(-0.1,1.08,r'Time =  '+r'${:.1f}$'.format(time),transform=ax.transAxes)
    if zoom==1:
        #axes = fig.add_axes([0.2,0.2,0.85,0.9])
        ax.set_xlim(-5.,5.)
        ax.set_ylim(0.0,10.0)
    
    if varname == 'temperature':
        plt.title(r'$T \left( \frac{K}{2 \times 10^6 K} \right)$',fontsize='small')
    elif varname == 'rho':
        plt.title(r'$log_{10}(\rho) \left( \frac{g.cm^{-3}}{\rho_0} \right)$', fontsize='small')
    
    # plt.draw()
    # plt.pause(0.2)
    imgname = 'TemporaryImageForMovie'+str(step).zfill(4)+'.png'
    print(imgname)
    fig.savefig(imgname)
    
# Wait that all cpu finishes their job
comm_mpi.Barrier()

#%%
# Make a movie
# The commands below should be passed to the shell, so we need to add the option shell=True
if (rank_mpi==0):
    res = subprocess.call(r"""ffmpeg -i 'TemporaryImageForMovie%04d.png' -crf 16 -c:v libx264 -pix_fmt yuv420p -filter:v "setpts=2*PTS" """+movname,shell=True) 
    if res == 0 : subprocess.call(r"rm TemporaryImageForMovie*.png",shell=True) ## erase temporary images

    plt.rcParams["font.size"] = fs_old ## set the font size to the original value
    plt.close('all')

    # %%
