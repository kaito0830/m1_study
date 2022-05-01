#%%
import plotlib
## make sure that you need to import athena_read for the latest public version
import imp
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import subprocess
import athena_read

InputForQL_cart \
    = imp.load_source('InputForQL_cart','./InputForQL_cart.py')

#filenames, tag_mov, path_athena_read, InputParams \
filenames, tag_mov, InputParams \
    = InputForQL_cart.InputParameters()
# print(path_athena_read)
# athena_read=imp.load_source('athena_read',path_athena_read)

#%%
nfile = len(filenames)
varname  = InputParams['varname']
vmin     = InputParams['vmin']
vmax     = InputParams['vmax']
colormap = InputParams['colormap']
cnorm    = InputParams['cnorm']
plane    = InputParams['plane']
xlim     = InputParams['xlim']
ylim     = InputParams['ylim']
figsize  = InputParams['figsize']
flag_tex = InputParams['flag_tex']
flag_mov = InputParams['flag_mov']
flag_one = InputParams['flag_one']
nstart   = InputParams['nstart']
nend     = InputParams['nend']
skip     = InputParams['skip']

if (nend < 0): nend = nfile-1
if (flag_one): nend = nstart

movname = varname + "_" + tag_mov + ".mov"

if (plane == "xy"):
    xlabel=r'x'
    ylabel=r'y'
elif (plane == "yz"):
    xlabel=r'y'
    ylabel=r'z'
elif (plane == "xz"):
    xlabel=r'x'
    ylabel=r'z'
else:
    print("Inappropriate parameter for plane")
    
if(flag_tex):
    xlabel = plotlib.AddFontCommand(xlabel)
    ylabel = plotlib.AddFontCommand(ylabel)

#%%

plt.close('all')
if(flag_mov): 
    plt.ioff()
else:
    plt.ion()
if(flag_tex): plotlib.prep_tex() ## for LaTeX
fs = 20
fs_old = plt.rcParams["font.size"]
plt.rcParams["font.size"] = fs

#%%
try:
    flag = 0
    for n in range(nstart,nend+1,skip):
        filename = filenames[n]
        var, x1, x2, time = athena_read.Get2Ddata(filename,varname,plane)

        if flag == 0:
            flag = 1
            fig = plt.figure(1,figsize=figsize)
            fig.subplots_adjust(bottom=0.2,top=0.9,left=0.2,right=0.85)
            ax1 = fig.add_subplot(111)

            im1 = ax1.pcolormesh(x1, x2, var, 
                vmin=vmin,vmax=vmax,cmap=colormap,norm=cnorm,shading='nearest')
    
            divider1 = make_axes_locatable(ax1)
            cax1 = divider1.append_axes("right", size="5%", pad=0.5)
            cb1  = fig.colorbar(im1, cax=cax1, orientation='vertical')
        else:
            ax1.clear()
            im1 = ax1.pcolormesh(x1, x2, var, 
                vmin=vmin,vmax=vmax,cmap=colormap,norm=cnorm,shading='nearest')

        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim)
        ax1.set_aspect('equal')

        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel,rotation=0)

        text0 = r'Time = '+'{:.3f}'.format(time)
        if(flag_tex): text0 = plotlib.AddFontCommand(text0)
        ax1.text(0.0, 1.03, text0, transform = ax1.transAxes)
        
        if not (flag_mov): plt.pause(0.1)

        if(flag_mov):
            imgname='ImgForMovie'+str(n-nstart).zfill(4)+'.png'
            print(imgname)
            fig.savefig(imgname,dpi=200)

    if(flag_mov):
        res = subprocess.call(r"""ffmpeg -y -i 'ImgForMovie%04d.png' \
                            -crf 16 -c:v libx264 -pix_fmt yuv420p -filter:v "setpts=2*PTS" """ \
                            +movname,shell=True)
        if res == 0 : subprocess.call(r"rm ImgForMovie*.png",shell=True)
        print(">>> "+movname+" has been created.")


## finally block will be always executed even though exceptions occur.
finally:
    if(flag_mov): plt.close('all')
    plotlib.reset_rcParams()
    plt.rcParams["font.size"] = fs_old
    plt.ion()