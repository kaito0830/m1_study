import glob
import matplotlib.colors as colors

def InputParameters():
    filenames = glob.glob('OrszagTang.out2.*.athdf')
    filenames.sort() # dont't forget to do this
    tag_mov = 'muscl2'
    #path_athena_read = '/Users/shinsuketakasao/Simulation/athena2019august/vis/python/athena_read.py'
    
    ## Parameters for colormap
    varname = 'rho'
    vmin  = 0.
    vmax  = 0.5
    colormap = 'gray'
    cnorm = colors.Normalize() ## colors.LogNorm() or colors.Normalize()

    ## Parameters for plotted domain
    #flag_slice = False
    plane = 'xy' # yz, xz
    xlim = [-0.5,0.5]
    #xlim = [-0.2,0.2]
    ylim = [-0.5,0.5]

    ## General parameters
    figsize = (9.0,7.0)
    flag_mov = True 
    flag_tex = False
    flag_one = False # when True, data only at n=nstart is shown.
    nstart = 90
    nend = -1
    skip = 1

    InputParams = {}
    InputParams['varname'] = varname
    InputParams['vmin'] = vmin
    InputParams['vmax'] = vmax
    InputParams['colormap'] = colormap
    InputParams['cnorm'] = cnorm
    InputParams['plane'] = plane
    InputParams['xlim'] = xlim
    InputParams['ylim'] = ylim
    InputParams['figsize'] = figsize
    InputParams['flag_tex'] = flag_tex
    InputParams['flag_mov'] = flag_mov
    InputParams['flag_one'] = flag_one
    
    InputParams['nstart'] = nstart
    InputParams['nend'] = nend
    InputParams['skip'] = skip

    #return filenames, tag_mov, path_athena_read, InputParams
    return filenames, tag_mov, InputParams