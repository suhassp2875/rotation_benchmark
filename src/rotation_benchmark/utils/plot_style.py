
import matplotlib.pyplot as plt
import matplotlib as mpl

# Tol's colorblind-friendly palette
COLORS = {
    # Representations
    "6D": "#D55E00",      # Vermillion
    "Lie_FullFix": "#0072B2", # Blue
    "Lie_LogGeo": "#56B4E9",  # Sky Blue
    "Lie_Bounded": "#E69F00", # Orange
    "Atlas": "#009E73",       # Bluish Green (Novel Method)
    "Euler": "#CC79A7",       # Reddish Purple
    "Quat": "#F0E442",        # Yellow
    
    # General
    "GT": "#000000",
    "Pred": "#D55E00"
}

def apply_plot_style():
    """Applies clean, minimal style for benchmark figures."""
    
    # Reset
    plt.style.use('default')
    
    # Fonts
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
    mpl.rcParams['font.size'] = 10
    mpl.rcParams['axes.labelsize'] = 10
    mpl.rcParams['axes.titlesize'] = 10
    mpl.rcParams['xtick.labelsize'] = 8
    mpl.rcParams['ytick.labelsize'] = 8
    mpl.rcParams['legend.fontsize'] = 8
    
    # Lines
    mpl.rcParams['lines.linewidth'] = 1.5
    mpl.rcParams['lines.markersize'] = 4
    
    # Grid
    mpl.rcParams['axes.grid'] = True
    mpl.rcParams['grid.alpha'] = 0.3
    mpl.rcParams['grid.linestyle'] = '--'
    
    # Saving
    mpl.rcParams['savefig.dpi'] = 300
    mpl.rcParams['savefig.bbox'] = 'tight'
    mpl.rcParams['savefig.pad_inches'] = 0.05
    
def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    return tuple(i/inch for i in tupl)
