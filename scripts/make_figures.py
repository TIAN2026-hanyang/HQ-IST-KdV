"""Rebuild manuscript figures from archived numerical arrays only.
Each panel is an independent Matplotlib figure. No smoothing or resimulation.
The space-time maps use the actual (occasionally nonuniform) time coordinates.
"""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'.mplconfig'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
ROOT=Path(__file__).resolve().parents[1]
SETTINGS={
 'single': {'snap':[0,.8,1.2,2], 'xlim':(-8,16)},
 'double': {'snap':[0,1.5,3,7], 'xlim':(-8,34)},
 'triple': {'snap':[0,.125,.25,.5], 'xlim':(-6,21)},
}

def save(fig,name):
    out=ROOT/'figures';out.mkdir(exist_ok=True)
    for ext in ('pdf','png'):
        fig.savefig(out/f'{name}.{ext}',dpi=240,bbox_inches='tight',pad_inches=.035)
    plt.close(fig)

def main():
    for case,config in SETTINGS.items():
        with np.load(ROOT/'data'/f'{case}.npz',allow_pickle=False) as d:
            x=d['x']; times=d['times']; ex=d['u_exact']; num=d['u_hybrid']
            if ex.shape!=num.shape or ex.shape!=(len(times),len(x)):
                raise ValueError('Inconsistent numerical archives')
        for j,t in enumerate(config['snap']):
            idx=int(np.argmin(np.abs(times-t)))
            if not np.isclose(times[idx],t,rtol=0,atol=1e-12):
                raise ValueError(f'{case}: requested t={t} is not saved')
            fig=plt.figure(figsize=(3.55,2.42));ax=fig.add_subplot(111)
            ax.plot(x,ex[idx],linewidth=1.2,label='Exact')
            stride=max(1,len(x)//200)
            ax.plot(x[::stride],num[idx,::stride],linestyle='None',marker='o',
                    markersize=2.25,markerfacecolor='none',markeredgewidth=.65,label='Hybrid')
            ax.set(xlim=config['xlim'],ylim=(-.04*ex.max(),1.08*ex.max()),xlabel='$x$',ylabel='$u(x,t)$')
            ax.set_title(f'({chr(97+j)}) $t={t:g}$',fontsize=10,pad=5)
            ax.tick_params(labelsize=8);ax.xaxis.label.set_size(9);ax.yaxis.label.set_size(9)
            ax.legend(loc='upper left',fontsize=8,frameon=False,handlelength=1.7)
            ax.xaxis.set_major_locator(MaxNLocator(5));fig.tight_layout(pad=.6)
            save(fig,f'{case}_profile_{j}')
        lo=min(ex.min(),num.min());hi=max(ex.max(),num.max())
        maps=[('exact',ex,'Exact solution'),('hybrid',num,'Hybrid reconstruction'),
              ('error',np.abs(num-ex),'Absolute error')]
        for j,(tag,field,title) in enumerate(maps):
            fig=plt.figure(figsize=(2.85,3.05));ax=fig.add_subplot(111)
            lim=(0,float(field.max())) if tag=='error' else (float(lo),float(hi))
            im=ax.pcolormesh(x,times,field,shading='nearest',vmin=lim[0],vmax=lim[1],rasterized=True)
            ax.set(xlim=config['xlim'],ylim=(times[0],times[-1]),xlabel='$x$',ylabel='$t$')
            ax.set_title(f'({chr(97+j)}) {title}',fontsize=9,pad=5)
            ax.tick_params(labelsize=8);ax.xaxis.label.set_size(9);ax.yaxis.label.set_size(9)
            ax.xaxis.set_major_locator(MaxNLocator(4));ax.yaxis.set_major_locator(MaxNLocator(5))
            cb=fig.colorbar(im,ax=ax,pad=.035,fraction=.065)
            cb.ax.tick_params(labelsize=7);cb.formatter.set_powerlimits((-3,3));cb.update_ticks()
            cb.ax.yaxis.get_offset_text().set_fontsize(7)
            fig.tight_layout(pad=.55)
            save(fig,f'{case}_spacetime_{tag}')
    print('Saved 21 independent panels in both PDF and PNG formats.')
if __name__=='__main__':main()
