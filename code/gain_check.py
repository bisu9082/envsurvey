"""R2: 40K photopeak centroid in each field spectrum (gain-stability check, SI Section S7 / Fig. S3).
Method: Gaussian + linear continuum fitted to 1380-1540 keV, Poisson weights, recorded energy scale
(MAESTRO channel index from 0). Acquisition start times from the $DATE_MEA header of each .Spe file.
Usage: python3 gain_check.py   -> gain_check.log / gain_check.csv"""
import numpy as np, os, io, contextlib, importlib.util, datetime
from scipy.optimize import curve_fit
HERE=os.path.dirname(os.path.abspath(__file__))
sp=importlib.util.spec_from_file_location("d",os.path.join(HERE,"dose_r2.py"))
with contextlib.redirect_stdout(io.StringIO()):
    d=importlib.util.module_from_spec(sp); sp.loader.exec_module(d)
LOGDIR=os.path.join(HERE,"..","logs") if os.path.isdir(os.path.join(HERE,"..","logs")) else HERE
out=[]
def P(*a):
    s=" ".join(str(x) for x in a); print(s); out.append(s)
P("# gain_check.py run",datetime.datetime.now().isoformat(timespec='seconds'))
def tstart(fn):
    L=open(os.path.join(d.SPE,fn+".Spe"),errors='replace').read().split("\n")
    i=L.index("$DATE_MEA:"); return L[i+1].strip()
g=lambda x,A,mu,s,c0,c1: A*np.exp(-0.5*((x-mu)/s)**2)+c0+c1*(x-mu)
rows=[]
for (c,lt,a,b),pid,fn in zip(d.SP,d.ids,d.files):
    E=a+b*np.arange(len(c)); m=(E>=1380)&(E<=1540); x=E[m]; y=c[m]
    p,cov=curve_fit(g,x,y,p0=[y.max()-y.min(),1475,25,y.min(),0],sigma=np.sqrt(np.maximum(y,1)),maxfev=40000)
    rows.append((tstart(fn),pid,fn,p[1],np.sqrt(cov[1,1]),2.355*abs(p[2])))
rows.sort()
P("%-20s %-6s %-6s %9s %6s %8s"%("start (SPE header)","point","file","centroid","1sig","FWHM"))
for t,pid,fn,mu,s,fw in rows: P("%-20s %-6s %-6s %9.1f %6.1f %8.1f"%(t,pid,fn,mu,s,fw))
mus=np.array([r[3] for r in rows]); sig=np.array([r[4] for r in rows])
P("mean %.1f keV (nominal 1460.8 -> %+.1f keV, %+.2f %%) ; SD %.1f keV ; peak-to-peak %.1f keV (%.1f channels) ; median fit 1sig %.1f keV"%(mus.mean(),mus.mean()-1460.8,100*(mus.mean()/1460.8-1),mus.std(ddof=1),mus.max()-mus.min(),(mus.max()-mus.min())/3.0734,np.median(sig)))
chi=np.sum(((mus-np.average(mus,weights=1/sig**2))/sig)**2); P("chi2 of constant centroid: %.1f for %d dof"%(chi,len(mus)-1))
tm=np.array([int(r[0][11:13])*60+int(r[0][14:16]) for r in rows]); from scipy import stats
s_=stats.linregress(tm,mus); P("linear trend with time: %.2f keV/h (p = %.2f)"%(60*s_.slope,s_.pvalue))
P("note: field log lists P3E5m at 11:40 and P3W5m at 12:03; the SPE headers of 3동쪽/3서쪽 give 12:03:33 and 11:37:33")
open(os.path.join(LOGDIR,"gain_check.log"),"w").write("\n".join(out)+"\n")
with open(os.path.join(LOGDIR,"gain_check.csv"),"w") as f:
    f.write("start,point,file,centroid_keV,sig_keV,fwhm_keV\n")
    for r in rows: f.write("%s,%s,%s,%.4f,%.4f,%.2f\n"%r)
