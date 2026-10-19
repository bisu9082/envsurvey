"""R2: how much cosmic-ray signal is already inside the 50 keV-3.1 MeV G(E) summation window?
Above the 2614.5 keV Tl-208 line there are no terrestrial full-energy peaks; counts in 2.75-3.1 MeV are
taken as cosmic-dominated (plus a small Tl-208 tail/sum contribution, which makes the estimate an upper bound).
Upper bound for the in-window cosmic dose: the cosmic count density per keV measured in 2.75-3.1 MeV,
extrapolated FLAT down to 50 keV and converted with the same G(E)*k weights as the dose."""
import numpy as np, os, sys, datetime
HERE=os.path.dirname(os.path.abspath(__file__))
sys.argv=[sys.argv[0]]
import importlib.util
cand=[os.path.join(HERE,"..","r2_dose","dose_r2.py"), os.path.join(HERE,"dose_r2.py")]
spec=importlib.util.spec_from_file_location("d", next(p for p in cand if os.path.exists(p)))
import io, contextlib
buf=io.StringIO()
with contextlib.redirect_stdout(buf):
    d=importlib.util.module_from_spec(spec); spec.loader.exec_module(d)
LOGDIR=os.path.join(HERE,"..","logs") if os.path.isdir(os.path.join(HERE,"..","logs")) else HERE
out=[]
def P(*a):
    s=" ".join(str(x) for x in a); print(s); out.append(s)
P("# cosmic_in_window.py run", datetime.datetime.now().isoformat(timespec='seconds'))
R=d.R; ids=d.ids
_f=lambda z: 0.21*np.exp(-1.649*z)+0.79*np.exp(0.4528*z)
CEST=(30*_f(0.664)+32*_f(0.691))/2/1000   # same central estimate as cosmic_r2.py
P("%-6s %8s %9s %9s %8s %9s %9s"%("point","n(2.75-3.1)","rate cps","last ch","H_G(E)","H_hi","H_flat_ub"))
Hhi=[];Hub=[];HG=[]
for (c,lt,a,b),pid in zip(d.SP,ids):
    E=(a+b*(np.arange(len(c))+1))/1000.     # same channel convention as dose_r2 [1]
    w=d.Gpoly(E)*d.hk_lin(E)
    v=(E>0.05)&(E<3.1); hi=(E>=2.75)&(E<3.1)
    nhi=c[hi].sum(); dens=nhi/hi.sum()          # counts per channel, cosmic-dominated
    H=np.sum(c[v]*w[v])/lt*R
    H_hi=np.sum(c[hi]*w[hi])/lt*R
    H_ub=np.sum(dens*w[v])/lt*R                  # flat extrapolation over whole window
    HG.append(H);Hhi.append(H_hi);Hub.append(H_ub)
    P("%-6s %8.0f %9.4f %9.0f %8.4f %9.5f %9.5f"%(pid,nhi,nhi/lt,c[-1],H,H_hi,H_ub))
HG=np.array(HG);Hhi=np.array(Hhi);Hub=np.array(Hub)
P("\nmean G(E) dose %.4f uSv/h ; dose from 2.75-3.1 MeV channels %.5f uSv/h (%.1f %% of G(E) dose)"%(HG.mean(),Hhi.mean(),100*Hhi.mean()/HG.mean()))
P("upper bound of cosmic dose already inside the window (flat extrapolation): %.4f uSv/h = %.1f %% of the cosmic estimate"%(Hub.mean(),100*Hub.mean()/CEST))
# sensitivity: continuum rising toward low energy as (2.9 MeV / E)^g
for g in (0.5,1.0):
    Hs=[]
    for (c,lt,a,b) in d.SP:
        E=(a+b*(np.arange(len(c))+1))/1000.; w=d.Gpoly(E)*d.hk_lin(E)
        v=(E>0.05)&(E<3.1); hi=(E>=2.75)&(E<3.1); dens=c[hi].sum()/hi.sum()
        Hs.append(np.sum(dens*(2.9/E[v])**g*w[v])/lt*R)
    P("  sensitivity: continuum ~E^-%.1f below 2.9 MeV -> in-window cosmic dose %.4f uSv/h (%.1f %% of estimate)"%(g,np.mean(Hs),100*np.mean(Hs)/CEST))
# effect on corrected comparison: add only the part of the cosmic component not already in the window
F=d.FH
for lab,inwin in [("no overlap",0.0),("upper-bound overlap",Hub.mean())]:
    C=CEST-inwin; r=(HG+C)/F
    P("  %-20s added %.4f uSv/h -> ratio %.3f +/- %.3f, remaining difference %.1f %%"%(lab,C,r.mean(),r.std(ddof=1),100*(1-r.mean())))
open(os.path.join(LOGDIR,"cosmic_in_window.log"),"w").write("\n".join(out)+"\n")
