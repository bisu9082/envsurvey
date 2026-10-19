"""R2: cosmic-ray ionizing component at the site and its effect on the G(E)/FH40G comparison.
Sources (verified 2026-10-19):
  UNSCEAR 2000 Annex B, para 20: sea-level absorbed dose rate in free air from directly ionizing + photon
    component 32 nGy/h; ~10 % geomagnetic latitude effect -> 30 nGy/h below 30 deg; w_R = 1.
  UNSCEAR 2000 Annex A, Eq. (12) (Bouville & Lowder): E_I(z) = E_I(0) [0.21 exp(-1.649 z) + 0.79 exp(0.4528 z)], z in km.
  UNSCEAR 2000 Annex B, para 21 / Annex A para 48: neutron component ~9 nSv/h at sea level (50 deg N) - photon
    survey meters are not designed to respond to it; listed only.
Site elevation: SRTM 30 m 691/671 m, ASTER 30 m 679/664 m at P3/P1 (OpenTopoData API, queried 2026-10-19).
"""
import numpy as np, os, datetime, sys
HERE=os.path.dirname(os.path.abspath(__file__)); out=[]
LOGDIR=os.path.join(HERE,"..","logs") if os.path.isdir(os.path.join(HERE,"..","logs")) else HERE
def P(*a):
    s=" ".join(str(x) for x in a); print(s); out.append(s)
P("# cosmic_r2.py run", datetime.datetime.now().isoformat(timespec='seconds'))
f=lambda z: 0.21*np.exp(-1.649*z)+0.79*np.exp(0.4528*z)
elev={"P3":(0.691,0.679),"P1":(0.671,0.664)}
P("[1] altitude factor (UNSCEAR 2000 Annex A Eq. 12)")
for k,(a,b) in elev.items(): P("  %s  z=%.3f-%.3f km  factor %.3f-%.3f"%(k,b,a,f(b),f(a)))
zlo,zhi=0.664,0.691
lo=30*f(zlo); hi=32*f(zhi)
P("  site cosmic ionizing + photon component: %.1f-%.1f nGy/h (sea level 30-32 nGy/h); taken as ~nSv/h H*(10)"%(lo,hi))
import importlib.util, io, contextlib
_c=[os.path.join(HERE,"..","r2_dose","dose_r2.py"), os.path.join(HERE,"dose_r2.py")]
_sp=importlib.util.spec_from_file_location("d", next(p for p in _c if os.path.exists(p)))
with contextlib.redirect_stdout(io.StringIO()):
    _d=importlib.util.module_from_spec(_sp); _sp.loader.exec_module(_d)
H=_d.run(1)[0]   # G(E) dose rates exactly as in dose_r2.log [1]
F=_d.FH
d=H-F
P("\n[2] G(E) excludes depositions >3 MeV; FH40G responds to the ionizing component.")
P("  observed mean difference %.4f uSv/h ; cosmic estimate = %.0f-%.0f %% of it"%(d.mean(),100*lo/1000/abs(d.mean()),100*hi/1000/abs(d.mean())))
P("  fraction f of the cosmic component registered by FH40G but not by G(E):")
for c in (lo,(lo+hi)/2,hi):
    for fr in (0.5,1.0):
        r=(H+fr*c/1000)/F
        P("   cosmic %.1f nSv/h x f=%.1f -> ratio %.3f +/- %.3f, remaining difference %.1f %%"%(c,fr,r.mean(),r.std(ddof=1),100*(1-r.mean())))
r=(H+hi/1000)/F; P("  terrestrial-only comparison (FH40G minus %.1f nSv/h): ratio %.3f"%(hi,(H/(F-hi/1000)).mean()))
P("  terrestrial-only comparison (FH40G minus %.1f nSv/h): ratio %.3f"%(lo,(H/(F-lo/1000)).mean()))

# ---------------- [3] corrected comparison (cosmic component added to G(E), f = 1) ----------------
from scipy import stats
C=(lo+hi)/2/1000; uC=np.sqrt(((hi-lo)/2/1000/np.sqrt(3))**2+(0.10*C)**2)   # range half-width (rect.) + 10 % solar-cycle/latitude (UNSCEAR)
Hc=H+C; rc=Hc/F; dc=Hc-F
W,pW=stats.wilcoxon(Hc,F); tc=stats.t.ppf(0.975,9); se=dc.std(ddof=1)/np.sqrt(10)
P("\n[3] G(E) + cosmic (%.1f +/- %.1f nSv/h, k=1) vs FH40G, assuming FH40G registers the cosmic component fully"%(C*1000,uC*1000))
P("  %-6s %7s %7s %6s"%("point","H+C","FH40G","ratio"))
ids=["P3","P3N5m","P3E5m","P3W5m","P3S5m","P1","P1E5m","P1N5m","P1W5m","P1S5m"]
for i in range(10): P("  %-6s %7.4f %7.4f %6.3f"%(ids[i],Hc[i],F[i],rc[i]))
P("  ratio %.3f +/- %.3f (range %.3f-%.3f) ; difference %.1f %%"%(rc.mean(),rc.std(ddof=1),rc.min(),rc.max(),100*(1-rc.mean())))
P("  points with ratio < 1: %d/10 ; Wilcoxon W=%.0f p=%.3f ; mean diff %.4f (95%% CI %.4f to %.4f)"%((rc<1).sum(),W,pW,dc.mean(),dc.mean()-tc*se,dc.mean()+tc*se))
u_ratio_C=100*uC/F.mean()/rc.mean()
u_prev=11.0   # dose_r2.log [7], combined k=1 with MCNP 7.5 %
uc=np.sqrt(u_prev**2+u_ratio_C**2)
P("  cosmic-estimate term on ratio %.1f %% ; combined with Table 2 (11.0 %%): %.1f %% (k=1), %.1f %% (k=2)"%(u_ratio_C,uc,2*uc))
P("  residual difference %.1f %% vs expanded %.1f %% -> %s"%(100*(1-rc.mean()),2*uc,"within" if 100*(1-rc.mean())<2*uc else "outside"))
uc15=np.sqrt(7.5**2+15**2+1.74**2+1.0**2+1.97**2+0.37**2+u_ratio_C**2)
P("  MCNP term doubled to 15 %%: combined %.1f %% (k=1), %.1f %% (k=2)"%(uc15,2*uc15))
P("  1/ratio range (FH40G over-reading needed to explain each point alone): %.2f-%.2f"%((1/rc).min(),(1/rc).max()))
P("  if FH40G cosmic response f=0.5: residual %.1f %% (%s)"%(100*(1-((H+0.5*C)/F).mean()),"within" if 100*(1-((H+0.5*C)/F).mean())<2*uc else "outside"))
fmin=None
for fr in np.arange(0,1.001,0.01):
    if 100*(1-((H+fr*C)/F).mean())<2*uc: fmin=fr; break
P("  smallest FH40G cosmic response for which the residual falls within the expanded uncertainty: f = %.2f"%fmin)
r3=rc[:5]; r1=rc[5:]; U_,pU=stats.mannwhitneyu(r3,r1,alternative='two-sided',method='exact')
P("  by cluster: P3 %.3f +/- %.3f ; P1 %.3f +/- %.3f ; Mann-Whitney U=%.0f p=%.3f"%(r3.mean(),r3.std(ddof=1),r1.mean(),r1.std(ddof=1),U_,pU))
s1=stats.linregress((Hc+F)/2,dc); P("  Bland-Altman slope %.2f (95%% CI %.2f to %.2f) p=%.2f"%(s1.slope,s1.slope-stats.t.ppf(0.975,8)*s1.stderr,s1.slope+stats.t.ppf(0.975,8)*s1.stderr,s1.pvalue))

open(os.path.join(LOGDIR,"cosmic_r2.log"),"w").write("\n".join(out)+"\n")
