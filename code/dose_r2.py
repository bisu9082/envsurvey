"""R2 (ARI-D-26-00647R1) reproducible dose analysis + uncertainty-term bases + bias-structure tests.
Reproduces the R1 numbers (G(E) x ICRP74 H*(10)/Ka, 50 keV-3 MeV, gross spectrum) and adds
the quantities requested by Reviewer 1 (comments 1, 4, 5).  Usage: python3 dose_r2.py
"""
import numpy as np, os, re, glob, datetime
from scipy.interpolate import interp1d
from scipy import stats
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.join(HERE,"..","..")
_repo=os.path.isdir(os.path.join(HERE,"..","data","spe"))
SPE=os.path.join(HERE,"..","data","spe") if _repo else os.path.join(ROOT,"'26 환경탐사_집중훈련_spe_국화사")
MCNP=os.path.join(HERE,"..","data","mcnp_3inch") if _repo else os.path.join(ROOT,"'25 G-factor 실습 set","2. G-factor 평가 실습","3inch nai response function")
LOGDIR=os.path.join(HERE,"..","logs") if _repo else HERE
out=[]
def P(*a):
    s=" ".join(str(x) for x in a); print(s); out.append(s)
P("# dose_r2.py run", datetime.datetime.now().isoformat(timespec='seconds'))
# ---------------- inputs ----------------
G_E=np.array([0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0])
G_V=np.array([3.8397e-3,3.9544e-3,9.1529e-3,1.5974e-2,2.4581e-2,3.4559e-2,4.5207e-2,5.5782e-2,6.6227e-2,
              8.5199e-2,1.0112e-1,1.1675e-1,1.3164e-1,1.4537e-1,1.5601e-1,1.6961e-1,1.8091e-1,1.8960e-1,1.9379e-1,2.2778e-1])
P6=np.polyfit(G_E,np.log(G_V),6)
Gpoly=lambda E: np.exp(np.polyval(P6,np.clip(E,0.05,3.0)))
_gl=interp1d(np.log(G_E),np.log(G_V)); Gloglog=lambda E: np.exp(_gl(np.log(np.clip(E,0.05,3.0))))
R=0.00877  # uGy per uR (air)
E_hk=np.array([0.010,0.015,0.020,0.030,0.040,0.050,0.060,0.080,0.100,0.150,0.200,0.300,0.400,0.500,0.600,0.800,
               1.000,1.500,2.000,3.000,4.000,5.000,6.000,8.000,10.000])
V_hk=np.array([0.008,0.26,0.61,1.10,1.47,1.67,1.74,1.72,1.65,1.49,1.40,1.31,1.26,1.23,1.21,1.19,1.17,1.15,1.14,
               1.13,1.12,1.11,1.11,1.11,1.10])   # ICRP 74 Table A.21, H*(10)/Ka in Sv/Gy
_hk=interp1d(E_hk,V_hk); hk_lin=lambda E:_hk(np.clip(E,0.05,3.0))
_hl=interp1d(np.log(E_hk),np.log(V_hk)); hk_log=lambda E: np.exp(_hl(np.log(np.clip(E,0.05,3.0))))
ids=["P3","P3N5m","P3E5m","P3W5m","P3S5m","P1","P1E5m","P1N5m","P1W5m","P1S5m"]
files=["3원점","3북쪽","3동쪽","3서쪽","3남쪽","1원점","1동쪽","1북쪽","1서쪽","1남쪽"]
FH=np.array([0.0932,0.0911,0.0973,0.0985,0.0823,0.0890,0.1190,0.0911,0.1040,0.1020])  # field log, uSv/h
def parse(p):
    L=open(p,'r',errors='replace').read().split('\n'); c=[];a=b=None;lt=None;i=0
    while i<len(L):
        s=L[i].strip()
        if s=='$ENER_FIT:': a,b=map(float,L[i+1].split()[:2])
        elif s=='$MEAS_TIM:': lt=float(L[i+1].split()[0])
        elif s=='$DATA:':
            i+=2
            while i<len(L):
                l=L[i].strip()
                if l.startswith('$'): break
                if l: c.append(int(l))
                i+=1
            continue
        i+=1
    return np.array(c,float),lt,a,b
SP=[parse(os.path.join(SPE,f+".Spe")) for f in files]
def dose(c,lt,Ekev,G=Gpoly,hk=hk_lin,lo=0.05,hi=3.1):
    E=Ekev/1000.; v=(E>lo)&(E<hi)
    w=G(E[v])*hk(E[v]); H=np.sum(c[v]*w)/lt*R
    keff=np.sum(c[v]*w)/np.sum(c[v]*G(E[v])); urel=np.sqrt(np.sum(c[v]*w**2))/np.sum(c[v]*w)
    return H,keff,urel
def run(chan_offset=1,G=Gpoly,hk=hk_lin,Emap=None):
    H=[];K=[];U=[]
    for c,lt,a,b in SP:
        Erec=a+b*(np.arange(len(c))+chan_offset)
        Ek=Emap(Erec) if Emap else Erec
        h,k,u=dose(c,lt,Ek,G,hk); H.append(h);K.append(k);U.append(u)
    return np.array(H),np.array(K),np.array(U)
def summ(H): r=H/FH; return r.mean(),r.std(ddof=1)
# ---------------- [1] reproduce R1 ----------------
H,K,U=run(1)
r=H/FH; d=H-FH; W,pW=stats.wilcoxon(H,FH); dz=d.mean()/d.std(ddof=1); tc=stats.t.ppf(0.975,9); se=d.std(ddof=1)/np.sqrt(10)
rp,pp=stats.pearsonr(H,FH)
P("\n[1] R1 analysis reproduced (channel index as in R1 code, 1-based)")
P("  %-6s %8s %8s %6s %6s"%("point","H_G(E)","FH40G","ratio","keff"))
for i in range(10): P("  %-6s %8.4f %8.4f %6.3f %6.4f"%(ids[i],H[i],FH[i],r[i],K[i]))
P("  ratio %.3f +/- %.3f ; difference %.1f %% ; 1/ratio %.2f"%(r.mean(),r.std(ddof=1),100*(1-r.mean()),1/r.mean()))
P("  Wilcoxon W=%.0f p=%.4f ; dz=%.2f ; bias %.4f (95%% CI %.4f to %.4f) ; LoA %.4f to %.4f"%(W,pW,dz,d.mean(),d.mean()-tc*se,d.mean()+tc*se,d.mean()-1.96*d.std(ddof=1),d.mean()+1.96*d.std(ddof=1)))
P("  Pearson r=%.3f p=%.2f ; keff %.4f +/- %.4f"%(rp,pp,K.mean(),K.std(ddof=1)))
# ---------------- [2] R1-1: k(E) meaning ----------------
P("\n[2] k(E)=H*(10)/Ka (Sv/Gy, ICRP 74) at selected channel energies (linear interpolation)")
for e in [0.05,0.1,0.2,0.3,0.6,0.662,1.0,1.461,2.0,2.614,3.0]: P("  %.3f MeV  k=%.3f"%(e,hk_lin(np.array([e]))[0]))
c,lt,a,b=SP[0]
P("  channel->energy: E_i = %.4f + %.6f*i keV ; channel width %.2f keV ; channels in 50-3000 keV: %d"%(a,b,b,int(((a+b*(np.arange(1024)+1))>50).sum()-((a+b*(np.arange(1024)+1))>3100).sum())))
# ---------------- [3] R1-5: bias structure ----------------
m=(H+FH)/2
s1=stats.linregress(m,d); s2=stats.linregress(FH,H); s3=stats.linregress(FH,r)
P("\n[3] proportional-bias tests (n=10, FH40G span %.3f-%.3f, factor %.2f)"%(FH.min(),FH.max(),FH.max()/FH.min()))
P("  Bland-Altman d vs mean : slope %.3f (95%% CI %.3f to %.3f) p=%.2f"%(s1.slope,s1.slope-tc*1.0*s1.stderr*stats.t.ppf(0.975,8)/tc,s1.slope+s1.stderr*stats.t.ppf(0.975,8),s1.pvalue))
P("  OLS H_G(E) vs FH40G    : slope %.3f (95%% CI %.3f to %.3f) p=%.2f"%(s2.slope,s2.slope-s2.stderr*stats.t.ppf(0.975,8),s2.slope+s2.stderr*stats.t.ppf(0.975,8),s2.pvalue))
P("  ratio vs FH40G          : slope %.2f per uSv/h (95%% CI %.2f to %.2f) p=%.2f"%(s3.slope,s3.slope-s3.stderr*stats.t.ppf(0.975,8),s3.slope+s3.stderr*stats.t.ppf(0.975,8),s3.pvalue))
cv_d=d.std(ddof=1)/abs(d.mean()); cv_r=r.std(ddof=1)/r.mean()
P("  CV of differences %.1f %% ; CV of ratios %.1f %%"%(100*cv_d,100*cv_r))
# minimum detectable BA slope (80 % power, two-sided 0.05) ~ (t_.975+t_.8)*SE
P("  minimum detectable BA slope at 80%% power ~ +/- %.2f"%((stats.t.ppf(0.975,8)+stats.t.ppf(0.8,8))*s1.stderr))
# ---------------- [4] R1-4: uncertainty bases ----------------
P("\n[4] bases of the non-geometric uncertainty terms (relative, on the site-mean ratio)")
base=summ(H)[0]
# 4a counting
P("  (a) counting: per-spectrum u_rel(sum c*G*k) = %.2f-%.2f %% ; on 10-point mean %.2f %%"%(100*U.min(),100*U.max(),100*np.sqrt(np.sum((U*r)**2))/10/base))
# 4b G(E) fit
res=np.log(G_V)-np.polyval(P6,G_E)
Hl,_,_=run(1,G=Gloglog)
P("  (b) G(E) 6th-order ln-poly: rms residual at 20 nodes %.2f %% (max %.2f %%); poly vs log-log node interpolation changes mean ratio by %+.2f %%"%(100*np.sqrt(np.mean(res**2)),100*np.max(abs(res)),100*(summ(Hl)[0]/base-1)))
# 4c hk interpolation
Hh,Kh,_=run(1,hk=hk_log)
P("  (c) H*(10)/Ka linear vs log-log interpolation: keff %.4f vs %.4f ; mean ratio change %+.2f %%"%(K.mean(),Kh.mean(),100*(summ(Hh)[0]/base-1)))
# 4d energy scale: channel index convention and measured non-linearity
H0,_,_=run(0)
P("  (d) energy scale:")
P("      0-based vs 1-based channel index (one channel, %.2f keV): mean ratio change %+.2f %%"%(b,100*(summ(H0)[0]/base-1)))
# natural-line centroids (0-based recorded scale, from cs137_consistent_roi.log and sum spectrum)
tot=sum(s[0] for s in SP); Er0=a+b*np.arange(len(tot))
from scipy.optimize import curve_fit
def g1(x,A,mu,s,c0,c1): return A*np.exp(-0.5*((x-mu)/s)**2)+c0+c1*(x-mu)
obs=[]
for Et,lo_,hi_ in [(351.9,325,385),(609.3,600,645),(911.2,870,990),(1120.3,1090,1190),(1460.8,1420,1540),(2614.5,2560,2740)]:
    mm=(Er0>=lo_)&(Er0<=hi_); q,_=curve_fit(g1,Er0[mm],tot[mm],p0=[tot[mm].max()-tot[mm].min(),Er0[mm][np.argmax(tot[mm])],18,tot[mm].min(),0],maxfev=40000)
    obs.append((Et,q[1]))
obs=np.array(obs)
P("      natural lines (true -> recorded, 0-based): "+"; ".join("%.0f->%.1f (%+.1f %%)"%(t,o,100*(o/t-1)) for t,o in obs))
emap=interp1d(np.r_[0,obs[:,1],4000],np.r_[0,obs[:,0],4000*obs[-1,0]/obs[-1,1]],fill_value="extrapolate")
Hc,_,_=run(0,Emap=emap)
P("      recompute with energies re-mapped through the natural lines: mean ratio %.3f (change %+.2f %% vs R1)"%(summ(Hc)[0],100*(summ(Hc)[0]/base-1)))
# 4e MCNP statistics
P("  (e) MCNP tally statistics (3-inch response decks, F8 total):")
errs=[]
for f in sorted(glob.glob(os.path.join(MCNP,"NaI_3inch_*keV.out")),key=lambda s:int(re.findall(r"_(\d+)keV",s)[0])):
    t=open(f,errors='replace').read()
    nps=re.findall(r"run terminated when\s+(\d+)\s+particle",t); tot_=re.findall(r"\n\s+total\s+([\d.E+-]+)\s+([\d.]+)",t.split("1tally        8")[1]) if "1tally        8" in t else []
    ver=re.findall(r"Code Name & Version = (\S+)",t)
    errs.append(float(tot_[0][1]) if tot_ else np.nan)
    P("      %-22s %s nps=%s total=%s relerr=%s"%(os.path.basename(f),ver[0] if ver else "?",nps[0] if nps else "?",tot_[0][0] if tot_ else "?",tot_[0][1] if tot_ else "?"))
P("      max total relative error %.4f -> statistical part negligible; 7.5 %% term is a Type B modelling allowance"%np.nanmax(errs))
open(os.path.join(LOGDIR,"dose_r2.log"),"w").write("\n".join(out)+"\n")

# ---------------- [6] R1-1: channel-wise k(E) vs incident-energy k(E0), using the MCNP response matrix ----------------
P("\n[6] channel-wise application of k(E) checked against the MCNP 20-bin response matrix")
P("    for each incident energy E0: q(E0) = sum_j G_j k(E_j) R(E0,j) / [k(E0) sum_j G_j R(E0,j)]  (q=1 means exact)")
Rm=[]
for e0 in G_E:
    f=os.path.join(MCNP,"NaI_3inch_%dkeV.out"%round(e0*1000)); t=open(f,errors='replace').read().split("1tally        8")[1]
    rows=re.findall(r"\n\s+([\d.]+E[+-]\d+)\s+([\d.]+E[+-]\d+)\s+[\d.]+",t.split("total")[0])
    v=np.array([float(y) for x,y in rows]); Rm.append(v[2:22])   # drop 0 and 1e-3 bins -> 20 bins aligned with G_E
Rm=np.array(Rm); kE=hk_lin(G_E)
X=Rm@G_V
q=(Rm@(G_V*kE))/(kE*X)
for e0,qq in zip(G_E,q): P("    E0 %.2f MeV  k(E0) %.3f  q %.3f"%(e0,hk_lin(np.array([e0]))[0],qq))
P("    q range over 0.2-3.0 MeV: %.3f-%.3f ; at 0.6 MeV %.3f ; at 1.4/1.6 MeV %.3f/%.3f ; at 2.6 MeV %.3f"%(q[2:].min(),q[2:].max(),q[6],q[11],q[12],q[17]))
open(os.path.join(LOGDIR,"dose_r2.log"),"w").write("\n".join(out)+"\n")

# 4f combined
terms={"FH40G (+/-15 %, k=2)":7.5,"MCNP modelling (Type B)":7.5}
P("\n[7] combined budget; geometry-independent terms from [4] (rounded up)")
u_cnt=max(0.3,100*U.max()); u_fit=max(1.0,abs(100*(summ(Hl)[0]/base-1))); u_hk=max(1.0,abs(100*(summ(Hh)[0]/base-1)),100*(q[1:].max()-1)); u_E=max(1.5,abs(100*(summ(Hc)[0]/base-1)))
for name,u in [("FH40G",7.5),("MCNP modelling",7.5),("energy scale",u_E),("G(E) fit",u_fit),("H*/Ka applic.",u_hk),("counting",u_cnt)]: P("  %-16s %.2f %%"%(name,u))
for mc in [7.5,15.0]:
    uc=np.sqrt(7.5**2+mc**2+u_E**2+u_fit**2+u_hk**2+u_cnt**2)
    P("  MCNP term %.1f %% -> combined %.1f %% (k=1), %.1f %% (k=2); observed difference %.1f %% = %.2fx expanded"%(mc,uc,2*uc,100*(1-base),100*(1-base)/(2*uc)))
P("  linear sum (MCNP 7.5 %%): %.1f %%"%(7.5+7.5+u_E+u_fit+u_hk+u_cnt))
open(os.path.join(LOGDIR,"dose_r2.log"),"w").write("\n".join(out)+"\n")

# ---------------- [8] cluster structure (R1-5 limitation; Fig. 4b label) ----------------
r3=r[:5]; r1=r[5:]
U_,pU=stats.mannwhitneyu(r3,r1,alternative='two-sided',method='exact')
P("\n[8] ratio by cluster: P3 %.3f +/- %.3f (n=5) ; P1 %.3f +/- %.3f (n=5) ; Mann-Whitney U=%.0f p=%.3f (exact, two-sided)"%(r3.mean(),r3.std(ddof=1),r1.mean(),r1.std(ddof=1),U_,pU))
P("    Wilcoxon minimum attainable two-sided p for n=10: %.4f"%(2/2**10))
open(os.path.join(LOGDIR,"dose_r2.log"),"w").write("\n".join(out)+"\n")

# ---------------- [9] SI Table S5: without H*(10)/Ka (k = 1), 10 keV-3.1 MeV as in the original submission ----------------
Hu=[]
for c,lt,a,b in SP:
    E=(a+b*(np.arange(len(c))+1))/1000.; v=(E>0.01)&(E<3.1)
    Hu.append(np.sum(c[v]*Gpoly(E[v]))/lt*R)
Hu=np.array(Hu); ru=Hu/FH; du=Hu-FH
P("\n[9] without H*(10)/Ka (k=1, 10 keV lower limit, original submission)")
for i in range(10): P("  %-6s %.4f  ratio %.3f"%(ids[i],Hu[i],ru[i]))
P("  ratio %.3f +/- %.3f ; difference %.1f %% ; dz %.2f ; Wilcoxon p %.4f ; corrected/uncorrected dose %.3f"%(ru.mean(),ru.std(ddof=1),100*(1-ru.mean()),du.mean()/du.std(ddof=1),stats.wilcoxon(Hu,FH).pvalue,(H/Hu).mean()))
open(os.path.join(LOGDIR,"dose_r2.log"),"w").write("\n".join(out)+"\n")

# ---------------- [10] values quoted in the text ----------------
P("\n[10] supporting values")
fr=[];f30=[]
for c,lt,a,b in SP:
    E=(a+b*(np.arange(len(c))+1))/1000.; w=Gpoly(E)*hk_lin(E); allv=(E>0)&(E<3.1)
    fr.append(np.sum((c*w)[(E>0)&(E<=0.05)])/np.sum((c*w)[allv]))
    f30.append(np.sum((c*w)[(E>3.0)&(E<3.1)])/np.sum((c*w)[(E>0.05)&(E<3.1)]))
P("  share of channels below 50 keV in a 0-3.1 MeV sum: %.1f-%.1f %%"%(100*min(fr),100*max(fr)))
P("  share of the 3.0-3.1 MeV channels in the 50 keV-3.1 MeV sum: %.2f-%.2f %%"%(100*min(f30),100*max(f30)))
f28=[np.sum((c*Gpoly((a+b*(np.arange(len(c))+1))/1000.)*hk_lin((a+b*(np.arange(len(c))+1))/1000.))[((a+b*(np.arange(len(c))+1))/1000.>2.9)&((a+b*(np.arange(len(c))+1))/1000.<3.1)])/np.sum((c*Gpoly((a+b*(np.arange(len(c))+1))/1000.)*hk_lin((a+b*(np.arange(len(c))+1))/1000.))[((a+b*(np.arange(len(c))+1))/1000.>0.05)&((a+b*(np.arange(len(c))+1))/1000.<3.1)]) for c,lt,a,b in SP]
P("  share of the 2.9-3.1 MeV channels: %.2f-%.2f %%"%(100*min(f28),100*max(f28)))
from scipy.stats import nct, t as tdist
def pw(dz,n=10,al=0.05):
    df=n-1; tc=tdist.ppf(1-al/2,df); nc=dz*np.sqrt(n); return 1-nct.cdf(tc,df,nc)+nct.cdf(-tc,df,nc)
for tg in (0.8,0.9):
    dzm=next(x for x in np.arange(0.5,2,0.001) if pw(x)>=tg); P("  minimum detectable paired dz (n=10, alpha 0.05 two-sided) at %d %% power: %.2f"%(100*tg,dzm))
a_,b_,c_=-0.01392,0.06968,-0.05862; e=0.6617; fw=a_+b_*np.sqrt(e+c_*e*e); P("  GEB FWHM at 661.7 keV: %.1f keV = %.2f %%"%(1000*fw,100*fw/e))
open(os.path.join(LOGDIR,"dose_r2.log"),"w").write("\n".join(out)+"\n")
