"""R2 (ARI-D-26-00647R1) Cs-137 re-analysis with ONE analysis definition applied to all ten SPE files.
Reason: field-recorded MAESTRO ROIs were set manually per point (widths differ; gross 1516-8704),
so the recorded net counts (incl. P3E5m 1960) are not comparable across points.
Usage: python3 cs137_consistent_roi.py   -> writes cs137_consistent_roi.log / .csv to ../logs (repo) or next to this file
"""
import numpy as np, os, sys, datetime, hashlib
from scipy.optimize import curve_fit
HERE=os.path.dirname(os.path.abspath(__file__))
SPE=next(p for p in [os.path.join(HERE,"..","data","spe"),
                      os.path.join(HERE,"..","..","'26 환경탐사_집중훈련_spe_국화사")] if os.path.isdir(p))
LOGDIR=os.path.join(HERE,"..","logs") if os.path.isdir(os.path.join(HERE,"..","logs")) else HERE
ID=[("3원점","P3"),("3북쪽","P3N5m"),("3동쪽","P3E5m"),("3서쪽","P3W5m"),("3남쪽","P3S5m"),
    ("1원점","P1"),("1동쪽","P1E5m"),("1북쪽","P1N5m"),("1서쪽","P1W5m"),("1남쪽","P1S5m")]
REC={"P3":245,"P3N5m":183,"P3E5m":1960,"P3W5m":35,"P3S5m":29,"P1":44,"P1E5m":51,"P1N5m":72,"P1W5m":120,"P1S5m":67}
REC_GROSS={"P3":6713,"P3N5m":2555,"P3E5m":8704,"P3W5m":1518,"P3S5m":1516,"P1":2269,"P1E5m":2525,"P1N5m":2029,"P1W5m":2542,"P1S5m":2346}
out=[]
def P(*a):
    s=" ".join(str(x) for x in a); print(s); out.append(s)
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
S={};LT={};cal=set()
P("# Cs-137 consistent-ROI re-analysis  run:",datetime.datetime.now().isoformat(timespec='seconds'))
for f,pid in ID:
    p=os.path.join(SPE,f+".Spe"); c,lt,a,b=parse(p); S[pid]=c; LT[pid]=lt; cal.add((a,b))
    P("  %-6s %-10s live=%.0f s  total=%d  sha1=%s"%(pid,f+".Spe",lt,c.sum(),hashlib.sha1(open(p,'rb').read()).hexdigest()[:12]))
assert len(cal)==1, cal
a,b=cal.pop(); ch=np.arange(len(S["P3"]))   # MAESTRO channel index 0-based
E=a+b*ch
P("  calibration from headers: E = %.4f + %.6f*ch (identical in all files)"%(a,b))

# ---- step 0: locate the 583/609/662 keV peaks on the recorded scale from the SUM of 10 spectra ----
# (the 2-parameter calibration is not exact across the full range; positions are taken from the data)
tot=sum(S.values())
def m3(x,A1,A2,A3,E1,E2,E3,s0,c0,c1):
    y=c0+c1*(x-640)
    for Ak,Ek in [(A1,E1),(A2,E2),(A3,E3)]:
        s=s0*np.sqrt(Ek/662.); y=y+Ak*np.exp(-0.5*((x-Ek)/s)**2)
    return y
m=(E>=540)&(E<=780); x=E[m]; y=tot[m]
p0,c0v=curve_fit(m3,x,y,p0=[2000,3000,500,595,622,675,18,y.min(),0],sigma=np.sqrt(y),maxfev=100000)
TL,BI,CS=p0[3],p0[4],p0[5]; FW=2.355*p0[6]*np.sqrt(CS/662.)
P("\n[0] sum-spectrum 3-peak fit (540-780 keV, centroids free):")
P("  Tl-208 583.2 -> %.1f +/- %.1f keV; Bi-214 609.3 -> %.1f +/- %.1f keV; Cs-137 661.7 -> %.1f +/- %.1f keV; FWHM(Cs) %.1f keV"
  %(TL,c0v[3,3]**.5,BI,c0v[4,4]**.5,CS,c0v[5,5]**.5,FW))
def g1(x,A,mu,s,c0,c1): return A*np.exp(-0.5*((x-mu)/s)**2)+c0+c1*(x-mu)
P("  other natural lines (single-peak fits, sum spectrum):")
for nm,Et,lo,hi in [("Pb-214",351.9,325,385),("Ac-228",911.2,870,990),("Bi-214",1120.3,1090,1190),("K-40",1460.8,1420,1540)]:
    mm=(E>=lo)&(E<=hi); q,_=curve_fit(g1,E[mm],tot[mm],p0=[tot[mm].max()-tot[mm].min(),E[mm][np.argmax(tot[mm])],18,tot[mm].min(),0],maxfev=40000)
    P("    %-7s %7.1f -> %7.1f keV"%(nm,Et,q[1]))

# ---- method A: fixed window, same for all points ----
P("\n[A] fixed ROI centred at %.1f keV, width = 1 FWHM of the sum-spectrum Cs peak (%.1f keV), linear background from 3 channels each side"%(CS,FW))
lo=int(np.argmin(abs(E-(CS-FW/2)))); hi=int(np.argmin(abs(E-(CS+FW/2)))); w=hi-lo+1
P("  channels %d-%d (%.1f-%.1f keV), n=%d"%(lo,hi,E[lo],E[hi],w))
A={}
for _,pid in ID:
    c=S[pid]; g=c[lo:hi+1].sum(); bl=c[lo-3:lo].sum(); br=c[hi+1:hi+4].sum()
    bk=(bl+br)/6.0*w; var=g+(w/6.0)**2*(bl+br); n=g-bk
    LC=2.33*np.sqrt(bk*(1+w/6.0)); LD=2.71+4.65*np.sqrt(bk*(1+w/6.0))   # Currie, paired background
    A[pid]=(g,bk,n,np.sqrt(var),LC,LD)
P("  %-6s %7s %7s %7s %6s %6s %6s  %s"%("point","gross","bkg","net","1sig","L_C","L_D","net>L_C"))
for _,pid in ID:
    g,bk,n,s,LC,LD=A[pid]; P("  %-6s %7.0f %7.0f %7.0f %6.0f %6.0f %6.0f  %s"%(pid,g,bk,n,s,LC,LD,"yes" if n>LC else "no"))

# ---- method B: simultaneous 3-peak fit (Tl-208, Bi-214, Cs-137) ----
P("\n[B] simultaneous Gaussian fit 540-780 keV: Tl-208, Bi-214, Cs-137 at mapped positions,"
  "\n    common shift dE, common width scaled as sqrt(E), linear continuum; Poisson weights")
def model(x,A1,A2,A3,dE,s0,c0,c1):
    y=c0+c1*(x-640)
    for Ak,Ek in [(A1,TL),(A2,BI),(A3,CS)]:
        s=s0*np.sqrt(Ek/CS); y=y+Ak*np.exp(-0.5*((x-(Ek+dE))/s)**2)
    return y
B={};BIA={}
m=(E>=540)&(E<=780); x=E[m]
for _,pid in ID:
    y=S[pid][m]
    p,cov=curve_fit(model,x,y,p0=[150,200,20,0,FW/2.355,y.min(),0],
        bounds=([0,0,0,-8,10,-1e4,-1e3],[5e3,5e3,5e3,8,30,1e5,1e3]),sigma=np.sqrt(np.maximum(y,1)),maxfev=80000)
    s3=p[4]; area=p[2]*s3*np.sqrt(2*np.pi)/b; dA=np.sqrt(cov[2,2])*s3*np.sqrt(2*np.pi)/b
    chi=np.sum(((y-model(x,*p))/np.sqrt(np.maximum(y,1)))**2)/(len(x)-7)
    B[pid]=(area,dA,p[3],2.355*s3,chi)
    BIA[pid]=p[1]*s3*np.sqrt(BI/CS)*np.sqrt(2*np.pi)/b
P("  %-6s %8s %6s %6s %7s %6s"%("point","Cs area","1sig","dE","FWHM","chi2/v"))
for _,pid in ID:
    ar,dA,dE,fw,chi=B[pid]; P("  %-6s %8.0f %6.0f %+6.1f %7.1f %6.2f"%(pid,ar,dA,dE,fw,chi))
P("  check: Bi-214 665.4 keV (1.53 %) under the Cs peak, scaled from fitted 609 keV area (45.5 %), eff ratio ~609/665:")
for _,pid in ID:
    x665=BIA[pid]*0.0153/0.455*(609.3/665.4); P("  %-6s Bi609 %5.0f -> Bi665 %3.0f (%.0f %% of 662-keV component)"%(pid,BIA[pid],x665,100*x665/B[pid][0]))

# ---- method C: model-free count fraction ----
P("\n[C] counts in %.0f-%.0f keV divided by total counts (x1e4)"%(CS-20,CS+20))
Cf={}
for _,pid in ID:
    c=S[pid]; mm=(E>=CS-20)&(E<=CS+20); Cf[pid]=1e4*c[mm].sum()/c.sum(); P("  %-6s %.2f"%(pid,Cf[pid]))

# ---- summary: is P3E5m distinct? ----
P("\n[summary] P3E5m vs the other nine points")
for lab,v in [("A net",{k:A[k][2] for k in A}),("B area",{k:B[k][0] for k in B}),("C frac",Cf)]:
    o=np.array([v[k] for _,k in ID if k!="P3E5m"]); r=1+sum(v[k]>v["P3E5m"] for _,k in ID)
    P("  %-7s P3E5m %8.2f | others mean %8.2f SD %7.2f | z %+5.2f | rank %d/10"%(lab,v["P3E5m"],o.mean(),o.std(ddof=1),(v["P3E5m"]-o.mean())/o.std(ddof=1),r))
for lab,v in [("A net",{k:A[k][2] for k in A}),("B area",{k:B[k][0] for k in B})]:
    allv=np.array([v[k] for _,k in ID]); P("  %-7s all ten: mean %.0f SD %.0f range %.0f-%.0f"%(lab,allv.mean(),allv.std(ddof=1),allv.min(),allv.max()))
P("\n[field record] manual MAESTRO ROI gross/net (not comparable between points):")
for _,pid in ID: P("  %-6s gross %5d net %5d"%(pid,REC_GROSS[pid],REC[pid]))
P("  P3E5m recorded gross %d vs fixed-ROI gross %.0f -> recorded ROI was %.1fx wider in counts"%(REC_GROSS["P3E5m"],A["P3E5m"][0],REC_GROSS["P3E5m"]/A["P3E5m"][0]))
with open(os.path.join(LOGDIR,"cs137_consistent_roi.csv"),"w") as f:
    f.write("point,live_s,A_gross,A_bkg,A_net,A_sig,A_LC,A_LD,B_area,B_sig,B_chi2nu,C_frac_x1e4,rec_gross,rec_net\n")
    for _,pid in ID:
        g,bk,n,s,LC,LD=A[pid]; ar,dA,_,_,chi=B[pid]
        f.write("%s,%.0f,%.0f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.3f,%.3f,%d,%d\n"%(pid,LT[pid],g,bk,n,s,LC,LD,ar,dA,chi,Cf[pid],REC_GROSS[pid],REC[pid]))
open(os.path.join(LOGDIR,"cs137_consistent_roi.log"),"w").write("\n".join(out)+"\n")
