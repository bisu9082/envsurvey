"""R2 figures: ARI_Fig2.png (manuscript Fig. 3) and ARI_Fig3.png (manuscript Fig. 4).
Reads the same inputs as dose_r2.py / cs137_consistent_roi.py / cosmic_r2.py.  Usage: python3 make_figures_r2.py [outdir]"""
import numpy as np, os, sys, io, contextlib, importlib.util, csv
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
HERE=os.path.dirname(os.path.abspath(__file__))
def load(name,cands):
    p=next(c for c in cands if os.path.exists(c)); sp=importlib.util.spec_from_file_location(name,p)
    with contextlib.redirect_stdout(io.StringIO()):
        m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    return m
d=load("d",[os.path.join(HERE,"..","r2_dose","dose_r2.py"),os.path.join(HERE,"dose_r2.py")])
csvp=next(c for c in [os.path.join(HERE,"..","r2_cs137","cs137_consistent_roi.csv"),os.path.join(HERE,"..","logs","cs137_consistent_roi.csv")] if os.path.exists(c))
OUT=sys.argv[1] if len(sys.argv)>1 else HERE
plt.rcParams.update({"font.size":10,"font.family":"DejaVu Sans"})
ids=d.ids; SP=d.SP
# ---------------- Fig. 3 (file ARI_Fig2.png) ----------------
fig,ax=plt.subplots(1,2,figsize=(13,5.2))
a,b=SP[0][2],SP[0][3]
E=a+b*np.arange(1024)       # keV, recorded scale (0-based, MAESTRO)
for k,col,lab in [(0,"#3b6ea5","P3"),(5,"#c0504d","P1")]:
    c,lt=SP[k][0],SP[k][1]
    ax[0].semilogy(E,np.maximum(c,0.5),color=col,lw=0.9,label="%s (%d s live)"%(lab,lt))
for nm,Ex in [("$^{214}$Pb 352",349.4),("$^{214}$Bi 609",614.9),("$^{137}$Cs 662",669.1),("$^{228}$Ac 911",927.5),("$^{40}$K 1461",1480.2),("$^{208}$Tl 2615",2648.4)]:   # recorded-scale positions (dose_r2.log [4d], cs137 log [0])
    ax[0].axvline(Ex,color="0.6",ls=":",lw=0.8); ax[0].text(Ex,2.5e5,nm,rotation=90,ha="right",va="top",fontsize=7.5,color="0.3")
ax[0].set_xlim(30,3100); ax[0].set_ylim(0.8,3e5); ax[0].set_xlabel("Energy (keV, recorded scale)"); ax[0].set_ylabel("Counts per channel")
ax[0].legend(loc="lower left",fontsize=8.5,frameon=True)
# inset: sum spectrum fit
tot=sum(s[0] for s in SP)
def m3(x,A1,A2,A3,E1,E2,E3,s0,c0,c1):
    y=c0+c1*(x-640); parts=[]
    for Ak,Ek in [(A1,E1),(A2,E2),(A3,E3)]:
        s=s0*np.sqrt(Ek/662.); parts.append(Ak*np.exp(-0.5*((x-Ek)/s)**2))
    return y+sum(parts),parts
f=lambda x,*p: m3(x,*p)[0]
m=(E>=540)&(E<=780); x=E[m]; y=tot[m]
p,_=curve_fit(f,x,y,p0=[2000,3000,500,595,622,675,18,y.min(),0],sigma=np.sqrt(y),maxfev=100000)
ins=ax[0].inset_axes([0.47,0.50,0.30,0.30])
ins.errorbar(x,y,yerr=np.sqrt(y),fmt=".",ms=3,color="k",lw=0.6,label="sum of 10 spectra")
xx=np.linspace(540,780,500); yy,parts=m3(xx,*p); base=p[7]+p[8]*(xx-640)
ins.plot(xx,yy,color="#444",lw=1.2,label="total fit"); ins.plot(xx,base,color="#444",lw=0.8,ls=":",label="continuum")
for pt,col,lab in zip(parts,["#8e7cc3","#e69138","#cc0000"],["$^{208}$Tl","$^{214}$Bi","$^{137}$Cs"]):
    ins.fill_between(xx,base,base+pt,color=col,alpha=0.35,label=lab)
ins.set_xlim(540,780); ins.tick_params(labelsize=7); ins.set_xlabel("keV",fontsize=7); ins.legend(fontsize=6.5,loc="upper left",bbox_to_anchor=(1.01,1.0),frameon=False)
ins.set_title("540–780 keV, three-component fit",fontsize=7.5)
# panel b: contribution at P3
c,lt=SP[0][0],SP[0][1]
Eb=(a+b*(np.arange(1024)+1))/1000.   # same convention as dose_r2 [1]
v=(Eb>0.05)&(Eb<3.1); w=d.Gpoly(Eb)*d.hk_lin(Eb); contrib=c*w*v
edges=np.arange(0.05,3.15,0.15); h=[contrib[(Eb>=lo)&(Eb<hi)].sum() for lo,hi in zip(edges[:-1],edges[1:])]
h=np.array(h)/contrib.sum()*100
ax[1].bar((edges[:-1]+edges[1:])/2,h,width=0.13,color="#9fc5e8",edgecolor="#3d85c6",label="contribution per 150 keV interval")
ax[1].plot(edges[1:],np.cumsum(h),"o-",color="#674ea7",ms=3.5,label="cumulative contribution")
ax[1].set_xlabel("Energy (MeV)"); ax[1].set_ylabel("Contribution to $G(E)$-derived $\\dot H^*(10)$ at P3 (%)"); ax[1].set_ylim(0,105); ax[1].set_xlim(0,3.1)
a2=ax[1].twinx(); Eg=np.linspace(0.05,3.0,300); a2.plot(Eg,d.Gpoly(Eg),color="#cc0000",lw=1.8,label="$G(E)$")
a2.plot(d.G_E,d.G_V,"s",color="#cc0000",ms=3.5,mfc="white")
a2.set_ylabel("$G(E)$ ($\\mu$R h$^{-1}$ per count s$^{-1}$)",color="#cc0000"); a2.tick_params(axis="y",colors="#cc0000"); a2.set_ylim(0,0.25)
h1,l1=ax[1].get_legend_handles_labels(); h2,l2=a2.get_legend_handles_labels(); ax[1].legend(h1+h2,l1+l2,loc="upper left",bbox_to_anchor=(0.42,0.45),fontsize=8.5)
for k,A in enumerate(ax): A.text(-0.09,1.02,"ab"[k],transform=A.transAxes,fontsize=15,fontweight="bold")
fig.tight_layout(); fig.savefig(os.path.join(OUT,"ARI_Fig2.png"),dpi=300); plt.close(fig)
# ---------------- Fig. 4 (file ARI_Fig3.png) ----------------
rows={r["point"]:r for r in csv.DictReader(open(csvp))}
order=["P3","P3N5m","P3S5m","P3W5m","P3E5m","P1","P1N5m","P1S5m","P1E5m","P1W5m"]
A=np.array([float(rows[k]["B_area"]) for k in order]); S=np.array([float(rows[k]["B_sig"]) for k in order])
fig,ax=plt.subplots(1,2,figsize=(13,5.0))
cols=["#c0504d"]*5+["#3b6ea5"]*5
ax[0].bar(range(10),A,yerr=S,color=cols,alpha=0.85,capsize=3)
ax[0].axhline(A.mean(),color="0.3",ls="--",lw=1); ax[0].text(9.4,A.mean()+15,"mean %.0f"%A.mean(),ha="right",fontsize=8.5,color="0.3")
ax[0].set_xticks(range(10)); ax[0].set_xticklabels(order,rotation=40,ha="right"); ax[0].set_ylabel("Fitted $^{137}$Cs photopeak area (counts per 1198 s)")
ax[0].set_ylim(0,1250); ax[0].set_title("Same three-component fit for every spectrum",fontsize=9.5)
H=d.run(1)[0]; F=d.FH
fz=lambda z: 0.21*np.exp(-1.649*z)+0.79*np.exp(0.4528*z); C=(30*fz(0.664)+32*fz(0.691))/2/1000
r=H/F; rc=(H+C)/F
rng=np.random.default_rng(1)
for g,(sl,col,name) in enumerate([(slice(0,5),"#c0504d","P3 cluster"),(slice(5,10),"#3b6ea5","P1 cluster")]):
    jit=rng.uniform(-0.07,0.07,5)
    ax[1].plot(g-0.15+jit,r[sl],"o",color=col,ms=7,label=name+" (uncorrected)")
    ax[1].plot(g+0.15+jit,rc[sl],"o",mfc="white",mec=col,ms=7,mew=1.5,label=name+" (+ cosmic)")
    ax[1].plot([g-0.27,g-0.03],[r[sl].mean()]*2,color=col,lw=2); ax[1].plot([g+0.03,g+0.27],[rc[sl].mean()]*2,color=col,lw=2,ls="--")
U1,p1=stats.mannwhitneyu(r[:5],r[5:],method="exact"); U2,p2=stats.mannwhitneyu(rc[:5],rc[5:],method="exact")
ax[1].plot([-0.15,0.85],[0.70,0.70],color="k",lw=0.8); ax[1].text(0.35,0.705,"Mann–Whitney U = %.0f, p = %.3f"%(U1,p1),ha="center",va="bottom",fontsize=8.5)
ax[1].plot([0.15,1.15],[1.10,1.10],color="k",lw=0.8); ax[1].text(0.65,1.105,"U = %.0f, p = %.2f"%(U2,p2),ha="center",va="bottom",fontsize=8.5)
ax[1].axhline(1.0,color="0.5",ls=":",lw=1)
ax[1].set_xticks([0,1]); ax[1].set_xticklabels(["P3 cluster (n = 5)","P1 cluster (n = 5)"]); ax[1].set_xlim(-0.6,1.6); ax[1].set_ylim(0.3,1.2)
ax[1].set_ylabel("$G(E)$/FH40G ratio"); ax[1].legend(fontsize=7.5,loc="lower right",ncol=2)
for k,Ax in enumerate(ax): Ax.text(-0.09,1.02,"ab"[k],transform=Ax.transAxes,fontsize=15,fontweight="bold")
fig.tight_layout(); fig.savefig(os.path.join(OUT,"ARI_Fig3.png"),dpi=300); plt.close(fig)
print("U1 %.0f p1 %.3f | U2 %.0f p2 %.3f | Cs mean %.0f | cosmic %.4f"%(U1,p1,U2,p2,A.mean(),C))

# ---------------- Fig. 2 (file ARI_Fig1.png): comparison + Bland-Altman ----------------
H=d.run(1)[0]; F=d.FH; ids=d.ids; r=H/F; dd=H-F; m=(H+F)/2
fig,ax=plt.subplots(1,2,figsize=(13,5.6))
isP3=np.array([k.startswith("P3") for k in ids])
lo,hi=0.03,0.125
ax[0].plot([lo,hi],[lo,hi],"--",color="0.6",lw=1.2,label="Line of equality")
sl=stats.linregress(F,H); xx=np.linspace(lo,hi,200); yy=sl.intercept+sl.slope*xx
n=len(F); t=stats.t.ppf(0.975,n-2); res=H-(sl.intercept+sl.slope*F); s_=np.sqrt(np.sum(res**2)/(n-2))
se=s_*np.sqrt(1/n+(xx-F.mean())**2/np.sum((F-F.mean())**2))
ax[0].fill_between(xx,yy-t*se,yy+t*se,color="#3b6ea5",alpha=0.12,lw=0)
ax[0].plot(xx,yy,color="#3b6ea5",lw=1.8,label="Regression (slope = %.2f)"%sl.slope)
ax[0].plot(F[isP3],H[isP3],"o",color="#c0504d",ms=8,mec="white",label="P3 cluster")
ax[0].plot(F[~isP3],H[~isP3],"o",color="#3b6ea5",ms=8,mec="white",label="P1 cluster")
offA={"P3":(-6,-14),"P3N5m":(-38,6),"P3E5m":(-14,-15),"P3W5m":(6,4),"P3S5m":(-8,-14),"P1":(-18,4),"P1N5m":(4,6),"P1E5m":(-10,8),"P1W5m":(4,6),"P1S5m":(-12,-14)}
for x,y,k,p3 in zip(F,H,ids,isP3): ax[0].annotate(k,(x,y),xytext=offA[k],textcoords="offset points",fontsize=8,color="#c0504d" if p3 else "#3b6ea5")
rp=stats.pearsonr(F,H)[0]; rmse=np.sqrt(np.mean(dd**2))
ax[0].text(0.40,0.97,"Pearson r = %.3f\nRMSE = %.4f $\\mu$Sv h$^{-1}$\nMean ratio = %.3f $\\pm$ %.3f"%(rp,rmse,r.mean(),r.std(ddof=1)),transform=ax[0].transAxes,va="top",fontsize=9.5,bbox=dict(boxstyle="round",fc="white",ec="0.8"))
ax[0].set_xlim(lo,hi); ax[0].set_ylim(lo,hi); ax[0].set_xlabel("FH40G dose rate ($\\mu$Sv h$^{-1}$)"); ax[0].set_ylabel("$G(E)$-derived dose rate ($\\mu$Sv h$^{-1}$)")
ax[0].legend(loc="upper left",fontsize=8.5)
bias=dd.mean(); sd=dd.std(ddof=1); tc=stats.t.ppf(0.975,n-1); ci=tc*sd/np.sqrt(n)
ax[1].axhspan(bias-1.96*sd,bias+1.96*sd,color="#f6b26b",alpha=0.10,lw=0)
ax[1].axhspan(bias-ci,bias+ci,color="#c0504d",alpha=0.15,lw=0,label="95% CI of mean difference")
ax[1].axhline(bias,color="#c0504d",lw=2); ax[1].text(0.121,bias+0.0008,"Mean difference\n(%.4f)"%bias,ha="right",va="bottom",fontsize=8.5,color="#c0504d")
for v,lab in [(bias+1.96*sd,"+1.96 SD"),(bias-1.96*sd,"$-$1.96 SD")]:
    ax[1].axhline(v,color="#e69138",ls="--",lw=1.5); ax[1].text(0.121,v+0.0006,"%s (%.4f)"%(lab,v),ha="right",va="bottom",fontsize=8.5,color="#e69138")
s2=stats.linregress(m,dd); xm=np.linspace(m.min()-0.005,m.max()+0.005,50)
ax[1].plot(xm,s2.intercept+s2.slope*xm,"-.",color="#8e7cc3",lw=1.3,label="Regression of difference on mean (slope = %.2f, p = %.2f)"%(s2.slope,s2.pvalue))
ax[1].plot(m[isP3],dd[isP3],"o",color="#c0504d",ms=8,mec="white"); ax[1].plot(m[~isP3],dd[~isP3],"o",color="#3b6ea5",ms=8,mec="white")
offB={"P3":(-22,-4),"P3N5m":(6,4),"P3E5m":(-44,-15),"P3W5m":(6,-13),"P3S5m":(6,4),"P1":(-22,-4),"P1N5m":(6,4),"P1E5m":(6,-4),"P1W5m":(6,-11),"P1S5m":(-12,7)}
for x,y,k,p3 in zip(m,dd,ids,isP3): ax[1].annotate(k,(x,y),xytext=offB[k],textcoords="offset points",fontsize=8,color="#c0504d" if p3 else "#3b6ea5")
ax[1].set_xlim(0.045,0.123); ax[1].set_ylim(-0.085,-0.015)
ax[1].set_xlabel("Mean of $G(E)$-derived and FH40G ($\\mu$Sv h$^{-1}$)"); ax[1].set_ylabel("Difference: $G(E)$ $-$ FH40G ($\\mu$Sv h$^{-1}$)")
ax[1].legend(loc="upper left",fontsize=8.2)
for k,A in enumerate(ax): A.text(-0.10,1.02,"ab"[k],transform=A.transAxes,fontsize=15,fontweight="bold")
fig.tight_layout(); fig.savefig(os.path.join(OUT,"ARI_Fig1.png"),dpi=300); plt.close(fig)
print("Fig2: r %.3f RMSE %.4f ratio %.3f+-%.3f slope %.2f BA slope %.2f p %.2f bias %.4f LoA %.4f %.4f"%(rp,rmse,r.mean(),r.std(ddof=1),sl.slope,s2.slope,s2.pvalue,bias,bias-1.96*sd,bias+1.96*sd))

# ---------------- Fig. S3 (file ARI_FigS3.png): 40K centroid vs time ----------------
gcsv=next(c for c in [os.path.join(HERE,"..","r2_dose","gain_check.csv"),os.path.join(HERE,"..","logs","gain_check.csv")] if os.path.exists(c))
G_=[l.split(",") for l in open(gcsv).read().strip().split("\n")[1:]]
tt_=np.array([int(r[0][11:13])+int(r[0][14:16])/60 for r in G_]); mu=np.array([float(r[3]) for r in G_]); sg=np.array([float(r[4]) for r in G_]); pts=[r[1] for r in G_]
fig,ax=plt.subplots(figsize=(10,4.6))
ax.axhspan(mu.mean()-mu.std(ddof=1),mu.mean()+mu.std(ddof=1),color="#3b6ea5",alpha=0.12,lw=0)
ax.axhline(mu.mean(),color="#3b6ea5",lw=1.6,label="Campaign mean (%.1f keV)"%mu.mean())
ax.axhline(1460.8,color="#c0504d",ls="--",lw=1.4,label="Nominal 1460.8 keV")
ax.plot(tt_,mu,"-",color="0.6",lw=1)
for x,y,e,p in zip(tt_,mu,sg,pts):
    col="#c0504d" if p.startswith("P3") else "#3b6ea5"
    ax.errorbar(x,y,yerr=e,fmt="o",color=col,mec="white",ms=8,capsize=3)
    ax.text(x,y+e+0.5,p,ha="center",va="bottom",fontsize=8.5,color=col)
ax.plot([],[],"o",color="#c0504d",label="P3 cluster"); ax.plot([],[],"o",color="#3b6ea5",label="P1 cluster")
ax.set_xticks([11,12,13,14]); ax.set_xticklabels(["11:00","12:00","13:00","14:00"]); ax.set_xlim(10.6,15.1); ax.set_ylim(1455,1492)
ax.set_xlabel("Acquisition start time (spectrum-file header, 29 April 2026)"); ax.set_ylabel("$^{40}$K photopeak centroid (keV)")
ax.text(0.01,0.97,"SD = %.1f keV   peak-to-peak = %.1f keV (%.1f channels)"%(mu.std(ddof=1),mu.max()-mu.min(),(mu.max()-mu.min())/3.0734),transform=ax.transAxes,va="top",fontsize=9,bbox=dict(boxstyle="round",fc="white",ec="0.6"))
ax.legend(loc="lower left",fontsize=8.5)
fig.tight_layout(); fig.savefig(os.path.join(OUT,"ARI_FigS3.png"),dpi=300); plt.close(fig)
print("FigS3 mean %.1f SD %.1f p2p %.1f"%(mu.mean(),mu.std(ddof=1),mu.max()-mu.min()))
