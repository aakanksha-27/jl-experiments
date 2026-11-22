#imports
from sklearn.datasets import fetch_kddcup99
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

seed = 0
rng = np.random.default_rng(seed) #random draws come from this generator

#data preprocessing
kdd = fetch_kddcup99(subset=None,as_frame=True)
X=kdd.data

text_feats = ['protocol_type','service','flag']
num_feats = X.drop(columns=text_feats).astype(float) #38 num feats

#one hot encoding
text_feats_encoded = pd.get_dummies(X[text_feats].astype(str)).astype(float)  
D = pd.concat([num_feats,text_feats_encoded],axis=1) 

#drop constant feats
D = D.loc[:,D.nunique()>1] 
D = D.to_numpy(dtype=np.float64)

#scale to [0,1]
D = (D-D.min(axis=0))/(D.max(axis=0)-D.min(axis=0)) 

y = (kdd.target!=b"normal.").to_numpy(dtype=float)
n,d=D.shape
print("number of rows n: ",n)
print("number of columns d: ",d)

def loss(D,y,a):
    lossv = np.sum((D@a-y)**2)
    return lossv

def solve(A,z):
    ans=np.linalg.lstsq(A,z,rcond=None)[0]
    return ans

def countsketch(D,y,m,rng):
    n=D.shape[0]
    #bucket for row
    h=rng.integers(0,m,size=n)
    #random sign for row
    sig=rng.choice([-1.0,1.0],size=n)
    sd = np.zeros((m,D.shape[1]))
    sy = np.zeros(m)
    np.add.at(sd,h,sig[:,None]*D)
    np.add.at(sy,h,sig*y)
    return sd,sy

#a 
t0=time.time()
#b=solve(D,y) #gave extra time
b,_,rank,_=np.linalg.lstsq(D,y,rcond=None)
time_b=time.time()-t0
loss_b=loss(D,y,b)
print("a) Loss(D,y,b): ",loss_b)
print("Time taken: ",time_b,"s")
print("Rank(D): ",rank)

#b and c
m = [50,100,200,500,1000,2000,5000]
laps=5
rows=[] #one per (m,lap)

for mm in m:
    for lap in range(laps):
        t0=time.time()
        sd,sy=countsketch(D,y,mm,rng)
        #solve small problem
        a=solve(sd,sy)
        time_taken=time.time()-t0
        loss_a=loss(D,y,a)
        sketched=np.sum((sd@a-sy)**2)
        rows.append(dict(m=mm,lap=lap,loss=loss_a,ratio=loss_a/loss_b,sketched_loss=sketched,time_taken=time_taken))
        #print("m: ",mm," lap: ",lap," loss: ",loss_a," time taken: ",time_taken,"s") #debugging

res=pd.DataFrame(rows)
print("b) all runs: ")
print(res.to_string(index=False))

summary=res.groupby("m").agg(mean_ratio=("ratio","mean"),min_ratio=("ratio","min"),max_ratio=("ratio","max"),mean_sketched_loss=("sketched_loss","mean"),mean_time=("time_taken","mean"))
print("summary per m:")
print(summary.to_string())

print("c) time to compute b in a): ",time_b,"s")
print("mean sketch time per m (draw s + form sd, sy + solve):")
print(summary["mean_time"])

#plotting mean ratio vs m, error bars for min/max
mean_rati=summary["mean_ratio"].to_numpy()
l = mean_rati-summary["min_ratio"].to_numpy()
h = summary["max_ratio"].to_numpy()-mean_rati
plt.figure()
plt.errorbar(m,mean_rati,yerr=[l,h],fmt='o-',capsize=4)
plt.axvline(rank,color="gray",linestyle="--",label=f"rank(D) = {rank}")
plt.xscale("log")
plt.xlabel("m")
plt.yscale("log")
plt.ylabel("loss(D,y,a~) / loss(D,y,b)") 
plt.title("Q2(b): loss ratio vs sketch size")
plt.legend()
#save plot
plt.savefig("q2b_loss_ratio_vs_m.png",dpi=150)

#d
v=D@b-y
v_norm2=np.sum(v**2)

fig,axes=plt.subplots(1,3,figsize=(14,4))
for ax,m in zip(axes,[50,500,5000]):
    vals = []
    for x in range(100):
        h = rng.integers(0,m,size=n)
        sig = rng.choice([-1.0,1.0],size=n)
        sv=np.bincount(h,weights=sig*v,minlength=m) #sv w/o building s
        vals.append(np.sum(sv**2)/v_norm2)
    vals = np.array(vals)
    print("d) m = ",m,": mean = ", vals.mean(), ", std = ", vals.std())
    ax.hist(vals,bins=20)
    ax.set_title("m = "+str(m))
    ax.set_xlabel("||Sv||^2 / ||v||^2")
plt.tight_layout()
#save plot
plt.savefig("q2d_histogram.png",dpi=150)

#e
m = 500
subset=np.random.default_rng(seed).choice(n,size=50000,replace=False) #fixed
dp = D[subset]
yp = y[subset]
bp = solve(dp,yp)
loss_p=loss(dp,yp,bp)

out={"countsketch":[],"dense gaussian":[]}
for lap in range(laps):
    #countsketch
    t0=time.time()
    sd,sy=countsketch(dp,yp,m,rng)
    time_taken=time.time()-t0
    a=solve(sd,sy)
    loss_a=loss(dp,yp,a)
    out["countsketch"].append((time_taken,loss_a/loss_p))
    
    #dense s w n(0,1/m) entries
    t0=time.time()
    s=rng.normal(0.0,1.0/np.sqrt(m),size=(m,50000))
    sd = s@dp
    sy = s@yp
    time_taken=time.time()-t0
    a=solve(sd,sy)
    loss_a=loss(dp,yp,a)
    out["dense gaussian"].append((time_taken,loss_a/loss_p))

print("e) m=500 on 50k rows over 5 laps")
for method, vals in out.items():
    vals=np.array(vals)
    print(method,": sketch time = ",vals[:,0].mean(),"s, loss ratio = ",vals[:,1].mean())