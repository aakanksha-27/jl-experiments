import random
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_kddcup99
import time
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import matplotlib.pyplot as plt

RANDOM_SEED=42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
kdd=fetch_kddcup99(subset=None, as_frame=True)

X_raw=kdd.data.copy()
target=kdd.target

print(f"Raw feature matrix shape: {X_raw.shape}")

cat_col=["protocol_type","service","flag"]          
numeric_col=[                        #converting to floating-point
    col for col in X_raw.columns
    if col not in cat_col
]

for col in numeric_col:
    X_raw[col]=pd.to_numeric(X_raw[col], errors="raise").astype(np.float32)

X_encoded=pd.get_dummies(X_raw,columns=cat_col,dtype=np.float32) #one-hot encoding

print(f"Shape after one-hot encoding:{X_encoded.shape}")

non_const_col=X_encoded.nunique(dropna=False)>1  #dropping constant cols
D_df=X_encoded.loc[:, non_const_col].copy()

print(f"Shape after dropping constant columns:{D_df.shape}")

col_min=D_df.min(axis=0)
col_max=D_df.max(axis=0)
col_range=col_max-col_min
D_df=(D_df-col_min)/col_range

D=D_df.to_numpy(dtype=np.float32) #storing D in float32 to reduce memory

y=(target!=b"normal.").to_numpy(dtype=float)
n,d=D.shape

print("\nPreprocessing complete.")
print(f"Number of rows (n): {n}")
print(f"Number of columns (d): {d}")
print(f"D shape: {D.shape}")
print(f"y shape: {y.shape}")
print(f"D data type: {D.dtype}")
print(f"Minimum value in D: {D.min():.6f}")
print(f"Maximum value in D: {D.max():.6f}")
print(f"Number of attack rows: {int(y.sum())}")
print(f"Number of normal rows: {int(n - y.sum())}")

#Q1(a)
k=15
start_time=time.perf_counter()

kmeans_org=KMeans(n_clusters=k,n_init=5,random_state=42)
kmeans_org.fit(D)
B=kmeans_org.cluster_centers_
clusters_D=kmeans_org.labels_
cost_org=kmeans_org.inertia_
time_org=time.perf_counter()-start_time

print("org dataset res:")
print("Number of clusters:",k)
print("org cost:",cost_org)
print("Execution time:",time_org,"seconds")

#Q1(b) and (c)
m_val=[5,10,15,20,25]
num_runs=5
k=15
rng=np.random.default_rng(42)
res=[]
q1b_times={}
pair_distance_ratios={}
cluster_cost_ratios={}


def calc_cost(X,centers):
    nearest_centres,dist=pairwise_distances_argmin_min(X,centers,metric="euclidean")
    return np.sum(dist**2)

def part_cost(X,labels,k):
    tot_cost=0.0

    for j in range(k):
        cluster_points=X[labels==j]
        if len(cluster_points)==0:
            continue

        mean=np.mean(cluster_points,axis=0)
        tot_cost+=np.sum((cluster_points-mean)**2)

    return tot_cost


for m in m_val:
    print(f"\nRunning experiments for m={m}")
    bhat_ratios=[]
    bplus_ratios=[]
    costs=[]
    run_times=[]

    for run in range(num_runs):
        A=rng.normal(loc=0,scale=1/np.sqrt(m),size=(m,d)).astype(np.float32)
        start_time=time.perf_counter()
        E=D@A.T

        kmeans_proj=KMeans(n_clusters=k,n_init=5,random_state=42+run)
        clusters_E=kmeans_proj.fit_predict(E)
        elapsed=time.perf_counter()-start_time
        run_times.append(elapsed)

        B_hat=np.zeros((k,d),dtype=np.float32)

        for j in range(k):
            cluster_points=D[clusters_E==j]
            B_hat[j]=np.mean(cluster_points,axis=0)

        cost_bhat=calc_cost(D, B_hat)
        ratio_bhat=cost_bhat/cost_org
        bhat_ratios.append(ratio_bhat)
        costs.append(cost_bhat)
        
        #Q1(c)
        A_pinv=np.linalg.pinv(A)
        proj_centres=kmeans_proj.cluster_centers_
        B_plus=(A_pinv@proj_centres.T).T

        cost_bplus=calc_cost(D,B_plus)
        ratio_bplus=cost_bplus/cost_org
        bplus_ratios.append(ratio_bplus)

        print(f"Run {run+1}: "
            f"B_hat ratio={ratio_bhat:.6f}, "
            f"B_plus ratio={ratio_bplus:.6f}")

    q1b_times[m]=np.mean(run_times)
    res.append({
        "m": m,
        "Mean Cost B_hat": np.mean(costs),
        "Mean B_hat Ratio": np.mean(bhat_ratios),
        "Min B_hat Ratio": np.min(bhat_ratios),
        "Max B_hat Ratio": np.max(bhat_ratios),
        "Mean B_plus Ratio": np.mean(bplus_ratios),
        "Min B_plus Ratio": np.min(bplus_ratios),
        "Max B_plus Ratio": np.max(bplus_ratios)})

res_df=pd.DataFrame(res)
print(res_df.to_string(index=False))
res_df.to_csv("Q1_parts_b_c_res.csv", index=False)


#Q(d)
print("org K-means time:",time_org)

for m in m_val:
    print(f"m={m}: "
        f"Average projection+K-means time= "
        f"{q1b_times[m]:.4f} seconds")

plt.figure(figsize=(8,5))
times=[q1b_times[m] for m in m_val]
plt.plot(m_val,times,marker="o",label="Projection+K-means")
plt.axhline(y=time_org,color="red",linestyle="--",label="org K-means on D")
plt.xlabel("Projection dimension (m)")
plt.ylabel("Execution time (seconds)")
plt.title("Q1(d): Execution Time Comparison")
plt.xticks(m_val)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("Q1_part_d_timing.png", dpi=300)
plt.show()


#Q1(e)
num_pairs=10**4
pair_i_list=[]
pair_l_list=[]

while sum(len(x) for x in pair_i_list)<num_pairs:
    rem=num_pairs-sum(len(x) for x in pair_i_list)
    batch_size=max(rem*2,1000)
    candidate_i=rng.integers(0,n,size=batch_size)
    candidate_l=rng.integers(0,n,size=batch_size)
    valid=candidate_i!=candidate_l
    candidate_i=candidate_i[valid]
    candidate_l=candidate_l[valid]

    valid=np.any(D[candidate_i]!=D[candidate_l],axis=1)

    candidate_i=candidate_i[valid]
    candidate_l=candidate_l[valid]
    pair_i_list.append(candidate_i)
    pair_l_list.append(candidate_l)

pair_i=np.concatenate(pair_i_list)[:num_pairs]
pair_l=np.concatenate(pair_l_list)[:num_pairs]

print("Number of valid pairs selected:",len(pair_i))

org_diff=D[pair_i]-D[pair_l]
org_squared_dist=np.sum(org_diff**2,axis=1)

org_part_cost=part_cost(D,clusters_D,k)

print("original partition cost:",org_part_cost)

for m in m_val:
    print(f"\nQ1(e): Running experiments for m={m}")
    A=rng.normal(loc=0,scale=1/np.sqrt(m),size=(m,d)).astype(np.float32)
    E=D@A.T
    proj_diff=E[pair_i]-E[pair_l]
    proj_squared_dist=np.sum(proj_diff**2,axis=1)

    ratios=(proj_squared_dist/org_squared_dist)
    pair_distance_ratios[m]=ratios

    cost_ratios=[]
    for repetition in range(20):
        A=rng.normal(loc=0,scale=1/np.sqrt(m),size=(m,d)).astype(np.float32)
        E=D@A.T
        proj_part_cost=part_cost(E,clusters_D,k)

        cost_ratio=(proj_part_cost/org_part_cost)
        cost_ratios.append(cost_ratio)

    cluster_cost_ratios[m]=cost_ratios

    print("Pairwise distance ratio: "
        f"Mean={np.mean(ratios):.4f}, "
        f"Std={np.std(ratios):.4f}")

    print(
        "Cluster cost ratio: "
        f"Mean={np.mean(cost_ratios):.4f}, "
        f"Std={np.std(cost_ratios):.4f}")

#boxplot
plt.figure(figsize=(11,6))
pos_pair=np.arange(len(m_val))*2.0+1
pos_cost=pos_pair+0.55
pair_data=[pair_distance_ratios[m] for m in m_val]
cost_data=[cluster_cost_ratios[m] for m in m_val]

plt.boxplot(pair_data,positions=pos_pair,widths=0.45)
plt.boxplot(cost_data,positions=pos_cost,widths=0.45)
plt.xticks(pos_pair+0.275,m_val)
plt.xlabel("Projection dimension (m)")
plt.ylabel("Squared distance/cost ratio")
plt.title("Q1(e): Pairwise Distance and Cluster Cost Preservation")
plt.legend(handles=[
        plt.Line2D(
            [0],[0],color="black",marker="o",linestyle="",label="Pairwise distance ratios"),
        plt.Line2D([0],[0],color="black",marker="s",linestyle="",label="Cluster cost ratios")])
plt.grid(axis="y",alpha=0.3)
plt.tight_layout()
plt.savefig("Q1_part_e_boxplot.png",dpi=300)
plt.show()

q1e_summary=[]
for m in m_val:
    q1e_summary.append({
        "m":m,"Pairwise Mean": np.mean(pair_distance_ratios[m]),"Pairwise Std": np.std(pair_distance_ratios[m]),
        "Cluster Cost Mean": np.mean(cluster_cost_ratios[m]),"Cluster Cost Std": np.std(cluster_cost_ratios[m])})

q1e_df=pd.DataFrame(q1e_summary)
print(q1e_df.to_string(index=False))
q1e_df.to_csv("Q1_part_e_res.csv",index=False)