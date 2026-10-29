# 24BSC10009-Smart-Campus-Student-Behaviour-Performance-and-Risk-Pattern-Mining-
Name: Gagandeep Kaur
Roll No.: 24BSC10009
Course / Branch: [BSC CS] – [Computing]
Subject: Data Mining
Submitted to: [Ms. Ankita]
Institution: [Chandigarh University] – [university institute of computing]
Index
1. Aim, Objectives and Tools	2
2. Imports and Helper Functions	3
3. Dataset Generation	4
4. Data Cleaning	4
5. Data Integration	5
6. Data Reduction	5
7. Data Transformation	6
8. Discretization	6
9. Statistical Analysis	7
10. Similarity and Dissimilarity	8
11. Frequent Patterns and Association Rules	9
12. Clustering	10
13. Classification and Evaluation	12
14. Linear Regression	13
15. Dashboard	15
16. Dashboard Output	15
17. Result Summary and Conclusion	15
1. Aim, Objectives and Tools
Aim
To build a complete data-mining system on smart-campus data that cleans and prepares the data, discovers behaviour patterns, groups students, predicts students who are at risk and presents the results in a dashboard.
Objectives
•	Apply data cleaning, integration, reduction, transformation and discretization.
•	Perform statistical analysis and compute similarity / dissimilarity.
•	Mine frequent patterns (Apriori, FP-Growth) and association rules.
•	Cluster students with K-Means, K-Medoids, Hierarchical clustering and DBSCAN.
•	Classify at-risk students (Decision Tree, Naive Bayes, k-NN) and evaluate with confusion matrix, accuracy, precision, recall and ROC.
•	Predict final score with Linear Regression and show everything on a dashboard.
Software and libraries
Table 1: Environment
Item	Details
Language	Python 3
Libraries	pandas, NumPy, SciPy, scikit-learn, mlxtend, matplotlib
Install	pip install pandas numpy scipy scikit-learn mlxtend matplotlib
Run	python smart_campus_mining.py
Outputs	academic_raw.csv, behaviour_raw.csv, integrated_clean.csv, risk_rules.csv, dashboard.html

Pipeline
Dataset → Cleaning → Integration → Reduction → Transformation → Discretization → Statistics → Similarity → Apriori / FP-Growth → Association Rules → K-Means, K-Medoids, Hierarchical, DBSCAN → Decision Tree, Naive Bayes, k-NN → Linear Regression → Confusion Matrix, Accuracy, Precision, Recall, ROC → Dashboard
Dataset
A synthetic dataset of 600 students is generated in two tables (academic and behaviour) with injected duplicates, missing values and outliers. Attributes: attendance, assignment_pct, midterm, final_score, department, hostel, study_hours, lms_logins, library_visits, wifi_hours, sleep_hours, complaints. The class label at_risk = 1 for the bottom 25% of final scores.
2. Imports and Helper Functions
Aim
Import all libraries and define logging and dashboard helpers.
Source Code
"""
Smart Campus Student Behaviour, Performance and Risk Pattern Mining
Using Data Mining Techniques  --  end-to-end pipeline + HTML dashboard.
 
Run:  pip install pandas numpy scikit-learn matplotlib mlxtend scipy
      python smart_campus_mining.py
Output: outputs/ (CSV files, dashboard.html)
"""
import os, io, base64, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean, cityblock, cosine
from sklearn.preprocessing import MinMaxScaler, StandardScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import (silhouette_score, confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, roc_curve, roc_auc_score, r2_score,
                             mean_absolute_error, mean_squared_error)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from scipy.cluster.hierarchy import linkage, dendrogram
 
warnings.filterwarnings("ignore")
OUT = "outputs"; os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(42)
LOG, FIGS, TABLES = [], [], []
 
def log(step, msg): LOG.append((step, msg)); print(f"[{step}] {msg}")
def fig_to_html(title, fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=90, bbox_inches="tight"); plt.close(fig)
    FIGS.append((title, base64.b64encode(buf.getvalue()).decode()))
def add_table(title, df, index=True): TABLES.append((title, df.to_html(classes="t", float_format="%.3f", index=index)))
3. Dataset Generation
Aim
Create the academic and behaviour tables with realistic relationships and deliberate noise.
Source Code
def make_dataset(n=600):
    ids = [f"S{i:04d}" for i in range(n)]
    dept = rng.choice(["CSE", "ECE", "ME", "Civil", "MBA"], n)
    hostel = rng.choice(["Day Scholar", "Hostel"], n, p=[.45, .55])
    att = np.clip(rng.normal(75, 15, n), 20, 100)
    study = np.clip(rng.normal(3 + (att - 75) / 25, 1.5, n), 0, 10)
    lms = np.clip(rng.poisson(20 + study * 4), 0, None)
    lib = np.clip(rng.poisson(4 + study), 0, None)
    wifi = np.clip(rng.normal(6 - study / 2, 2, n), 0.5, 14)
    sleep = np.clip(rng.normal(6.8, 1.2, n), 3, 10)
    assign = np.clip(rng.normal(60 + att / 4 + study * 2, 12, n), 0, 100)
    compl = rng.poisson(np.clip(1.5 - att / 80 + wifi / 10, .1, None))
    mid = np.clip(15 + .25 * att + 3 * study + .1 * assign + rng.normal(0, 7, n), 0, 100)
    final = np.clip(.8 * mid + 8 + 1.5 * study - 1.5 * compl + rng.normal(0, 5, n), 0, 100)
    academic = pd.DataFrame(dict(student_id=ids, department=dept, hostel=hostel, attendance=att,
                                 assignment_pct=assign, midterm=mid, final_score=final))
    behaviour = pd.DataFrame(dict(student_id=ids, study_hours=study, lms_logins=lms, library_visits=lib,
                                  wifi_hours=wifi, sleep_hours=sleep, complaints=compl))
    # inject dirt: missing values, duplicates, outliers
    for df, cols in [(academic, ["attendance", "midterm"]), (behaviour, ["study_hours", "sleep_hours"])]:
        for c in cols: df.loc[rng.choice(n, 25, replace=False), c] = np.nan
    behaviour.loc[rng.choice(n, 5, replace=False), "wifi_hours"] = 60   # outliers
    academic = pd.concat([academic, academic.sample(15, random_state=1)])  # duplicates
    academic.to_csv(f"{OUT}/academic_raw.csv", index=False); behaviour.to_csv(f"{OUT}/behaviour_raw.csv", index=False)
    log("1 Dataset", f"academic {academic.shape}, behaviour {behaviour.shape} (with injected noise)")
    return academic, behaviour
 
academic, behaviour = make_dataset()
Output (console)
[1 Dataset] academic (615, 7), behaviour (600, 7) (with injected noise)
4. Data Cleaning
Aim
Remove duplicates, fill missing values with the median and cap outliers using the IQR rule.
Source Code
def clean(academic, behaviour):
    d = academic.duplicated().sum(); academic = academic.drop_duplicates()
    miss = int(academic.isna().sum().sum() + behaviour.isna().sum().sum())
    for df in (academic, behaviour):
        for c in df.select_dtypes("number"): df[c] = df[c].fillna(df[c].median())
    out = 0
    for c in ["wifi_hours", "study_hours", "lms_logins", "library_visits"]:       # IQR winsorising
        q1, q3 = behaviour[c].quantile([.25, .75]); lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        out += int(((behaviour[c] < lo) | (behaviour[c] > hi)).sum()); behaviour[c] = behaviour[c].clip(lo, hi)
    log("2 Cleaning", f"removed {d} duplicates, imputed {miss} missing (median), capped {out} outliers (IQR)")
    return academic, behaviour
academic, behaviour = clean(academic, behaviour)
Output (console)
[2 Cleaning] removed 15 duplicates, imputed 100 missing (median), capped 36 outliers (IQR)
5. Data Integration
Aim
Merge the two tables on student_id and create the at_risk class label.
Source Code
df = academic.merge(behaviour, on="student_id", how="inner")
df["at_risk"] = (df.final_score < df.final_score.quantile(.25)).astype(int)   # class label
log("3 Integration", f"merged on student_id -> {df.shape}; at-risk students = {df.at_risk.sum()} ({df.at_risk.mean():.0%})")
df.to_csv(f"{OUT}/integrated_clean.csv", index=False)
NUM = ["attendance", "assignment_pct", "midterm", "study_hours", "lms_logins", "library_visits",
       "wifi_hours", "sleep_hours", "complaints"]
Output (console)
[3 Integration] merged on student_id -> (600, 14); at-risk students = 150 (25%)
6. Data Reduction
Aim
Drop highly correlated attributes and apply PCA.
Source Code
corr = df[NUM].corr().abs(); upper = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool))
dropped = [c for c in upper.columns if (upper[c] > .85).any()]
Xs = StandardScaler().fit_transform(df[NUM]); pca = PCA().fit(Xs)
k95 = int(np.argmax(np.cumsum(pca.explained_variance_ratio_) >= .90) + 1)
X_pca = PCA(2).fit_transform(Xs)
log("4 Reduction", f"correlation-drop (>0.85): {dropped or 'none'}; PCA needs {k95} comps for 90% variance")
fig, ax = plt.subplots(figsize=(5, 3)); ax.bar(range(1, len(NUM) + 1), pca.explained_variance_ratio_)
ax.set(title="PCA explained variance", xlabel="component"); fig_to_html("Data Reduction - PCA", fig)
Output (console)
[4 Reduction] correlation-drop (>0.85): none; PCA needs 7 comps for 90% variance
Results
 
Figure 1: PCA explained variance
7. Data Transformation
Aim
Normalise attributes using min-max and z-score scaling.
Source Code
X_mm = pd.DataFrame(MinMaxScaler().fit_transform(df[NUM]), columns=NUM)
X_z = pd.DataFrame(Xs, columns=NUM)
log("5 Transformation", "min-max (0-1) and z-score normalisation applied")
Output (console)
[5 Transformation] min-max (0-1) and z-score normalisation applied
8. Discretization
Aim
Convert numeric attributes into Low / Medium / High categories using equal-width and equal-frequency binning.
Source Code
disc = pd.DataFrame({
    "Attendance": pd.cut(df.attendance, [0, 60, 80, 100], labels=["Low", "Med", "High"]),        # equal-domain bins
    "Study": pd.qcut(df.study_hours, 3, labels=["Low", "Med", "High"]),                           # equal-frequency
    "LMS": pd.qcut(df.lms_logins, 3, labels=["Low", "Med", "High"]),
    "Sleep": pd.cut(df.sleep_hours, [0, 6, 8, 12], labels=["Low", "Med", "High"]),
    "WiFi": pd.qcut(df.wifi_hours, 3, labels=["Low", "Med", "High"]),
    "Complaints": pd.cut(df.complaints, [-1, 0, 2, 99], labels=["None", "Few", "Many"]),
    "Result": pd.qcut(df.final_score, 3, labels=["Poor", "Avg", "Good"]),
    "Hostel": df.hostel.map({"Hostel": "Hostel", "Day Scholar": "Day"}),
})
disc["Risk"] = df.at_risk.map({1: "AtRisk", 0: "Safe"})
log("6 Discretization", "binning: equal-width (attendance, sleep, complaints) + equal-frequency (study, LMS, WiFi)")
Output (console)
[6 Discretization] binning: equal-width (attendance, sleep, complaints) + equal-frequency (study, LMS, WiFi)
9. Statistical Analysis
Aim
Compute descriptive statistics, skewness, correlation heatmap and boxplots.
Source Code
desc = df[NUM + ["final_score"]].describe().T; desc["skew"] = df[NUM + ["final_score"]].skew()
desc["median"] = df[NUM + ["final_score"]].median(); add_table("Descriptive Statistics", desc)
fig, ax = plt.subplots(figsize=(6, 5)); cm = df[NUM + ["final_score"]].corr()
im = ax.imshow(cm, cmap="coolwarm", vmin=-1, vmax=1); ax.set_xticks(range(len(cm))); ax.set_yticks(range(len(cm)))
ax.set_xticklabels(cm.columns, rotation=60, ha="right", fontsize=7); ax.set_yticklabels(cm.columns, fontsize=7)
fig.colorbar(im); ax.set_title("Correlation heatmap"); fig_to_html("Statistical Analysis - Correlation", fig)
fig, ax = plt.subplots(figsize=(5, 3)); df.boxplot("final_score", by="department", ax=ax); plt.suptitle("")
ax.set_title("Final score by department"); fig_to_html("Statistical Analysis - Boxplot", fig)
log("7 Statistics", f"strongest correlate of final_score: {cm.final_score.drop('final_score').abs().idxmax()}")
Output (console)
[7 Statistics] strongest correlate of final_score: midterm
Results
Table 2: Descriptive statistics
Attribute	Mean	Std	Min	Max	Skew
attendance	74.84	14.48	20.27	100.00	-0.41
assignment_pct	82.93	12.23	40.64	100.00	-0.43
midterm	50.13	10.27	19.51	78.63	-0.12
study_hours	2.79	1.55	0.00	6.98	0.17
lms_logins	31.17	8.64	9.00	55.00	0.26
library_visits	6.64	2.74	0.50	12.50	0.16
wifi_hours	4.74	2.08	0.50	10.49	0.14
sleep_hours	6.78	1.24	3.00	10.00	-0.12
complaints	1.07	1.12	0.00	6.00	1.08
final_score	50.77	11.44	10.43	85.15	-0.10

 
Figure 2: Correlation heatmap
 
Figure 3: Final score by department
10. Similarity and Dissimilarity
Aim
Compute Euclidean, Manhattan, cosine and Jaccard measures and find nearest students.
Source Code
a, b = X_z.iloc[0].values, X_z.iloc[1].values
ba, bb = (disc.iloc[0] == "High").values, (disc.iloc[1] == "High").values
jac = (ba & bb).sum() / max((ba | bb).sum(), 1)
sim = pd.DataFrame({"Measure": ["Euclidean", "Manhattan", "Cosine distance", "Jaccard (High-items)"],
                    "Student S0000 vs S0001": [euclidean(a, b), cityblock(a, b), cosine(a, b), 1 - jac]})
add_table("Dissimilarity between two students", sim, index=False)
D = np.linalg.norm(Xs[:, None] - Xs[None, :], axis=2); np.fill_diagonal(D, np.inf)
nn = D[0].argsort()[:3]; log("8 Similarity", f"nearest students to S0000: {[df.student_id[i] for i in nn]}")
Output (console)
[8 Similarity] nearest students to S0000: ['S0254', 'S0384', 'S0472']
Results
Table 3: Dissimilarity between students S0000 and S0001
Measure	Student S0000 vs S0001
Euclidean	3.27
Manhattan	7.295
Cosine distance	0.424
Jaccard (High-items)	1.0

11. Frequent Patterns and Association Rules
Aim
Mine frequent itemsets with Apriori and FP-Growth and generate rules leading to At-Risk.
Source Code
basket = pd.get_dummies(disc.astype(str), prefix_sep="=").astype(bool)
import time
t = time.time(); fi_ap = apriori(basket, min_support=.08, use_colnames=True); t_ap = time.time() - t
t = time.time(); fi_fp = fpgrowth(basket, min_support=.08, use_colnames=True); t_fp = time.time() - t
log("9 Frequent patterns", f"Apriori {len(fi_ap)} itemsets in {t_ap:.3f}s | FP-Growth {len(fi_fp)} itemsets in {t_fp:.3f}s")
rules = association_rules(fi_fp, metric="confidence", min_threshold=.6)
rules = rules[rules.consequents.apply(lambda s: "Risk=AtRisk" in s)].sort_values("lift", ascending=False)
rules["antecedents"] = rules.antecedents.apply(lambda s: ", ".join(sorted(s)))
rules["consequents"] = rules.consequents.apply(lambda s: ", ".join(sorted(s)))
r_show = rules[["antecedents", "consequents", "support", "confidence", "lift"]].head(10)
r_show.to_csv(f"{OUT}/risk_rules.csv", index=False)
add_table("Top Association Rules -> At-Risk", r_show, index=False)
add_table("Apriori vs FP-Growth", pd.DataFrame({"Algorithm": ["Apriori", "FP-Growth"], "Itemsets": [len(fi_ap), len(fi_fp)],
                                                "Time (s)": [t_ap, t_fp]}), index=False)
Output (console)
[9 Frequent patterns] Apriori 660 itemsets in 0.009s | FP-Growth 660 itemsets in 0.027s
Results
Table 4: Apriori vs FP-Growth
Algorithm	Itemsets	Time (s)
Apriori	660	0.009
FP-Growth	660	0.024

Table 5: Top rules leading to At-Risk
Antecedent	Consequent	Support	Confidence	Lift
Hostel=Hostel, LMS=Low, Result=Poor	Risk=AtRisk, Study=Low	0.09	0.82	4.49
Hostel=Hostel, Result=Poor, Study=Low	LMS=Low, Risk=AtRisk	0.09	0.67	4.21
LMS=Low, Result=Poor	Risk=AtRisk, Study=Low	0.14	0.74	4.03
Result=Poor, Study=Low	LMS=Low, Risk=AtRisk	0.14	0.63	3.95
Result=Poor, WiFi=High	Risk=AtRisk, Study=Low	0.10	0.66	3.62
Hostel=Hostel, LMS=Low, Result=Poor, Study=Low	Risk=AtRisk	0.09	0.90	3.61
Hostel=Hostel, LMS=Low, Result=Poor	Risk=AtRisk	0.10	0.90	3.59
Result=Poor, Sleep=Med, Study=Low	Risk=AtRisk	0.11	0.87	3.47
LMS=Low, Result=Poor, Sleep=Med	Risk=AtRisk	0.09	0.86	3.45
Attendance=Med, LMS=Low, Result=Poor	Risk=AtRisk	0.09	0.85	3.40

12. Clustering
Aim
Run K-Means (elbow and silhouette), K-Medoids, Hierarchical (Ward) and DBSCAN.
Source Code
def kmedoids(X, k, iters=50):
    D = np.linalg.norm(X[:, None] - X[None, :], axis=2); med = rng.choice(len(X), k, replace=False)
    for _ in range(iters):
        lab = D[:, med].argmin(1); new = med.copy()
        for j in range(k):
            idx = np.where(lab == j)[0]
            if len(idx): new[j] = idx[D[np.ix_(idx, idx)].sum(1).argmin()]
        if (new == med).all(): break
        med = new
    return D[:, med].argmin(1), med
 
ks = range(2, 8); inertia, sil = [], []
for k in ks:
    km = KMeans(k, n_init=10, random_state=0).fit(Xs); inertia.append(km.inertia_); sil.append(silhouette_score(Xs, km.labels_))
best_k = list(ks)[int(np.argmax(sil))]
fig, ax = plt.subplots(1, 2, figsize=(8, 3)); ax[0].plot(ks, inertia, "o-"); ax[0].set(title="Elbow", xlabel="k")
ax[1].plot(ks, sil, "o-r"); ax[1].set(title="Silhouette", xlabel="k"); fig_to_html("K-Means model selection", fig)
 
lab_km = KMeans(best_k, n_init=10, random_state=0).fit_predict(Xs)
lab_kmed, medoids = kmedoids(Xs, best_k)
lab_hc = AgglomerativeClustering(best_k, linkage="ward").fit_predict(Xs)
lab_db = DBSCAN(eps=2.2, min_samples=8).fit_predict(Xs)
n_noise = int((lab_db == -1).sum())
res = {"K-Means": lab_km, "K-Medoids": lab_kmed, "Hierarchical (Ward)": lab_hc, "DBSCAN": lab_db}
rows = []
for name, l in res.items():
    ok = len(set(l) - {-1}) > 1
    rows.append([name, len(set(l) - {-1}), silhouette_score(Xs[l != -1], l[l != -1]) if ok else np.nan,
                 df.at_risk.groupby(l).mean().max()])
add_table("Clustering comparison", pd.DataFrame(rows, columns=["Algorithm", "Clusters", "Silhouette", "Max at-risk rate in a cluster"]), index=False)
fig, ax = plt.subplots(1, 4, figsize=(15, 3.2))
for a_, (name, l) in zip(ax, res.items()):
    a_.scatter(X_pca[:, 0], X_pca[:, 1], c=l, cmap="viridis", s=8); a_.set_title(name, fontsize=9)
fig_to_html("Clusters in PCA space", fig)
sample = np.random.default_rng(0).choice(len(Xs), 60, replace=False)
fig, ax = plt.subplots(figsize=(10, 3)); dendrogram(linkage(Xs[sample], "ward"), ax=ax, no_labels=True); ax.set_title("Dendrogram (60 students)")
fig_to_html("Hierarchical dendrogram", fig)
df["cluster"] = lab_km
prof = df.groupby("cluster")[NUM + ["final_score", "at_risk"]].mean(); add_table(f"K-Means (k={best_k}) cluster profiles", prof)
log("10 Clustering", f"best k={best_k} (silhouette {max(sil):.3f}); DBSCAN noise points={n_noise}")
Output (console)
[10 Clustering] best k=2 (silhouette 0.199); DBSCAN noise points=42
Results
 
Figure 4: Elbow and silhouette
Table 6: Clustering comparison
Algorithm	Clusters	Silhouette	Max at-risk rate in a cluster
K-Means	2	0.199	0.486
K-Medoids	2	0.184	0.465
Hierarchical (Ward)	2	0.165	0.393
DBSCAN	1	n/a	0.357

 
Figure 5: Clusters in PCA space
 
Figure 6: Dendrogram (60 students)
Table 7: K-Means cluster profiles
Cluster	Attend.	Assign.	Mid	Study h	LMS	WiFi h	Final	At-risk
0	67.29	76.92	43.29	1.67	26.00	5.15	42.89	0.49
1	81.62	88.34	56.28	3.81	35.83	7.98	57.85	0.04

13. Classification and Evaluation
Aim
Train Decision Tree, Naive Bayes and k-NN; evaluate with confusion matrix, accuracy, precision, recall, F1 and ROC.
Source Code
FEATS = ["attendance", "assignment_pct", "midterm", "study_hours", "lms_logins", "library_visits",
         "wifi_hours", "sleep_hours", "complaints"]
Xtr, Xte, ytr, yte = train_test_split(df[FEATS], df.at_risk, test_size=.3, stratify=df.at_risk, random_state=7)
sc = StandardScaler().fit(Xtr)
models = {"Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=0),
          "Naive Bayes": GaussianNB(), "k-NN (k=7)": KNeighborsClassifier(7)}
metrics, fig_roc, fig_cm = [], plt.subplots(figsize=(5, 4)), plt.subplots(1, 3, figsize=(11, 3))
for i, (name, m) in enumerate(models.items()):
    m.fit(sc.transform(Xtr), ytr); p = m.predict(sc.transform(Xte)); pr = m.predict_proba(sc.transform(Xte))[:, 1]
    metrics.append([name, accuracy_score(yte, p), precision_score(yte, p), recall_score(yte, p), f1_score(yte, p), roc_auc_score(yte, pr)])
    fpr, tpr, _ = roc_curve(yte, pr); fig_roc[1].plot(fpr, tpr, label=f"{name} (AUC={metrics[-1][-1]:.2f})")
    c = confusion_matrix(yte, p); ax = fig_cm[1][i]; ax.imshow(c, cmap="Blues"); ax.set_title(name, fontsize=9)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    for (r, q), v in np.ndenumerate(c): ax.text(q, r, v, ha="center", va="center", fontsize=12)
fig_roc[1].plot([0, 1], [0, 1], "k--"); fig_roc[1].set(title="ROC curves", xlabel="FPR", ylabel="TPR"); fig_roc[1].legend(fontsize=7)
fig_to_html("ROC Curves", fig_roc[0]); fig_to_html("Confusion Matrices", fig_cm[0])
mt = pd.DataFrame(metrics, columns=["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]); add_table("Classification metrics (risk prediction)", mt, index=False)
dt = DecisionTreeClassifier(max_depth=3, random_state=0).fit(Xtr, ytr)
fig, ax = plt.subplots(figsize=(11, 4.5)); plot_tree(dt, feature_names=FEATS, class_names=["Safe", "AtRisk"], filled=True, fontsize=7, ax=ax)
fig_to_html("Decision Tree (interpretable rules)", fig)
log("11 Classification", "best model: " + mt.sort_values("ROC-AUC").iloc[-1].Model)
Output (console)
[11 Classification] best model: Naive Bayes
Results
Table 8: Classification metrics on the test set
Model	Accuracy	Precision	Recall	F1	ROC-AUC
Decision Tree	0.844	0.667	0.756	0.708	0.886
Naive Bayes	0.867	0.672	0.911	0.774	0.943
k-NN (k=7)	0.872	0.72	0.8	0.758	0.926

 
Figure 7: Confusion matrices
 
Figure 8: ROC curves
 
Figure 9: Decision tree
14. Linear Regression
Aim
Predict final score and report R², MAE and RMSE.
Source Code
Xtr_r, Xte_r, ytr_r, yte_r = train_test_split(df[FEATS], df.final_score, test_size=.3, random_state=7)
lr = LinearRegression().fit(Xtr_r, ytr_r); pr_ = lr.predict(Xte_r)
reg = pd.DataFrame({"Metric": ["R²", "MAE", "RMSE"], "Value": [r2_score(yte_r, pr_), mean_absolute_error(yte_r, pr_),
                                                             np.sqrt(mean_squared_error(yte_r, pr_))]})
add_table("Linear regression: predicting final score", reg, index=False)
add_table("Regression coefficients", pd.DataFrame({"Feature": FEATS, "Coefficient": lr.coef_}), index=False)
fig, ax = plt.subplots(figsize=(4.5, 4)); ax.scatter(yte_r, pr_, s=10); ax.plot([0, 100], [0, 100], "r--")
ax.set(title="Actual vs Predicted", xlabel="Actual", ylabel="Predicted"); fig_to_html("Linear Regression", fig)
log("12 Regression", f"R2={r2_score(yte_r, pr_):.3f}")
Output (console)
[12 Regression] R2=0.787
Results
Table 9: Regression performance
Metric	Value
R²	0.787
MAE	3.98
RMSE	5.109

Table 10: Regression coefficients
Feature	Coefficient
attendance	0.012
assignment_pct	0.031
midterm	0.714
study_hours	2.289
lms_logins	-0.012
library_visits	-0.054
wifi_hours	0.141
sleep_hours	0.107
complaints	-1.409

 
Figure 10: Actual vs predicted final score
15. Dashboard
Aim
Build the HTML dashboard with KPIs, tables and all charts.
Source Code
kpi = [("Students", len(df)), ("At-risk", f"{df.at_risk.mean():.0%}"), ("Avg attendance", f"{df.attendance.mean():.1f}%"),
       ("Best model AUC", f"{mt['ROC-AUC'].max():.2f}"), ("Regression R²", f"{reg.Value[0]:.2f}"), ("Risk rules", len(rules))]
html = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Smart Campus Mining Dashboard</title><style>
body{{font-family:system-ui,sans-serif;margin:0;background:#f4f6fa;color:#1c2333}}
header{{background:#1f3a8a;color:#fff;padding:20px 28px}} h1{{margin:0;font-size:22px}}
main{{max-width:1200px;margin:auto;padding:20px}} .k{{display:flex;gap:12px;flex-wrap:wrap}}
.kp{{flex:1;min-width:140px;background:#fff;border-radius:10px;padding:14px;box-shadow:0 1px 3px #0002}}
.kp b{{font-size:24px;display:block;color:#1f3a8a}} .c{{background:#fff;border-radius:10px;padding:16px;margin:16px 0;box-shadow:0 1px 3px #0002;overflow-x:auto}}
h2{{font-size:16px;margin:0 0 10px}} img{{max-width:100%}} table.t{{border-collapse:collapse;font-size:12px}}
.t th,.t td{{border:1px solid #dde;padding:4px 8px}} .t th{{background:#eef1fb}} li{{font-size:13px;margin:3px 0}}
</style></head><body><header><h1>Smart Campus Student Behaviour, Performance &amp; Risk Pattern Mining</h1>
<div style="opacity:.8;font-size:13px">Data-mining pipeline dashboard</div></header><main>
<div class="k">{''.join(f'<div class="kp"><b>{v}</b>{k}</div>' for k, v in kpi)}</div>
<div class="c"><h2>Pipeline log</h2><ul>{''.join(f'<li><b>{s}:</b> {m}</li>' for s, m in LOG)}</ul></div>
{''.join(f'<div class="c"><h2>{t}</h2>{h}</div>' for t, h in TABLES)}
{''.join(f'<div class="c"><h2>{t}</h2><img src="data:image/png;base64,{b}"></div>' for t, b in FIGS)}
</main></body></html>"""
open(f"{OUT}/dashboard.html", "w", encoding="utf-8").write(html)
log("13 Dashboard", f"saved {OUT}/dashboard.html")
Output (console)
[13 Dashboard] saved outputs/dashboard.html
16. Dashboard Output
Running the script writes dashboard.html, which opens in any browser. It shows KPI cards (students, at-risk %, average attendance, best model AUC, regression R², number of rules), the pipeline log, every table and every chart shown in this file.
17. Result Summary and Conclusion
Table 11: Summary of results
Item	Result
Dataset after cleaning and merging	600 students, 14 attributes, 25% at risk
Strongest correlate of final score	Mid-term marks
Frequent itemsets (min support 8%)	660 (Apriori and FP-Growth agree)
Strongest risk rule	Hostel + low LMS + poor result + low study → At-Risk (confidence 0.90, lift 3.6)
Best number of clusters	2 (silhouette 0.199); the low-engagement cluster has 49% at-risk students
Best classifier	Naive Bayes: accuracy 0.87, recall 0.91, ROC-AUC 0.94
Linear regression	R² = 0.79, MAE = 3.98

Conclusion
The project implements the full data-mining process on smart-campus data. Low LMS activity and low study hours are the strongest warning signs of academic risk; students separate into engaged and disengaged groups; Naive Bayes detects at-risk students best; and linear regression predicts final marks to within about 4 marks.
Limitations
•	The dataset is synthetic; results must be validated on real data.
•	Mid-term marks are used as a feature and at_risk is derived from final marks, so classification results are optimistic.
•	Association rules show correlation, not causation.
