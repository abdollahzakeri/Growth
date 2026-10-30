"""R16: looser fragment grouping for a leakage sensitivity check: in addition to the r02 links (IoU >= 0.3, any gap),
link every pair of sequences whose masks overlap with IoU >= 0.1 within 24 h (end of one -> start of the other).
Usage: python r16_loose_groups.py <in seqs.pkl> <tracks.pkl> <out seqs.pkl>
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import sys, pickle
import numpy as np
from growth.common import to_poly
INP, TR, OUT = sys.argv[1:4]
D = pickle.load(open(INP, "rb")); S = D["seqs"]; T = pickle.load(open(TR, "rb"))["tracks"]
tid = {t["id"]: i for i, t in enumerate(T)}
par = {}
def find(x):
    par.setdefault(x, x)
    while par[x] != x:
        par[x] = par[par[x]]; x = par[x]
    return x
for s in S:
    find(s["group"])
idx = [tid[s["id"]] for s in S]
lp = [to_poly(T[i]["last_pts"]) for i in idx]; fp = [to_poly(T[i]["first_pts"]) for i in idx]
lf = np.array([T[i]["frame"][-1] for i in idx]); ff = np.array([T[i]["frame"][0] for i in idx])
n_new = 0
for b in range(len(S)):
    if fp[b] is None: continue
    for a in np.where((lf < ff[b]) & (ff[b] - lf <= 96))[0]:
        if lp[a] is None: continue
        i = lp[a].intersection(fp[b]).area
        if i <= 0: continue
        if i / (lp[a].area + fp[b].area - i) >= 0.1:
            ga, gb = find(S[a]["group"]), find(S[b]["group"])
            if ga != gb:
                par[gb] = ga; n_new += 1
for s in S:
    s["group"] = find(s["group"])
D["stats"]["n_groups"] = len({s["group"] for s in S}); D["stats"]["loose_merges"] = n_new
pickle.dump(D, open(OUT, "wb")); print(D["stats"]["n_groups"], "groups,", n_new, "merges")
