"""Exact grouped-Shapley attribution of pursuit capability to input-feature
bands, for an oracle-trained vs a self-play-trained pursuer (Ignatenko's SHAP
idea).  Demonstrates that Shapley DIAGNOSES the oracle gap (locates which
feature bands each pursuer relies on) but does not CLOSE it (it is computed on
already-trained networks; attribution changes no weights).

Players (feature groups) over phi(t) = [sin_k, cos_k (k=1..12), t/t_max]:
  linear = {t/t_max};  low = k1-4;  mid = k5-8;  high = k9-12.
Coalition value v(S) = pursuit performance using only the bands in S (other
feature columns masked to their time-mean), pursuer rolled out vs the FIXED
analytic-optimal evader (same task for both pursuers -> a fair comparison).
Exact Shapley over the 4 groups (16 coalitions).
"""
import os, sys, json, itertools, numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from pursuit_evasion_numpy import (TrainConfig, fourier_features, forward_dirs,
    backward_dirs, rollout, backward_rollout, forward_loss, backward_loss,
    init_params, sample_evader_ICs, Adam)

N=int(os.environ.get("SHAP_ITERS",5000)); SEED=int(sys.argv[1]) if len(sys.argv)>1 else 0
cfg=TrainConfig(n_h=16,n_b=16,n_freqs=12,t_max=4.0,n_t=64,L=2.0,v_E=1.0,v_P=1.2,
                eps=0.08,tau=5e-3,no_capture_penalty=2.0,lam_d=5.0,seed=0)
tg=np.linspace(0,cfg.t_max,cfg.n_t+1); phi=fourier_features(tg,cfg.t_max,cfg.n_freqs)
x_E0=sample_evader_ICs(cfg.n_h,cfg.L,0.4,seed=0)
d=(x_E0+cfg.L/2)%cfg.L-cfg.L/2
dhat=d/np.clip(np.linalg.norm(d,axis=-1,keepdims=True),1e-9,None)
uE_opt=np.repeat(dhat[:,None,:],cfg.n_t+1,axis=1)
R=(cfg.v_P-cfg.v_E)*cfg.t_max+cfg.eps
d0=np.sqrt(((d)**2).sum(-1)); capturable=d0<=R

def step(pP,optP,pE=None,optE=None,uEf=None,who="P"):
    uP,cP=forward_dirs(pP,phi)
    uE = uEf if uEf is not None else forward_dirs(pE,phi)[0]
    xE,xP=rollout(uE,uP,x_E0,cfg.L,cfg.v_E,cfg.v_P,cfg.t_max,cfg.n_t)
    _,_,cL=forward_loss(xE,xP,cfg.L,cfg.eps,cfg.tau,cfg.t_max,cfg.n_t,cfg.no_capture_penalty,lam_d=cfg.lam_d)
    ip=(who=="P"); dxE,dxP=backward_loss(1.0,cL,cfg.n_h,cfg.no_capture_penalty,cfg.t_max,is_pursuer=ip)
    duE,duP=backward_rollout(dxE,dxP,cfg.L,cfg.v_E,cfg.v_P,cfg.t_max,cfg.n_t)
    if ip: optP.step(pP,backward_dirs(pP,cP,duP),sign=-1.0)
    else: optE.step(pE,backward_dirs(pE,cE if False else forward_dirs(pE,phi)[1],duE),sign=-1.0)

def train_oracle(seed):
    rng=np.random.default_rng(seed); D=2*cfg.n_freqs+1
    pP=init_params(cfg.n_h,cfg.n_b,D,rng); optP=Adam(pP,1e-2)
    for _ in range(N): step(pP,optP,uEf=uE_opt,who="P")
    return pP

def train_selfplay(seed):     # honest 1:1 simultaneous (the arm with the gap)
    rng=np.random.default_rng(seed); D=2*cfg.n_freqs+1
    pP=init_params(cfg.n_h,cfg.n_b,D,rng); pE=init_params(cfg.n_h,cfg.n_b,D,rng)
    optP=Adam(pP,1e-2); optE=Adam(pE,1e-3)
    for _ in range(N):
        uP,cP=forward_dirs(pP,phi); uE,cE=forward_dirs(pE,phi)
        xE,xP=rollout(uE,uP,x_E0,cfg.L,cfg.v_E,cfg.v_P,cfg.t_max,cfg.n_t)
        _,_,cL=forward_loss(xE,xP,cfg.L,cfg.eps,cfg.tau,cfg.t_max,cfg.n_t,cfg.no_capture_penalty,lam_d=cfg.lam_d)
        dxEp,dxPp=backward_loss(1.0,cL,cfg.n_h,cfg.no_capture_penalty,cfg.t_max,True)
        _,duP=backward_rollout(dxEp,dxPp,cfg.L,cfg.v_E,cfg.v_P,cfg.t_max,cfg.n_t)
        dxEe,dxPe=backward_loss(1.0,cL,cfg.n_h,cfg.no_capture_penalty,cfg.t_max,False)
        duE,_=backward_rollout(dxEe,dxPe,cfg.L,cfg.v_E,cfg.v_P,cfg.t_max,cfg.n_t)
        optP.step(pP,backward_dirs(pP,cP,duP),sign=-1.0)
        optE.step(pE,backward_dirs(pE,cE,duE),sign=-1.0)
    return pP

# feature-column groups
cols={"linear":[24],
      "low":[0,1,2,3, 12,13,14,15],
      "mid":[4,5,6,7, 16,17,18,19],
      "high":[8,9,10,11, 20,21,22,23]}
groups=["linear","low","mid","high"]
phi_mean=phi.mean(0,keepdims=True)

def masked_phi(active):
    keep=set()
    for g in active: keep|=set(cols[g])
    m=phi.copy()
    for c in range(phi.shape[1]):
        if c not in keep: m[:,c]=phi_mean[0,c]
    return m

def perf(pP,active):
    m=masked_phi(active); uP,_=forward_dirs(pP,m)
    xE,xP=rollout(uE_opt,uP,x_E0,cfg.L,cfg.v_E,cfg.v_P,cfg.t_max,cfg.n_t)
    disp=(xE-xP+cfg.L/2)%cfg.L-cfg.L/2; dmin=np.sqrt((disp**2).sum(-1)).min(1)
    cap=float((dmin[capturable]<=cfg.eps).mean()); gap=float(dmin[capturable].mean())
    return cap,gap

def shapley(pP,which):   # which: 0=cap (max), 1=neg-gap
    val={}
    for r in range(len(groups)+1):
        for S in itertools.combinations(groups,r):
            c,g=perf(pP,set(S)); val[frozenset(S)]=(c if which==0 else -g)
    n=len(groups); phe={}; import math
    for gi in groups:
        rest=[x for x in groups if x!=gi]; tot=0.0
        for r in range(len(rest)+1):
            for S in itertools.combinations(rest,r):
                S=frozenset(S); w=math.factorial(len(S))*math.factorial(n-len(S)-1)/math.factorial(n)
                tot+=w*(val[S|{gi}]-val[S])
        phe[gi]=tot
    return phe, val[frozenset()], val[frozenset(groups)]

pP_o=train_oracle(SEED); pP_s=train_selfplay(SEED)
out={"seed":SEED,"iters":N}
for name,pP in [("oracle",pP_o),("selfplay",pP_s)]:
    capF,gapF=perf(pP,set(groups)); capE,gapE=perf(pP,set())
    sh_cap,_,_=shapley(pP,0); sh_gap,_,_=shapley(pP,1)
    out[name]={"cap_full":round(capF,3),"gap_full":round(gapF,4),
               "cap_baseline(DC)":round(capE,3),"gap_baseline(DC)":round(gapE,4),
               "shap_cap":{k:round(v,3) for k,v in sh_cap.items()},
               "shap_gapred":{k:round(v,4) for k,v in sh_gap.items()}}
print(json.dumps(out,indent=2))
json.dump(out,open(os.path.join(HERE,f"shap_seed{SEED}.json"),"w"),indent=2)
