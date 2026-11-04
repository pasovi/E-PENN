import numpy as np, sys
sys.path.insert(0,'.')
from pursuit_evasion_numpy import (fourier_features, forward_dirs, rollout,
    forward_loss, backward_loss, backward_rollout, backward_dirs, init_params,
    sample_evader_ICs, Adam)
K,t_max,n_t,L,vE,vP,eps=12,4.0,64,2.0,1.0,1.2,0.08
tg=np.linspace(0,t_max,n_t+1); phi=fourier_features(tg,t_max,K)
x_E0=sample_evader_ICs(16,L,0.4,seed=0)
d0v=(x_E0+L/2)%L-L/2; dhat=d0v/np.clip(np.linalg.norm(d0v,axis=-1,keepdims=True),1e-9,None)
uE_opt=np.repeat(dhat[:,None,:],n_t+1,axis=1)
R=(vP-vE)*t_max+eps; cap=np.sqrt((d0v**2).sum(-1))<=R
print("capturable heads:", int(cap.sum()), "of 16")
def PR(e): e=np.clip(e,0,None); return float(e.sum()**2/max((e**2).sum(),1e-12))
def strat_PR(u_or_params, idx=None):
    if isinstance(u_or_params,dict): _,c=forward_dirs(u_or_params,phi); Z=c["z_head"]
    else: Z=u_or_params
    if idx is not None: Z=Z[idx]
    nh,T,_=Z.shape; M=np.transpose(Z,(1,0,2)).reshape(T,2*nh); Mc=M-M.mean(0,keepdims=True)
    return PR(np.linalg.eigvalsh(Mc.T@Mc/T))
def head_PR(u, idx=None):
    if idx is not None: u=u[idx]
    nh,T,_=u.shape; M=u.reshape(nh,T*2); Mc=M-M.mean(0,keepdims=True)
    return PR(np.linalg.eigvalsh(Mc@Mc.T/(T*2)))
# baseline actor (v2)
z=np.load("v2_actor.npz"); pP_a={"B":z["BP"],"b_body":z["bP"],"W":z["WP"]}
uP_a,cA=forward_dirs(pP_a,phi)
# oracle pursuer (train seed 0)
rng=np.random.default_rng(0); pP=init_params(16,16,2*K+1,rng); opt=Adam(pP,1e-2)
for _ in range(6000):
    uP,cP=forward_dirs(pP,phi); xE,xP=rollout(uE_opt,uP,x_E0,L,vE,vP,t_max,n_t)
    _,_,cL=forward_loss(xE,xP,L,eps,5e-3,t_max,n_t,2.0,lam_d=5.0)
    dxE,dxP=backward_loss(1.0,cL,16,2.0,t_max,is_pursuer=True)
    _,duP=backward_rollout(dxE,dxP,L,vE,vP,t_max,n_t); opt.step(pP,backward_dirs(pP,cP,duP),sign=-1.0)
uP_o,_=forward_dirs(pP,phi)
ci=np.where(cap)[0]
print("\n           metric            ALL-16     CAPTURABLE-4")
print(f"saddle     head_PR         {head_PR(uE_opt):6.2f}      {head_PR(uE_opt,ci):6.2f}")
print(f"actor(v2)  head_PR         {head_PR(uP_a):6.2f}      {head_PR(uP_a,ci):6.2f}")
print(f"oracle     head_PR         {head_PR(uP_o):6.2f}      {head_PR(uP_o,ci):6.2f}")
print(f"actor(v2)  PR(Z)           {strat_PR(pP_a):6.2f}      {strat_PR(cA['z_head'],ci):6.2f}")
print(f"oracle     PR(Z)           {strat_PR(pP):6.2f}      {strat_PR(forward_dirs(pP,phi)[1]['z_head'],ci):6.2f}")
# also: oracle capture on capturable heads
xE,xP=rollout(uE_opt,uP_o,x_E0,L,vE,vP,t_max,n_t)
dmin=np.sqrt((((xE-xP+L/2)%L-L/2)**2).sum(-1)).min(1)
print("oracle capture on capturable heads:", float((dmin[cap]<=eps).mean()))

# ---- winding within-class distance spread ----
print("\n=== winding within-class distances (baseline v2) ===")
def cum(u,v): um=0.5*(u[:,:-1]+u[:,1:]); return (dt:=t_max/n_t)*v*um
pE_a={"B":z["BE"],"b_body":z["bE"],"W":z["WE"]}; uE_a,_=forward_dirs(pE_a,phi)
net=(t_max/n_t)*vE*0.5*(uE_a[:,:-1]+uE_a[:,1:]); net=net.sum(1)
cls=np.round(net/L).astype(int)
Dd=uE_a[:,None]-uE_a[None]; Dm=np.sqrt((Dd**2).sum(-1).mean(-1))
from collections import defaultdict
byc=defaultdict(list)
for j in range(16): byc[tuple(cls[j])].append(j)
wi=[]
for c_,members in byc.items():
    for a in range(len(members)):
        for b in range(a+1,len(members)):
            wi.append((c_,round(Dm[members[a],members[b]],3)))
print("within-class pairs (class, L2dist):", wi)
ac=[Dm[j,k] for j in range(16) for k in range(j+1,16) if tuple(cls[j])!=tuple(cls[k])]
print(f"within: n={len(wi)} vals={[w[1] for w in wi]}  mean={np.mean([w[1] for w in wi]):.3f} max={max(w[1] for w in wi):.3f}")
print(f"across: n={len(ac)} mean={np.mean(ac):.3f} min={min(ac):.3f}")
