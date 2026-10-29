import numpy as np, sys
from collections import Counter
sys.path.insert(0,'.')
from pursuit_evasion_numpy import fourier_features, forward_dirs
K,t_max,n_t,L,vE,vP=12,4.0,64,2.0,1.0,1.2
tg=np.linspace(0,t_max,n_t+1); phi=fourier_features(tg,t_max,K); dt=t_max/n_t
def load(p):
    z=np.load(p); return ({"B":z["BE"],"b_body":z["bE"],"W":z["WE"]},{"B":z["BP"],"b_body":z["bP"],"W":z["WP"]})
def cum(u,v): umid=0.5*(u[:,:-1]+u[:,1:]); return (dt*v*umid).sum(1)
def l2(u): d=u[:,None]-u[None]; return np.sqrt((d**2).sum(-1).mean(-1))
for path in ["v2_actor.npz","e1b_actor.npz","c1_s1.npz","e7_actor.npz","e7_4k.npz"]:
    try: pE,pP=load(path)
    except Exception as e: print(path,"err",e); continue
    uE,_=forward_dirs(pE,phi); uP,_=forward_dirs(pP,phi); nh=uE.shape[0]
    cls=np.round(cum(uE,vE)/L).astype(int)
    c=Counter(map(tuple,cls)); mult=sorted(c.values(),reverse=True)
    npair=sum(m*(m-1)//2 for m in c.values()); D=l2(uE); wi=[];ac=[]
    for j in range(nh):
        for k in range(j+1,nh): (wi if tuple(cls[j])==tuple(cls[k]) else ac).append(D[j,k])
    wi=np.array(wi);ac=np.array(ac); rr=ac.mean()/wi.mean() if len(wi) and len(ac) else float('nan')
    print(f"{path:16s} n_h={nh} {len(c)} classes mult={mult} within-pairs={npair} L2 within={wi.mean():.3f} across={ac.mean():.3f} ratio={rr:.2f}")
