"""PR(Z) for the oracle-trained pursuer vs the mediocre self-play actor
(Pedro Emb #12): is PR(Z)~2 a property of near-optimal (saddle) play, or a
coincidence of the 23%-capture equilibrium?
"""
import os, sys, json, numpy as np
sys.path.insert(0,'.')
from pursuit_evasion_numpy import (TrainConfig, fourier_features, forward_dirs,
    backward_dirs, rollout, backward_rollout, forward_loss, backward_loss,
    init_params, sample_evader_ICs, Adam)
K,t_max,n_t,L,vE,vP,eps=12,4.0,64,2.0,1.0,1.2,0.08
tg=np.linspace(0,t_max,n_t+1); phi=fourier_features(tg,t_max,K)
x_E0=sample_evader_ICs(16,L,0.4,seed=0)
d0=(x_E0+L/2)%L-L/2; dhat=d0/np.clip(np.linalg.norm(d0,axis=-1,keepdims=True),1e-9,None)
uE_opt=np.repeat(dhat[:,None,:],n_t+1,axis=1)
R=(vP-vE)*t_max+eps; capturable=np.sqrt((d0**2).sum(-1))<=R

def PR(e): e=np.clip(e,0,None); return float(e.sum()**2/max((e**2).sum(),1e-12))
def strat_PR(params):                      # PR of temporal cov of stacked Z=W H
    _,c=forward_dirs(params,phi); Z=c["z_head"]; nh,T,_=Z.shape
    M=np.transpose(Z,(1,0,2)).reshape(T,2*nh); Mc=M-M.mean(0,keepdims=True)
    return PR(np.linalg.eigvalsh(Mc.T@Mc/T))
def head_PR(u):                            # PR of the 16 unit-heading curves
    nh,T,_=u.shape; M=u.reshape(nh,T*2); Mc=M-M.mean(0,keepdims=True)
    return PR(np.linalg.eigvalsh(Mc@Mc.T/(T*2)))

def train_oracle(seed,N=6000):
    rng=np.random.default_rng(seed); D=2*K+1
    pP=init_params(16,16,D,rng); optP=Adam(pP,1e-2)
    for _ in range(N):
        uP,cP=forward_dirs(pP,phi)
        xE,xP=rollout(uE_opt,uP,x_E0,L,vE,vP,t_max,n_t)
        _,_,cL=forward_loss(xE,xP,L,eps,5e-3,t_max,n_t,2.0,lam_d=5.0)
        dxE,dxP=backward_loss(1.0,cL,16,2.0,t_max,is_pursuer=True)
        _,duP=backward_rollout(dxE,dxP,L,vE,vP,t_max,n_t)
        optP.step(pP,backward_dirs(pP,cP,duP),sign=-1.0)
    uP,_=forward_dirs(pP,phi)
    xE,xP=rollout(uE_opt,uP,x_E0,L,vE,vP,t_max,n_t)
    dmin=np.sqrt((((xE-xP+L/2)%L-L/2)**2).sum(-1)).min(1)
    cap=float((dmin[capturable]<=eps).mean())
    return {"seed":seed,"cap":round(cap,2),"prZ_P":round(strat_PR(pP),2),"prHead_P":round(head_PR(uP),2)}

if __name__=="__main__":
    if len(sys.argv)>1 and sys.argv[1]=="baseline":
        z=np.load("v2_actor.npz")
        pE={"B":z["BE"],"b_body":z["bE"],"W":z["WE"]}; pP={"B":z["BP"],"b_body":z["bP"],"W":z["WP"]}
        uP,_=forward_dirs(pP,phi)
        print(json.dumps({"what":"baseline_actor(v2)","prZ_P":round(strat_PR(pP),2),
            "prZ_E":round(strat_PR(pE),2),"prHead_P":round(head_PR(uP),2)}))
        # analytic saddle: pursuer optimal heading = constant dhat_j
        print(json.dumps({"what":"analytic_saddle","prHead_P_constDir":round(head_PR(uE_opt),2)}))
    else:
        print(json.dumps(train_oracle(int(sys.argv[1]))))
