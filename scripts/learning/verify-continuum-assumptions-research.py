#!/usr/bin/env python3
"""Exact finite kinetic-energy budgets and paired-parent counting geometry.
Standard library only, no product imports. These cases complement the manuscript proofs.
"""
from fractions import Fraction as F
import json

def weights(z,h):
    if abs(z)>=h:return F(0),F(0)
    return 15/(16*h)*(1-z*z/(h*h))**2,-15*z/(4*h**3)*(1-z*z/(h*h))

def raw(particles,x,t,h):
    values=[];dx=[];dt=[]
    for order in range(4):
        value=gx=gt=F(0)
        for mass,initial,v in particles:
            w,wx=weights(x-initial-v*t,h)
            value+=mass*v**order*w;gx+=mass*v**order*wx;gt-=mass*v**(order+1)*wx
        values.append(value);dx.append(gx);dt.append(gt)
    return values,dx,dt

def exact_parent_integrals(ell,a,kappa):
    cuts=sorted({x+sign*a/2 for x in [F(0),ell,2*ell] for sign in [-1,1]})
    raw=[F(0)]*3
    def count(c,lo,hi):return int(lo<=c-a/2<hi)+int(lo<=c+a/2<hi)
    for lo,hi in zip(cuts,cuts[1:]):
        c=(lo+hi)/2;w=count(c,F(0),ell);v=count(c,ell,2*ell)
        for i,value in enumerate([w,w*w,w*v]):raw[i]+=(hi-lo)*value
    return [kappa*ell*ell*x for x in raw]

def energy_checks():
    p=[(F(1),F(-1,4),F(-1)),(F(2),F(0),F(1,2)),(F(3),F(1,2),F(2))];checks=[];empty=0
    for x in [F(-1,4),F(0),F(1,4),F(1,2),F(3,4)]:
        for t in [F(-1,4),F(0),F(1,4)]:
            v,dx,dt=raw(p,x,t,F(3,4));rho,j,Q,M3=v
            if rho==0:
                assert v==[0,0,0,0] and dx==[0,0,0,0] and dt==[0,0,0,0]
                empty+=1
                continue
            assert rho>0
            u=j/rho;ux=(dx[1]*rho-j*dx[0])/rho**2;ut=(dt[1]*rho-j*dt[0])/rho**2
            K=j*j/(2*rho);Kx=j*dx[1]/rho-j*j*dx[0]/(2*rho*rho);Kt=j*dt[1]/rho-j*j*dt[0]/(2*rho*rho)
            S=Q-j*j/rho;Sx=dx[2]-2*Kx;St=dt[2]-2*Kt
            q=(M3-3*u*Q+2*rho*u**3)/2
            qx=(dx[3]-3*ux*Q-3*u*dx[2]+2*dx[0]*u**3+6*rho*u*u*ux)/2
            assert dt[2]+dx[3]==0
            assert M3/2==(Q/2+S)*u+q
            assert Kt+(Kx+Sx)*u+(K+S)*ux==S*ux
            assert St/2+Sx*u/2+S*ux/2+qx==-S*ux
            for b in [F(-2),F(0),F(3,2)]:
                boosted=[(m,X,velocity+b) for m,X,velocity in p];vb,_,_=raw(boosted,x+b*t,t,F(3,4));rb,jb,Qb,M3b=vb;ub=jb/rb
                assert rb==rho and ub==u+b and Qb-jb*jb/rb==S
                assert (M3b-3*ub*Qb+2*rb*ub**3)/2==q
                assert Qb/2==Q/2+b*j+b*b*rho/2
            checks.append(dict(x=str(x),t=str(t),rho=str(rho),mean=str(u),dispersion=str(S),centralThirdFlux=str(q),transfer=str(S*ux)))
    assert len(checks)==14 and empty==1
    return dict(exactLocalEnergyBudgets=len(checks),exactEnergyBoosts=3*len(checks),emptySupportCases=empty)

def counting_checks():
    rows=[]
    for ell in [F(1,4),F(1,2),F(1),F(3,2),F(2),F(4)]:
        for a in [F(0),F(1,8),F(1,2),F(1),F(2),F(3),F(4),F(8)]:
            for n in [F(1,4),F(1),F(4)]:
                volume=ell**3;overlap=max(ell-a,F(0))*ell**2;kappa=n/2
                mean,var,cov=exact_parent_integrals(ell,a,kappa)
                assert mean==n*volume and var==n*(volume+overlap)
                assert cov==kappa*ell**2*max(ell-abs(a-ell),F(0))
                one_rate=2*kappa*(volume-overlap);two_rate=kappa*overlap
                assert mean==one_rate+2*two_rate and var==one_rate+4*two_rate
                assert one_rate>=0 and two_rate>=0 and 1<=var/mean<=2
                rows.append(dict(ell=str(ell),a=str(a),n=str(n),mean=str(mean),variance=str(var),adjacentCovariance=str(cov)))
    # A positive gap a=ell gives Poisson marginal counts and nonzero neighbouring covariance.
    assert exact_parent_integrals(F(1),F(1),F(2))==[4,4,2]
    # Exact sign bracket for count tolerance n=4, epsilon=1/4, a=1.
    def polynomial(ell):return ell**4/4-2*ell+1
    lo,hi=F(7,4),F(15,8);assert polynomial(lo)<0<polynomial(hi)
    for _ in range(80):
        mid=(lo+hi)/2
        if polynomial(mid)<0:lo=mid
        else:hi=mid
    assert polynomial(lo)<0<polynomial(hi)
    return dict(exactParentGeometryCases=len(rows),exactThresholdBisections=80,dependentPoissonMarginalCases=1)

def verify():
    out={**energy_checks(),**counting_checks()}
    count=0
    for a in [F(1,4),F(1,2),F(1),F(2),F(4)]:
        wm,wp=1/(1+a*a),a*a/(1+a*a)
        vminus,vplus=-a,1/a
        assert wm>0 and wp>0 and wm+wp==1
        assert wm*vminus+wp*vplus==0
        assert wm*vminus**2+wp*vplus**2==1
        assert wm*vminus**3+wp*vplus**3==1/a-a
        count+=1
    out['positiveSameLowerMomentPreparations']=count
    return dict(sorted(out.items()))

if __name__=='__main__':
    print(json.dumps(verify(),sort_keys=True))
