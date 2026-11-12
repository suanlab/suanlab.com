#!/usr/bin/env python3
"""Supplementary weak-window and entropy-jump checks; standard library only.
Exact finite examples complement the manuscript derivations; no product imports.
"""
from fractions import Fraction as F
from collections import Counter
import math,json


def mass(t,left,right,wl,wr,s,rhol,rhor,area):
    l,r=left+wl*t,right+wr*t
    if r<=l or min(rhol,rhor,area)<=0:
        raise ValueError('Positive ordered window, densities and area required')
    return area*(rhol*(r-l)+(rhor-rhol)*(max(r-s*t,F(0))-max(l-s*t,F(0))))


def integrated_flux(t0,t1,left,right,wl,wr,s,rhol,rhor,area,transport=None):
    if t1<t0:
        return -integrated_flux(t1,t0,left,right,wl,wr,s,rhol,rhor,area,transport)
    velocity=s if transport is None else transport
    cuts={t0,t1}
    for x,w in [(left,wl),(right,wr)]:
        if w!=s:
            crossing=-x/(w-s)
            if t0<crossing<t1:
                cuts.add(crossing)
    result=F(0);cuts=sorted(cuts)
    # Density traces on open intervals; coincident material faces have zero relative flux.
    for a,b in zip(cuts,cuts[1:]):
        t=(a+b)/2;l,r=left+wl*t,right+wr*t
        rho_l=rhol if l<s*t else rhor
        rho_r=rhol if r<s*t else rhor
        result+=area*(rho_l*(velocity-wl)-rho_r*(velocity-wr))*(b-a)
    return result


def conserved(r,u,p,g):
    return [r,r*u,p/(g-1)+r*u*u/2]


def flux(r,u,p,g):
    U=conserved(r,u,p,g)
    return [r*u,r*u*u+p,(U[2]+p)*u]


def verify():
    out=Counter()
    for v in [F(-1),F(0),F(1),F(2)]:
        for wl,wr in [(F(0),F(0)),(v,v),(F(-1,4),F(1,4))]:
            for t0,t1 in [(F(0),F(1)),(F(1),F(0)),(F(1,4),F(3,4)),(F(1,2),F(1,2))]:
                for rhor in [F(1),F(3)]:
                    args=(F(-1,2),F(1,2),wl,wr,v,F(1),rhor,F(2))
                    assert mass(t1,*args)-mass(t0,*args)==integrated_flux(t0,t1,*args)
                    out['exactMovingWindowBudgets']+=1
    for v in [F(-1),F(0),F(1)]:
        args=(F(0),F(1),v,v,v,F(1),F(3),F(1))
        assert mass(F(1),*args)==mass(F(0),*args)==3
        assert integrated_flux(F(0),F(1),*args)==0
        out['coincidentMaterialFaceCases']+=1
    for speed in [F(-1,2),F(0),F(1,2)]:
        args=(F(-1,2),F(1,2),F(0),F(0),speed,F(1),F(3),F(1))
        t0,t1=F(0),F(1,4)
        defect=mass(t1,*args)-mass(t0,*args)-integrated_flux(t0,t1,*args,transport=F(1))
        assert defect==(1-speed)*2*(t1-t0) and defect!=0
        out['missedJumpSourceNegativeControls']+=1
    try:
        mass(F(0),F(0),F(0),F(0),F(0),F(1),F(1),F(3),F(1))
    except ValueError:
        out['zeroWidthRejections']+=1
    else:
        raise AssertionError('Zero-width observation accepted')
    for g in [F(7,5),F(5,3),F(2)]:
        for m in [F(5,4),F(3,2),F(2),F(3),F(4)]:
            z=m*m;r=(g+1)*z/((g-1)*z+2);P=(2*g*z-(g-1))/(g+1)
            r1,p1,v1=F(1),1/g,m;r2,p2,v2=r,p1*P,m/r
            entropy=math.log(float(P))-float(g)*math.log(float(r))
            assert entropy>0 and (g-1)*z+2<2*g*z-(g-1)
            # Differentiate the two logarithms, then compare with the manifestly positive factorization.
            dP=2*g/(g+1);dr=2*(g+1)/((g-1)*z+2)**2
            derivative=dP/P-g*dr/r
            positive=2*g*(g-1)*(z-1)**2/(z*((g-1)*z+2)*(2*g*z-(g-1)))
            assert derivative==positive and positive>0
            out['exactEntropyDerivativeCases']+=1
            h1=g*p1/((g-1)*r1);h2=g*p2/((g-1)*r2)
            assert h1+v1*v1/2==h2+v2*v2/2
            assert h2-h1==(v1*v1-v2*v2)/2>0
            out['exactEnthalpyConversionCases']+=1
            for speed in [F(-2),F(0),F(3,4),F(3)]:
                left=(r1,v1+speed,p1,g);right=(r2,v2+speed,p2,g)
                UL,UR=conserved(*left),conserved(*right);FL,FR=flux(*left),flux(*right)
                assert all(FR[i]-FL[i]==speed*(UR[i]-UL[i]) for i in range(3))
                assert all(FL[i]-FR[i]==speed*(UL[i]-UR[i]) for i in range(3))
                assert r1*v1*entropy>0 and r2*v2*(-entropy)<0
                out['exactBoostedShockJumpCases']+=1
                out['reversedConservationPassesEntropyFails']+=1
                base=conserved(r1,v1,p1,g)
                assert UL[1]==base[1]+speed*base[0]
                assert UL[2]==base[2]+speed*base[1]+speed*speed*base[0]/2
                out['exactStoredQuantityBoostCases']+=1
    return dict(sorted(out.items()))


if __name__=='__main__':
    print(json.dumps(verify(),sort_keys=True))
