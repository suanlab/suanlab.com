"""Independent finite checks of the two declared thermal inverse problems.

Integrate the physical balances with RK4 and quadrature. For the gas, integrate
pressure with respect to physical volume, not the manuscript's closed work
formula. Rational basis moments check the observation null direction separately.
No production calculator is imported. Finite checks are not general certificates.
"""
from math import exp, log, isclose
from fractions import Fraction as F
import json


def close(a, b, tolerance=2e-9):
    assert isclose(a, b, rel_tol=tolerance, abs_tol=tolerance), (a, b)


def quadrature(fn, start, end, steps=2048):
    width = (end-start)/steps
    return width/3 * (fn(start)+fn(end)+sum((4 if i % 2 else 2)*fn(start+i*width) for i in range(1,steps)))


def field(y, C1, C2, K, P):
    transfer=K*(y[0]-y[1])
    return [(P-transfer)/C1, transfer/C2]


def rk4(initial, duration, C1, C2, K, P):
    y=list(initial); dt=duration/2048
    for _ in range(2048):
        a=field(y,C1,C2,K,P)
        b=field([v+dt*w/2 for v,w in zip(y,a)],C1,C2,K,P)
        c=field([v+dt*w/2 for v,w in zip(y,b)],C1,C2,K,P)
        d=field([v+dt*w for v,w in zip(y,c)],C1,C2,K,P)
        y=[v+dt*(aa+2*bb+2*cc+dd)/6 for v,aa,bb,cc,dd in zip(y,a,b,c,d)]
    return y


def closed(t, initial, C1, C2, K, P=0):
    lam=K*(1/C1+1/C2)
    d0=initial[0]-initial[1]
    # All main cases are connected. Disconnected limits are tested separately.
    equilibrium_difference=P/(C1*lam)
    difference=equilibrium_difference+(d0-equilibrium_difference)*exp(-lam*t)
    mean=(C1*initial[0]+C2*initial[1]+P*t)/(C1+C2)
    return [mean+C2/(C1+C2)*difference, mean-C1/(C1+C2)*difference]


thermal_cases=0; entropy_cases=0; reconstructions=0
for parameters in [(2,3,.4),(.7,4,1.1),(5,1,.05)]:
    for initial in [(330,270),(270,330),(300,300)]:
        for scale in [.5,1,2]:
            C1,C2,K=[scale*x for x in parameters]
            a,b=K/C1,K/C2
            for duration in [.2,1,3]:
                unforced=closed(duration,initial,C1,C2,K)
                for u,v in zip(unforced,closed(duration,initial,*parameters)):close(u,v)
                for power in [0,2]:
                    actual=rk4(initial,duration,C1,C2,K,power)
                    expected=closed(duration,initial,C1,C2,K,power)
                    for u,v in zip(actual,expected):close(u,v)
                    close(C1*(expected[0]-initial[0])+C2*(expected[1]-initial[1]),power*duration)
                    def production(t):
                        t1,t2=closed(t,initial,C1,C2,K,power)
                        assert 0<t1<600 and 0<t2<600
                        return K*(t1-t2)**2/(t1*t2)+power*(1/t1-1/600)
                    entropy=C1*log(expected[0]/initial[0])+C2*log(expected[1]/initial[1])-power*duration/600
                    close(quadrature(production,0,duration),entropy)
                    assert entropy>=-1e-12
                    entropy_cases+=1
                if initial[0]!=initial[1]:
                    state=closed(duration/2,initial,C1,C2,K)
                    before=field(state,C1,C2,K,0);after=field(state,C1,C2,K,2)
                    inferred_a=-before[0]/(state[0]-state[1]);inferred_b=before[1]/(state[0]-state[1])
                    inferred_C1=2/(after[0]-before[0]);inferred_K=inferred_a*inferred_C1
                    for u,v in zip([inferred_C1,inferred_K,inferred_K/inferred_b],[C1,K,C2]):close(u,v)
                    close(after[1]-before[1],0)
                    reconstructions+=1
                thermal_cases+=1

# Exact rank and common-scale direction; no detector model is introduced.
J=[[-1,0,1],[0,-1,1]]
assert all(sum(row)==0 for row in J)
assert J[0][0]*J[1][1]-J[0][1]*J[1][0]==1
for K,initial in [(0,(330,270)),(0,(300,300)),(.4,(300,300))]:
    assert field(initial,2,3,K,0)==[0,0]
    assert rk4(initial,3,2,3,K,0)==list(initial)

# Independent exact moments of polynomial f=u-u² and h=u-3u²+2u³.
f=[F(0),F(1),F(-1)];h=[F(0),F(1),F(-3),F(2)]
def poly(coeffs,u):return sum(c*u**i for i,c in enumerate(coeffs))
assert sum(c/F(i+1) for i,c in enumerate(f))==F(1,6)
assert sum(c/F(i+1) for i,c in enumerate(h))==0
assert poly(h,F(0))==poly(h,F(1))==poly(h,F(1,2))==0
assert poly(h,F(1,4))==F(3,32)
assert F(1,6)*F(3,32)==F(1,64)

R=8.31446261815324;gas_cases=0;equal_volume_cases=0
for n,T0,T1,V0,V1 in [(1,300,330,.01,.02),(.5,330,290,.02,.01),(2,300,300,.01,.03)]:
    L=log(V1/V0)
    for b,c in [(0,0),(40,0),(40,80),(40,-80),(-30,60)]:
        assert abs(b)+abs(c)<4*min(T0,T1)
        def temperature(u):return T0+(T1-T0)*u+b*u*(1-u)+c*u*(1-u)*(1-2*u)
        def derivative(u):return T1-T0+b*(1-2*u)+c*(1-6*u+6*u*u)
        def pressure(V):return n*R*temperature(log(V/V0)/L)/V
        work=quadrature(pressure,V0,V1)
        close(work,n*R*L*((T0+T1)/2+b/6))
        def heat_per_volume(V):
            u=log(V/V0)/L
            return n*1.5*R*derivative(u)/(L*V)+pressure(V)
        heat=quadrature(heat_per_volume,V0,V1)
        close(heat,n*1.5*R*(T1-T0)+work)
        entropy=quadrature(lambda V:heat_per_volume(V)/temperature(log(V/V0)/L),V0,V1)
        close(entropy,n*1.5*R*log(T1/T0)+n*R*L)
        inferred_b=6*(work/(n*R*L)-(T0+T1)/2)
        inferred_c=32/3*(temperature(.25)-T0-(T1-T0)/4-3*inferred_b/16)
        close(inferred_b,b,1e-8);close(inferred_c,c,1e-8)
        close(temperature(.5),(T0+T1)/2+b/4)
        gas_cases+=1
for b,c in [(0,0),(40,80),(40,-80)]:
    # Equal volume means no volume work; reversible heat integral is endpoint U.
    close(quadrature(lambda u:1.5*R*(30+b*(1-2*u)+c*(1-6*u+6*u*u)),0,1),1.5*R*30)
    equal_volume_cases+=1

print(json.dumps({'status':'passed','thermalTrajectories':thermal_cases,
    'rk4AndEntropyBudgets':entropy_cases,'calibratedReconstructions':reconstructions,
    'unforcedDegenerateCases':3,'physicalVolumePathQuadratures':gas_cases,
    'equalVolumeHeatCases':equal_volume_cases,'exactPolynomialNullDirection':True}))
