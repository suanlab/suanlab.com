"""Reproduce independent exact/60-digit action integrals; no production imports.
Run: python3 scripts/learning/generate-stationary-action-references.py
Requires mpmath only for the oscillator quadrature (fixtures are portable JSON).
"""
from fractions import Fraction as F
from pathlib import Path
import json, random
import mpmath as mp
mp.mp.dps=60
rng=random.Random(20261104)
def label(x): return str(x)
def boole(fn, end=F(1)):
    return end*sum(w*fn(end*F(j,4)) for j,w in enumerate([7,32,12,32,7]))/90

def free(p):
    m,tc,ac,bc,Ac,Bc,gc,u0c,u1c,Cc,Dc,ec,sc=p
    T=F(tc,10);a,b,A,B,g,u0,u1,C,D=[F(x,10) for x in [ac,bc,Ac,Bc,gc,u0c,u1c,Cc,Dc]]
    eps=F(ec,100);s=F(sc,100)
    def shape(s,a,b,A,B):
        q=a+(b-a)*s+A*s*(1-s)+B*s*(1-s)*(2*s-1)
        v=((b-a)+A*(1-2*s)+B*(-6*s*s+6*s-1))/T
        acc=(-2*A+B*(6-12*s))/(T*T)
        return q,v,acc
    def L(s):
        q,v,_=shape(s,a,b,A,B);return m*v*v/2+g*q*v
    def varied(s):
        q,v,_=shape(s,a+eps*u0,b+eps*u1,A+eps*C,B+eps*D);return m*v*v/2+g*q*v
    def first(s):
        q,v,_=shape(s,a,b,A,B);xi,dv,_=shape(s,u0,u1,C,D)
        return m*v*dv+g*(v*xi+q*dv)
    def second(s):
        xi,dv,_=shape(s,u0,u1,C,D);return m*dv*dv+2*g*xi*dv
    q,v,acc=shape(s,a,b,A,B)
    out=dict(action=T*boole(L),variedAction=T*boole(varied),firstVariation=T*boole(first),secondVariation=T*boole(second),residualMeanSquare=boole(lambda z:(m*shape(z,a,b,A,B)[2])**2))
    query=dict(position=q,velocity=v,acceleration=acc,kinetic=m*v*v/2,canonicalMomentum=m*v+g*q,cumulativeAction=T*boole(L,s),eulerLagrangeResidual=m*acc)
    return dict(p=p,exact={k:label(v) for k,v in out.items()},query={k:label(v) for k,v in query.items()})

def harmonic(p):
    m,f,tc,ac,bc,cc,dc,ec,sc=p
    T=mp.mpf(tc)/10;A=mp.mpf(ac)/10;B=mp.mpf(bc)/10;C=mp.mpf(cc)/10;D=mp.mpf(dc)/10;e=mp.mpf(ec)/100;s=mp.mpf(sc)/100;w=mp.pi*f/4;k=m*w*w
    def state(z,a=A,b=B):
        q=a*mp.sin(mp.pi*z)+b*mp.sin(3*mp.pi*z)
        v=mp.diff(lambda x:a*mp.sin(mp.pi*x/T)+b*mp.sin(3*mp.pi*x/T),z*T)
        acc=mp.diff(lambda x:a*mp.sin(mp.pi*x/T)+b*mp.sin(3*mp.pi*x/T),z*T,2)
        return q,v,acc
    def L(z,a=A,b=B):
        q,v,_=state(z,a,b);return m*v*v/2-k*q*q/2
    def first(z):
        q,v,_=state(z);xi,dv,_=state(z,C,D);return m*v*dv-k*q*xi
    def second(z):
        xi,dv,_=state(z,C,D);return m*dv*dv-k*xi*xi
    q,v,acc=state(s)
    out=dict(action=T*mp.quad(L,[0,mp.mpf('.5'),1]),variedAction=T*mp.quad(lambda z:L(z,A+e*C,B+e*D),[0,mp.mpf('.5'),1]),firstVariation=T*mp.quad(first,[0,mp.mpf('.5'),1]),secondVariation=T*mp.quad(second,[0,mp.mpf('.5'),1]),residualMeanSquare=mp.quad(lambda z:(m*state(z)[2]+k*state(z)[0])**2,[0,mp.mpf('.5'),1]))
    query=dict(position=q,velocity=v,acceleration=acc,kinetic=m*v*v/2,potential=k*q*q/2,eulerLagrangeResidual=m*acc+k*q,cumulativeAction=T*mp.quad(L,[0,s]))
    return dict(p=p,numeric={key:mp.nstr(value,55) for key,value in out.items()},query={key:mp.nstr(value,55) for key,value in query.items()})
free_ps=[[1,10,0,10,0,0,0,0,0,10,0,50,50],[1,10,0,10,0,0,-20,0,10,0,0,50,0],[1,10,0,10,10,0,0,0,0,0,10,50,50]]
for _ in range(80): free_ps.append([rng.randint(a,b) for a,b in [(1,4),(5,40),*([(-20,20)]*4),(-30,30),*([(-10,10)]*4),(-100,100),(0,100)]])
harmonic_ps=[[1,2,t,10,0,10,0,50,s] for t in [10,20,25] for s in [0,17,50,100]]
for _ in range(32): harmonic_ps.append([rng.randint(a,b) for a,b in [(1,4),(0,4),(5,25),(-20,20),(-20,20),(-10,10),(-10,10),(-100,100),(0,100)]])
result=dict(methods=dict(free='Fraction Boole quadrature exact through degree five; direct prescribed derivatives',harmonic='mpmath 60-digit direct Lagrangian/variation quadrature; independent differentiation'),free=[free(p) for p in free_ps],harmonic=[harmonic(p) for p in harmonic_ps])
target=Path(__file__).resolve().parents[1]/'fixtures'/'stationary-action-independent.json'
target.write_text(json.dumps(result,indent=2)+'\n')
print(f'{len(free_ps)} exact free-particle and {len(harmonic_ps)} 60-digit harmonic references -> {target.name}')
