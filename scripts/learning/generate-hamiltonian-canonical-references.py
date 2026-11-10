"""Independent symbolic Legendre elimination, 60-digit derivatives and exact free quadrature.
Run: python3 scripts/learning/generate-hamiltonian-canonical-references.py
Requires sympy and mpmath; Node tests consume the portable JSON only.
"""
from pathlib import Path
from fractions import Fraction as Q
import json,random
import sympy as s
import mpmath as mp
mp.mp.dps=60;rng=random.Random(20261104428)
m,r=s.symbols('m r',positive=True);k=s.symbols('k',nonnegative=True)
rd,w,pr,pt=s.symbols('rd w pr pt',real=True)
L=m*(rd*rd+r*r*w*w)/2-k*r*r/2
sol=s.solve([pr-s.diff(L,rd),pt-s.diff(L,w)],[rd,w],dict=True)[0]
H=s.simplify(pr*sol[rd]+pt*sol[w]-L.subs(sol))
pvars=(m,k,r,rd,w)
Pfn=s.lambdify(pvars,s.Matrix([s.diff(L,rd),s.diff(L,w)]),'mpmath')
Hfn=s.lambdify((m,k,r,pr,pt),H,'mpmath');HPfn=s.lambdify((m,k,r,pr,pt),s.Matrix([s.diff(H,pr),s.diff(H,pt),-s.diff(H,r)]),'mpmath')
Dfn=s.lambdify((m,r),s.det(s.hessian(L,[rd,w])),'mpmath')
polar_inputs=[[1,0,2,0,2,2,0],[1,2,2,0,0,2,20],[1,2,2,0,0,-2,20],[1,2,2,0,0,0,20],[1,0,1,0,-3,0,0],[1,3,2,0,0,0,40]]
polar_inputs += [[rng.randint(lo,hi)for lo,hi in [(1,4),(0,4),(1,6),(-4,4),(-4,4),(-4,4),(0,40)]]for _ in range(234)]
polar=[]
for p in polar_inputs:
    mass,f,rc,ac,vc,wc,tc=p;M=mp.mpf(mass);Om=mp.pi*f/4;K=M*Om*Om;R=mp.mpf(rc)/2;V=mp.mpf(vc)/2;W=mp.mpf(wc)/2*(Om if f else 1);th=mp.pi*ac/2;T=mp.mpf(tc)/20
    x0=mp.matrix([R*mp.cos(th),R*mp.sin(th)]);v0=mp.matrix([V*mp.cos(th)-R*W*mp.sin(th),V*mp.sin(th)+R*W*mp.cos(th)])
    def X(t):return x0+v0*t if f==0 else x0*mp.cos(Om*t)+v0*mp.sin(Om*t)/Om
    def Vel(t):return mp.matrix([mp.diff(lambda u:X(u)[i],t)for i in range(2)])
    pos=X(T);vel=Vel(T);energy=M*(vel.T*vel)[0]/2+K*(pos.T*pos)[0]/2;Lz=M*(pos[0]*vel[1]-pos[1]*vel[0])
    origin=wc==0 and (rc*20+vc*tc==0 if f==0 else vc==0 and (f*tc)%80==40)
    canonical=None
    if not origin:
        rad=lambda t:mp.sqrt((X(t).T*X(t))[0]);rr=rad(T);vr=mp.diff(rad,T);ar=mp.diff(rad,T,2)
        def wr(t):
            x=X(t);v=Vel(t);return (x[0]*v[1]-x[1]*v[0])/(x.T*x)[0]
        ww=wr(T);alpha=mp.diff(wr,T);mom=Pfn(M,K,rr,vr,ww);hv=Hfn(M,K,rr,*mom);rates=HPfn(M,K,rr,*mom)
        canonical={name:mp.nstr(val,55)for name,val in dict(radius=rr,radialMomentum=mom[0],angularMomentum=mom[1],hamiltonian=hv,radialVelocity=rates[0],angularVelocity=rates[1],radialMomentumRate=rates[2],radialAccelerationFromHamilton=ar,angularAccelerationFromHamilton=alpha,velocityHessianDeterminant=Dfn(M,rr)).items()}
    polar.append(dict(p=p,physicalEnergy=mp.nstr(energy,55),physicalAngularMomentum=mp.nstr(Lz,55),origin=origin,canonical=canonical))
print('240 independent polar Legendre maps and60digit trajectory derivatives.',flush=True)
x,v,t,gamma,P=s.symbols('x v t gamma P',real=True)
F=gamma*t*x*x/2;L0=m*v*v/2-k*x*x/2;Lg=L0+s.diff(F,t)+s.diff(F,x)*v
vexpr=s.solve(s.Eq(P,s.diff(Lg,v)),v)[0];Hg=s.simplify((P*v-Lg).subs(v,vexpr))
qP=s.lambdify((m,gamma,x,v,t),s.diff(Lg,v),'mpmath');qL=s.lambdify((m,k,gamma,x,v,t),Lg,'mpmath');qH=s.lambdify((m,k,gamma,x,P,t),Hg,'mpmath')
qPartials=s.lambdify((m,k,gamma,x,P,t),s.Matrix([s.diff(Hg,x),s.diff(Hg,P),s.diff(Hg,t)]),'mpmath')
gauge_inputs=[[1,2,2,0,0,20],[1,2,2,0,8,0],[1,2,2,0,8,20],[1,0,0,2,4,20],[1,0,2,0,4,40],[1,2,0,0,8,40],[2,3,-2,4,-8,13]]
gauge_inputs += [[rng.randint(lo,hi)for lo,hi in [(1,4),(0,4),(-4,4),(-4,4),(-8,8),(0,40)]]for _ in range(173)]
gauge=[]
for p in gauge_inputs:
    mass,f,xc,vc,gc,tc=p;M=mp.mpf(mass);Om=mp.pi*f/4;K=M*Om*Om;X0=mp.mpf(xc)/2;V0=mp.mpf(vc)/2;G=mp.mpf(gc)/2;T=mp.mpf(tc)/20
    def X(u):return X0+V0*u if f==0 else X0*mp.cos(Om*u)+V0*mp.sin(Om*u)/Om
    def V(u):return mp.diff(X,u)
    def Mom(u):return qP(M,G,X(u),V(u),u)
    def Hamilton(u):return qH(M,K,G,X(u),Mom(u),u)
    xx=X(T);vv=V(T);pp=Mom(T);parts=qPartials(M,K,G,xx,pp,T);E=M*vv*vv/2+K*xx*xx/2
    out=dict(position=xx,velocity=vv,canonicalMomentum=pp,canonicalMomentumRate=mp.diff(Mom,T),physicalEnergy=E,hamiltonian=Hamilton(T),partialPosition=parts[0],partialMomentum=parts[1],partialTime=parts[2],hamiltonianTotalTime=mp.diff(Hamilton,T),gaugeFunction=G*T*xx*xx/2,baseCumulativeAction=mp.quad(lambda u:M*V(u)**2/2-K*X(u)**2/2,[0,T]),gaugeCumulativeAction=mp.quad(lambda u:qL(M,K,G,X(u),V(u),u),[0,T]))
    gauge.append(dict(p=p,query={name:mp.nstr(val,55)for name,val in out.items()}))
print('180 independent60digit time-gauge canonical derivatives and direct action quadratures.',flush=True)
free=[]
for j in range(64):
    p=[1+j%4,0,-4+j%9,-4+(3*j)%9,-8+(5*j)%17,j%41];mass,f,xc,vc,gc,tc=p;M=Q(mass);X0=Q(xc,2);V=Q(vc,2);G=Q(gc,2);T=Q(tc,20);X=X0+V*T;E=M*V*V/2;Pc=M*V+G*T*X
    def Lbase(u):return M*V*V/2
    def Lgauge(u):
        q=X0+V*u;return Lbase(u)+G*q*q/2+G*u*q*V
    def boole(fn):return T*sum(w*fn(T*Q(i,4))for i,w in enumerate([7,32,12,32,7]))/90
    out=dict(position=X,velocity=V,mechanicalMomentum=M*V,canonicalMomentum=Pc,physicalEnergy=E,hamiltonian=E-G*X*X/2,canonicalMomentumRate=G*X+G*T*V,partialTime=-G*X*V,baseCumulativeAction=boole(Lbase),gaugeCumulativeAction=boole(Lgauge))
    free.append(dict(p=p,exact={name:str(val)for name,val in out.items()}))
result=dict(methods=dict(polar='Symbolic velocity elimination and canonical partial derivatives, independent60digit Cartesian/radial derivatives',gauge='Symbolic Legendre elimination;60digit time/partial derivatives and direct quadrature',free='Fraction Boole quadrature exact for free-polynomial Lagrangian'),polar=polar,gauge=gauge,free=free)
target=Path(__file__).resolve().parents[1]/'fixtures'/'hamiltonian-canonical-independent.json';target.write_text(json.dumps(result,indent=2)+'\n');print('64exact free-gauge cases -> '+target.name,flush=True)
