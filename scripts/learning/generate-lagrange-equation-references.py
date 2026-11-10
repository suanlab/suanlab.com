"""Independent Lagrange references: symbolic Cartesian energies + DOP853 + 60-digit checks.
Run from repo root with numpy, scipy, sympy, mpmath. No production-code imports.
Fixtures let Node tests run without these Python dependencies.
"""
from pathlib import Path
import json,random,math
import numpy as np
import sympy as s
import mpmath as mp
from scipy.integrate import solve_ivp
mp.mp.dps=60;rng=random.Random(20261104426)
a,b,wa,wb,aa,ab,m1,m2,l1,l2=s.symbols('a b wa wb aa ab m1 m2 l1 l2',real=True)
P=s.Matrix([l1*s.sin(a),-l1*s.cos(a),l1*s.sin(a)+l2*s.sin(b),-l1*s.cos(a)-l2*s.cos(b)])
J=P.jacobian([a,b]);vel=J*s.Matrix([wa,wb]);W=s.diag(m1,m1,m2,m2)
K=(vel.T*W*vel)[0]/2;U=10*((m1+m2)*l1*(1-s.cos(a))+m2*l2*(1-s.cos(b)));L=K-U
Dt=lambda e:s.diff(e,a)*wa+s.diff(e,b)*wb+s.diff(e,wa)*aa+s.diff(e,wb)*ab
EL=s.Matrix([Dt(s.diff(L,w))-s.diff(L,q) for q,w in [(a,wa),(b,wb)]])
M=s.simplify(EL.jacobian([aa,ab]));rem=s.simplify(EL.subs({aa:0,ab:0}))
args=(m1,m2,l1,l2,a,b,wa,wb)
Mfn=s.lambdify(args,M,'numpy');remfn=s.lambdify(args,rem,'numpy');Pfn=s.lambdify(args,P,'numpy');Jfn=s.lambdify(args,J,'numpy');Kfn=s.lambdify(args,K,'numpy');Ufn=s.lambdify(args,U,'numpy')
Mmp=s.lambdify(args,M,'mpmath');remmp=s.lambdify(args,rem,'mpmath')
Hfn=[s.lambdify(args,s.hessian(P[i],[a,b]),'numpy')for i in range(4)]
def parameters(p):return [p[0],p[1],p[2]/2,p[3]/2]
def initial(p):return np.array([p[4]*math.pi/12,p[5]*math.pi/12,p[6]/10,p[7]/10])
def state(par,z):
    vals=[*par,*z];mat=np.array(Mfn(*vals),dtype=float);rhs=np.array(remfn(*vals),dtype=float).ravel();acc=np.linalg.solve(mat,-rhs)
    pos=np.array(Pfn(*vals),dtype=float).ravel();jac=np.array(Jfn(*vals),dtype=float);v=jac@z[2:]
    hess=[np.array(Hfn[i](*vals),dtype=float)for i in range(4)]
    accxy=jac@acc+np.array([z[2:]@h@z[2:] for h in hess])
    R2=par[1]*(accxy[2:]+[0,10]);R1=par[0]*(accxy[:2]+[0,10])+R2
    return dict(angles=z[:2].tolist(),rates=z[2:].tolist(),accelerations=acc.tolist(),matrix=mat.tolist(),pose=pos.tolist(),velocities=v.tolist(),cartesianAccelerations=accxy.tolist(),rod1Force=R1.tolist(),rod2Force=R2.tolist(),kinetic=float(Kfn(*vals)),potential=float(Ufn(*vals)),energy=float(Kfn(*vals)+Ufn(*vals)))
inputs=[[1,1,2,2,0,0,0,0,20,2],[1,1,2,2,12,12,0,0,20,2],[1,1,2,2,6,0,10,-10,20,2],[4,1,1,6,-6,6,10,10,40,4],[1,4,6,1,11,-11,-10,10,40,4]]
inputs += [[rng.randint(lo,hi)for lo,hi in [(1,4),(1,4),(1,6),(1,6),(-12,12),(-12,12),(-10,10),(-10,10),(0,40),(1,4)]]for _ in range(115)]
instant=[]
for p in inputs:
    par=parameters(p);z=initial(p);v=state(par,z)
    mm=[mp.mpf(n)for n in par]+[mp.pi*p[4]/12,mp.pi*p[5]/12,mp.mpf(p[6])/10,mp.mpf(p[7])/10]
    accmp=mp.lu_solve(Mmp(*mm),-remmp(*mm));v['highPrecisionAccelerations']=[mp.nstr(x,55)for x in accmp]
    instant.append(dict(p=p,initial=v))
print('120 symbolic Cartesian-energy/linear-solve states;60digit accelerations checked.',flush=True)
trajectories=[]
for p in inputs[:5]+inputs[5:18]:
    par=parameters(p);z=initial(p)
    def rhs(t,z):
        vals=[*par,*z];return np.r_[z[2:],np.linalg.solve(np.array(Mfn(*vals),dtype=float),-np.array(remfn(*vals),dtype=float).ravel())]
    times=np.linspace(0,2,21)
    sol=solve_ivp(rhs,(0,2),z,method='DOP853',rtol=2e-13,atol=2e-14,t_eval=times,max_step=.01)
    check=solve_ivp(rhs,(0,2),z,method='DOP853',rtol=3e-14,atol=3e-15,t_eval=times,max_step=.005)
    assert sol.success and check.success
    gap=np.max(np.abs(sol.y-check.y));assert gap<1e-9,gap
    energies=[state(par,x)['energy']for x in check.y.T];drift=max(abs(e-energies[0])for e in energies)
    trajectories.append(dict(p=p,times=times.tolist(),states=check.y.T.tolist(),solverRefinementGap=float(gap),maximumEnergyDrift=drift))
print('18 independently refined DOP853 physical-time paths,21samples each.',flush=True)
polar=[]
polar_inputs=[[1,0,2,0,2,2,20],[1,2,2,0,0,2,20],[1,2,2,0,0,0,20],[4,4,6,4,-4,-4,40]]+[[rng.randint(lo,hi)for lo,hi in [(1,4),(0,4),(1,6),(-4,4),(-4,4),(-4,4),(0,40)]]for _ in range(236)]
for p in polar_inputs:
    mass,f,rc,ac,vc,wc,tc=p;T=mp.mpf(tc)/20;Om=mp.pi*f/4;ref=Om if f else mp.mpf(1);r0=mp.mpf(rc)/2;rd0=mp.mpf(vc)/2;w0=mp.mpf(wc)*ref/2;th=mp.pi*ac/2
    X0=mp.matrix([r0*mp.cos(th),r0*mp.sin(th)]);V0=mp.matrix([rd0*mp.cos(th)-r0*w0*mp.sin(th),rd0*mp.sin(th)+r0*w0*mp.cos(th)])
    X=lambda t:X0+V0*t if f==0 else X0*mp.cos(Om*t)+V0*mp.sin(Om*t)/Om
    pos=X(T);vel=mp.matrix([mp.diff(lambda t:X(t)[i],T)for i in range(2)]);acc=mp.matrix([mp.diff(lambda t:X(t)[i],T,2)for i in range(2)])
    analyticOrigin=wc==0 and (rc*20+vc*tc==0 if f==0 else vc==0 and (f*tc)%80==40)
    # phase=pi*f*tc/80; cos vanishes for f*tc=40 mod80.
    if analyticOrigin:pos=mp.matrix([0,0])
    K=mass*(vel.T*vel)[0]/2;U=mass*Om**2*(pos.T*pos)[0]/2;rad=mp.sqrt((pos.T*pos)[0]);pol=None
    if not analyticOrigin:
        rr=lambda t:mp.sqrt((X(t).T*X(t))[0]);dr=mp.diff(rr,T);ddr=mp.diff(rr,T,2)
        angular=(pos[0]*vel[1]-pos[1]*vel[0])/(rad*rad)
        def angularFn(t):
            x=X(t);v=mp.matrix([mp.diff(lambda u:X(u)[i],t)for i in range(2)]);return (x[0]*v[1]-x[1]*v[0])/(x.T*x)[0]
        alpha=mp.diff(angularFn,T)
        pol=dict(radialRate=mp.nstr(dr,55),angularRate=mp.nstr(angular,55),radialAcceleration=mp.nstr(ddr,55),angularAcceleration=mp.nstr(alpha,55))
    origins=[]
    if wc==0:
        if f==0:
            if vc<0 and -mp.mpf(rc)/vc<=2: origins=[-mp.mpf(rc)/vc]
        else:
            phase=mp.atan(rd0/(Om*r0))
            for n in range(4):
                root=mp.mpf(2+4*n)/f if vc==0 else (mp.pi/2+phase+n*mp.pi)/Om
                if root>2: break
                origins.append(root)
    polar.append(dict(p=p,originTimes=[mp.nstr(t,55)for t in origins],position=[mp.nstr(x,55)for x in pos],velocity=[mp.nstr(x,55)for x in vel],acceleration=[mp.nstr(x,55)for x in acc],radius=mp.nstr(rad,55),kinetic=mp.nstr(K,55),potential=mp.nstr(U,55),angularMomentum=mp.nstr(mass*(pos[0]*vel[1]-pos[1]*vel[0]),55),analyticOrigin=analyticOrigin,polar=pol))
print('240 independent 60digit Cartesian-flow/radial-derivative polar states.',flush=True)
result=dict(methods=dict(instantaneous='SymPy EL differentiated from Cartesian kinetic/potential energies;NumPy linear solve;mpmath60digit accelerations',trajectories='Independent SymPy RHS;SciPy DOP853 tolerance/step refinement',polar='mpmath60digit direct Cartesian-flow differentiation and radius derivatives'),instantaneous=instant,trajectories=trajectories,polar=polar)
target=Path(__file__).resolve().parents[1]/'fixtures'/'lagrange-equations-independent.json';target.write_text(json.dumps(result,indent=2)+'\n');print(target.name,flush=True)
