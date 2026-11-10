"""Independent generalized-eigenproblem and physical-flow references.

Run from repository root. No TypeScript product code is imported.
Uses SciPy eigh/expm/DOP853 refinement, real flux quadrature and SymPy identities.
"""
import json
import math
import random
from pathlib import Path

import numpy as np
import scipy
import sympy as sp
from scipy.linalg import eigh, expm
from scipy.integrate import solve_ivp, quad

rng = random.Random(20261106)
q1, q2, v1, v2, m1, m2, k1, k2, coupling = sp.symbols('q1 q2 v1 v2 m1 m2 k1 k2 coupling', real=True)
acc = [(-(k1 + coupling)*q1 + coupling*q2)/m1, (coupling*q1 - (k2 + coupling)*q2)/m2]
local = [m1*v1**2/2 + k1*q1**2/2 + coupling*(q2-q1)**2/4,
         m2*v2**2/2 + k2*q2**2/2 + coupling*(q2-q1)**2/4]
flux = coupling*(q1-q2)*(v1+v2)/2
ddt = lambda f: sp.diff(f,q1)*v1 + sp.diff(f,q2)*v2 + sp.diff(f,v1)*acc[0] + sp.diff(f,v2)*acc[1]
assert sp.simplify(ddt(local[0])+flux) == 0
assert sp.simplify(ddt(local[1])-flux) == 0

def physical(s):
    M = np.diag([s['mass1'], s['mass2']])
    c = s['coupling']
    K = np.array([[s['stiffness1']+c,-c],[-c,s['stiffness2']+c]])
    flow = np.block([[np.zeros((2,2)), np.eye(2)], [-np.linalg.solve(M,K),np.zeros((2,2))]])
    return M, K, flow

def mechanical(s, y):
    q, v = y[:2], y[2:4]
    bare = [s['mass1']*v[0]**2/2 + s['stiffness1']*q[0]**2/2,
            s['mass2']*v[1]**2/2 + s['stiffness2']*q[1]**2/2]
    ec = s['coupling']*(q[1]-q[0])**2/2
    return dict(position=q.tolist(), velocity=v.tolist(), bareEnergy=bare, couplingEnergy=ec,
                assignedEnergy=[v+ec/2 for v in bare], energy=sum(bare)+ec,
                flux12=s['coupling']*(q[0]-q[1])*(v[0]+v[1])/2)

systems = [(1,2,1,1,1),(1,2,0,0,1),(1,2,1,2,0),(1,2,0,0,0),
           (1,1,1,1,.0625),(.25,4,4,4,4),(4,.25,0,4,4),(4,4,4,4,.0625)]
for i in range(112):
    systems.append((rng.randint(1,16)/4,rng.randint(1,16)/4,rng.randint(0,64)/16,
                    rng.randint(0,64)/16,rng.randint(0,64)/16))
normal = []
for index, raw in enumerate(systems):
    s = dict(zip(['mass1','mass2','stiffness1','stiffness2','coupling'],raw))
    M,K,flow = physical(s)
    initial = np.array([rng.randint(-4,4)/2 for _ in range(4)],dtype=float)
    duration = 128. if index==5 else 0. if index==3 else rng.uniform(.5,16)
    times = np.linspace(0,duration,9)
    lam, phi = eigh(K,M)
    exact_K = sp.Matrix([[sp.Rational(str(v)) for v in row] for row in K])
    rank = int(exact_K.rank())
    if rank==0:
        lam[:] = 0
    elif rank==1:
        lam[0] = 0
    # Zero eigenvalues follow exact structural rank, not a floating threshold.
    assert np.min(lam)>=0
    repeated = s['coupling']==0 and s['stiffness1']*s['mass2']==s['stiffness2']*s['mass1']
    if duration==0:
        solution = refined = np.tile(initial,(9,1))
    else:
        def solve(tol):
            result=solve_ivp(lambda t,y:flow@y,[0,duration],initial,method='DOP853',rtol=tol,atol=tol*.04,t_eval=times)
            assert result.success,result.message
            return result.y.T
        solution,refined=solve(2e-11),solve(3e-13)
    init_q=phi.T@M@initial[:2]
    init_v=phi.T@M@initial[2:]
    modal_energy=(init_v**2+lam*init_q**2)/2
    states=[]; maximum_crosscheck=0.
    for t,y in zip(times,refined):
        exact=expm(flow*t)@initial
        maximum_crosscheck=max(maximum_crosscheck,float(np.max(np.abs(exact-y))))
        row=mechanical(s,y)
        row.update(time=float(t),modalEnergies=modal_energy.tolist())
        states.append(row)
    assert maximum_crosscheck<2e-9
    projectors=[np.outer(phi[:,j],phi[:,j])@M for j in range(2)]
    normal.append(dict(system=s,initial=dict(position=initial[:2].tolist(),velocity=initial[2:].tolist()),duration=duration,
                       rank=rank,degenerate=repeated,squaredFrequencies=lam.tolist(),projectors=None if repeated else [p.tolist()for p in projectors],
                       states=states,refinementDifference=float(np.max(np.abs(solution-refined))),independentExpmDifference=maximum_crosscheck))

beating=[]
parameters=[(1,1,.0625,1),(1,0,1,1),(1,1,0,1),(1,1,1,0),(1,0,0,1),(4,4,.0625,1),(.25,4,4,4),(1,1,1,1)]
parameters += [(rng.randint(1,16)/4,rng.randint(0,64)/16,rng.randint(0,64)/16,rng.randint(0,8)/2)for _ in range(64)]
for index,(m,k,c,A) in enumerate(parameters):
    s=dict(mass=m,stiffness=k,coupling=c,amplitude=A)
    system=dict(mass1=m,mass2=m,stiffness1=k,stiffness2=k,coupling=c)
    M,K,flow=physical(system)
    wm,wp=math.sqrt(k/m),math.sqrt((k+2*c)/m)
    split=wp-wm
    swap=math.pi/split if c>0 and A>0 else None
    duration=128. if index==6 else 64. if index==0 else rng.uniform(.5,20)
    times=np.linspace(0,duration,9)
    initial=np.array([A,0,0,0],dtype=float)
    def rhs(t,y):
        return np.r_[flow@y[:4],c*(y[0]-y[1])*(y[2]+y[3])/2]
    def solve(tol):
        result=solve_ivp(rhs,[0,duration],np.r_[initial,0],method='DOP853',rtol=tol,atol=tol*.04,t_eval=times)
        assert result.success,result.message
        return result.y.T
    coarse,refined=solve(2e-11),solve(3e-13)
    def flux_closed(t):
        # Separate two-cosine solution, not the product-envelope product code.
        qdiff=A*math.cos(wp*t)
        vsum=-A*wm*math.sin(wm*t)
        return c*qdiff*vsum/2
    states=[]
    for t,y in zip(times,refined):
        actual=mechanical(system,y[:4]);actual['time']=float(t)
        integral=quad(flux_closed,0,float(t),epsabs=2e-10,epsrel=2e-11,limit=600)[0]if t>0 else 0.
        assert abs(integral-y[4])<2e-8*(1+abs(integral))
        actual['netTransfer']=integral
        if wm>0:
            approximate_wp=wm+c/(m*wm)
            position=[A*(math.cos(wm*t)+math.cos(approximate_wp*t))/2,
                      A*(math.cos(wm*t)-math.cos(approximate_wp*t))/2]
            velocity=[-A*(wm*math.sin(wm*t)+approximate_wp*math.sin(approximate_wp*t))/2,
                      -A*(wm*math.sin(wm*t)-approximate_wp*math.sin(approximate_wp*t))/2]
            actual['approximation']=dict(position=position,velocity=velocity)
        else:
            actual['approximation']=None
        states.append(actual)
    swap_state=None
    if swap is not None:
        state=expm(flow*swap)@initial
        swap_state=dict(time=swap,**mechanical(system,state))
    beating.append(dict(system=s,duration=duration,states=states,firstEnvelopeSwap=swap_state,
                        refinementDifference=float(np.max(np.abs(coarse-refined)))))

data=dict(provenance=dict(generator='scripts/learning/generate-coupled-modes-references.py',seed=20261106,
                         scipy=scipy.__version__,sympy=sp.__version__,methods=['generalized symmetric eigh','expm','DOP853 tolerance refinement','independent real flux quadrature','symbolic local-energy conservation'],symbolicFluxIdentities=True),
          normal=normal,beating=beating)
path=Path('scripts/fixtures/coupled-modes-independent.json')
path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
print(f'{path}: {len(normal)} normal systems / {len(normal)*9} states; {len(beating)} beating systems / {len(beating)*9} states')
