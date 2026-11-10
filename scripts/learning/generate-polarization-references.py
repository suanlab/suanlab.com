"""Independent polarization references, no product imports. NumPy vector
projections/SVD/eigensystems, actual real-field cycle integrals with QUADPACK,
and Mueller columns from independently prepared analyzer-basis states.
"""
from pathlib import Path
import json
import numpy as np
from scipy.integrate import quad
import sympy as sp
rng=np.random.default_rng(20261110)
orders=['retarder-only','analyzer-only','retarder-then-analyzer','analyzer-then-retarder']
def pair(z):return [float(z.real),float(z.imag)]
def vec(v):return [pair(z) for z in v]
def prep(s):return np.array([s['amplitudeX'],s['amplitudeY']*np.exp(1j*s['relativePhase'])])*np.exp(1j*s['globalPhase'])
def propagate(v,s):
 f=np.array([np.cos(s['retarderAngle']),np.sin(s['retarderAngle'])]);slow=np.array([-f[1],f[0]])
 n=np.array([np.cos(s['analyzerAngle']),np.sin(s['analyzerAngle'])])
 plate=lambda e:f*np.dot(f,e)+np.exp(1j*s['retardance'])*slow*np.dot(slow,e)
 analyzer=lambda e:n*np.dot(n,e)
 if s['order']=='retarder-only':return plate(v),plate(v)
 if s['order']=='analyzer-only':return analyzer(v),analyzer(v)
 if s['order']=='retarder-then-analyzer':return plate(v),analyzer(plate(v))
 return analyzer(v),plate(analyzer(v))
def coherency(e,p):return sum(w*np.outer(v,v.conj()) for w,v in zip(p,e))
def stokes(c):return [float((c[0,0]+c[1,1]).real),float((c[0,0]-c[1,1]).real),float(2*c[0,1].real),float(-2*c[0,1].imag)]
def cjson(c):return dict(xx=float(c[0,0].real),yy=float(c[1,1].real),xy=pair(c[0,1]))
analyzers=[np.array([1,0],complex),np.array([0,1],complex),np.array([1,1],complex)/np.sqrt(2),np.array([1,-1],complex)/np.sqrt(2),np.array([1,1j],complex)/np.sqrt(2),np.array([1,-1j],complex)/np.sqrt(2)]
pauli=[np.eye(2),np.diag([1.,-1.]),np.array([[0,1],[1,0]],complex),np.array([[0,-1j],[1j,0]],complex)]
def actual_intensity(v):
 return 2*quad(lambda tau:float(np.sum(np.real(v*np.exp(-1j*tau))**2)),0,2*np.pi,epsabs=2e-12,epsrel=2e-12)[0]/(2*np.pi)
jones=[]
for case in range(48):
 s=dict(amplitudeX=float(rng.uniform(.1,1.8)),amplitudeY=float(rng.uniform(.1,1.8)),relativePhase=float(rng.uniform(-np.pi,np.pi)),globalPhase=float(rng.uniform(-np.pi,np.pi)),retarderAngle=float(rng.uniform(-np.pi,np.pi)),retardance=float(rng.uniform(-2*np.pi,2*np.pi)),analyzerAngle=float(rng.uniform(-np.pi,np.pi)),order=orders[case%4],observationPhase=float(rng.uniform(0,2*np.pi)))
 if case==0:s.update(amplitudeX=1,amplitudeY=0,retarderAngle=0,retardance=0,analyzerAngle=np.pi/2,order='analyzer-only')
 if case==1:s.update(amplitudeX=1/np.sqrt(2),amplitudeY=1/np.sqrt(2),relativePhase=0,globalPhase=0,retarderAngle=0,retardance=np.pi/2,order='retarder-only')
 if case==2:s.update(amplitudeX=0,amplitudeY=0)
 e=prep(s);middle,out=propagate(e,s);matrix=np.column_stack([out.real,out.imag]);axes=np.linalg.svd(matrix,compute_uv=False)
 trajectory=[dict(phase=float(t),input=np.real(e*np.exp(-1j*t)).tolist(),intermediate=np.real(middle*np.exp(-1j*t)).tolist(),output=np.real(out*np.exp(-1j*t)).tolist()) for t in np.linspace(0,2*np.pi,17)]
 jones.append(dict(system=s,input=vec(e),intermediate=vec(middle),output=vec(out),outputStokes=stokes(np.outer(out,out.conj())),axes=axes.tolist(),cycleIntensity=actual_intensity(out),trajectory=trajectory))
ensembles=[]
for case in range(48):
 population=[dict(amplitudeX=float(rng.uniform(0,1.8)),amplitudeY=float(rng.uniform(0,1.8)),relativePhase=float(rng.uniform(-np.pi,np.pi)),globalPhase=float(rng.uniform(-np.pi,np.pi)),weight=float(rng.integers(1,5))) for _ in range(1+case%4)]
 if case==0:population=[dict(amplitudeX=1,amplitudeY=0,relativePhase=0,globalPhase=0,weight=1),dict(amplitudeX=0,amplitudeY=1,relativePhase=0,globalPhase=0,weight=1)]
 if case==1:population=[dict(amplitudeX=1/np.sqrt(2),amplitudeY=1/np.sqrt(2),relativePhase=np.pi/2,globalPhase=0,weight=1)]
 if case==47:
  for r in population:r.update(amplitudeX=0,amplitudeY=0)
 s=dict(population=population,retainedPolarization=[0,.25,.8,1][case%4],retarderAngle=float(rng.uniform(-np.pi,np.pi)),retardance=float(rng.uniform(-2*np.pi,2*np.pi)),analyzerAngle=float(rng.uniform(-np.pi,np.pi)),order=orders[(case//4)%4])
 total=sum(r['weight'] for r in population);p=np.array([r['weight']/total for r in population]);e=[prep(r) for r in population];d=s['retainedPolarization'];weights=[(1+3*d)/4,*([(1-d)/4]*3)]
 expanded=[(prob*w,u@v) for prob,v in zip(p,e) for w,u in zip(weights,pauli)]
 out=[propagate(v,s)[1] for _,v in expanded];probs=[w for w,_ in expanded]
 cin=coherency(e,p);cd=coherency([v for _,v in expanded],probs);co=coherency(out,probs)
 measurements=[float(sum(w*actual_intensity(np.array([np.vdot(a,v)])) for w,v in zip(probs,out))) for a in analyzers]
 # Prepare six physical input states; infer Mueller columns by differences.
 basis_outputs=[]
 for a in analyzers:
  fields=[propagate(u@a,s)[1] for u in pauli]
  basis_outputs.append(np.array(stokes(coherency(fields,weights))))
 h,v,diag,anti,left,right=basis_outputs
 mueller=np.column_stack([(h+v)/2,(h-v)/2,(diag-anti)/2,(left-right)/2])
 eigen=np.linalg.eigvalsh(co)
 degree=None if np.trace(co).real==0 else float((eigen[1]-eigen[0])/sum(eigen))
 ensembles.append(dict(system=s,input=cjson(cin),depolarized=cjson(cd),output=cjson(co),stokes=stokes(co),degree=degree,measurements=measurements,mueller=mueller.tolist()))
alpha,beta=sp.symbols('alpha beta',real=True)
malus=sp.trigsimp((sp.cos(alpha)*sp.cos(beta)+sp.sin(alpha)*sp.sin(beta))**2-sp.cos(alpha-beta)**2)
assert malus==0
choi=np.zeros((4,4))
for i in range(2):
 for j in range(2):
  eij=np.zeros((2,2));eij[i,j]=1;choi+=np.kron(eij,eij.T)
report=dict(provenance=dict(productImports=False,generator='scripts/learning/generate-polarization-references.py',jones='vector projections and SVD; real time-domain QUADPACK cycle integral',ensembles='real analyzer integrals; Mueller columns from six physical basis preparations; eigensystem degree',symbolicMalus=str(malus)),jones=jones,ensembles=ensembles,transposeChoiEigenvalues=np.linalg.eigvalsh(choi).tolist())
Path('scripts/fixtures/polarization-independent.json').write_text(json.dumps(report,indent=2)+'\n')
print(f'Wrote{len(jones)}Jones systems/{sum(len(r["trajectory"]) for r in jones)}real-field samples and{len(ensembles)}ensembles/288independentanalyzerintegrals. SymbolicMalus={malus}.')
