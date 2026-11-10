"""Independent incidence/mass matrices, matrix powers and physical cell integrals.

No production import. Production iterates cumulative changes with signed logs;
this oracle forms dense high-precision operators and applies (I+dt R)^n.
"""
import argparse,json,random,math
from pathlib import Path
import mpmath as m
parser=argparse.ArgumentParser();parser.add_argument('--output',default='scripts/fixtures/continuum-volume-independent.json');target=Path(parser.parse_args().output)
if target.exists():raise SystemExit('Refusing to overwrite reference')
base=dict(cellCountExponent=3,length=2.,area=1.,meshWarp=.6,velocity=.5,diffusivity=.1,meanConcentration=1.,relativeContrast=.8,mode=1,phaseCycles=.125,timeStep=.01,steps=8)
rng=random.Random(261110)
def obs(v):
 v=m.mpf(v);return dict(value=str(v),sign=int(m.sign(v)),logMagnitude=None if v==0 else str(m.log(abs(v))))
def mat(volumes,conductances,flow):
 N=len(volumes);B=m.zeros(N);up=m.zeros(N)
 for i in range(N):B[i,i]=1;B[(i+1)%N,i]=-1;up[i,i if flow>=0 else(i+1)%N]=flow
 face=up+m.diag(conductances)*B.T
 return -m.diag([1/v for v in volumes])*B*face,-B*face/(sum(volumes)/N),face

def run(p):
 N=2**p['cellCountExponent'];L=m.mpf(p['length']);A=m.mpf(p['area']);a=m.mpf(p['velocity']);D=m.mpf(p['diffusivity']);mean=m.mpf(p['meanConcentration']);eps=m.mpf(p['relativeContrast']);mode=p['mode'];phase=m.mpf(p['phaseCycles']);dt=m.mpf(p['timeStep']);steps=p['steps'];k=2*m.pi*mode/L
 # Ordinary floating-point coordinates are explicitly the model geometry; oracle
 # field integration and matrix arithmetic are then high precision on that mesh.
 edges=[0. if i==0 else float(L)if i==N else float(L)*(i/N+p['meshWarp']*math.sin(2*math.pi*i/N)/(2*math.pi))for i in range(N+1)]
 X=list(map(m.mpf,edges));h=[X[i+1]-X[i]for i in range(N)];centers=[(X[i]+X[i+1])/2 for i in range(N)];volumes=[A*x for x in h];dist=[centers[i+1]-centers[i]if i<N-1 else L-centers[i]+centers[0]for i in range(N)];gamma=[D*A/d for d in dist]
 R,W,F=mat(volumes,gamma,a*A)
 def exact(time):return m.matrix([mean+mean*eps*m.exp(-D*k*k*time)*(m.sinpi(2*(mode*X[i+1]/L+phase-a*mode*time/L))-m.sinpi(2*(mode*X[i]/L+phase-a*mode*time/L)))/(k*h[i])for i in range(N)])
 initial=exact(0);time=dt*steps;current=(m.eye(N)+dt*R)**steps*initial;wrong=(m.eye(N)+dt*W)**steps*initial;continuum=exact(time)
 if mean==0 or eps==0 or a==D==0:current=initial.copy();wrong=initial.copy()
 rates=R*current;wrongRates=W*wrong
 if mean==0 or eps==0 or a==D==0:rates=m.zeros(N,1);wrongRates=m.zeros(N,1)
 change=current-initial;wrongChange=wrong-initial;delta=initial-m.matrix([mean]*N);dep=current-m.matrix([mean]*N);wrongDep=wrong-m.matrix([mean]*N)
 dot=lambda xs:sum(volumes[i]*xs[i]for i in range(N));norm=lambda xs:dot(m.matrix([x*x for x in xs]))/2
 lossDiff=sum(gamma[i]*(current[i]-current[(i+1)%N])**2 for i in range(N));lossAdv=abs(a*A)/2*sum((current[i]-current[(i+1)%N])**2 for i in range(N));positive=dt*dt*dot(m.matrix([x*x for x in rates]))/2
 exactChange=continuum-initial;errors=change-exactChange;wrongErrors=wrongChange-exactChange
 scale=max(m.mpf(1),*map(abs,current),*map(abs,wrong));tol=m.mpf(10)**(-(m.mp.dps-50))*scale
 assert abs(dot(change))<tol,(dot(change),tol)
 assert abs(sum(wrongChange))<tol
 assert m.almosteq(dot(m.matrix([current[i]*rates[i]for i in range(N)])),-lossDiff-lossAdv,abs_eps=tol*scale)
 mass0=mean*sum(volumes)
 def budget(values,changes,departure,isWrong=False):
  massChange=dot(changes);centerChange=norm(departure)-norm(delta)
  if not isWrong:massChange=m.mpf(0)
  return dict(assembledMass=dot(values),massFromConservationLedger=mass0+massChange,absoluteCellMassSum=dot(m.matrix(list(map(abs,values)))),centeredNorm=norm(departure),quadraticNorm=norm(values),stableCenteredNormChange=centerChange,normChangeUsingMassLedger=centerChange+mean*massChange,minimumConcentration=min(values),maximumConcentration=max(values))
 correct=budget(current,change,dep);bad=budget(wrong,wrongChange,wrongDep,True)
 if p['meshWarp']==0:bad['massFromConservationLedger']=mass0
 cells=[]
 for i in range(N):
  cells.append({key:obs(value)for key,value in dict(initialDeparture=delta[i],initialConcentration=initial[i],departure=dep[i],concentration=current[i],change=change[i],rate=rates[i],wrongConcentration=wrong[i],wrongChange=wrongChange[i],wrongRate=wrongRates[i],exactConcentration=continuum[i],exactDeparture=continuum[i]-mean,exactChange=exactChange[i],error=errors[i],wrongError=wrongErrors[i]).items()})
 cfl=[dt*(abs(a*A)+gamma[(i-1)%N]+gamma[i])/volumes[i]for i in range(N)]
 flux=F*current;flux0=F*initial;fluxChange=F*change
 summary=dict(diffusiveNormLoss=lossDiff,upwindNormLoss=lossAdv,normRateFromFaces=-lossDiff-lossAdv,positiveEulerCorrection=positive,predictedNextNormChangeFromFaces=dt*(-lossDiff-lossAdv)+positive,massRateFromFaces=0,wrongMassRateFromFaces=dot(wrongRates),l1CellAverageError=dot(m.matrix(list(map(abs,errors)))),linfinityCellAverageError=max(map(abs,errors)),wrongL1CellAverageError=dot(m.matrix(list(map(abs,wrongErrors)))))
 if p['meshWarp']==0:summary['wrongMassRateFromFaces']=m.mpf(0)
 if dt==0 or steps==0:
  for c in cells:
   for key in ['change','wrongChange','exactChange','error','wrongError']:c[key]=obs(0)
  for key in ['l1CellAverageError','linfinityCellAverageError','wrongL1CellAverageError']:summary[key]=m.mpf(0)
 # Known constant/no-flow invariants only; no general magnitude-based clipping.
 if mean==0 or eps==0 or a==D==0:
  for c in cells:
   for key in ['change','rate','wrongChange','wrongRate','exactChange','error','wrongError']:c[key]=obs(0)
  for key in summary:summary[key]=m.mpf(0)
  for b in [correct,bad]:b['stableCenteredNormChange']=m.mpf(0);b['normChangeUsingMassLedger']=m.mpf(0)
 out=dict(parameters=p,mesh=dict(edges=edges,volumes=list(map(str,volumes)),distances=list(map(str,dist)),conductances=[obs(x)for x in gamma]),cells=cells,correct={key:obs(v)for key,v in correct.items()},wrong={key:obs(v)for key,v in bad.items()},summary={key:obs(v)for key,v in summary.items()},faces=[dict(full=obs(flux[i]),departure=obs(flux[i]-a*A*mean),change=obs(fluxChange[i]))for i in range(N)],localCfl=[obs(v)for v in cfl],positivitySufficient=max(cfl)<=1)
 return out
ordinary=[];limits=[]
for digits,rows in [(160,ordinary),(1600,limits)]:
 m.mp.dps=digits
 if digits==160:
  params=[base,{**base,'meshWarp':0.},{**base,'velocity':-.5},{**base,'diffusivity':0.},{**base,'velocity':0.},{**base,'velocity':0.,'diffusivity':0.},{**base,'timeStep':0.},{**base,'steps':0},{**base,'relativeContrast':0.},{**base,'meanConcentration':0.},{**base,'velocity':2.,'diffusivity':2.,'meshWarp':.75,'timeStep':.25,'steps':16},{**base,'meshWarp':-.6,'phaseCycles':.25}]
  for i in range(12):
   p=dict(base);p.update(cellCountExponent=3+i%2,length=rng.uniform(.5,4),area=rng.uniform(.25,4),meshWarp=rng.uniform(-.7,.7),velocity=rng.uniform(-2,2),diffusivity=rng.uniform(0,.5),meanConcentration=rng.uniform(.1,4),relativeContrast=rng.uniform(-1,1),mode=1+i%3,phaseCycles=rng.uniform(0,1),timeStep=rng.uniform(.0001,.002),steps=1+i%16);params.append(p)
 else:
  tiny=float.fromhex('0x0.0000000000001p-1022')
  params=[{**base,'timeStep':tiny},{**base,'relativeContrast':tiny},{**base,'relativeContrast':tiny,'timeStep':tiny},{**base,'meanConcentration':tiny,'relativeContrast':tiny,'timeStep':tiny},{**base,'meanConcentration':1e-200,'relativeContrast':1e-200,'timeStep':tiny},{**base,'velocity':tiny,'diffusivity':0.,'area':.25},{**base,'velocity':0.,'diffusivity':tiny,'area':.25},{**base,'phaseCycles':tiny,'meshWarp':0.},{**base,'velocity':-tiny,'diffusivity':tiny,'timeStep':tiny},{**base,'steps':0,'timeStep':tiny},{**base,'relativeContrast':-tiny,'meanConcentration':tiny},{**base,'meanConcentration':0.,'relativeContrast':tiny,'diffusivity':tiny}]
 for i,p in enumerate(params):rows.append(run(p));print(f'{digits} digits case {i+1}/{len(params)}',flush=True)
# Independent exact rational generic one-step examples, also useful without a mesh constructor.
m.mp.dps=1600;ring=[]
for velocity,dt,initial in [(0.,.1,[1.,0.,0.]),(0.,.25,[1.,0.,0.]),(1.,.1,[1.,.5,0.]),(-1.,.1,[1.,.5,0.]),(0.,0.,[1.,0.,0.]),(0.,float.fromhex('0x0.0000000000001p-1022'),[1.,0.,0.])]:
 volumes=list(map(m.mpf,[1,2,1]));gamma=list(map(m.mpf,[1,1,1]));R,W,F=mat(volumes,gamma,m.mpf(velocity));q=m.matrix(initial);step=m.mpf(dt);r=R*q;w=W*q;dot=lambda x:sum(volumes[i]*x[i]for i in range(3))
 ring.append(dict(volumes=[1,2,1],conductances=[1,1,1],concentration=initial,velocityArea=velocity,timeStep=dt,correct=dict(rates=[obs(x)for x in r],values=[obs(x)for x in q+step*r],assembledPhysicalMass=obs(dot(q+step*r)),stableNormChange=obs(dot(m.matrix([(q[i]+step*r[i])**2-q[i]**2 for i in range(3)]))/2)),wrong=dict(rates=[obs(x)for x in w],values=[obs(x)for x in q+step*w],assembledPhysicalMass=obs(dot(q+step*w)),stableNormChange=obs(dot(m.matrix([(q[i]+step*w[i])**2-q[i]**2 for i in range(3)]))/2))))
result=dict(method='independent dense incidence/mass operators and matrix powers on declared coordinates; analytic physical cell integrals and continuum solution',ordinaryDigits=160,limitDigits=1600,ordinaryCases=ordinary,limitCases=limits,ringCases=ring)
target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(result,indent=2)+'\n');print(f'Wrote {target}:24 ordinary,12 limits,6 generic ring cases.',flush=True)
