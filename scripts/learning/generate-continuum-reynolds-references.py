"""Independent physical derivatives, actual eight-phase fields and spatial Gauss integrals.

No production import. Ordinary cases use 160 digits; limits use 2200 digits.
The 8-point Gauss rule exactly integrates these polynomial budgets (degree <= 10).
"""
import argparse,json,random
from pathlib import Path
import mpmath as m
parser=argparse.ArgumentParser();parser.add_argument('--output',default='scripts/fixtures/continuum-reynolds-independent.json');parser.add_argument('--edges-only',action='store_true');args=parser.parse_args();target=Path(args.output)
if target.exists():raise SystemExit('Refusing to overwrite reference')
base=dict(height=2.,wavenumber=1.5,spanWidth=1.,density=1.,dynamicViscosity=.1,bulkVelocity=.8,meanAsymmetry=.25,firstStreamfunctionAmplitude=1.,secondStreamfunctionAmplitude=.7,candidateEddyViscosity=.01)
rng=random.Random(261108)
def observation(v):
 v=m.mpf(v)
 return dict(value=str(v),sign=int(m.sign(v)),logMagnitude=None if v==0 else str(m.log(abs(v))))
def run(p,xi,xcycle,phase,nodes,weights):
 H,k,W,rho,mu,Ub,c,a,b,nu=[m.mpf(p[key])for key in base];y=m.mpf(xi)*H;x=m.mpf(xcycle)*2*m.pi/k
 phi=lambda z:(z/H)**2*(1-z/H)**2
 chi=lambda z:phi(z)*(2*z/H-1)
 U=lambda z:6*Ub*(z/H)*(1-z/H)*(1+c*(2*z/H-1))
 # Shift a sine/cosine pair instead of rounding the sum of tiny and quarter phases.
 def tr(f):
  f=m.mpf(f)%1
  if f==0:return m.mpf(0),m.mpf(1)
  if f==m.mpf('.25'):return m.mpf(1),m.mpf(0)
  if f==m.mpf('.5'):return m.mpf(0),m.mpf(-1)
  if f==m.mpf('.75'):return m.mpf(-1),m.mpf(0)
  return m.sin(2*m.pi*f),m.cos(2*m.pi*f)
 def flow(z,xf,pf):
  sx,cx=tr(xf);sp,cp=tr(pf);s=sx*cp+cx*sp;co=cx*cp-sx*sp
  up=a*m.diff(phi,z)*s+b*m.diff(chi,z)*co
  return U(z)+up,-a*k*phi(z)*co+b*k*chi(z)*s,up
 phases=[m.mpf(j)/8 for j in range(8)]
 def cov(z):
  values=[flow(z,0,pf)for pf in phases]
  return sum(v[2]**2 for v in values)/8,sum(v[1]**2 for v in values)/8,sum(v[2]*v[1]for v in values)/8
 def force(z,xf,pf):
  sx,cx=tr(xf);sp,cp=tr(pf);s=sx*cp+cx*sp;co=cx*cp-sx*sp
  # Differentiate the actual streamfunction in physical x and y.
  psi=lambda xx,yy:a*phi(yy)*(m.sin(k*xx)*co+m.cos(k*xx)*s)+b*chi(yy)*(m.cos(k*xx)*co-m.sin(k*xx)*s)
  velocity=lambda xx,yy:(U(yy)+m.diff(lambda zz:psi(xx,zz),yy),-m.diff(lambda zz:psi(zz,yy),xx))
  v=velocity(0,z);dx=[m.diff(lambda xx:velocity(xx,z)[i],0)for i in range(2)];dy=[m.diff(lambda zz:velocity(0,zz)[i],z)for i in range(2)]
  lap=[m.diff(lambda xx:velocity(xx,z)[i],0,2)+m.diff(lambda zz:velocity(0,zz)[i],z,2)for i in range(2)]
  return dict(u=v[0],v=v[1],uPrime=flow(z,xf,pf)[2],derivativeUX=dx[0],derivativeUY=dy[0],derivativeVX=dx[1],derivativeVY=dy[1],laplacianU=lap[0],laplacianV=lap[1],forceX=rho*(v[0]*dx[0]+v[1]*dy[0])-mu*lap[0],forceY=rho*(v[0]*dx[1]+v[1]*dy[1])-mu*lap[1])
 Ruu,Rvv,Ruv=cov(y);shear=m.diff(U,y);curvature=m.diff(U,y,2)
 RuvY=m.diff(lambda z:cov(z)[2],y);RvvY=m.diff(lambda z:cov(z)[1],y)
 G=rho*RuvY-mu*curvature;Gy=rho*RvvY;modeledR=-nu*shear;modeledG=-(mu+rho*nu)*curvature
 # Covariance determinant from the two independent phase directions, avoiding cancellation.
 v0=flow(y,0,0);v1=flow(y,0,m.mpf('.25'));det=(v0[2]*v1[1]-v1[2]*v0[1])**2/4
 local=dict(position=y,meanU=U(y),meanV=0,meanShear=shear,meanCurvature=curvature,covarianceUU=Ruu,covarianceVV=Rvv,covarianceUV=Ruv,covarianceUVGradient=RuvY,covarianceVVGradient=RvvY,covarianceDeterminant=det,fluctuationEnergyPerMass=(Ruu+Rvv)/2,meanForceX=G,meanForceY=Gy,molecularShearStress=mu*shear,reynoldsShearStress=-rho*Ruv,totalMeanShearStress=mu*shear-rho*Ruv,modeledCovarianceUV=modeledR,covarianceDefect=Ruv-modeledR,modeledForceX=modeledG,forceDefect=G-modeledG,modeledTotalShearStress=(mu+rho*nu)*shear,meanForcingWorkDensity=G*U(y),molecularMeanLossDensity=mu*shear**2,meanToFluctuationTransferDensity=-rho*Ruv*shear,modeledTransferDensity=rho*nu*shear**2,forcingWorkDefectDensity=(G-modeledG)*U(y))
 inst=force(y,xcycle,phase)
 def close(u,v):assert m.almosteq(u,v,abs_eps=m.mpf('1e-130')if m.mp.dps==160 else m.mpf('1e-2100')), (str(u),str(v))
 forces=[force(y,0,pf)for pf in phases]
 averaged={key:sum(v[key]for v in forces)/8 for key in ['u','v','forceX','forceY']}
 close(averaged['u'],U(y));close(averaged['v'],0);close(averaged['forceX'],G);close(averaged['forceY'],Gy)
 integrate=lambda f: H/2*sum(w*f(H*(z+1)/2)for z,w in zip(nodes,weights))
 area=2*m.pi/k*W
 visc=area*integrate(lambda z:mu*m.diff(U,z)**2);prod=area*integrate(lambda z:-rho*cov(z)[2]*m.diff(U,z));work=area*integrate(lambda z:(rho*m.diff(lambda q:cov(q)[2],z)-mu*m.diff(U,z,2))*U(z))
 modeled=area*integrate(lambda z:rho*nu*m.diff(U,z)**2)
 close(work,visc+prod)
 budgets=dict(meanKineticEnergy=area*integrate(lambda z:rho*U(z)**2/2),fluctuationKineticEnergy=area*integrate(lambda z:rho*sum(cov(z)[:2])/2),molecularMeanLoss=visc,meanToFluctuationTransfer=prod,meanForcingWork=work,modeledTransfer=modeled,modeledForcingWork=visc+modeled,forcingWorkDefect=prod-modeled)
 # Symmetry-exact zeros can acquire only integration/phase rounding residue.
 if c==0 or a==0 or b==0 or Ub==0:
  close(prod,0);budgets['meanToFluctuationTransfer']=m.mpf(0);budgets['meanForcingWork']=visc;budgets['forcingWorkDefect']=-modeled
 if xi in [0.,.5,1.]:
  for key in ['covarianceUVGradient','covarianceVVGradient','meanForceY']:close(local[key],0);local[key]=m.mpf(0)
 if xi in [0.,1.]:
  # These are exact boundary identities, not a magnitude threshold for general observations.
  for key in ['u','v','uPrime','derivativeUX','derivativeVX','derivativeVY']:
   close(inst[key],0);inst[key]=m.mpf(0)
  for key in ['meanU','meanForcingWorkDensity','forcingWorkDefectDensity','covarianceUU','covarianceVV','covarianceUV','covarianceDeterminant','meanToFluctuationTransferDensity','fluctuationEnergyPerMass']:close(local[key],0);local[key]=m.mpf(0)
 if xi==.5:close(local['covarianceDeterminant'],0);local['covarianceDeterminant']=m.mpf(0)
 return dict(parameters=p,xi=xi,xCycles=xcycle,phaseCycles=phase,expected={key:observation(v)for key,v in local.items()},instantaneous={key:observation(v)for key,v in inst.items()},budgets={key:observation(v)for key,v in budgets.items()},inferredEddyViscosity=observation(-Ruv/shear)if shear!=0 else None,inferredStatus='defined'if shear!=0 else 'indeterminate'if Ruv==0 else 'nonzero-stress-at-zero-shear')
cases=[];limits=[]
for digits,output in ([(2200,limits)]if args.edges_only else [(160,cases),(2200,limits)]):
 m.mp.dps=digits;nodes,weights=m.gauss_quadrature(8,'legendre')
 if digits==160:
  inputs=[(base,.35,.125,.25),({**base,'meanAsymmetry':0.},.5,0.,0.),({**base,'meanAsymmetry':0.,'firstStreamfunctionAmplitude':0.,'secondStreamfunctionAmplitude':0.},.5,.25,.5),(base,0.,0.,0.),(base,1.,.25,.75),({**base,'secondStreamfunctionAmplitude':-.7},.6,.5,.125),({**base,'dynamicViscosity':0.},.2,.25,.25),({**base,'bulkVelocity':0.},.25,0.,.5)]
  for i in range(16):
   p=dict(base);p.update(height=rng.uniform(.25,4),wavenumber=rng.uniform(.25,4),spanWidth=rng.uniform(.25,4),density=rng.uniform(.25,4),dynamicViscosity=rng.uniform(.01,2),bulkVelocity=rng.uniform(-2,2),meanAsymmetry=rng.uniform(-1,1),firstStreamfunctionAmplitude=rng.uniform(-2,2),secondStreamfunctionAmplitude=rng.uniform(-2,2),candidateEddyViscosity=rng.uniform(0,4));inputs.append((p,rng.uniform(.05,.95),rng.uniform(0,1),rng.uniform(0,1)))
 else:
  tiny=float.fromhex('0x0.0000000000001p-1022')
  inputs=[({**base,'firstStreamfunctionAmplitude':1e-200,'secondStreamfunctionAmplitude':-1e-200},.35,0.,.25),({**base,'firstStreamfunctionAmplitude':tiny,'secondStreamfunctionAmplitude':tiny},.5,0.,0.),(base,1e-200,0.,.25),(base,tiny,0.,0.),({**base,'meanAsymmetry':tiny},.5,0.,0.),({**base,'dynamicViscosity':tiny},.35,tiny,.25),({**base,'firstStreamfunctionAmplitude':tiny},.5,0.,0.),({**base,'bulkVelocity':tiny,'candidateEddyViscosity':tiny},.35,0.,0.)]
 if args.edges_only:inputs=[({**base,'meanAsymmetry':1.},tiny,0.,0.),({**base,'firstStreamfunctionAmplitude':1.,'secondStreamfunctionAmplitude':1.},tiny,0.,.125)]
 for i,case_args in enumerate(inputs):
  output.append(run(*case_args,nodes,weights));print(f'{digits} digits case {i+1}/{len(inputs)}',flush=True)
result=dict(method='actual 8-phase streamfunction fields; physical derivatives; independent 8-point spatial Gauss integration',ordinaryDigits=160,limitDigits=2200,ordinaryCases=cases,limitCases=limits,phaseForceChecks=len(cases)+len(limits),spatialIntegrals=(len(cases)+len(limits))*5)
target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(result,indent=2)+'\n');print(f'Wrote {target}: {len(cases)} ordinary and {len(limits)} limit cases, {(len(cases)+len(limits))*5} spatial integrals.',flush=True)
