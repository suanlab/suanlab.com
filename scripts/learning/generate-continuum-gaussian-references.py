"""Independent finite Fourier polynomial algebra, physical periodic quadrature and Gaussian convolution.

Production uses positive covariance factors and stable amplitude combinations. This script
instead multiplies Laurent polynomials, applies Gaussian multipliers to every actual mode,
and differentiates/integrates those physical fields. No production import.
"""
import argparse,json,random
from pathlib import Path
import mpmath as m
parser=argparse.ArgumentParser();parser.add_argument('--output',default='scripts/fixtures/continuum-gaussian-independent.json');args=parser.parse_args();target=Path(args.output)
if target.exists():raise SystemExit('Refusing to overwrite reference')
base=dict(wavenumber=1.,filterWidth=.5,kinematicViscosity=.1,initialFirstAmplitude=1.,initialSecondAmplitude=-.5,density=1.,area=1.,frameVelocity=0.)
rng=random.Random(261109)
def add(*polys):
 out={}
 for p in polys:
  for n,c in p.items():out[n]=out.get(n,m.mpc(0))+c
 return out
def scale(p,c):return{n:v*c for n,v in p.items()}
def multiply(a,b):
 out={}
 for n,c in a.items():
  for j,d in b.items():out[n+j]=out.get(n+j,m.mpc(0))+c*d
 return out
def observe(v):
 v=m.re(v);return dict(value=str(v),sign=int(m.sign(v)),logMagnitude=None if v==0 else str(m.log(abs(v))))
def run(p,time,xcycle,phase,convolution=False):
 k,width,nu,a0,b0,rho,area,C=[m.mpf(p[key])for key in base];time=m.mpf(time);xcycle=m.mpf(xcycle);phase=m.mpf(phase)
 A=a0*m.exp(-nu*k*k*time);B=b0*m.exp(-4*nu*k*k*time);L=2*m.pi/k;s2=(k*width)**2
 departure={1:A/(2j),-1:-A/(2j),2:B/(2j),-2:-B/(2j)};u=add({0:C},departure)
 filt=lambda p,times=1:{n:c*m.exp(-times*n*n*s2/2)for n,c in p.items()}
 dx=lambda p:{n:1j*n*k*c for n,c in p.items()}
 dt=lambda p:{n:(-nu*n*n*k*k-1j*n*k*C)*c for n,c in p.items()}
 w=filt(u);wt=dt(w);wx=dx(w);wxx=dx(wx);vx=dx(departure);ut=dt(u)
 force=multiply(departure,vx);F=filt(force);variance=add(filt(multiply(u,u)),scale(multiply(w,w),-1));varianceX=dx(variance)
 transfer=scale(multiply(variance,wx),-m.mpf('.5'));loss=scale(multiply(wx,wx),nu);forceWork=multiply(w,F)
 flux=add(scale(multiply(multiply(w,w),w),m.mpf(1)/3),scale(multiply(w,wx),-nu),scale(multiply(w,variance),m.mpf('.5')))
 initialArg=2*(xcycle+phase)-k*C*time/m.pi
 def value(poly,shift=0):return m.re(sum(c*m.mpc(m.cospi(n*(initialArg+shift)),m.sinpi(n*(initialArg+shift)))for n,c in poly.items()))
 local=dict(firstAmplitude=A,secondAmplitude=B,firstAmplitudeChange=A-a0,secondAmplitudeChange=B-b0,firstTransferFunction=m.exp(-s2/2),secondTransferFunction=m.exp(-2*s2),velocity=value(u),velocityDeparture=value(departure),velocityGradient=value(vx),velocityCurvature=value(dx(vx)),velocityTimeRate=value(ut),filteredVelocity=value(w),filteredDeparture=value(filt(departure)),filteredGradient=value(wx),filteredCurvature=value(wxx),filteredTimeRate=value(wt),twiceFilteredVelocity=value(filt(u,2)),residualVelocity=value(add(u,scale(w,-1))),filteredResidualVelocity=value(add(w,scale(filt(u,2),-1))),variance=value(variance),varianceGradient=value(varianceX),filteredVelocitySquare=value(filt(multiply(u,u))),force=value(force),filteredForce=value(F),resolvedNonlinearFluxGradient=value(multiply(w,wx)),filteredNonlinearFluxGradient=value(filt(multiply(u,vx))),subgridForce=-value(varianceX)/2,transferPerMass=value(transfer),resolvedMolecularLossPerMass=value(loss),filteredForcingWorkPerMass=value(forceWork),resolvedEnergyPerMass=value(multiply(w,w))/2,residualEnergyPerMass=value(variance)/2,resolvedEnergyRatePerMass=value(multiply(w,wt)),resolvedEnergyFlux=value(flux),resolvedEnergyFluxGradient=value(dx(flux)))
 # Exact Fourier integration and a separate physical-position periodic trapezoidal rule.
 # 16 equispaced points integrate all products here, whose highest frequency is <=6.
 def integral(poly):
  exact=L*m.re(poly.get(0,0));quad=L*sum(value(poly,m.mpf(j)/8)for j in range(16))/16
  tol=m.mpf('1e-125')if m.mp.dps==160 else m.mpf('1e-3100')
  assert m.almosteq(exact,quad,abs_eps=tol,rel_eps=tol),(str(exact),str(quad))
  return rho*area*exact
 fullLoss=scale(multiply(vx,vx),nu);filteredFullLoss=filt(fullLoss);residualLoss=add(filteredFullLoss,scale(loss,-1))
 totals=dict(totalKineticEnergy=integral(scale(multiply(u,u),m.mpf('.5'))),resolvedKineticEnergy=integral(scale(multiply(w,w),m.mpf('.5'))),residualKineticEnergy=integral(scale(variance,m.mpf('.5'))),totalMolecularLoss=integral(fullLoss),resolvedMolecularLoss=integral(loss),residualMolecularLoss=integral(residualLoss),totalEnergyRate=integral(multiply(u,ut)),resolvedEnergyRate=integral(multiply(w,wt)),residualEnergyRate=integral(add(filt(multiply(u,ut)),scale(multiply(w,wt),-1))),totalForcingWork=integral(multiply(u,force)),resolvedForcingWork=integral(forceWork),residualForcingWork=integral(add(filt(multiply(u,force)),scale(forceWork,-1))),resolvedToResidualTransfer=integral(transfer))
 tol=m.mpf('1e-125')if m.mp.dps==160 else m.mpf('1e-3100')
 assert m.almosteq(local['velocityTimeRate']+local['velocity']*local['velocityGradient'],nu*local['velocityCurvature']+local['force'],abs_eps=tol)
 assert m.almosteq(local['resolvedEnergyRatePerMass']+local['resolvedEnergyFluxGradient'],-local['resolvedMolecularLossPerMass']+local['filteredForcingWorkPerMass']-local['transferPerMass'],abs_eps=tol)
 for part in ['total','resolved','residual']:
  transferWork=totals['resolvedToResidualTransfer']*(-1 if part=='resolved'else 1 if part=='residual'else 0)
  assert m.almosteq(totals[part+'EnergyRate'],totals[part+'ForcingWork']-totals[part+'MolecularLoss']+transferWork,abs_eps=tol)
 # Exact point-product identities, not a general small-value threshold.
 for output,left,right in [('filteredForcingWorkPerMass','filteredVelocity','filteredForce'),('resolvedEnergyPerMass','filteredVelocity','filteredVelocity'),('resolvedEnergyRatePerMass','filteredVelocity','filteredTimeRate'),('resolvedMolecularLossPerMass','filteredGradient','filteredGradient'),('transferPerMass','variance','filteredGradient')]:
  if local[left]==0 or local[right]==0:
   assert abs(local[output])<tol;local[output]=m.mpf(0)
 if width==0:
  for key in ['variance','varianceGradient','residualVelocity','filteredResidualVelocity','subgridForce','transferPerMass','residualEnergyPerMass']:assert abs(local[key])<tol;local[key]=m.mpf(0)
  for key in ['residualKineticEnergy','residualMolecularLoss','residualEnergyRate','residualForcingWork','resolvedToResidualTransfer','resolvedForcingWork']:assert abs(totals[key])<tol;totals[key]=m.mpf(0)
 # Symmetry-exact mean work is zero for the periodic manufactured force.
 assert abs(totals['totalForcingWork'])<tol;totals['totalForcingWork']=m.mpf(0)
 out=dict(parameters=p,time=float(time),xCycles=float(xcycle),phaseCycles=float(phase),expected={n:observe(v)for n,v in local.items()},budgets={n:observe(v)for n,v in totals.items()})
 if convolution and width!=0:
  angle=m.pi*initialArg
  actual=lambda z:C+A*m.sin(angle+k*width*z)+B*m.sin(2*(angle+k*width*z))
  actualGradient=lambda z:k*(A*m.cos(angle+k*width*z)+2*B*m.cos(2*(angle+k*width*z)))
  normal=lambda z:m.exp(-z*z/2)/m.sqrt(2*m.pi)
  integrate=lambda f:m.quad(lambda z:normal(z)*f(z),[-m.inf,-4,0,4,m.inf])
  mean=integrate(actual);var=integrate(lambda z:(actual(z)-mean)**2);forc=integrate(lambda z:(actual(z)-C)*actualGradient(z))
  for a,b in [(mean,local['filteredVelocity']),(var,local['variance']),(forc,local['filteredForce'])]:assert m.almosteq(a,b,abs_eps=m.mpf('1e-125'),rel_eps=m.mpf('1e-125')),(a,b)
  out['gaussianConvolution']={n:observe(v)for n,v in dict(mean=mean,variance=var,force=forc).items()}
 return out
ordinary=[];limits=[]
for digits,output in [(160,ordinary),(3200,limits)]:
 m.mp.dps=digits
 if digits==160:
  inputs=[(base,.3,.125,0.),({**base,'initialSecondAmplitude':.5},.3,.3,0.),({**base,'filterWidth':0.},.2,.25,0.),({**base,'initialSecondAmplitude':0.},0.,.25,0.),({**base,'initialFirstAmplitude':0.},.5,.125,0.),({**base,'kinematicViscosity':0.},2.,.125,0.),({**base,'frameVelocity':1.},.3,.125,.25),({**base,'wavenumber':4.,'filterWidth':4.,'initialSecondAmplitude':0.},0.,.125,0.)]
  for i in range(16):
   p=dict(base);p.update(wavenumber=rng.uniform(.5,2),filterWidth=rng.uniform(.05,1),kinematicViscosity=rng.uniform(0,1),initialFirstAmplitude=rng.uniform(-2,2),initialSecondAmplitude=rng.uniform(-2,2),density=rng.uniform(.25,4),area=rng.uniform(.25,4),frameVelocity=rng.uniform(-2,2));inputs.append((p,rng.uniform(0,4),rng.uniform(0,1),rng.uniform(0,1)))
 else:
  tiny=float.fromhex('0x0.0000000000001p-1022')
  inputs=[({**base,'filterWidth':1e-20},0.,0.,0.),({**base,'filterWidth':tiny},0.,0.,0.),({**base,'filterWidth':tiny,'initialSecondAmplitude':0.},0.,.25,0.),({**base,'filterWidth':tiny,'initialFirstAmplitude':1e-200,'initialSecondAmplitude':-.5e-200},0.,0.,0.),({**base,'initialFirstAmplitude':tiny,'initialSecondAmplitude':tiny},.3,.125,0.),({**base,'frameVelocity':2.},tiny,0.,0.),({**base,'filterWidth':tiny,'frameVelocity':2.},tiny,0.,.25),({**base,'wavenumber':4.,'filterWidth':4.,'kinematicViscosity':2.},4.,.125,0.),({**base,'filterWidth':1e-200,'initialSecondAmplitude':0.},0.,.25,0.),({**base,'filterWidth':tiny},tiny,0.,0.),({**base,'initialFirstAmplitude':0.,'initialSecondAmplitude':0.,'frameVelocity':2.},tiny,0.,.25),({**base,'kinematicViscosity':tiny},tiny,.125,0.)]
 for i,params in enumerate(inputs):
  output.append(run(*params,convolution=digits==160 and i in [0,1,3,4,5,6]));print(f'{digits} digits case {i+1}/{len(inputs)}',flush=True)
result=dict(method='independent Laurent polynomial products, derivatives, Gaussian mode filtering and physical 16-point quadrature; six actual infinite-line Gaussian convolutions',ordinaryDigits=160,limitDigits=3200,ordinaryCases=ordinary,limitCases=limits,spatialIntegralChecks=36*13,actualGaussianConvolutionCases=6)
target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(result,indent=2)+'\n');print(f'Wrote {target}:24 ordinary and12 limit cases.',flush=True)
