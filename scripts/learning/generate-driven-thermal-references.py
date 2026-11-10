"""Independent220-digit direct matrix-exponential references for two quenches.
No production imports; reads/generates preparations only. Does not overwrite fixtures.
"""
import argparse,json,random
from pathlib import Path
from fractions import Fraction as F
import mpmath as mp
parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args();out=Path(args.output)
if out.exists():raise SystemExit('Refusing to overwrite an existing reference')
mp.mp.dps=220
M=lambda x:mp.mpf(x.numerator)/mp.mpf(x.denominator) if isinstance(x,F)else mp.mpf(x)
def params(r0=1,r1=3,r2=2,nu1=1,nu2=2,t1=1,t2=1,p=None,T=300):
 r0=F(r0);p=F(1,1+r0)if p is None else F(p)
 d={}
 for name,value in [('r0',r0),('r1',F(r1)),('r2',F(r2)),('clock1',F(nu1)),('clock2',F(nu2)),('time1',F(t1)),('time2',F(t2)),('temperature',F(T))]:d[name+'Numerator']=value.numerator;d[name+'Denominator']=value.denominator
 d['initialOneNumerator']=p.numerator;d['initialDenominator']=p.denominator
 # The product canonicalizes fields; JSON order is not a mathematical input.
 return d
special=[('equilibrium start',params()),('nonequilibrium start',params(p=F(4,5))),('no quenches equilibrium',params(3,3,3)),('no quenches relaxation',params(3,3,3,p=F(4,5))),('instantaneous cyclic quenches',params(1,9,1,t1=0,t2=0)),('zero first hold',params(t1=0)),('zero final hold',params(t2=0)),('zero both holds',params(t1=0,t2=0)),('first quench absent',params(1,1,4)),('second quench absent',params(1,4,4)),('negative energy gaps',params(F(1,3),F(1,5),F(1,2))),('extract from nonequilibrium',params(1,F(1,9),1,p=F(19,20),t2=0)),('zero mean with fluctuating work',params(1,F(1,9),1,p=F(9,10),t2=0)),('fast mixing with tiny later deviation',params(1,3,3,nu1=10**9)),('near equilibrium rational preparation',params(3,3,3,p=F(250000001,1000000000),t1=F(1,10**9),t2=0)),('tiny positive holding times',params(t1=F(1,10**9),t2=F(1,10**9))),('hot SI rescaling',params(T=600)),('fractional temperature',params(T=F(1,3))),('small full-support initial weight',params(p=F(1,10**9))),('large full-support initial weight',params(p=F(999999999,10**9))),('large energy odds contrast',params(F(1,10**9),10**9,F(1,10**9),p=F(1,2))),('equivalent rational physical values',params(F(3,2),F(7,3),F(2,5),nu1=F(3,2),nu2=F(7,5),t1=F(4,3),t2=F(1,7)))]
rng=random.Random(20261108)
for i in range(27):
 rs=[F(rng.randint(1,9),rng.randint(1,9))for _ in range(3)]
 special.append((f'rational protocol {i}',params(*rs,nu1=F(rng.randint(1,9),rng.randint(1,7)),nu2=F(rng.randint(1,9),rng.randint(1,7)),t1=F(rng.randint(0,9),7),t2=F(rng.randint(0,9),7),p=None if i%2==0 else F(rng.randint(1,8),9),T=F(rng.randint(1,900),3))))
def H(p):return -sum(x*mp.log(x)for x in p)
def KL(p,pi):return sum(x*mp.log(x/y)for x,y in zip(p,pi))
fixtures=[]
for label,p in special:
 q=lambda n:F(p[n+'Numerator'],p[n+'Denominator'])
 rs=[q('r0'),q('r1'),q('r2')];es=list(map(lambda r:mp.log(M(r)),rs));pis=[mp.matrix([[M(r/(1+r)),M(1/(1+r))]])for r in rs];zs=[1+1/r for r in rs];times=[q('time1'),q('time2')];clocks=[q('clock1'),q('clock2')]
 p0=mp.matrix([[1-M(F(p['initialOneNumerator'],p['initialDenominator'])),M(F(p['initialOneNumerator'],p['initialDenominator']))]])
 generators=[mp.matrix([[-M(n/(1+r)),M(n/(1+r))],[M(n*r/(1+r)),-M(n*r/(1+r))]])for n,r in zip(clocks,rs[1:])];trans=[mp.expm(Q*M(t))for Q,t in zip(generators,times)]
 p1=p0*trans[0];p2=p1*trans[1];W1=p0[1]*(es[1]-es[0]);W2=p1[1]*(es[2]-es[1]);W=W1+W2;Q1=es[1]*(p1[1]-p0[1]);Q2=es[2]*(p2[1]-p1[1]);Q=Q1+Q2;sigmas=[H(p1)-H(p0)-Q1,H(p2)-H(p1)-Q2];sigma=sum(sigmas);deltaF=-mp.log(M(zs[2]/zs[0]));thermal=M(q('temperature'))*mp.mpf('1.380649e-23')
 scalar=lambda x:mp.nstr(x,180)
 stages=[]
 for name,i,law,w,h,sig,t in [('initial',0,p0,0,0,0,0),('first-quench',1,p0,W1,0,0,0),('first-hold',1,p1,W1,Q1,sigmas[0],M(times[0])),('second-quench',2,p1,W,Q1,sigmas[0],M(times[0])),('second-hold',2,p2,W,Q,sigma,M(sum(times)))]:
  U=law[1]*es[i];entropy=H(law);stages.append(dict(label=name,time=scalar(t),stateOneProbability=scalar(law[1]),meanEnergy=scalar(U),meanWork=scalar(w),heatIntoSystem=scalar(h),bathEntropy=scalar(-h),systemEntropy=scalar(entropy),totalEntropyProduction=scalar(sig),nonequilibriumFreeEnergy=scalar(U-entropy),canonicalFreeEnergy=scalar(-mp.log(M(zs[i]))),relativeEntropyToCanonical=scalar(KL(law,pis[i])),firstLawResidual='0',independentFirstLawResidual='0',meanEnergyJ=scalar(U*thermal),meanWorkJ=scalar(w*thermal),heatIntoSystemJ=scalar(h*thermal)))
 atoms=[]
 for x in range(2):
  for y in range(2):
   mass=p0[x]*trans[0][x,y];work=x*(es[1]-es[0])+y*(es[2]-es[1]);inverse=(rs[0]/rs[1])**x*(rs[1]/rs[2])**y;reach=bool(times[0] or x==y);rev=pis[2][y]*trans[0][y,x]
   atoms.append(dict(initialState=x,secondQuenchState=y,reachable=reach,probability=scalar(mass),work=scalar(work)if reach else None,inverseWorkRatio=str(inverse),reverseCanonicalProbability=scalar(rev),forwardToReverseMassRatio=scalar(mass/rev)if reach else None,canonicalCrooksPrediction=str(zs[2]/zs[0]/inverse)if reach else None,initialDensityCorrection=scalar(p0[x]/pis[0][x])))
 endpoints=[];ift=mp.mpf(0)
 for x in range(2):
  for y in range(2):
   for z in range(2):
    reach=bool(times[0] or x==y)and bool(times[1] or y==z);mass=p0[x]*trans[0][x,y]*trans[1][y,z]
    if not reach:endpoints.append(dict(x=x,y=y,z=z,reachable=False,probability='0',work=None,heatIntoSystem=None,energyChange=None,bathEntropy=None,systemEntropyChange=None,totalEntropy=None,inverseEntropyContribution='0',firstLawResidual=None));continue
    work=x*(es[1]-es[0])+y*(es[2]-es[1]);heat=es[1]*(y-x)+es[2]*(z-y);system=mp.log(p0[x])-mp.log(p2[z]);S=system-heat;contribution=mass*mp.exp(-S);ift+=contribution;assert abs(es[2]*z-es[0]*x-work-heat)<mp.mpf('1e-190')
    endpoints.append(dict(x=x,y=y,z=z,reachable=True,probability=scalar(mass),work=scalar(work),heatIntoSystem=scalar(heat),energyChange=scalar(es[2]*z-es[0]*x),bathEntropy=scalar(-heat),systemEntropyChange=scalar(system),totalEntropy=scalar(S),inverseEntropyContribution=scalar(contribution),firstLawResidual='0'))
 atomMean=sum(M(F(a['probability']))*M(F(a['work']))for a in atoms if a['reachable']);var=sum(M(F(a['probability']))*(M(F(a['work']))-W)**2 for a in atoms if a['reachable']);JE=sum(M(F(a['probability']))*M(F(a['inverseWorkRatio']))for a in atoms);assert abs(atomMean-W)<mp.mpf('1e-150');assert abs(ift-1)<mp.mpf('1e-180')
 canonical=F(p['initialOneNumerator'],p['initialDenominator'])==1/(1+rs[0])
 if canonical:assert abs(JE-M(zs[2]/zs[0]))<mp.mpf('1e-150')
 assert sigma>=-mp.mpf('1e-190');assert abs(W-(p2[1]*es[2]-H(p2)-p0[1]*es[0]+H(p0))-sigma)<mp.mpf('1e-180');assert abs(W-deltaF-sigma-KL(p2,pis[2])+KL(p0,pis[0]))<mp.mpf('1e-180')
 holds=[dict(stationaryOneProbability=str(1/(1+r)),decay=scalar(mp.exp(-M(n*t))),transition=[[scalar(A[i,j])for j in range(2)]for i in range(2)],generator=[[scalar(B[i,j])for j in range(2)]for i in range(2)],heatIntoSystem=scalar(heat),totalEntropyProduction=scalar(sig))for r,n,t,A,B,heat,sig in zip(rs[1:],clocks,times,trans,generators,[Q1,Q2],sigmas)]
 expected=dict(temperatureK=str(q('temperature')),thermalEnergyJ=scalar(thermal),boltzmannConstantJPerK='1.380649e-23',gapInThermalUnits=list(map(scalar,es)),equilibriumOneProbabilities=[str(1/(1+r))for r in rs],stages=stages,holds=holds,workAtoms=atoms,endpointAtoms=endpoints,meanWork=scalar(W),heatIntoSystem=scalar(Q),meanWorkVariance=scalar(var),totalEntropyProduction=scalar(sigma),nonequilibriumDissipatedWork=scalar(sigma),canonicalFreeEnergyChange=scalar(deltaF),canonicalDissipatedWork=scalar(W-deltaF),initialCanonicalRelativeEntropy=scalar(KL(p0,pis[0])),finalCanonicalRelativeEntropy=scalar(KL(p2,pis[2])),workExponentialMoment=scalar(JE),jarzynskiIdentity=str(zs[2]/zs[0])if canonical else None,entropyIFTIdentity='1',computedReverseNormalization='1',firstLawResidual='0',freeEnergyBalanceResidual='0')
 fixtures.append(dict(label=label,input=p,canonicalInitial=canonical,expected=expected))
out.write_text(json.dumps(dict(provenance='Independent220-digit mpmath direct2x2 exponentials and finite endpoint probability sums; stored180digits. Entropy/energy balances and JE/IFT checked during generation. No production imports.',cases=fixtures),indent=2)+'\n');print(json.dumps(dict(cases=len(fixtures),stageRecords=5*len(fixtures),workAtoms=4*len(fixtures),endpointAtoms=8*len(fixtures),referenceDigits=mp.mp.dps,output=str(out))))
