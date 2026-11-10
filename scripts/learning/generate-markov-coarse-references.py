"""Independent Fraction scalar-hidden-coordinate oracle; no production imports."""
from fractions import Fraction as F
from pathlib import Path
import json,argparse
base=dict(weight01=1,weight02=1,weight10=1,weight12=3,weight20=1,weight21=3,transitionDenominator=4,initialWeights=[1,0,0],liftNumerator=1,liftDenominator=2,steps=8,memoryDepth=1)
specs=[('stationary-matched nonlumpable',{}),('same macro different hidden',dict(initialWeights=[0,1,0])),('stationary law',dict(initialWeights=[1,1,1])),('lifted initial law',dict(initialWeights=[1,1,0])),('strong lumpability',dict(weight12=1)),('strong other lifting',dict(weight12=1,liftNumerator=0)),('zero steps',dict(steps=0)),('memoryless approximation',dict(memoryDepth=0)),('full finite memory',dict(memoryDepth=32)),('lift boundary zero',dict(liftNumerator=0)),('lift boundary one',dict(liftNumerator=2)),('identity',dict(weight01=0,weight02=0,weight10=0,weight12=0,weight20=0,weight21=0)),('absorbing coarse state',dict(weight20=0,weight21=0)),('deterministic hidden swap',dict(weight01=4,weight02=0,weight10=4,weight12=0)),('slow alternating hidden',dict(weight01=999999,weight02=1,weight10=999998,weight12=2,weight20=0,weight21=1,transitionDenominator=1000000,steps=32,memoryDepth=2)),('long positive hidden',dict(weight01=0,weight02=1,weight10=0,weight12=2,weight20=1,weight21=0,transitionDenominator=1000000,steps=32)),('persistent hidden mode',dict(weight01=0,weight02=1,weight10=0,weight12=0,weight20=0,weight21=1,liftNumerator=2,steps=16)),('all initial mass B',dict(initialWeights=[0,0,1])),('mixed initial law',dict(initialWeights=[2,5,7],steps=12,memoryDepth=3)),('nonreduced coefficients',dict(weight01=2,weight02=2,weight10=2,weight12=6,weight20=2,weight21=6,transitionDenominator=8,liftNumerator=2,liftDenominator=4)),('asymmetric lift',dict(liftNumerator=1,liftDenominator=3)),('one step',dict(steps=1,memoryDepth=0)),('two steps full',dict(steps=2,memoryDepth=1)),('maximum records',dict(steps=32)),('no internal transitions',dict(weight01=0,weight10=0)),('nearly lumpable',dict(weight01=1,weight02=500000,weight10=1,weight12=500001,weight20=2,weight21=3,transitionDenominator=1000000)),('tiny lift',dict(liftNumerator=1,liftDenominator=1000000,steps=32)),('pure state1 equal exits',dict(weight12=1,initialWeights=[0,1,0]))]
specs += [('negative truncated pseudo-law',dict(weight01=0,weight02=4,weight10=1,weight12=0,weight20=2,weight21=2,liftNumerator=0,memoryDepth=0)),('nonzero underflow memory',dict(weight01=0,weight02=1000000,weight10=0,weight12=999999,weight20=500000,weight21=500000,transitionDenominator=1000000,liftNumerator=1,liftDenominator=1000000,steps=32,memoryDepth=32))]
def add(a,b):return [x+y for x,y in zip(a,b)]
def mul(m,v):return [sum(a*b for a,b in zip(row,v))for row in m]
def encode(x):
 if isinstance(x,F):return str(x)
 if isinstance(x,list):return [encode(v)for v in x]
 if isinstance(x,dict):return {k:encode(v)for k,v in x.items()}
 return x
cases=[]
for label,patch in specs:
 p={**base,**patch};den=p['transitionDenominator'];a,b,c,d,e,g=[F(p['weight'+s],den)for s in ['01','02','10','12','20','21']];alpha=F(p['liftNumerator'],p['liftDenominator']);lam=1-a-c-(1-alpha)*b-alpha*d;out=alpha*b+(1-alpha)*d;A=[[1-out,e+g],[out,1-e-g]]
 # Scalar h=p0-alpha*(p0+p1) eliminates the three-dimensional projector.
 bvec=[d-b,b-d];dvec=[(1-alpha)*c-alpha*a+alpha*(1-alpha)*(d-b),(1-alpha)*e-alpha*g]
 kernels=[[[bvec[i]*lam**lag*dvec[j]for j in range(2)]for i in range(2)]for lag in range(p['steps'])]
 initial=[F(v,sum(p['initialWeights']))for v in p['initialWeights']];h0=initial[0]-alpha*(initial[0]+initial[1]);micro=initial;naive=[micro[0]+micro[1],micro[2]];truncated=naive[:];hist=[];mh=[];th=[];basis0=[F(1),F(0),F(0)];basis1=[F(0),F(1),F(0)]
 def advance(v):return [(1-a-b)*v[0]+c*v[1]+e*v[2],a*v[0]+(1-c-d)*v[1]+g*v[2],b*v[0]+d*v[1]+(1-e-g)*v[2]]
 for k in range(p['steps']+1):
  macro=[micro[0]+micro[1],micro[2]];mh.append(macro);th.append(truncated);instant=mul(A,macro);force=[v*lam**k*h0 for v in bvec];memory=[F(0),F(0)];truncmemory=[F(0),F(0)];omitted=[F(0),F(0)]
  for j in range(k):
   lag=k-1-j;term=mul(kernels[lag],mh[j]);memory=add(memory,term)
   if lag<p['memoryDepth']:truncmemory=add(truncmemory,mul(kernels[lag],th[j]))
   else:omitted=add(omitted,term)
  nextmicro=advance(micro);nextmacro=[nextmicro[0]+nextmicro[1],nextmicro[2]];reconstruct=add(add(instant,force),memory);w0=[basis0[0]+basis0[1],basis0[2]];w1=[basis1[0]+basis1[1],basis1[2]]
  hist.append(dict(micro=micro,macro=macro,hidden=[micro[0]-alpha*macro[0],micro[1]-(1-alpha)*macro[0],F(0)],naiveMarkov=naive,initialAwareTruncated=truncated,naiveErrorTV=sum(abs(x-y)for x,y in zip(macro,naive))/2,truncatedErrorHalfL1=sum(abs(x-y)for x,y in zip(macro,truncated))/2,truncatedIsProbability=sum(truncated)==1 and min(truncated)>=0,instantaneousTerm=instant,initialHiddenTerm=force,fullMemoryTerm=memory,omittedMemoryOnExactHistory=omitted,exactNextMacro=nextmacro,reconstructedNextMacro=reconstruct,reconstructionResidual=[x-y for x,y in zip(reconstruct,nextmacro)],microMassResidual=sum(micro)-1,macroMassResidual=sum(macro)-1,witnessFromState0=w0,witnessFromState1=w1,witnessDistanceTV=sum(abs(x-y)for x,y in zip(w0,w1))/2))
  micro=nextmicro;naive=mul(A,naive);truncated=add(add(mul(A,truncated),force),truncmemory);basis0=advance(basis0);basis1=advance(basis1)
 cases.append(dict(label=label,input=p,expected=encode(dict(hiddenEigenvalue=lam,lumpabilityExitDefect=b-d,instantaneousMacroTransition=A,memoryKernels=[dict(lag=i,matrix=m)for i,m in enumerate(kernels)],history=hist))))
parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args();path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps({'provenance':'Independent Python Fraction scalar hidden-coordinate recurrence and explicit three-component transition equations; no production imports and no rounded reference values.','cases':cases},indent=2)+'\n');print(json.dumps({'preparations':len(cases),'records':sum(len(c['expected']['history'])for c in cases)}))
