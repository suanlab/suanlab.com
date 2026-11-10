#!/usr/bin/env python3
"""Independent matrix-square-root/Kraus accounts; a different prepare decomposition."""
import argparse,json
from pathlib import Path
import sympy as S
import mpmath as mp
mp.mp.dps=250
I=S.eye(2);X=S.Matrix([[0,1],[1,0]]);Y=S.Matrix([[0,-S.I],[S.I,0]]);Z=S.diag(1,-1);paulis=[X,Y,Z]
def base():return dict(blochX=0,blochY=1,blochZ=0,blochDenominator=1,axisReal0=1,axisImag0=0,axisReal1=0,axisImag1=0,followReal0=1,followImag0=0,followReal1=1,followImag1=0,strengthNumerator=3,strengthDenominator=5,minusReal=1,minusX=0,minusY=0,minusZ=0,plusReal=1,plusX=0,plusY=0,plusZ=0)
def cases():
 rows=[]
 def put(name,p):rows.append(dict(label=name,input=p.copy()))
 for name,r in [('positive X',(1,0,0)),('positive Y',(0,1,0)),('positive Z',(0,0,1)),('negative Y',(0,-1,0)),('maximally mixed',(0,0,0))]:
  for kn,kd in [(0,1),(3,5),(1,1)]:
   p=base();p.update(blochX=r[0],blochY=r[1],blochZ=r[2],strengthNumerator=kn,strengthDenominator=kd);put(f'{name} strength {kn}/{kd}',p)
 p=base();p.update(strengthNumerator=1,strengthDenominator=2);put('irrational retention',p)
 for name,patch in [
 ('complex mixed state',dict(blochX=1,blochY=2,blochZ=2,blochDenominator=4)),
 ('rational pure tilted state',dict(blochX=3,blochY=0,blochZ=4,blochDenominator=5)),
 ('measure along Y',dict(axisReal1=0,axisImag1=1)),
 ('follow along Y',dict(followReal1=0,followImag1=1)),
 ('measure and follow same Z',dict(followReal1=0)),
 ('complex directions',dict(axisReal0=2,axisImag0=3,axisReal1=-4,axisImag1=5,followReal0=-3,followImag0=2,followReal1=5,followImag1=-1)),
 ('quarter turn feedback',dict(minusReal=1,minusZ=1,plusReal=1,plusX=1)),
 ('unequal feedback axes',dict(minusReal=1,minusX=1,minusY=1,minusZ=1,plusReal=2,plusX=-1,plusY=3,plusZ=-2)),
 ('same global feedback sign',dict(minusReal=-1,plusReal=-1)),
 ('raw axis phase',dict(axisReal0=0,axisImag0=1)),
 ('zero outcome positive Z',dict(blochX=0,blochY=0,blochZ=1,strengthNumerator=1,strengthDenominator=1)),
 ('zero outcome negative Z',dict(blochX=0,blochY=0,blochZ=-1,strengthNumerator=1,strengthDenominator=1)),
 ('no information feedback dephasing',dict(blochX=1,blochY=0,strengthNumerator=0,strengthDenominator=1,plusReal=0,plusZ=1)),
 ('sharp flip feedback',dict(blochX=0,blochY=0,blochZ=1,followReal1=0,strengthNumerator=1,strengthDenominator=1,minusReal=0,minusX=1,plusReal=0,plusX=1)),
 ('sharp reset feedback',dict(blochX=0,blochY=0,blochZ=0,followReal1=0,strengthNumerator=1,strengthDenominator=1,minusReal=0,minusX=1)),
 ('tiny strength',dict(strengthNumerator=1,strengthDenominator=1000000)),
 ('nearly sharp',dict(strengthNumerator=999999,strengthDenominator=1000000)),
 ('nearly orthogonal followup',dict(followReal0=1,followReal1=1000000)),
 ('near compatible tiny disturbance',dict(strengthNumerator=1,strengthDenominator=1000000,followReal0=1000000,followReal1=1)),
 ('near zero conditional mass',dict(blochX=0,blochY=0,blochZ=1,axisReal0=1000000,axisReal1=1,strengthNumerator=1,strengthDenominator=1)),
 ('near identity feedback',dict(strengthNumerator=0,strengthDenominator=1,plusReal=1000000,plusX=1)),
 ('large complex feedback',dict(axisReal0=1000000,axisImag0=999999,axisReal1=-999998,axisImag1=999997,followReal0=-999996,followImag0=999995,followReal1=999994,followImag1=-999993,minusReal=1000000,minusX=-999999,minusY=999998,minusZ=-999997,plusReal=-999996,plusX=999995,plusY=-999994,plusZ=999993))]:
  q=p.copy();q.update(patch);put(name,q)
 # A case with a very small but exactly positive initial determinant.
 q=base();q.update(blochX=1000,blochY=0,blochZ=999999,blochDenominator=1000000,strengthNumerator=1,strengthDenominator=2);put('nearly pure valid state',q)
 q=base();q.update(blochX=1,blochY=0,blochZ=0,blochDenominator=1000000,strengthNumerator=2,strengthDenominator=3);put('tiny nonzero radius',q)
 return rows

def simp(x):return S.simplify(S.expand_complex(x))
def num(x):return str(simp(x))
def cv(x):return dict(real=num(S.re(x)),imaginary=num(S.im(x)))
def mat(x):return [[cv(x[i,j])for j in range(x.cols)]for i in range(x.rows)]
def mpv(x):
 z=S.N(x,255);return mp.mpc(str(S.re(z)),str(S.im(z)))
def dec(x):return 'decimal:'+mp.nstr(x,210)
def numerical(x):return dec(mp.re(mpv(x)))
def nm(x):return [[dict(real=dec(mp.re(mpv(x[i,j]))),imaginary=dec(mp.im(mpv(x[i,j]))))for j in range(x.cols)]for i in range(x.rows)]
def sqrt_positive(m):
 m=m.applyfunc(simp)
 result=S.zeros(2)
 for eigenvalue,rank,vectors in m.eigenvects():
  V=S.Matrix.hstack(*vectors);P=V*(V.H*V).inv()*V.H;result+=S.sqrt(eigenvalue)*P
 return result.applyfunc(simp)

def evaluate(p):
 r=S.Matrix([S.Rational(p[k],p['blochDenominator'])for k in ['blochX','blochY','blochZ']]);rho=(I+sum((r[j]*paulis[j]for j in range(3)),S.zeros(2)))/2;k=S.Rational(p['strengthNumerator'],p['strengthDenominator']);q=1-k*k;h=S.sqrt(q);hcoef,hrad=h.as_coeff_Mul();field_cache={}
 def field(x):
  x=simp(x)
  if x in field_cache:return field_cache[x]
  if h.is_Rational:aa=x;bb=S.Integer(0)
  else:
   x=S.radsimp(x);other=simp(x.xreplace({hrad:-hrad}));aa=simp((x+other)/2);bb=simp((x-other)/(2*h))
  assert aa.is_Rational and bb.is_Rational,(x,h,aa,bb)
  result=dict(rationalPart=str(aa),rootPart=str(bb),exactSign=0 if x==0 else 1 if x>0 else -1,observation=str(aa)if bb==0 else numerical(x));field_cache[x]=result;return result
 def fc(z):return dict(real=field(S.re(z)),imaginary=field(S.im(z)))
 def fm(M):return [[fc(M[i,j])for j in range(M.cols)]for i in range(M.rows)]
 def fv(v):return [field(x)for x in v]
 def axis(prefix):
  z=S.Matrix([p[f'{prefix}Real{i}']+S.I*p[f'{prefix}Imag{i}']for i in range(2)]);P=z*z.H/(z.H*z)[0];return(P,2*P-I,S.Matrix([simp(S.trace(P*v))for v in paulis]))
 P,A,n=axis('axis');PB,B,b=axis('follow');signs=[-1,1];effects=[(I+s*k*A)/2 for s in signs];probs=[simp(S.trace(rho*E))for E in effects];roots=[sqrt_positive(E)for E in effects];m=simp(S.trace(rho*A));mb=simp(S.trace(rho*B));va=1-m*m;vb=1-mb*mb;comm=simp(S.trace(rho*(A*B-B*A))/(2*S.I));C=abs(comm);std=lambda x:num(S.sqrt(x))if S.sqrt(x).is_Rational else numerical(S.sqrt(x))
 Us=[];rot=[];norms=[]
 for prefix in ['minus','plus']:
  w,x,y,z=[S.Integer(p[prefix+key])for key in ['Real','X','Y','Z']];N=w*w+x*x+y*y+z*z;U=(w*I-S.I*(x*X+y*Y+z*Z))/S.sqrt(N);Us.append(U);norms.append(N);rot.append(S.Matrix([[simp(S.trace(paulis[i]*U*paulis[j]*U.H)/2)for j in range(3)]for i in range(3)]))
 def meter(t):
  O1=sum((s*t*E for s,E in zip(signs,effects)),S.zeros(2));O2=t*t*I;noise=O2-O1*O1;bias=O1-A;mean=simp(S.trace(rho*O1));err=simp(S.trace(rho*(O2-O1*A-A*O1+A*A)))
  return dict(values=[num(s*t)for s in signs],mean=num(mean),secondMoment=num(S.trace(rho*O2)),variance=num(S.trace(rho*O2)-mean**2),firstMomentOperator=mat(O1),secondMomentOperator=mat(O2),intrinsicNoiseOperator=mat(noise),intrinsicNoiseExpectation=num(S.trace(rho*noise)),biasOperator=mat(bias),meanBias=num(S.trace(rho*bias)),rmsErrorSquared=num(err),rmsError=std(err),unbiasedForEveryInput=O1==A)
 raw_meter=meter(S.Integer(1));cal_meter=None if k==0 else meter(1/k)
 def ozawa(err,eta2):
  epsilon=S.sqrt(err);eta=S.sqrt(eta2);left=epsilon*eta+epsilon*S.sqrt(vb)+S.sqrt(va)*eta;naive=simp(err*eta2-C*C)
  assert mp.re(mpv(left-C))>=-mp.mpf('1e-200')
  return dict(error=std(err),disturbance=std(eta2),errorTimesDisturbance=numerical(epsilon*eta),correctedLeft=numerical(left),correctedSlack=numerical(left-C),naiveSquaredProductSlack=field(naive),naiveProductBoundViolated=bool(naive<0))
 def bloch(tau):return S.Matrix([simp(S.trace(tau*pauli))for pauli in paulis])
 instruments=[]
 for kind in ['luders','feedback','measure-prepare']:
  if kind=='luders':ks=[[M]for M in roots]
  elif kind=='feedback':ks=[[(Us[i]*roots[i]).applyfunc(simp)]for i in range(2)]
  else:
   # A different Kraus decomposition: |n_s><j|sqrt(E_s), computational j.
   ks=[]
   for i,s in enumerate(signs):
    projector=(I+s*A)/2;vector=projector.columnspace()[0];vector=vector/S.sqrt((vector.H*vector)[0]);ks.append([(vector*S.eye(2)[:,j].H*roots[i]).applyfunc(simp)for j in range(2)])
  allK=[K for group in ks for K in group];taus=[sum((K*rho*K.H for K in group),S.zeros(2)).applyfunc(simp)for group in ks];out=sum(taus,S.zeros(2)).applyfunc(simp);rv=bloch(out);difference=rv-r;distance2=simp((difference.T*difference)[0]/4);bchange=abs(simp(S.trace((out-rho)*B)))/2;dual=sum((K.H*B*K for K in allK),S.zeros(2)).applyfunc(simp);dual0=simp(S.trace(dual)/2);dualv=S.Matrix([simp(S.trace(dual*v)/2)for v in paulis]);eta2=simp(sum(S.trace(rho*(B*K-K*B).H*(B*K-K*B))for K in allK));leak=simp(sum(S.trace(((I-signs[i]*A)/2)*taus[i])for i in range(2)))
  J=S.zeros(4)
  for i in range(2):
   for j in range(2):
    E=S.zeros(2);E[i,j]=1;image=sum((K*E*K.H for K in allK),S.zeros(2)).applyfunc(simp)
    for a in range(2):
     for bidx in range(2):J[2*a+i,2*bidx+j]=image[a,bidx]
  rank=J.rank();shift=bloch(sum((K*K.H for K in allK),S.zeros(2))/2);T=S.Matrix([[simp(S.trace(paulis[i]*sum((K*paulis[j]*K.H for K in allK),S.zeros(2)))/2)for j in range(3)]for i in range(3)])
  outcomes=[]
  for i,s in enumerate(signs):
   cond=None if probs[i]==0 else (taus[i]/probs[i]).applyfunc(simp);joint=[simp(S.trace(((I+t*B)/2)*taus[i]))for t in signs]
   o=dict(label=s,probability=num(probs[i]),unnormalizedDensity=fm(taus[i]),unnormalizedBloch=fv(bloch(taus[i])),conditionalDensity=None if cond is None else fm(cond),conditionalBloch=None if cond is None else fv(bloch(cond)),conditionalPurity=None if cond is None else field(S.trace(cond*cond)),followupJointProbabilities=fv(joint),followupConditionalProbabilities=None if cond is None else fv([v/probs[i]for v in joint]))
   if kind!='measure-prepare':o['kraus']=[nm(K)for K in ks[i]]
   outcomes.append(o)
  identity_channel=shift==S.zeros(3,1)and T==S.eye(3);global_sharp=all(((I-signs[i]*A)*K).applyfunc(simp)==S.zeros(2)for i in range(2)for K in ks[i])
  instruments.append(dict(kind=kind,outcomes=outcomes,nonselectiveDensity=fm(out),nonselectiveBloch=fv(rv),nonselectivePurity=field(S.trace(out*out)),nonselectiveDeterminant=field(out.det()),followupMean=field(S.trace(out*B)),followupVariance=field(1-S.trace(out*B)**2),followupTotalVariation=field(bchange),traceDistanceToInitial=std(distance2),squaredHilbertSchmidtDifference=field(S.trace((out-rho)*(out-rho))),dataProcessingSquaredSlack=field(distance2-bchange*bchange),selectedInputSharpARepeatLeak=field(leak),disturbance=dict(dualObservable=fm(dual),dualConstant=field(dual0),dualVector=fv(dualv),rmsSquared=field(eta2),rms=std(eta2)),rawOzawa=ozawa(S.Rational(raw_meter['rmsErrorSquared']),eta2),calibratedOzawa=None if k==0 else ozawa(S.Rational(cal_meter['rmsErrorSquared']),eta2),channel=dict(affineShift=fv(shift),affineMatrix=[[field(T[i,j])for j in range(3)]for i in range(3)],choi=fm(J),choiRank=rank,minimalKrausCount=rank,suppliedKrausCount=len(allK),choiConvention='unnormalized output tensor input; index=2*output+input; trace=2'),classification=dict(selectedStateUnchanged=out==rho,channelUnital=shift==S.zeros(3,1),channelIdentity=identity_channel,selectedBDistributionUnchanged=bchange==0,rmsBUndisturbedOnSelectedInput=eta2==0,sharpARepeatableOnSelectedInput=leak==0,globallyPreparesOwnOutcomeEigenstate=global_sharp,completelyPositiveByKrausConstruction=True,tracePreserving=True,physicalTimeSpecified=False)))
 return dict(dimension=2,rootSquared=num(q),coherenceRetention=field(h),initialDensity=mat(rho),initialBloch=[num(v)for v in r],initialPurity=num(S.trace(rho*rho)),measurementAxis=[num(v)for v in n],followupAxis=[num(v)for v in b],observables=dict(a=mat(A),b=mat(B)),effects=[mat(E)for E in effects],effectProbabilities=[num(v)for v in probs],effectDeterminants=[num(E.det())for E in effects],effectRank=effects[0].rank(),strength=num(k),feedback=[dict(rawNormSquared=num(norms[i]),rotation=[[num(rot[i][a,bidx])for bidx in range(3)]for a in range(3)],unitary=nm(Us[i]))for i in range(2)],preparation=dict(meanA=num(m),meanB=num(mb),varianceA=num(va),varianceB=num(vb),stdA=std(va),stdB=std(vb),commutatorHalfOverI=num(comm),commutatorMagnitudeHalf=num(C),robertsonSquaredSlack=num(va*vb-C*C)),rawMeter=raw_meter,calibratedMeter=cal_meter,instruments=instruments)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',default='scripts/fixtures/povm-instrument-reference.json');args=parser.parse_args();rows=cases()
 for i,row in enumerate(rows):
  assert len(row['input'])==22 and all(type(x)is int and abs(x)<=1000000 for x in row['input'].values());row['expected']=evaluate(row['input']);print(f'{i+1}/{len(rows)} {row["label"]}',flush=True)
 Path(args.output).write_text(json.dumps(dict(source='Independent SymPy effect square roots and Kraus operator traces; alternate computational-basis preparation Kraus decomposition;250dps210-digit displays',cases=rows),indent=2)+'\n');print('Generated',len(rows),'cases',flush=True)
if __name__=='__main__':main()
