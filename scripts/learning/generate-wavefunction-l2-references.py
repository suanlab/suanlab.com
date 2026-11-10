"""Closed binomial Legendre coefficients and Fraction monomial moments.
Independent of production recurrence/Horner integrals;220digit sqrt coordinates."""
from pathlib import Path
from fractions import Fraction as F
from math import comb
import mpmath as mp,json,argparse
mp.mp.dps=220
base=dict(lengthNumerator=1,lengthDenominator=1,breakNumerator=1,breakDenominator=2,leftRealConstant=1,leftRealSlope=0,leftImagConstant=0,leftImagSlope=0,rightRealConstant=0,rightRealSlope=0,rightImagConstant=0,rightImagSlope=0,cutoffDegree=3,bins=8,sampleSteps=8)
specs=[('half interval step',{}),('constant projection of step',dict(cutoffDegree=0)),('linear projection of step',dict(cutoffDegree=1)),('degree twelve step',dict(cutoffDegree=12)),('zero projection of sign-changing state',dict(rightRealConstant=-1,cutoffDegree=0)),('sign-changing degree one',dict(rightRealConstant=-1,cutoffDegree=1)),('constant complex state',dict(leftRealConstant=3,leftImagConstant=4,rightRealConstant=3,rightImagConstant=4,cutoffDegree=0)),('global affine complex state',dict(leftRealConstant=1,leftRealSlope=2,leftImagConstant=-1,leftImagSlope=3,rightRealConstant=1,rightRealSlope=2,rightImagConstant=-1,rightImagSlope=3,cutoffDegree=1)),('affine state degree zero',dict(leftRealSlope=2,rightRealConstant=1,rightRealSlope=2,cutoffDegree=0)),('interior complex discontinuity',dict(breakNumerator=2,breakDenominator=3,leftRealConstant=-2,leftRealSlope=3,leftImagConstant=1,leftImagSlope=-2,rightRealConstant=3,rightRealSlope=-1,rightImagConstant=-4,rightImagSlope=5,cutoffDegree=7)),('small support missed by coarse samples',dict(breakDenominator=1000000,cutoffDegree=12)),('small right support',dict(breakNumerator=999999,breakDenominator=1000000,leftRealConstant=0,rightRealConstant=1,cutoffDegree=12)),('short physical interval',dict(lengthNumerator=1,lengthDenominator=1000000)),('long physical interval',dict(lengthNumerator=10)),('split at zero',dict(breakNumerator=0,leftRealConstant=999,rightRealConstant=1,rightImagConstant=1,cutoffDegree=0)),('split at one',dict(breakNumerator=1,breakDenominator=1,rightRealConstant=999,cutoffDegree=0)),('many bins and samples',dict(bins=32,sampleSteps=64)),('single bin and sample interval',dict(bins=1,sampleSteps=1)),('nonreduced split and length',dict(lengthNumerator=2,lengthDenominator=2,breakNumerator=2,breakDenominator=4)),('large coefficients with cancellation',dict(leftRealConstant=1000000,leftRealSlope=-1000000,rightRealConstant=999999,rightRealSlope=-1000000,leftImagConstant=-999999,leftImagSlope=1000000,rightImagConstant=1000000,rightImagSlope=-999999,cutoffDegree=12)),('imaginary step',dict(leftRealConstant=0,leftImagConstant=1)),('zero-integral linear state',dict(leftRealConstant=-1,leftRealSlope=2,rightRealConstant=-1,rightRealSlope=2,cutoffDegree=0))]
def padd(a,b):return[(a[i]if i<len(a)else F(0))+(b[i]if i<len(b)else F(0))for i in range(max(len(a),len(b)))]
def pmul(a,b):
 r=[F(0)]*(len(a)+len(b)-1)
 for i,x in enumerate(a):
  for j,y in enumerate(b):r[i+j]+=x*y
 return r
def area(poly,a,b):return sum(coef*(b**(i+1)-a**(i+1))/F(i+1)for i,coef in enumerate(poly))
def val(poly,u):return sum(coef*u**i for i,coef in enumerate(poly))
def dens(re,im):return padd(pmul(re,re),pmul(im,im))
def dec(x):return mp.mpf(x.numerator)/x.denominator
def coordinate(x,norm):
 # Exact rational simplification when both norm numerator/denominator are squares.
 from math import isqrt
 a,b=isqrt(norm.numerator),isqrt(norm.denominator)
 if a*a==norm.numerator and b*b==norm.denominator:return str(x/F(a,b))
 if not x:return '0'
 return 'decimal:'+mp.nstr(dec(x)/mp.sqrt(dec(norm)),170)
def normalized(re,im,norm):return dict(real=coordinate(re,norm),imaginary=coordinate(im,norm))
def encode(x):
 if isinstance(x,F):return str(x)
 if isinstance(x,list):return[encode(v)for v in x]
 if isinstance(x,dict):return{k:encode(v)for k,v in x.items()}
 return x
cases=[]
for label,patch in specs:
 p={**base,**patch};L=F(p['lengthNumerator'],p['lengthDenominator']);r=F(p['breakNumerator'],p['breakDenominator']);lr,li,rr,ri=[[F(p[side+part+'Constant']),F(p[side+part+'Slope'])]for side,part in [('left','Real'),('left','Imag'),('right','Real'),('right','Imag')]];ld,rd=dens(lr,li),dens(rr,ri)
 def integrate(left,right,a=F(0),b=F(1)):
  segments=[(a,min(b,r),left),(max(a,r),b,right)]
  return sum(area(poly,x,y)for x,y,poly in segments if x<y)
 Z=integrate(ld,rd);P=[[F((-1)**(n-k)*comb(n,k)*comb(n+k,k))for k in range(n+1)]for n in range(13)];coeff=[((2*n+1)*integrate(pmul(lr,poly),pmul(rr,poly)),(2*n+1)*integrate(pmul(li,poly),pmul(ri,poly)))for n,poly in enumerate(P)];mass=[(a*a+b*b)/F(2*n+1)for n,(a,b)in enumerate(coeff)];N=p['cutoffDegree'];M=sum(mass[:N+1]);hRe=[F(0)];hIm=[F(0)]
 for n,(a,b)in enumerate(coeff[:N+1]):hRe=padd(hRe,[a*v for v in P[n]]);hIm=padd(hIm,[b*v for v in P[n]])
 hd=dens(hRe,hIm);error=integrate(dens(padd(lr,[-x for x in hRe]),padd(li,[-x for x in hIm])),dens(padd(rr,[-x for x in hRe]),padd(ri,[-x for x in hIm])))
 history=[dict(degree=n,capturedProbability=sum(mass[:n+1])/Z,squaredL2Error=(Z-sum(mass[:n+1]))/Z)for n in range(13)]
 bins=[]
 for i in range(p['bins']):
  a,b=F(i,p['bins']),F(i+1,p['bins']);bins.append(dict(index=i,leftPosition=L*a,rightPosition=L*b,targetProbability=integrate(ld,rd,a,b)/Z,projectedMass=area(hd,a,b)/Z,renormalizedProjectedProbability=area(hd,a,b)/M if M else None))
 points=[]
 for i in range(p['sampleSteps']+1):
  u=F(i,p['sampleSteps']);re,im=(lr,li)if u<r else(rr,ri);gRe,gIm=val(re,u),val(im,u);pr,pi=val(hRe,u),val(hIm,u);points.append(dict(index=i,normalizedPosition=u,position=L*u,sourceBranch='left'if u<r else'right',wavefunction=normalized(gRe,gIm,L*Z),density=(gRe*gRe+gIm*gIm)/(L*Z),projectedWavefunction=normalized(pr,pi,L*Z),projectedDensity=(pr*pr+pi*pi)/(L*Z),renormalizedProjectedDensity=(pr*pr+pi*pi)/(L*M)if M else None))
 expected=dict(length=L,breakpoint=L*r,rawDimensionlessNormSquared=Z,rawPhysicalNormSquared=L*Z,selectedCapturedMass=M/Z,selectedSquaredL2Error=(Z-M)/Z,directSquaredL2Error=error/Z,normalizedProjectionFidelity=M/Z if M else None,coefficients=[dict(degree=n,rawLegendreCoefficient=dict(real=a,imaginary=b),orthonormalCoefficient=normalized(a,b,(2*n+1)*Z),modalProbability=mass[n]/Z)for n,(a,b)in enumerate(coeff)],cutoffHistory=history,selectedProjectionRealPolynomial=hRe,selectedProjectionImaginaryPolynomial=hIm,shiftedLegendreGram=[[area(pmul(a,b),F(0),F(1))for b in P[:N+1]]for a in P[:N+1]],probabilityBins=bins,points=points,positionMean=L*integrate(pmul(ld,[F(0),F(1)]),pmul(rd,[F(0),F(1)]))/Z,positionSecondMoment=L*L*integrate(pmul(ld,[F(0),F(0),F(1)]),pmul(rd,[F(0),F(0),F(1)]))/Z)
 # Branch labels are metadata, not numeric observations.
 cases.append(dict(label=label,input=p,expected=encode(expected)))
result=dict(method='Closed binomial shifted-Legendre coefficients; Fraction explicit monomial powers/integrals; mpmath220digit normalized coordinate evaluation. No production recurrence or imports.',cases=cases)
a=argparse.ArgumentParser();a.add_argument('--output',default='scripts/fixtures/wavefunction-l2-reference.json');args=a.parse_args();Path(args.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(preparations=len(cases),points=sum(len(c['expected']['points'])for c in cases))))
