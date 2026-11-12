#!/usr/bin/env python3
"""Independent positive-modal / boundary-order calculations. Standard library only.
No product imports; finite numerical checks complement the manuscript proofs.
"""
import math,cmath,json,sys
from collections import Counter
I=[[1+0j,0j],[0j,1+0j]]
def add(a,b):return [[a[i][j]+b[i][j] for j in range(2)] for i in range(2)]
def scale(a,c):return [[c*x for x in row] for row in a]
def mul(a,b):return [[sum(a[i][k]*b[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
def norm(a):return max(sum(abs(x) for x in row) for row in a)
def sub(a,b):return add(a,scale(b,-1))
def mv(a,v):return [sum(x*y for x,y in zip(row,v)) for row in a]
def inv(a):
 det=a[0][0]*a[1][1]-a[0][1]*a[1][0]
 return scale([[a[1][1],-a[0][1]],[-a[1][0],a[0][0]]],1/det)
def power(a,n):
 p=I
 for _ in range(n):p=mul(a,p)
 return p

def layer(e,width,w):
 n=math.sqrt(e);phase=w*n*width
 return [[cmath.cos(phase),1j*cmath.sin(phase)/n],[1j*n*cmath.sin(phase),cmath.cos(phase)]]
def layer_exp(e,width,w):
 # Matrix exponential from generator powers, independent of the trigonometric form.
 generator=[[0j,1j*w*width],[1j*w*width*e,0j]];total=I;term=I
 for k in range(1,90):term=scale(mul(term,generator),1/k);total=add(total,term)
 return total

def log_near_identity(a,terms=80):
 x=sub(a,I);q=norm(x)
 if q>=1:raise ValueError('This log-series comparison requires norm(M-I)<1')
 total=[[0j,0j],[0j,0j]];term=I
 for k in range(1,terms+1):term=mul(term,x);total=add(total,scale(term,(-1)**(k+1)/k))
 tail=q**(terms+1)/((terms+1)*(1-q))
 return total,tail

def boundary_solve(m):
 a,b=m[0];c,d=m[1]
 system=[[a-b,-1],[c-d,-1]];rhs=[-a-b,-c-d]
 return mv(inv(system),rhs)
def scattering(m):
 a,b=m[0];c,d=m[1];den=a+d-b-c
 return (d-a+c-b)/den,2/den

def modal(theta,n,c_total=1.,travel=1.):
 if not isinstance(n,int) or n<0 or n>32 or not all(math.isfinite(x) for x in [theta,c_total,travel]) or c_total<=0 or travel<=0 or theta<0:raise ValueError('Positive finite scales; 0<=N<=32 integer; theta>=0 required')
 omega=theta/travel;caps=[c_total*8/(math.pi**2*(2*j+1)**2) for j in range(n)];poles=[(2*j+1)*math.pi/(2*travel) for j in range(n)];tail=c_total-math.fsum(caps)
 if theta>=(2*n+1)*math.pi/2:raise ValueError('Omitted-mode bound requires frequency below first omitted pole')
 if any(abs(1-(omega/p)**2)<1e-12 for p in poles):raise ValueError('Near retained pole requires a separate precision calculation')
 xs=[(omega/p)**2 for p in poles];b=omega*(tail+math.fsum(c/(1-x) for c,x in zip(caps,xs)));slope=tail+math.fsum(c*(1+x)/(1-x)**2 for c,x in zip(caps,xs))
 rho=theta/((2*n+1)*math.pi/2);x=rho*rho
 return dict(caps=caps,poles=poles,tail=tail,susceptance=b,slope=slope,portBound=omega*tail*x/(1-x),slopeBound=tail*x*(3-x)/(1-x)**2)

def field_at(e1,e2,f,w,n,x):
 m=power(mul(layer_exp(e2,1-f,w),layer_exp(e1,f,w)),n);r,t=boundary_solve(m);v=[1+r,1-r]
 if x==n:return [t,t]
 walked=0.
 for _ in range(n):
  for e,width in [(e1,f),(e2,1-f)]:
   distance=max(0.,min(width,x-walked));v=mv(layer_exp(e,distance,w),v)
   if x<walked+width:return v
   walked+=width
 return v

def close(a,b,tol=1e-10):assert abs(a-b)<=tol*max(1.,abs(b)),(a,b)
def verify():
 out=Counter()
 for n in [0,1,2,4,8]:
  phases=[.1,.5,1.,1.4]+([1.6,2.,math.pi/2-.01,math.pi/2+.01] if n else [])
  for theta in phases:
   for c_total,travel in [(1.,1.),(1e-10,5e-9),(2.,.4)]:
    q=modal(theta,n,c_total,travel);w=theta/travel;exact=math.tan(theta)*c_total/travel;exact_slope=c_total/math.cos(theta)**2
    assert q['tail']>0 and all(c>0 for c in q['caps']);out['positiveFiniteRealizations']+=1
    normalized_port=(exact-q['susceptance'])/(w*c_total);normalized_slope=(exact_slope-q['slope'])/c_total
    assert normalized_port>=-2e-12 and normalized_port<=q['portBound']/(w*c_total)+2e-12;out['portTailBounds']+=1
    assert normalized_slope>=-2e-8 and normalized_slope<=q['slopeBound']/c_total+2e-8;out['energyTailBounds']+=1
    independent_b=0.;independent_energy=q['tail']/4
    for c,p in zip(q['caps'],q['poles']):
     inductance=1/(p*p*c);z=1j*w*inductance+1/(1j*w*c);current=1/z;vc=current/(1j*w*c)
     close(current*(z),1);independent_b+=current.imag;independent_energy+=c*abs(vc)**2/4+inductance*abs(current)**2/4
     close(inductance,travel*travel/(2*c_total));out['independentBranchImpedanceSolves']+=1
    close((independent_b+w*q['tail'])*travel/c_total,q['susceptance']*travel/c_total,2e-11)
    close(independent_energy/c_total,q['slope']/(4*c_total),2e-9);out['independentBranchEnergySums']+=1
 for n in [0,1,4]:
  q=modal(0,n);assert q['susceptance']==0;close(q['slope'],1);close((q['tail']+sum(q['caps']))/2,.5);out['staticVersusCycleLimitCases']+=1
 # A rounded first pole is not an exact symbolic pole. Reject within this verifier's comparison margin.
 for theta,n,ct,travel in [(2.,0,1,1),(.2,-1,1,1),(.2,1,0,1),(.2,1,1,0),(math.pi/2,1,1,1)]:
  try:modal(theta,n,ct,travel)
  except ValueError:out['invalidModalComparisonRejections']+=1
  else:raise AssertionError('invalid comparison accepted')
 for e1,e2 in [(1.,4.),(2.,8.),(4.,4.),(.5,9.)]:
  for f in [0.,.25,1/3,.75,1.]:
   for w in [0.,.1,.7,1.]:
    first=layer_exp(e1,f,w);second=layer_exp(e2,1-f,w);m=mul(second,first);reverse=mul(first,second)
    close(norm(sub(m,mul(layer(e2,1-f,w),layer(e1,f,w)))),0,1e-12);out['independentLayerExponentials']+=1
    close(norm(sub(reverse,mul(mul(first,m),inv(first)))),0,1e-12);out['cellSimilarityChecks']+=1
    close(m[0][0]+m[1][1],reverse[0][0]+reverse[1][1]);out['bulkTracePreservation']+=1
    for n in [0,1,3]:
     mn=power(m,n);rn=power(reverse,n);r,t=boundary_solve(mn);rr,tt=boundary_solve(rn);rf,tf=scattering(mn)
     close(r,rf);close(t,tf);close(abs(r)**2+abs(t)**2,1);out['independentBoundaryPowerBudgets']+=1
     close(t,tt);close(abs(r)**2,abs(rr)**2);out['reversedStackPowerInvariance']+=1
     close(mn[0][0],rn[1][1]);close(mn[1][1],rn[0][0]);close(mn[0][1],rn[0][1]);close(mn[1][0],rn[1][0]);out['reversedDiagonalSwaps']+=1
     den=mn[0][0]+mn[1][1]-mn[0][1]-mn[1][0];close(rr-r,2*(mn[0][0]-mn[1][1])/den);out['complexReflectionDifferences']+=1
     if abs(rr-r)>1e-6:out['nonzeroReflectionPhaseContrasts']+=1
     if e1==e2 or f in [0.,1.]:close(rr,r);out['homogeneousOrderControls']+=1
 for e1,e2,f in [(2.,8.,1/3),(1.,4.,.25),(2.,8.,.5)]:
  a1=[[0,1],[e1,0]];a2=[[0,1],[e2,0]];avg=add(scale(a1,f),scale(a2,1-f));comm=sub(mul(a2,a1),mul(a1,a2));previous=None
  for w in [.04,.02,.01,.005]:
   first=layer(e1,f,w);second=layer(e2,1-f,w);m=mul(second,first);sym=mul(mul(layer(e1,f/2,w),second),layer(e1,f/2,w));reverse=mul(first,second)
   log,tail=log_near_identity(m);logs,tails=log_near_identity(sym);logr,_=log_near_identity(reverse)
   base=scale(avg,1j*w);bch=add(base,scale(comm,-w*w*f*(1-f)/2))
   error=norm(sub(log,bch));symmetric_error=norm(sub(logs,base));assert tail<1e-30 and tails<1e-30;out['logSeriesTailBounds']+=1
   assert error/w**3<50 and symmetric_error/w**3<50;out['bchRemainderComparisons']+=1
   close(log[0][0]/w**2,-f*(1-f)*(e1-e2)/2,3e-3);close(logr[0][0],-log[0][0]);close(logs[0][0],0);out['originDependentCommutatorChecks']+=1
   if previous:
    assert error<previous[0]*.14 and symmetric_error<previous[1]*.14;out['thirdOrderRefinements']+=1
   previous=(error,symmetric_error)
 for w in [.2,.5,.8]:
  first=layer(2,1/3,w);second=layer(8,2/3,w);half=layer(2,1/6,w);m=mul(second,first);sym=mul(mul(half,second),half)
  close(norm(sub(sym,mul(mul(half,m),inv(half)))),0);close(sym[0][0],sym[1][1]);out['symmetricCellSimilarityChecks']+=1
  for n in [1,2,3]:
   r,t=boundary_solve(power(m,n));rs,ts=boundary_solve(power(sym,n));close(abs(rs)**2+abs(ts)**2,1)
   assert abs(abs(ts)**2-abs(t)**2)>1e-6;out['physicalTerminationPowerContrasts']+=1
 try:log_near_identity([[3,0],[0,1]])
 except ValueError:out['logDomainRejections']+=1
 else:raise AssertionError('out-of-domain log accepted')
 return dict(sorted(out.items()))

def encoded(z):return [z.real,z.imag]
def references():
 lines=[]
 for length in [.5,1.,2.]:
  for voltage in [0.,.5,-1.]:
   for theta in [.2,.8,1.4,2.]:
    lp=250e-9;cp=100e-12;travel=length*math.sqrt(lp*cp);ct=cp*length;q=modal(theta,2,ct,travel)
    lines.append(dict(length=length,voltage=voltage,theta=theta,capacitance=ct,travel=travel,exactSusceptance=math.tan(theta)*ct/travel,exactSlopeRatio=1/math.cos(theta)**2,approximateSusceptance=q['susceptance'],approximateSlope=q['slope'],portBound=q['portBound'],slopeBound=q['slopeBound']))
 stacks=[]
 for w in [.05,.2,.7,1.]:
  for n in [1,3]:
   for e1,e2,f in [(2.,8.,.25),(8.,2.,.75)]:
    m=mul(layer_exp(e2,1-f,w),layer_exp(e1,f,w));r,t=boundary_solve(power(m,n));points=[]
    for x in [0.,n*.37,float(n)]:
     v=field_at(e1,e2,f,w,n,x);points.append(dict(x=x,electric=encoded(v[0]),magnetic=encoded(v[1])))
    stacks.append(dict(e1=e1,e2=e2,f=f,frequency=w,periods=n,halfTrace=(m[0][0]+m[1][1]).real/2,r=encoded(r),t=encoded(t),points=points))
 return dict(lines=lines,stacks=stacks)
if __name__=='__main__':print(json.dumps(references() if '--references' in sys.argv else verify(),sort_keys=True))
