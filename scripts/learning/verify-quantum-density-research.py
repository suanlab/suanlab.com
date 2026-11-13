"""Independent sampling sums, constrained fits, Choi maps and static-noise dilation."""
from fractions import Fraction as Q
from math import comb
import itertools,json,math
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import expm
# Exact finite binomial moments of the unprojected estimator.
moment_cases=0
for n,p in itertools.product([1,2,5,10,20],[Q(0),Q(1,10),Q(3,10),Q(1,2),Q(4,5),Q(1)]):
    law=[Q(comb(n,k))*p**k*(1-p)**(n-k)for k in range(n+1)];mean=sum((Q(2*k,n)-1)*mass for k,mass in enumerate(law));second=sum((Q(2*k,n)-1)**2*mass for k,mass in enumerate(law));r=2*p-1
    assert sum(law)==1 and mean==r and second-mean*mean==(1-r*r)/n;moment_cases+=1
# Sum exact binomial masses inside each raw-coordinate confidence box.
coverage_cases=0;minimum_coverage=1.
for counts,ps,alpha in itertools.product([(3,5,10),(10,20,30),(30,10,20)],[(Q(1,10),Q(1,2),Q(4,5)),(Q(0),Q(1),Q(1,2)),(Q(1,5),Q(1,5),Q(1,5))],[.05,.2,.5]):
    coverage=Q(1)
    for n,p in zip(counts,ps):
        eps=math.sqrt(2*math.log(6/alpha)/n);r=2*p-1
        coverage*=sum((Q(comb(n,k))*p**k*(1-p)**(n-k)for k in range(n+1)if abs(float(Q(2*k,n)-1-r))<=eps),Q(0))
    assert float(coverage)>=1-alpha-1e-14;minimum_coverage=min(minimum_coverage,float(coverage));coverage_cases+=1
# General three-coordinate constrained optimization, not the manuscript's angle root.
fits=[]
for ratio in [1,2,5,10]:
    counts=np.array([10*ratio,10,10],float);positive=np.array([9*ratio,9,5],float);empirical=2*positive/counts-1
    def objective(r):
        q=(1+r)/2
        return float(-sum(positive*np.log(q)+(counts-positive)*np.log1p(-q)))
    def jacobian(r):return counts*(r-empirical)/(1-r*r)
    result=minimize(objective,[.5,.5,0.],jac=jacobian,bounds=[(-.999999,.999999)]*3,constraints=[{'type':'ineq','fun':lambda r:1-r@r,'jac':lambda r:-2*r}],method='SLSQP',options={'ftol':1e-12,'maxiter':1000})
    assert result.success,result.message;r=result.x;assert abs(r@r-1)<2e-10 and abs(r[2])<1e-9
    multiplier=-float(jacobian(r)@r);assert multiplier>0 and max(abs(jacobian(r)+multiplier*r))<3e-6
    radial=empirical/np.linalg.norm(empirical);gain=objective(radial)-objective(r);assert gain>=-1e-11
    fits.append({'ratio':ratio,'x':float(r[0]),'y':float(r[1]),'normSquared':float(r@r),'gain':gain})
# Direct channel superoperators and Choi construction with output-major indices.
I=np.eye(2);Z=np.diag([1.,-1.]);X=np.array([[0,1],[1,0]],complex);Y=np.array([[0,-1j],[1j,0]])
channel=lambda r,eta:((1+eta)*r+(1-eta)*Z@r@Z)/2
choi=lambda eta:sum((np.kron(channel(np.outer(I[:,i],I[:,j]),eta),np.outer(I[:,i],I[:,j]))for i in range(2)for j in range(2)),np.zeros((4,4),complex))
composition_cases=intermediate_cases=0
for a,b in itertools.product([-1.,-.6,0.,.2,.7,1.],repeat=2):
    for r in [I/2,(I+X)/2,(I+Y)/2,(I+Z)/2]:assert np.linalg.norm(channel(channel(r,b),a)-channel(r,a*b))<2e-14
    composition_cases+=1
for earlier,later in itertools.product([-1.,-.6,-.2,.2,.6,1.],[-1.,-.6,0.,.2,.6,1.]):
    k=later/earlier;eigen=np.linalg.eigvalsh(choi(k));assert np.max(abs(eigen-np.sort([0.,0.,1-k,1+k])))<2e-14
    assert (min(eigen)>=-1e-13)==(abs(k)<=1+1e-13)
    for r in [I/2,(I+X)/2,(I+Y)/2,(I+Z)/2]:assert np.linalg.norm(channel(channel(r,earlier),k)-channel(r,later))<2e-14
    intermediate_cases+=1
# A fixed two-state environment provides a separate 4x4 unitary dilation.
partial_trace=lambda r:np.trace(r.reshape(2,2,2,2),axis1=1,axis2=3)
trace_distance=lambda a,b:float(np.linalg.svd(a-b,compute_uv=False).sum()/2)
dilation_cases=0;maximum_dilation_residual=0.
for omega,t in itertools.product([.4,1.,2.],[0.,.3,math.pi/2,2.,math.pi]):
    H=omega*np.kron(Z,Z)/2;U=expm(-1j*t*H);eta=math.cos(omega*t)
    outputs=[]
    for r in [(I+X)/2,(I-X)/2,(I+Y)/2,(I+Z)/2,I/2]:
        actual=partial_trace(U@np.kron(r,I/2)@U.conj().T);expected=channel(r,eta);residual=np.linalg.norm(actual-expected);maximum_dilation_residual=max(maximum_dilation_residual,float(residual));assert residual<2e-14
        outputs.append(actual);dilation_cases+=1
    assert abs(trace_distance(outputs[0],outputs[1])-abs(eta))<2e-14
# Exact erasure is checked algebraically, rather than treating cos(pi/2) as exact zero.
plus,minus=(I+X)/2,(I-X)/2;assert np.array_equal(channel(plus,0.),channel(minus,0.));assert trace_distance(channel(plus,-1.),channel(minus,-1.))==1
rate_cases=0
for omega,t in itertools.product([.4,1.,2.],[.2,.7,1.1,2.]):
    eta=math.cos(omega*t)
    if abs(eta)>1e-8:
        rate=omega*math.tan(omega*t)/2;r=channel(plus,eta);derivative=channel(plus,1.)-channel(plus,0.);derivative*= -omega*math.sin(omega*t)
        assert np.linalg.norm(derivative-rate*(Z@r@Z-r))<2e-14;rate_cases+=1
print(json.dumps({'status':'passed','binomialMoments':moment_cases,'confidencePreparations':coverage_cases,'minimumCoverage':minimum_coverage,'maximumLikelihoodFits':fits,'compositionPairs':composition_cases,'intermediatePairs':intermediate_cases,'dilationPreparations':dilation_cases,'ratePreparations':rate_cases,'maxDilationResidual':maximum_dilation_residual},indent=2))
