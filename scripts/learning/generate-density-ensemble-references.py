"""Independent SymPy matrix/eigenspace/Berkowitz and mpmath250dps references.

No production modules, principal-minor rank rule or entry-wise tomography formulas
are used. A rational AB/BA companion checks Gram characteristic invariants.
"""
import argparse
import json
from pathlib import Path
import sympy as s
import mpmath as mp
mp.mp.dps=250

def exact(x):return str(s.cancel(s.simplify(x)))
def cv(x):return dict(real=exact(s.re(x)),imaginary=exact(s.im(x)))
def mat(a):return [[cv(a[i,j])for j in range(a.cols)]for i in range(a.rows)]
def mm(a):return mp.matrix([[mp.mpc(str(s.N(s.re(a[i,j]),270)),str(s.N(s.im(a[i,j]),270)))for j in range(a.cols)]for i in range(a.rows)])
def dec(x):return 'decimal:'+mp.nstr(mp.re(x),210,min_fixed=0,max_fixed=0)
def dcv(x):return dict(real=dec(mp.re(x)),imaginary=dec(mp.im(x)))
def dmat(a):return [[dcv(a[i,j])for j in range(a.cols)]for i in range(a.rows)]
def put(d,states,weights=None,probe=None,center=(0,1),gap=(1,1)):
 states=[list(z)for z in states]+[[]for _ in range(4-len(states))];weights=weights or [1 if z else 0 for z in states];probe=probe or [1]+[0]*(d-1)
 return dict(dimension=d,**{f'weight{i}':int(weights[i])for i in range(4)},**{k:int(v)for i in range(4)for j in range(4)for k,v in [(f'component{i}Real{j}',s.re(states[i][j])if j<len(states[i])else 0),(f'component{i}Imag{j}',s.im(states[i][j])if j<len(states[i])else 0)]},**{k:int(v)for j in range(4)for k,v in [(f'probeReal{j}',s.re(probe[j])if j<len(probe)else 0),(f'probeImag{j}',s.im(probe[j])if j<len(probe)else 0)]},centerNumerator=center[0],centerDenominator=center[1],gapNumerator=gap[0],gapDenominator=gap[1])
raw=[]
for d in range(1,5):
 basis=[[int(i==j)for i in range(d)]for j in range(d)]
 raw.extend([(f'dimension {d} uniform orthogonal',put(d,basis)),(f'dimension {d} pure',put(d,[basis[0]])),(f'dimension {d} repeated ray phases',put(d,[[((-1)**j)*(1+s.I)*(i+1)for i in range(d)]for j in range(4)],[1,2,3,4])),(f'dimension {d} complex full',put(d,[[int(i==j)+s.I*(1+i+j)for i in range(d)]for j in range(4)],[2,3,5,7],probe=[i+1+s.I*(-1)**i for i in range(d)])),(f'dimension {d} scalar measurement',put(d,basis,center=(2,3),gap=(0,1)))])
raw.extend([
 ('Z maximally mixed',put(2,[[1,0],[0,1]])),
 ('X maximally mixed',put(2,[[1,1],[1,-1]])),
 ('Y maximally mixed',put(2,[[1,s.I],[1,-s.I]])),
 ('mixed Z versus coherent plus',put(2,[[1,0],[0,1]],probe=[1,1])),
 ('pure plus same Z histogram',put(2,[[1,1]])),
 ('pure minus same Z histogram',put(2,[[1,-1]])),
 ('pure positive Y',put(2,[[1,s.I]])),
 ('pure negative Y',put(2,[[1,-s.I]])),
 ('nonorthogonal mixture',put(2,[[1,0],[1,1]])),
 ('unequal nonorthogonal mixture',put(2,[[1,0],[1,1]],[2,3,0,0])),
 ('zero outcome',put(2,[[0,1]])),
 ('zero weight nonzero component',put(2,[[1,0],[0,1],[1,s.I]],[1,0,0,0])),
 ('duplicate opposite vectors',put(2,[[1,s.I],[-1,-s.I]])),
 ('thin nonzero weight',put(2,[[1,0],[0,1]],[1000000,1,0,0])),
 ('near parallel rank two',put(2,[[1000000,1],[1000000,-1]])),
 ('negative gap qutrit',put(3,[[1,0,0],[0,1,0],[0,0,1]],center=(3,7),gap=(-5,11))),
 ('degenerate mixed conditional',put(3,[[1,0,0],[0,1,0],[0,0,1]])),
 ('four dimensional mixed conditional',put(4,[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])),
 ('rank two dependent qutrit',put(3,[[1,0,1],[0,1,1],[1,1,2],[2,-1,1]],[1,2,3,4])),
 ('rank two embedded four dimensional',put(4,[[1,0,1,0],[0,1,1,0],[1,1,2,0],[2,-1,1,0]],[1,2,3,4])),
 ('large complex coefficients',put(4,[[1000000-j-i+s.I*(-1)**(i+j)*(999996-j+i)for i in range(4)]for j in range(4)],[1000000,999999,1,2],probe=[1,1+s.I,2-s.I,3])),
 ('complex rational observable',put(3,[[1+s.I,-2,3*s.I],[0,1,1-s.I],[2,3,-1]],[2,5,7,0],probe=[1+s.I,2-s.I,3],center=(-5,13),gap=(7,19))),
 ('single dimension negative gap',put(1,[[3+4*s.I]],[3,0,0,0],center=(2,7),gap=(-3,5))),
])
# Keep intended named examples even if two ensembles have identical parameter arrays.
specs=raw;cases=[];eigenspaces={}
for label,p in specs:
 d=p['dimension'];I=s.eye(d);z=[s.Matrix([p[f'component{i}Real{j}']+s.I*p[f'component{i}Imag{j}']for j in range(d)])for i in range(4)];N=[s.simplify((v.conjugate().T*v)[0])for v in z];W=sum(p[f'weight{i}']for i in range(4));alpha=[s.Rational(p[f'weight{i}'],W)for i in range(4)];factors=[s.S(0)if alpha[i]==0 else alpha[i]/N[i]for i in range(4)];rho=s.simplify(sum((factors[i]*(z[i]*z[i].conjugate().T)for i in range(4)),s.zeros(d)));rank=rho.rank();purity=s.trace(rho*rho);v=s.Matrix([p[f'probeReal{j}']+s.I*p[f'probeImag{j}']for j in range(d)]);P=s.simplify(v*v.conjugate().T/(v.conjugate().T*v)[0]);a=s.Rational(p['centerNumerator'],p['centerDenominator']);b=s.Rational(p['gapNumerator'],p['gapDenominator']);A=s.simplify(a*I+b*(2*P-I));mean=s.simplify(s.trace(rho*A));second=s.simplify(s.trace(rho*A*A));variance=s.simplify(second-mean**2)
 assert all(isinstance(v,int) and abs(v)<=1000000 for v in p.values())
 assert all(alpha[i]==0 or N[i]>0 for i in range(4))
 S=s.Matrix.hstack(*z).conjugate().T*s.Matrix.hstack(*z);M=s.diag(*factors)*S
 minors=lambda m:[dict(indices=indices,determinant=cv(m.extract(indices,indices).det(method='berkowitz')))for mask in range(1,2**m.rows)for indices in [[j for j in range(m.rows)if mask&2**j]]]
 pair=[[None if N[i]==0 or N[j]==0 else exact(s.simplify(S[i,j]*s.conjugate(S[i,j])/(N[i]*N[j])))for j in range(4)]for i in range(4)];mag=[[s.simplify(factors[i]*factors[j]*S[i,j]*s.conjugate(S[i,j]))for j in range(4)]for i in range(4)]
 phi=mp.matrix(d,4)
 for j in range(4):
  for i in range(d):phi[i,j]=0 if alpha[j]==0 else mp.sqrt(mp.mpf(str(alpha[j].p))/int(alpha[j].q))*mm(z[j])[i]/mp.sqrt(mp.mpf(str(N[j])))
 gram=phi.H*phi
 components=[];within=s.S(0);between=s.S(0)
 for i in range(4):
  local=None if N[i]==0 else s.simplify((z[i].conjugate().T*A*z[i])[0]/N[i]);local2=None if N[i]==0 else s.simplify((z[i].conjugate().T*A*A*z[i])[0]/N[i]);var=None if N[i]==0 else s.simplify(local2-local**2)
  if alpha[i]!=0:within+=alpha[i]*var;between+=alpha[i]*(local-mean)**2
  amplitudes=None if N[i]==0 else[dcv(mm(z[i])[k]/mp.sqrt(mp.mpf(str(N[i]))))for k in range(d)]
  components.append(dict(index=i,occupied=bool(alpha[i]>0),probability=exact(alpha[i]),rawNormSquared=exact(N[i]),normalizedAmplitudes=amplitudes,pureDensity=None if N[i]==0 else mat(s.simplify(z[i]*z[i].conjugate().T/N[i])),observableMean=None if local is None else exact(local),observableVariance=None if var is None else exact(var)))
 key=str(A)
 if key not in eigenspaces:
  parts=[]
  for eigen,multiplicity,basis in sorted(A.eigenvects(),key=lambda e:e[0]):
   V=s.Matrix.hstack(*basis);projector=s.simplify(V*(V.conjugate().T*V).inv()*V.conjugate().T);parts.append((eigen,multiplicity,projector))
  eigenspaces[key]=parts
 outcomes=[];post=s.zeros(d)
 for eigen,multiplicity,projector in eigenspaces[key]:
  unnormalized=s.simplify(projector*rho*projector);prob=s.simplify(s.trace(unnormalized));conditional=None if prob==0 else s.simplify(unnormalized/prob);post+=unnormalized;outcomes.append(dict(eigenvalue=exact(eigen),rank=multiplicity,projector=mat(projector),probability=exact(prob),conditionalDensity=None if conditional is None else mat(conditional),conditionalPurity=None if conditional is None else exact(s.trace(conditional*conditional))))
 tomography=[]
 for i in range(d):
  for j in range(i+1,d):
   e=s.eye(d)[:,i];f=s.eye(d)[:,j];law={}
   for name,qv in [('xPlus',(e+f)/s.sqrt(2)),('xMinus',(e-f)/s.sqrt(2)),('yPlus',(e+s.I*f)/s.sqrt(2)),('yMinus',(e-s.I*f)/s.sqrt(2))]:law[name]=exact((qv.conjugate().T*rho*qv)[0])
   law['outside']=exact(s.trace(rho*(I-e*e.T-f*f.T)));tomography.append(dict(i=i,j=j,probabilities=law))
 diag=s.diag(*rho.diagonal());dist=lambda x:s.simplify(s.trace(x.conjugate().T*x));postpur=s.simplify(s.trace(post*post));elementaryRho=[(-1)**k*v for k,v in enumerate(rho.charpoly().all_coeffs())]+[s.S(0)]*(4-d);elementaryGram=[(-1)**k*v for k,v in enumerate(M.charpoly().all_coeffs())]
 expected=dict(dimension=d,ensembleSize=4,totalRawWeight=exact(W),weights=[exact(x)for x in alpha],components=components,density=mat(rho),standardProbabilities=[exact(rho[i,i])for i in range(d)],purity=exact(purity),linearEntropy=exact(1-purity),purityAboveMaximallyMixed=exact(purity-s.Rational(1,d)),rank=rank,principalMinors=minors(rho),characteristicPolynomial=dict(density=[exact(x)for x in rho.charpoly().all_coeffs()],gram=[exact(x)for x in M.charpoly().all_coeffs()]),pairFidelities=pair,gramSquaredMagnitudes=[[exact(x)for x in row]for row in mag],purityFromOverlaps=exact(sum(sum(row)for row in mag)),gramRank=M.rank(),gramPrincipalMinors=minors(M),weightedColumns=[[dcv(phi[k,i])for k in range(d)]for i in range(4)],gramMatrix=dmat(gram),observable=dict(matrix=mat(A),mean=exact(mean),secondMoment=exact(second),variance=exact(variance),withinEnsembleVariance=exact(within),betweenEnsembleMeanVariance=exact(between),outcomes=outcomes,nonselectiveDensity=mat(post),nonselectivePurity=exact(postpur),purityLost=exact(purity-postpur),disturbanceSquaredHilbertSchmidt=exact(dist(rho-post))),probeComparison=dict(pureDensity=mat(P),overlapProbability=exact(s.trace(rho*P)),squaredHilbertSchmidtDifference=exact(dist(rho-P)),basisProbabilities=[exact(P[k,k])for k in range(d)]),tomography=dict(pairs=tomography,reconstructedDensity=mat(rho)),standardBasisDephasing=dict(density=mat(diag),purity=exact(s.trace(diag*diag)),squaredOffDiagonalCoherence=exact(dist(rho-diag))),qubitBloch=[exact(s.trace(rho*a))for a in [s.Matrix([[0,1],[1,0]]),s.Matrix([[0,-s.I],[s.I,0]]),s.diag(1,-1)]]if d==2 else None,classification=dict(pureState=bool(purity==1),rankOne=rank==1,occupiedComponents=sum(1 for a in alpha if a>0),allOccupiedComponentsSameRay=rank==1,maximallyMixed=bool(rho==I/d)))
 assert all(s.simplify(x-y)==0 for x,y in zip(elementaryRho,elementaryGram))
 cases.append(dict(label=label,input=p,expected=expected));print('prepared '+label,flush=True)
parser=argparse.ArgumentParser();parser.add_argument('--output',default='scripts/fixtures/density-ensemble-reference.json');args=parser.parse_args();out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(dict(method='SymPy exact matrices,Gaussian rank,Berkowitz determinants/characteristic polynomials,rational AB/BA Gram companion,independent eigenspaces and projector tomography; mpmath250dps normalized columns and Gram displays,210digit outputs. No production imports.',cases=cases),indent=2)+'\n');print(json.dumps(dict(preparations=len(cases),dimensions=sorted(set(c['input']['dimension']for c in cases)))))
