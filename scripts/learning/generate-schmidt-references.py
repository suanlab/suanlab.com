"""Independent SymPy density/characteristic/rank and mpmath Hermitian eigenvectors."""
import json
import random
from pathlib import Path
import sympy as S
import mpmath as mp
mp.mp.dps=250

def prep(sites=4,budget=2):
 return dict(sites=sites,selectedCut=sites//2,retainedRank=budget,**{f'state{part}{i}':0 for i in range(16) for part in ['Real','Imag']})
def fraction(x):x=S.simplify(x);assert x.is_Rational,x;return str(x)
def cv(x):return dict(real=fraction(S.re(x)),imaginary=fraction(S.im(x)))
def matrix(a):return [[cv(a[i,j]) for j in range(a.cols)] for i in range(a.rows)]
def mp_value(x):return mp.mpf(str(S.N(x,255)))
def mpi(a):return mp.matrix([[mp.mpc(mp_value(S.re(a[i,j])),mp_value(S.im(a[i,j]))) for j in range(a.cols)] for i in range(a.rows)])
def dec(x):return 'decimal:'+mp.nstr(x,245)
def ci(x):return dict(real=dec(mp.re(x)),imaginary=dec(mp.im(x)))
def mi(a):return [[ci(a[i,j]) for j in range(a.cols)] for i in range(a.rows)]
def vi(a):return [ci(x) for x in a]
def normalized(a,N):return [ci(x/mp.sqrt(N)) for x in a]

def reference(p):
 sites=p['sites'];z=S.Matrix([p[f'stateReal{i}']+S.I*p[f'stateImag{i}'] for i in range(2**sites)]);N=S.expand((z.conjugate().T*z)[0]);mz=mpi(z);cuts=[]
 for k in range(1,sites):
  left,right=2**k,2**(sites-k);C=S.Matrix(left,right,list(z));rhoL=(C*C.conjugate().T/N).applyfunc(S.expand);rhoR=(C.T*C.conjugate()/N).applyfunc(S.expand);small='left' if left<=right else 'right';rho=rhoL if small=='left' else rhoR;size=rho.rows;rank=C.rank();poly=rho.charpoly().as_poly()
  intervals=poly.intervals(eps=S.Rational(1,10**270));assert all(a.is_Rational and b.is_Rational for (a,b),_ in intervals)
  vals,vectors=mp.eighe(mpi(rho));groups=[];index=0
  for (lo,hi),multiplicity in intervals:
   indices=list(range(index,index+multiplicity));index+=multiplicity;zero=bool(lo==hi==0);weight=mp.mpf(0) if zero else mp_value(lo) if lo==hi else mp.fsum(vals[i] for i in indices)/multiplicity
   P=mp.matrix(size,size)
   for i in range(size):
    for j in range(size):P[i,j]=mp.fsum(vectors[i,a]*mp.conj(vectors[j,a]) for a in indices)
   groups.append(dict(weight=weight,multiplicity=multiplicity,zero=zero,P=P))
  groups.reverse();assert sum(g['multiplicity'] for g in groups if not g['zero'])==rank
  remaining=min(p['retainedRank'],rank);retained=mp.mpf(0);Pkeep=mp.matrix(size,size);boundary=False;alloc=[]
  for gi,g in enumerate(groups):
   kept=0 if g['zero'] else min(remaining,g['multiplicity']);remaining-=kept;retained+=kept*g['weight'];boundary|=kept>0 and kept<g['multiplicity']
   if kept==g['multiplicity'] and not g['zero']:Pkeep+=g['P']
   alloc.append(dict(groupIndex=gi,keptMultiplicity=kept,discardedMultiplicity=0 if g['zero'] else g['multiplicity']-kept))
  # At full positive rank the exact retained mass is1, avoiding reference roundoff.
  if p['retainedRank']>=rank:retained=mp.mpf(1)
  if p['retainedRank']==0:retained=mp.mpf(0)
  discarded=1-retained;defined=min(p['retainedRank'],rank)>0;chosen=None
  if not boundary:
   mC=mpi(C);truncated=Pkeep*mC if small=='left' else mC*Pkeep.T
   rest=mC-truncated
   chosen=dict(retainedPositiveSupportProjector=mi(Pkeep),truncatedRawCoefficientMatrix=mi(truncated),amplitudesOnNormalizedInput=normalized([truncated[i,j] for i in range(left) for j in range(right)],mp_value(N)),normalizedAmplitudes=normalized([truncated[i,j] for i in range(left) for j in range(right)],mp_value(N)*retained) if defined else None,directSquaredError=dec(mp.fsum(abs(v)**2 for v in rest)/mp_value(N)),directRetainedNormSquared=dec(mp.fsum(abs(v)**2 for v in truncated)/mp_value(N)))
  purity=S.simplify(S.trace(rho*rho));entropy=-mp.fsum(g['multiplicity']*g['weight']*mp.log(g['weight']) for g in groups if not g['zero'])
  if rank==1:entropy=mp.mpf(0)
  cuts.append(dict(cut=k,leftDimension=left,rightDimension=right,smallerSide=small,smallerDimension=size,coefficientMatrix=matrix(C),rhoLeft=matrix(rhoL),rhoRight=matrix(rhoR),schmidtRank=rank,leftZeroMultiplicity=left-rank,rightZeroMultiplicity=right-rank,characteristicPolynomial=[fraction(v) for v in poly.all_coeffs()],purity=fraction(purity),linearEntropy=fraction(1-purity),entropyNats=dec(entropy),entropyBits=dec(entropy/mp.log(2)),spectrum=[dict(weight=dec(g['weight']),singularValue=dec(mp.sqrt(g['weight'])),multiplicity=g['multiplicity'],zeroWeight=g['zero'],projectorOnSmallerPhysicalSide=mi(g['P'])) for g in groups],truncation=dict(rankBudget=p['retainedRank'],retainedPositiveRank=min(p['retainedRank'],rank),allocation=alloc,boundaryDegenerate=bool(boundary),optimalUnnormalizedStateUnique=not boundary,normalizedStateDefinedForEveryOptimum=bool(defined),chosenStateAvailable=not boundary,retainedWeight=dec(retained),discardedWeight=dec(discarded),optimalUnnormalizedSquaredError=dec(discarded),optimalNormalizedFidelitySquared=dec(retained) if defined else None,optimalNormalizedDistanceSquared=dec(2*(1-mp.sqrt(retained))) if defined else None,chosen=chosen,noArbitraryDegenerateEigenvectorsChosen=True)))
 return dict(sites=sites,dimension=2**sites,state=[cv(x) for x in z],normalizedState=normalized(mz,mp_value(N)),rawNormSquared=fraction(N),selectedCut=p['selectedCut'],selectedCutIndex=p['selectedCut']-1,cuts=cuts)

cases=[]
def save(label,p):
 cases.append(dict(label=label,input=dict(p),expected=reference(p)));print('Verified '+label,flush=True)
rng=random.Random(20261109)
for sites in range(2,5):
 for sample in range(2):
  p=prep(sites,sample+1)
  for i in range(2**sites):p[f'stateReal{i}']=rng.randint(-3,3);p[f'stateImag{i}']=rng.randint(-3,3)
  save(f'general complex {sites} sites case{sample}',p)
for budget in range(5):
 p=prep(4,budget);p.update(stateReal0=3,stateReal5=2,stateReal10=1);save(f'diagonal weights9-4-1 rank budget{budget}',p)
p=prep(4,1);p.update(stateReal0=1,stateReal15=1);save('GHZ4 degenerate rank1 optimum',p)
for budget in [1,2,3,4]:
 p=prep(4,budget);p.update(stateReal0=1,stateReal5=1,stateReal10=1,stateReal15=1);save(f'fourfold equal middle weights budget{budget}',p)
for budget in [1,2,3]:
 p=prep(4,budget);p.update(stateReal0=2,stateReal5=2,stateReal10=1,stateReal15=1);save(f'two positive degenerate groups budget{budget}',p)
p=prep(3,1);p.update(selectedCut=2,stateReal0=1,stateImag1=1);save('right physical density complex product transpose',p)
p=prep(3,1);p.update(selectedCut=2,stateReal0=2,stateImag1=2,stateReal2=1,stateImag3=-1);save('right physical spectral truncation complex phase',p)
p=prep(2,1);p.update(stateReal0=1000000,stateReal1=999999,stateReal2=999999,stateReal3=999998);save('near rank deficient exact nonzero determinant',p)
p=prep(4,1);p.update(stateReal0=1000000,stateReal5=999999,stateReal10=1);save('close positive Schmidt weights not merged',p)
p=prep(4,4)
for i in range(16):p[f'stateReal{i}']=1000000;p[f'stateImag{i}']=-1000000
save('maximum complex integer product state',p)
p=prep(2,0);p.update(stateReal0=1);save('rank0 zero truncation no normalized optimum',p)
p=prep(3,4);p.update(stateReal0=1,stateImag1=1);save('rank budget above smaller side dimension',p)
output=Path('scripts/fixtures/schmidt-reference.json');output.write_text(json.dumps(dict(method='independent SymPy physical densities,exact characteristic/rank/root multiplicities;mpmath250dps Hermitian eigenspaces and projected states',seed=20261109,cases=cases),indent=2)+'\n');print(json.dumps(dict(cases=len(cases),output=str(output))),flush=True)
