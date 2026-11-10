"""Independent matrix-channel, symbolic spectra and mpmath250dps spectral references.

Uses operator-sum action, generic Hermitian eigensolvers and the singular-value
form of squared Uhlmann fidelity. No production Bloch, stable eigenvalue or log code.
"""
import argparse,json
from pathlib import Path
import sympy as s
import mpmath as mp
mp.mp.dps=250
I=s.eye(2);Z=s.diag(1,-1);SX=s.Matrix([[0,1],[1,0]]);SY=s.Matrix([[0,-s.I],[s.I,0]])
def exact(x):return str(s.cancel(s.simplify(x)))
def cv(x):return dict(real=exact(s.re(x)),imaginary=exact(s.im(x)))
def mat(a):return [[cv(a[i,j])for j in range(a.cols)]for i in range(a.rows)]
def scalar(x):return mp.mpc(str(s.N(s.re(x),270)),str(s.N(s.im(x),270)))
def mm(a):return mp.matrix([[scalar(a[i,j])for j in range(a.cols)]for i in range(a.rows)])
def dec(x):return 'decimal:'+mp.nstr(mp.re(x),210,min_fixed=0,max_fixed=0)
def dcv(x):return dict(real=dec(mp.re(x)),imaginary=dec(mp.im(x)))
def dmat(a):return [[dcv(a[i,j])for j in range(a.cols)]for i in range(a.rows)]
def put(r=(1,0,0),den=1,eta=(0,1),probe=(1,0),samples=4):return dict(blochX=r[0],blochY=r[1],blochZ=r[2],blochDenominator=den,probeReal0=int(s.re(probe[0])),probeImag0=int(s.im(probe[0])),probeReal1=int(s.re(probe[1])),probeImag1=int(s.im(probe[1])),etaNumerator=eta[0],etaDenominator=eta[1],sampleSteps=samples)
specs=[('pure X full dephasing',put()),('maximally mixed',put((0,0,0))),('pure Z invariant',put((0,0,1))),('negative Z invariant',put((0,0,-1))),('pure Y',put((0,1,0))),('negative Y',put((0,-1,0))),('negative X',put((-1,0,0))),('mixed X three fifths',put((3,0,0),5)),('mixed complex Bloch',put((1,2,2),4)),('pure complex Bloch',put((1,2,2),3)),('pure Pythagorean Bloch',put((2,3,6),7)),('identity channel',put(eta=(1,1))),('phase flip endpoint',put(eta=(-1,1))),('offgrid zero crossing',put(eta=(-1,1),samples=3)),('negative half',put(eta=(-1,2))),('positive two thirds',put(eta=(2,3))),('negative two thirds',put(eta=(-2,3))),('small eta',put(eta=(1,1000000))),('near identity',put(eta=(999999,1000000))),('near phase flip',put(eta=(-999999,1000000))),('sixteen samples',put((1,2,2),4,(-7,13),samples=16)),('single interval with crossing',put(eta=(-1,3),samples=1)),('X measurement basis',put(probe=(1,1))),('Y measurement basis',put((0,1,0),probe=(1,s.I))),('complex measurement basis',put((1,2,2),4,(-2,3),probe=(2+3*s.I,-4+5*s.I))),('raw probe phase',put(probe=(7*s.I,7*s.I))),('near pure Z',put((0,0,999999),1000000)),('tiny radius',put((1,0,0),1000000)),('tiny complex radius',put((1,-1,1),1000000,(-1,2))),('thin transverse change',put((1,0,999999),1000000,(999999,1000000))),('minimal positive determinant',put((500000,1000,0),500001,(999999,1000000))),('large exact pure coordinates',put((600000,800000,0),1000000,(-999999,1000000))),('large complex probe',put((2,-1,2),3,(-5,7),probe=(1000000+999999*s.I,-999998+999997*s.I))),('diagonal negative eta',put((0,0,3),5,(-1,1))),('nearly mixed path',put((1,0,0),1000000,(-999983,1000000),samples=16)),('mixed phase flip endpoint',put((1,2,2),4,(-1,1))),('pure state measured along itself',put((0,0,1),probe=(1,0))),('zero channel change mixed',put((1,2,2),4,(1,1)))]
cache={}
def spectral(rho):
 key=str(rho)
 if key in cache:return cache[key]
 groups=sorted(rho.eigenvals().items(),key=lambda e:e[0]);values,vectors=mp.eighe(mm(rho));position=0;entries=[];entropy=mp.mpf(0);root=mp.matrix(2,2)
 for eigen,rank in groups:
  lam=mp.re(scalar(eigen));P=mp.matrix(2,2)
  for j in range(position,position+rank):
   v=vectors[:,j];P+=v*v.H
  position+=rank;term=mp.mpf(0)if eigen==0 else -rank*lam*mp.log(lam);entropy+=term
  if eigen!=0:root+=mp.sqrt(lam)*P
  entries.append(dict(eigenvalue=dec(lam),rank=rank,projector=dmat(P),logEigenvalue=None if eigen==0 else dec(mp.log(lam)),entropyTermNats=dec(term),zeroEigenvalueExact=bool(eigen==0)))
 cache[key]=(entries,entropy,root);return cache[key]
def entropy(rho):return spectral(rho)[1]
cases=[]
for label,p in specs:
 assert all(isinstance(v,int)and abs(v)<=1000000 for v in p.values());r=s.Matrix([s.Rational(p[k],p['blochDenominator'])for k in ['blochX','blochY','blochZ']]);assert sum(x*x for x in r)<=1;end=s.Rational(p['etaNumerator'],p['etaDenominator']);rho0=s.simplify((I+r[0]*SX+r[1]*SY+r[2]*Z)/2);probe=s.Matrix([p['probeReal0']+s.I*p['probeImag0'],p['probeReal1']+s.I*p['probeImag1']]);N=s.simplify((probe.conjugate().T*probe)[0]);P=s.simplify(probe*probe.conjugate().T/N);A=2*P-I;axis=[s.simplify(s.trace(P*a))for a in [SX,SY,Z]];raw=s.Matrix.hstack(probe,s.Matrix([-s.conjugate(probe[1]),s.conjugate(probe[0])]));U=raw/s.sqrt(N);baseS=entropy(rho0);root0=spectral(rho0)[2];diagonal=s.diag(*rho0.diagonal());diagS=entropy(diagonal);baseC=diagS-baseS;initialR2=s.simplify(sum(x*x for x in r));transverse=s.simplify(r[0]**2+r[1]**2);fractions={s.Rational(i,p['sampleSteps'])for i in range(p['sampleSteps']+1)};crossing=None if end>=0 else 1/(1-end)
 if crossing is not None:fractions.add(crossing)
 rows=[]
 for index,u in enumerate(sorted(fractions)):
  eta=s.simplify(1+(end-1)*u);wI=(1+eta)/2;wZ=(1-eta)/2;K=[s.sqrt(wI)*I,s.sqrt(wZ)*Z];rho=s.simplify(sum((k*rho0*k.conjugate().T for k in K),s.zeros(2)));bloch=[s.simplify(s.trace(rho*a))for a in [SX,SY,Z]];r2=s.simplify(sum(x*x for x in bloch));spect,hs,root=spectral(rho);inprobe=s.simplify(U.conjugate().T*rho*U);probeDiag=s.diag(*inprobe.diagonal());probeS=entropy(probeDiag);coherence=diagS-hs;mean=s.simplify(s.trace(rho*A));var=s.simplify(s.trace(rho*A*A)-mean**2);delta=s.simplify(rho-rho0);distance=sum(abs(mp.re(scalar(e)))*rank for e,rank in delta.eigenvals().items())/2;singular=mp.svd(root0*root,compute_uv=False);fidelity=sum(singular)**2;choi=s.zeros(4)
  for k in K:
   v=s.Matrix([k[i,j]for i in range(2)for j in range(2)]);choi+=v*v.conjugate().T
  choi=s.simplify(choi);cg=sorted(choi.eigenvals().items(),key=lambda e:e[0]);pure=bool(s.det(rho)==0);maxmix=bool(rho==I/2)
  rows.append(dict(index=index,fraction=exact(u),eta=exact(eta),bloch=[exact(x)for x in bloch],radiusSquared=exact(r2),density=mat(rho),determinant=exact(rho.det()),purity=exact(s.trace(rho*rho)),linearEntropy=exact(1-s.trace(rho*rho)),spectralDirection=None if maxmix else[dec(mp.re(scalar(x))/mp.sqrt(mp.re(scalar(r2))))for x in bloch],spectrum=spect,entropyNats=dec(hs),entropyBits=dec(hs/mp.log(2)),standardBasisDiagonalEntropyNats=dec(diagS),standardBasisCoherenceNats=dec(coherence),probeBasis=dict(density=mat(inprobe),purity=exact(s.trace(inprobe*inprobe)),radiusSquared=exact(r2),diagonalEntropyNats=dec(probeS),coherenceNats=dec(probeS-hs)),measurement=dict(matrix=mat(A),projectors=[mat(I-P),mat(P)],axis=[exact(x)for x in axis],outcomes=[-1,1],probabilities=[exact(s.trace(rho*(I-P))),exact(s.trace(rho*P))],mean=exact(mean),variance=exact(var)),entropyGainFromInitialNats=dec(hs-baseS),coherenceLossFromInitialNats=dec(baseC-coherence),traceDistanceToInitial=dec(distance),squaredUhlmannFidelityToInitial=dec(fidelity),squaredHilbertSchmidtDifferenceToInitial=exact(s.trace(delta.conjugate().T*delta)),channel=dict(identityWeight=exact(wI),phaseFlipWeight=exact(wZ),krausOperators=[dmat(mm(k))for k in K],choi=mat(choi),choiSpectrum=[dict(eigenvalue=exact(e),rank=rank)for e,rank in cg],choiRank=choi.rank(),minimalKrausCount=choi.rank()),classification=dict(pureState=pure,maximallyMixed=maxmix,spectralEigenspacesMerged=maxmix,selectedStateUnchanged=bool(rho==rho0),spectrumUnchangedFromInitial=bool(rho.charpoly().all_coeffs()==rho0.charpoly().all_coeffs()),unitaryChannel=bool(choi.rank()==1),fullyDephasingChannel=bool(eta==0),selectedStateCommutesWithZ=bool(rho0*Z==Z*rho0))))
 expected=dict(dimension=2,initialDensity=mat(rho0),initialBloch=[exact(x)for x in r],initialRadiusSquared=exact(initialR2),endpointEta=exact(end),lnTwo=dec(mp.log(2)),probeBasisUnitary=dmat(mm(U)),records=rows,classification=dict(physicalTimeSpecified=False,markovianTimeLawSpecified=False,parameterPathIsTimeEvolution=False,zeroCrossingInserted=bool(crossing is not None and crossing not in {s.Rational(i,p['sampleSteps'])for i in range(p['sampleSteps']+1)}),continuousParameterPathEntropyNondecreasing=bool(end>=0 or transverse==0)))
 cases.append(dict(label=label,input=p,expected=expected));print('prepared '+label,flush=True)
parser=argparse.ArgumentParser();parser.add_argument('--output',default='scripts/fixtures/density-spectrum-reference.json');args=parser.parse_args();out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(dict(method='SymPy exact operator-sum channels,direct measurement traces and symbolic eigenspaces; mpmath250dps generic Hermitian eigensolvers,projectors,matrix square roots,singular-value squared Uhlmann fidelity and entropy;210digit outputs. No production imports or stable scalar eigenvalue/log routines.',cases=cases),indent=2)+'\n');print(json.dumps(dict(preparations=len(cases),uniquePreparations=len({json.dumps(c['input'],sort_keys=True)for c in cases}),samples=sum(len(c['expected']['records'])for c in cases))))
