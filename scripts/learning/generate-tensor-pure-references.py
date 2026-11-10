#!/usr/bin/env python3
"""Independent SymPy tensor traces and mpmath250dps generic Hermitian spectra."""
import argparse,json,random
from pathlib import Path
import sympy as S
import mpmath as mp
mp.mp.dps=250

def preparation(a,b,C):
 p=dict(dimensionA=a,dimensionB=b)
 for i in range(4):
  for j in range(4):
   z=C[i][j]if i<a and j<b else 0;p[f'stateReal{i}{j}']=int(S.re(z));p[f'stateImag{i}{j}']=int(S.im(z))
 for key,d in [('a',a),('b',b)]:
  for i in range(4):p[f'{key}ProbeReal{i}']=i+1 if i<d else 0;p[f'{key}ProbeImag{i}']=(-1)**i if i<d else 0
 return p

def cases():
 out=[];rng=random.Random(73201)
 def put(label,p):out.append(dict(label=label,input=p))
 for a in range(1,5):
  for b in range(1,5):
   put(f'{a}x{b} complex generic',preparation(a,b,[[rng.randint(-5,5)+S.I*rng.randint(-5,5)for j in range(b)]for i in range(a)]))
   put(f'{a}x{b} maximal Schmidt',preparation(a,b,[[int(i==j)for j in range(b)]for i in range(a)]))
   u=[i+1+S.I*(-1)**i for i in range(a)];v=[2-j+S.I*(j+1)for j in range(b)]
   put(f'{a}x{b} complex product',preparation(a,b,[[x*y for y in v]for x in u]))
 for label,C in [
 ('Bell plus',[[1,0],[0,1]]),('Bell minus',[[1,0],[0,-1]]),('Bell imaginary',[[1,0],[0,S.I]]),('complex B sign',[[1,S.I],[0,0]]),('unequal Schmidt',[[1,0],[0,2]]),('rank three diagonal',[[1,0,0],[0,2,0],[0,0,3]]),('rank four diagonal',[[1,0,0,0],[0,2,0,0],[0,0,3,0],[0,0,0,4]]),('repeated nonzero Schmidt',[[1,0,0],[0,1,0],[0,0,2]]),('zero outcome',[[1,0],[0,1]]),('tiny entanglement',[[1000000,1],[1,0]]),('near product diagonal',[[1000000,0],[0,1]]),('complex rank four',[[3,1+S.I,2,0],[1-S.I,4,0,S.I],[0,2,5,1],[S.I,0,2,6]]),('tiny fourth Schmidt',[[1000000,1,0,0],[0,1000000,1,0],[0,0,1000000,1],[0,0,0,1]]),('almost repeated weights',[[1000000,1],[0,1000000]]),('large complex coefficients',[[1000000+999999*S.I,-999998+999997*S.I],[999996-999995*S.I,-999994-999993*S.I]])]:
  a,b=len(C),len(C[0]);p=preparation(a,b,C)
  if label=='zero outcome':p=preparation(2,2,[[0,0],[1,0]]);p.update(bProbeReal0=0,bProbeImag0=0,bProbeReal1=1,bProbeImag1=0)
  put(label,p)
 p=preparation(2,2,[[0,0],[1,0]]);p.update(aProbeReal0=1,aProbeImag0=0,aProbeReal1=0,aProbeImag1=0);put('zero A event',p)
 p=preparation(2,3,[[1,S.I,0],[0,1,1]]);p.update(bProbeReal0=1,bProbeImag0=0,bProbeReal1=0,bProbeImag1=0,bProbeReal2=0,bProbeImag2=0);put('rank two complement conditional',p)
 p=preparation(2,2,[[1,0],[0,2]]);p.update(stateReal00=-1,stateReal11=-2);put('global raw sign',p)
 p=preparation(2,2,[[S.I,0],[0,2*S.I]]);put('global raw phase',p)
 return out

def num(x):return str(S.simplify(x))
def obs(x):return 'decimal:'+mp.nstr(x,210)
def mn(x):return mp.mpc(str(S.N(S.re(x),240)),str(S.N(S.im(x),240)))
def matrix(a):return[[dict(real=num(S.re(a[i,j])),imaginary=num(S.im(a[i,j])))for j in range(a.cols)]for i in range(a.rows)]
def mp_matrix(a):return mp.matrix([[mn(a[i,j])for j in range(a.cols)]for i in range(a.rows)])
def partial(rho,a,b,keep):
 if keep=='a':return S.Matrix(a,a,lambda i,j:sum(rho[i*b+k,j*b+k]for k in range(b)))
 return S.Matrix(b,b,lambda i,j:sum(rho[k*b+i,k*b+j]for k in range(a)))
def invariants(rho):
 d=rho.rows;minors=[]
 for mask in range(1,2**d):
  ids=[i for i in range(d)if mask>>i&1];minors.append(dict(indices=ids,determinant=dict(real=num(rho.extract(ids,ids).det()),imaginary='0')))
 return dict(density=matrix(rho),purity=num(S.trace(rho*rho)),rank=rho.rank(),principalMinors=minors,characteristicPolynomial=[num(v)for v in rho.charpoly().all_coeffs()])
def evaluate(p):
 a,b=p['dimensionA'],p['dimensionB'];C=S.Matrix(a,b,lambda i,j:p[f'stateReal{i}{j}']+S.I*p[f'stateImag{i}{j}']);z=S.Matrix(list(C));N=S.expand((z.H*z)[0]);rho=(z*z.H/N).applyfunc(S.expand);A=partial(rho,a,b,'a');B=partial(rho,a,b,'b');rank=C.rank()
 # Generic complex-Hermitian numerical eigensolver,independent of polynomial isolation.
 eigen=sorted([mp.re(v)for v in mp.eighe(mp_matrix(A),eigvals_only=True)]);positive=eigen[a-rank:];assert all(v>0 for v in positive)
 poly=S.Poly(A.charpoly().as_expr())
 while poly.eval(0)==0:poly=poly.exquo(S.Poly(poly.gens[0],*poly.gens))
 factors=poly.sqf_list()[1];multiplicities=[]
 for factor,mult in factors:
  if factor.as_expr()==factor.gens[0]:continue
  multiplicities.extend([mult]*factor.degree())
 # Group with a precision far below the smallest nonzero gap in the bounded cases.
 groups=[]
 for v in positive:
  if groups and abs(v-groups[-1][0])<mp.mpf('1e-210'):groups[-1][1]+=1
  else:groups.append([v,1])
 assert sorted(m for _,m in groups)==sorted(multiplicities)
 entropy=-sum(v*mp.log(v)for v in positive)if rank>1 else mp.mpf(0);negativity=sum(mp.sqrt(positive[i]*positive[j])for i in range(rank)for j in range(i+1,rank))
 pa=S.Matrix([p[f'aProbeReal{i}']+S.I*p[f'aProbeImag{i}']for i in range(a)]);pb=S.Matrix([p[f'bProbeReal{i}']+S.I*p[f'bProbeImag{i}']for i in range(b)]);PA=(pa*pa.H/S.expand((pa.H*pa)[0])).applyfunc(S.expand);PB=(pb*pb.H/S.expand((pb.H*pb)[0])).applyfunc(S.expand);IA=S.eye(a);IB=S.eye(b);ea=[PA,IA-PA];eb=[PB,IB-PB]
 joint=[[S.trace(rho*S.kronecker_product(x,y))for y in eb]for x in ea]
 def condition(on):
  rows=[];events=ea if on=='a'else eb;keep='b'if on=='a'else'a'
  for i,P in enumerate(events):
   K=S.kronecker_product(P,IB)if on=='a'else S.kronecker_product(IA,P);tau=(K*rho*K.H).applyfunc(S.expand);prob=S.simplify(S.trace(tau));local=partial(tau,a,b,keep);conditional=local/prob if prob!=0 else None;rows.append(dict(event=i,projectorRank=1 if i==0 else(a if on=='a'else b)-1,probability=num(prob),unnormalizedDensity=matrix(local),conditionalDensity=matrix(conditional)if conditional is not None else None,conditionalPurity=num(S.trace(conditional*conditional))if conditional is not None else None))
  return dict(rows=rows,nonselectiveDensity=matrix(B if keep=='b'else A),probabilityMass='0')
 HA=IA-2*PA;HB=IB-2*PB;tz=(S.kronecker_product(HA,HB)*z).applyfunc(S.expand);trho=(tz*tz.H/S.expand((tz.H*tz)[0])).applyfunc(S.expand);tA=partial(trho,a,b,'a');tB=partial(trho,a,b,'b');classical=S.diag(*[rho[i,i]for i in range(a*b)]);classA=partial(classical,a,b,'a');classB=partial(classical,a,b,'b')
 pt=S.Matrix(a*b,a*b,lambda u,v:rho[(u//b)*b+v%b,(v//b)*b+u%b]);pt_eigen=sorted([mp.re(v)for v in mp.eighe(mp_matrix(pt),eigvals_only=True)]);negative=pt_eigen[:rank*(rank-1)//2];assert abs(-sum(negative)-negativity)<mp.mpf('1e-200')
 pta=mp.mpf(0)if rank==1 else mp.log(1+2*negativity,2)
 return dict(dimensionA=a,dimensionB=b,jointDimension=a*b,rawNormSquared=num(N),rawCoefficients=matrix(C),jointDensity=matrix(rho),globalPurity='1',reducedA=invariants(A),reducedB=invariants(B),coefficientGramB=matrix(C.H*C/N),schmidtRank=rank,schmidtSpectrum=[dict(weight=obs(v),multiplicity=m,coefficient=obs(mp.sqrt(v)),logWeight=obs(mp.log(v)),entropyTermNats=obs(-v*mp.log(v)))for v,m in groups],zeroMultiplicityA=a-rank,zeroMultiplicityB=b-rank,entropyNats=obs(entropy),entropyBits=obs(entropy/mp.log(2)),localLinearEntropy=num(1-S.trace(A*A)),standardBasisProbabilities=[[num(rho[i*b+j,i*b+j])for j in range(b)]for i in range(a)],standardMarginalA=[num(A[i,i])for i in range(a)],standardMarginalB=[num(B[i,i])for i in range(b)],projectorTests=dict(a=matrix(PA),b=matrix(PB),probabilityA=num(S.trace(A*PA)),probabilityB=num(S.trace(B*PB)),jointProbabilities=[[num(v)for v in row]for row in joint],connected=num(joint[0][0]-S.trace(A*PA)*S.trace(B*PB)),conditionAOnB=condition('b'),conditionBOnA=condition('a')),localReflection=dict(a=matrix(HA),b=matrix(HB),transformedRawCoefficients=matrix(S.Matrix(a,b,list(tz))),transformedReducedA=matrix(tA),transformedReducedB=matrix(tB),transformedPurityA=num(S.trace(tA*tA)),transformedPurityB=num(S.trace(tB*tB))),basisDephasedReference=dict(density=matrix(classical),reducedA=matrix(classA),reducedB=matrix(classB),purity=num(S.trace(classical*classical)),separableByProductBasisMixture=True,sameReducedStates=bool(classA==A and classB==B),connectedProbability=[[num(rho[i*b+j,i*b+j]-A[i,i]*B[j,j])for j in range(b)]for i in range(a)]),partialTranspose=dict(matrix=matrix(pt),zeroMultiplicity=a*b-rank*rank,negativity=obs(negativity),logNegativityBits=obs(pta)))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',default='scripts/fixtures/tensor-pure-reference.json');args=ap.parse_args();rows=cases()
 for i,row in enumerate(rows):
  assert len(row['input'])==50 and all(type(v)is int and abs(v)<=1000000 for v in row['input'].values());row['expected']=evaluate(row['input']);print(f'{i+1}/{len(rows)} {row["label"]}',flush=True)
 Path(args.output).write_text(json.dumps(dict(source='Independent SymPy tensor products,partial traces,rank and principal minors;mpmath250dps generic Hermitian eigensolver;210-digit observations',cases=rows),indent=2)+'\n');print('Generated',len(rows),'cases',flush=True)
if __name__=='__main__':main()
