#!/usr/bin/env python3
"""Independent SymPy direct tensor algebra and mpmath250dps Hermitian spectra."""
import argparse,json,random
from pathlib import Path
import sympy as S
import mpmath as mp
mp.mp.dps=250
I2=S.eye(2)
PAULI=[S.Matrix([[0,1],[1,0]]),S.Matrix([[0,-S.I],[S.I,0]]),S.diag(1,-1)]

def canonical(a):return a.applyfunc(S.expand)
def num(x):return str(S.cancel(S.expand(x)))
def obs(x):return 'decimal:'+mp.nstr(x,210)
def matrix(a):return [[dict(real=num(S.re(a[i,j])),imaginary=num(S.im(a[i,j])))for j in range(a.cols)]for i in range(a.rows)]
def mp_matrix(a):
 a=canonical(a)
 return mp.matrix([[mp.mpc(str(S.N(S.re(a[i,j]),240)),str(S.N(S.im(a[i,j]),240)))for j in range(a.cols)]for i in range(a.rows)])
def partial(a,keep):return S.Matrix(2,2,lambda i,j:sum(a[2*i+k,2*j+k]if keep=='a'else a[2*k+i,2*k+j]for k in range(2)))
def preparation(H,settings=None):
 H=canonical(H);den=S.ilcm(*[S.denom(S.re(z))for z in H],*[S.denom(S.im(z))for z in H]);H=canonical(H*den)
 common=S.igcd(*[int(S.re(z))for z in H],*[int(S.im(z))for z in H]);H=H/common
 p={f'hReal{i}{j}':int(S.re(H[i,j]))for i in range(4)for j in range(i,4)}
 p.update({f'hImag{i}{j}':int(S.im(H[i,j]))for i in range(4)for j in range(i+1,4)})
 settings=settings or [[1,0],[1,1],[1,0],[1,S.I]]
 for name,z in zip(['a0','a1','b0','b1'],settings):
  for i in range(2):p[f'{name}Real{i}']=int(S.re(z[i]));p[f'{name}Imag{i}']=int(S.im(z[i]))
 assert len(p)==32 and all(type(v)is int and abs(v)<=1000000 for v in p.values())
 return p

def cases():
 rows=[];rng=random.Random(81052)
 def put(label,H,settings=None):rows.append(dict(label=label,input=preparation(H,settings)))
 def pure(z):z=S.Matrix(z);return canonical(z*z.H)
 bell=S.Matrix([1,0,0,1]);singlet=S.Matrix([0,1,-1,0]);
 put('Bell plus poor settings',pure(bell));put('Bell imaginary',pure([1,0,0,S.I]));put('Bell minus',pure([1,0,0,-1]));put('singlet poor settings',pure(singlet))
 put('Bell plus violating settings',pure(bell),[[1,0],[1,1],[5,2],[5,-2]])
 put('Bell plus negative violation',pure(bell),[[0,1],[1,-1],[5,2],[5,-2]])
 put('classical correlated',S.diag(1,0,0,1));put('independent maximally mixed',S.eye(4));put('pure product zero outcomes',pure([1,0,0,0]));put('complex product',pure(S.kronecker_product(S.Matrix([1,S.I]),S.Matrix([2+S.I,-1+2*S.I]))))
 put('unequal pure Schmidt',pure([1,0,0,2]));put('tiny pure entanglement',pure([1000,0,0,1]));put('tiny generic pure entanglement',pure([999,1,1,0]));put('complex pure generic',pure([2+S.I,-1+2*S.I,3-S.I,2-3*S.I]))
 for n,d in [(0,1),(1,3),(99999,300000),(100001,300000),(1,2),(353553,500000),(353554,500000),(1,1)]:
  put(f'Werner {n}/{d}',canonical((d-n)*S.eye(4)+2*n*singlet*singlet.H),[[1,0],[1,1],[5,2],[5,-2]])
 # Full-rank entangled CHSH boundary: T=diag(3/5,-4/5,1/2); M=1.
 # Strict positivity permits two-sided physical perturbations of t_x.
 boundary=canonical((S.eye(4)+S.Rational(3,5)*S.kronecker_product(PAULI[0],PAULI[0])-S.Rational(4,5)*S.kronecker_product(PAULI[1],PAULI[1])+S.Rational(1,2)*S.kronecker_product(PAULI[2],PAULI[2]))/4)
 put('entangled exact CHSH boundary',boundary)
 for delta in [-1,1]:
  put(f'CHSH boundary perturbation {delta}',canonical(boundary+S.Rational(delta,100000)*S.kronecker_product(PAULI[0],PAULI[0])))
 U=S.Matrix([[S.Rational(3,5),-S.Rational(4,5)],[S.Rational(4,5),S.Rational(3,5)]]);V=S.Matrix([[S.Rational(3,5),4*S.I/5],[4*S.I/5,S.Rational(3,5)]]);K=S.kronecker_product(U,V)
 put('rotated entangled CHSH boundary',canonical(K*boundary*K.H))
 put('classical anticorrelated',S.diag(0,1,1,0));put('asymmetric independent mixed',S.kronecker_product(S.diag(1,2),S.Matrix([[3,S.I],[-S.I,2]])))
 Z=S.Matrix([[1,999,0,0],[0,1,999,0],[0,0,1,999],[0,0,0,1]])
 put('ill-conditioned full rank bounded numerator',Z*Z.H)
 for rank in range(1,5):
  for k in range(4):
   Z=S.Matrix(4,rank,lambda i,j:rng.randint(-4,4)+S.I*rng.randint(-4,4))
   assert Z.rank()==rank
   settings=[[rng.randint(-4,4)+S.I*rng.randint(-4,4)for i in range(2)]for s in range(4)]
   assert all(any(z!=0 for z in a)for a in settings)
   put(f'general complex rank{rank} case{k}',canonical(Z*Z.H),settings)
 for k in range(6):
  H=S.zeros(4)
  for j in range(4):
   a=S.Matrix([1+j+S.I*(k+1),2-j*S.I]);b=S.Matrix([2+S.I*j,k-j+S.I]);z=S.kronecker_product(a,b);H+=z*z.H
  put(f'complex separable mixture{k}',canonical(H))
 # Same state, independently rescaled raw numerator and local representatives.
 p=dict(rows[4]['input']);p.update({k:2*v for k,v in p.items()if k.startswith('h')});rows.append(dict(label='scaled Bell numerator',input=p))
 p=dict(rows[4]['input']);p.update({k:-3*v for k,v in p.items()if not k.startswith('h')});rows.append(dict(label='scaled Bell local spinors',input=p))
 return rows

def invariants(a):
 minors=[]
 for mask in range(1,2**a.rows):
  ids=[i for i in range(a.rows)if mask>>i&1];minors.append(dict(indices=ids,determinant=num(a.extract(ids,ids).det())))
 return dict(principalMinors=minors,characteristicPolynomial=[num(v)for v in a.charpoly().all_coeffs()])

def spectral(a,positive=False):
 poly=a.charpoly().as_poly();x=poly.gens[0];zero_mult=0
 while poly.eval(0)==0:poly=poly.exquo(S.Poly(x,x));zero_mult+=1
 negative=sum(int(factor.count_roots(-S.oo,0))*mult for factor,mult in poly.sqf_list()[1])
 if positive:assert negative==0
 values=[mp.re(v)for v in mp.eighe(mp_matrix(a),eigvals_only=True)]
 for i in range(negative,negative+zero_mult):assert abs(values[i])<mp.mpf('1e-220');values[i]=mp.mpf(0)
 assert all(v<0 for v in values[:negative])and all(v>0 for v in values[negative+zero_mult:])
 groups=[]
 for v in values:
  if groups and abs(v-groups[-1][0])<mp.mpf('1e-210'):groups[-1][1]+=1
  else:groups.append([v,1])
 exact_mult=[zero_mult]if zero_mult else[]
 for factor,mult in poly.sqf_list()[1]:exact_mult.extend([mult]*factor.degree())
 assert sorted(exact_mult)==sorted(n for _,n in groups)
 return values,[dict(eigenvalue=obs(v),multiplicity=n,sign=0 if v==0 else(1 if v>0 else-1))for v,n in groups],negative

def density(a):
 vals,rows,_=spectral(a,True);rank=a.rank();entropy=-sum(v*mp.log(v)for v in vals if v>0)if rank>1 else mp.mpf(0)
 return dict(density=matrix(a),rank=rank,purity=num(S.trace(a*a)),**invariants(a),spectrum=rows,entropyNats=obs(entropy),entropyBits=obs(entropy/mp.log(2))),entropy

def evaluate(p):
 H=S.Matrix(4,4,lambda i,j:p[f'hReal{min(i,j)}{max(i,j)}']+(S.I*p[f'hImag{i}{j}']if i<j else(-S.I*p[f'hImag{j}{i}']if i>j else 0)))
 rho=H/S.trace(H);A=partial(rho,'a');B=partial(rho,'b');pt=S.Matrix(4,4,lambda u,v:rho[(u//2)*2+v%2,(v//2)*2+u%2])
 joint,Sjoint=density(rho);ra,Sa=density(A);rb,Sb=density(B);ptvals,ptrows,negative=spectral(pt)
 negativity=-sum(ptvals[:negative]);logneg=mp.log(1+2*negativity,2)if negative else mp.mpf(0)
 T=canonical(S.Matrix(3,3,lambda i,j:S.trace(rho*S.kronecker_product(PAULI[i],PAULI[j]))));G=canonical(T.T*T);gv,grows,_=spectral(G,True);M=gv[-1]+gv[-2]
 q=S.trace(G)-1;poly=G.charpoly().as_poly();x=poly.gens[0];equal=0
 while poly.eval(q)==0:poly=poly.exquo(S.Poly(x-q,x));equal+=1
 below=int(poly.count_roots(-S.oo,q))if poly.degree()>0 else 0;comparison=1 if below else(0 if equal else-1)
 if comparison==0:assert abs(M-1)<mp.mpf('1e-220')
 else:assert (M>1)==(comparison==1)
 ra['blochVector']=[num(S.trace(A*P))for P in PAULI];rb['blochVector']=[num(S.trace(B*P))for P in PAULI]
 settings=[]
 for name in ['a0','a1','b0','b1']:
  z=S.Matrix([p[f'{name}Real{i}']+S.I*p[f'{name}Imag{i}']for i in range(2)]);P=canonical(z*z.H/S.expand((z.H*z)[0]));O=2*P-I2
  settings.append(dict(name=name,P=P,events=[P,I2-P],observable=O))
 pairs=[];correlations=[]
 for a in settings[:2]:
  row=[];corr=[]
  for b in settings[2:]:
   probs=[[S.expand(S.trace(rho*S.kronecker_product(P,Q)))for Q in b['events']]for P in a['events']];E=S.expand(S.trace(rho*S.kronecker_product(a['observable'],b['observable'])))
   ma=[S.expand(S.trace(A*P))for P in a['events']];mb=[S.expand(S.trace(B*Q))for Q in b['events']]
   row.append(dict(jointProbabilities=[[num(v)for v in row]for row in probs],marginalA=[num(v)for v in ma],marginalB=[num(v)for v in mb],correlation=num(E),connected=num(E-S.trace(A*a['observable'])*S.trace(B*b['observable']))));corr.append(E)
  pairs.append(row);correlations.append(corr)
 selected=correlations[0][0]+correlations[0][1]+correlations[1][0]-correlations[1][1]
 conditions=[]
 for i,s in enumerate(settings):
  onA=i<2;prior=B if onA else A;keep='b'if onA else'a';rows=[]
  for event,P in enumerate(s['events']):
   K=S.kronecker_product(P,I2)if onA else S.kronecker_product(I2,P);tau=canonical(K*rho*K.H);prob=S.trace(tau);reduced=partial(tau,keep);conditional=reduced/prob if prob!=0 else None
   rows.append(dict(event=event,probability=num(prob),unnormalizedDensity=matrix(reduced),conditionalDensity=matrix(conditional)if conditional is not None else None,conditionalPurity=num(S.trace(conditional*conditional))if conditional is not None else None))
  conditions.append(dict(rows=rows,nonselectiveDensity=matrix(prior)))
 return dict(dimensionA=2,dimensionB=2,jointDimension=4,rawHermitianNumerator=matrix(H),normalizationTrace=num(S.trace(H)),joint=joint,reducedA=ra,reducedB=rb,mutualInformationNats=obs(Sa+Sb-Sjoint),mutualInformationBits=obs((Sa+Sb-Sjoint)/mp.log(2)),conditionalEntropyAGivenBNats=obs(Sjoint-Sb),conditionalEntropyBGivenANats=obs(Sjoint-Sa),correlationTensor=[[num(T[i,j])for j in range(3)]for i in range(3)],partialTranspose=dict(matrix=matrix(pt),**invariants(pt),spectrum=ptrows,positiveSemidefinite=negative==0,negativeMultiplicity=negative,negativity=obs(negativity),logNegativityBits=obs(logneg)),localTests=dict(settings=[dict(positiveProjector=matrix(s['P']),observable=matrix(s['observable']),blochDirection=[num(S.trace(s['P']*P))for P in PAULI])for s in settings],pairs=pairs,conditionals=conditions),chsh=dict(selectedSigned=num(selected),selectedAbsolute=num(abs(selected)),selectedViolates=bool(abs(selected)>2),gramMatrix=matrix(G),gramCharacteristicPolynomial=[num(v)for v in G.charpoly().all_coeffs()],gramSpectrum=grows,topTwoEigenvalueSum=obs(M),optimizedMaximum=obs(2*mp.sqrt(M)),optimizedComparisonToLocalBound=comparison,optimizedViolates=comparison==1,optimizedAtBoundary=comparison==0,exactBoundaryCertificate=dict(threshold=num(q),shiftedMatrix=matrix(G-q*S.eye(3)),**{'principalMinors':invariants(G-q*S.eye(3))['principalMinors']},positiveSemidefinite=comparison!=1,determinant=num((G-q*S.eye(3)).det()))),classification=dict(globallyPure=joint['rank']==1,separableTwoQubit=negative==0,entangledTwoQubit=negative>0,localAIsPure=ra['rank']==1,localBIsPure=rb['rank']==1,selectedChshViolation=bool(abs(selected)>2),optimizedChshViolation=comparison==1))

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',default='scripts/fixtures/mixed-entanglement-reference.json');args=parser.parse_args();rows=cases()
 for i,row in enumerate(rows):row['expected']=evaluate(row['input']);print(f'{i+1}/{len(rows)} {row["label"]}',flush=True)
 Path(args.output).write_text(json.dumps(dict(source='Independent SymPy direct tensor products,partial traces,principal minors and exact polynomial root counts;mpmath250dps Hermitian eigensolver;210-digit observations',cases=rows),indent=2)+'\n')
if __name__=='__main__':main()
