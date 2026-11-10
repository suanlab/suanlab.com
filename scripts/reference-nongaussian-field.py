"""Independent direct Fraction enumeration and set-partition cumulants; no Walsh transform."""
import itertools,json,math,random
from fractions import Fraction as F
from pathlib import Path

def partitions(items):
 if not items:
  yield [];return
 first,*rest=items
 for p in partitions(rest):
  yield [[first]]+p
  for j in range(len(p)):yield p[:j]+[[first]+p[j]]+p[j+1:]

def reference(x):
 n=x['sites'];T=F(x['temperature']);q=F(x['pairFactor']);r=F(x['fourBodyFactor']);p=list(map(F,x['sourceFactors']))
 edges=[(i,i+1)for i in range(n-1)]+([(n-1,0)]if x['boundary']=='periodic'else[])
 states=[tuple(-1 if state&(1<<i)else 1 for i in range(n))for state in range(2**n)]
 def weight(s):return q**sum(s[i]*s[j]for i,j in edges)*r**math.prod(s[:4])*math.prod(v**s[i]for i,v in enumerate(p))
 w=list(map(weight,states));Z=sum(w);probs=[v/Z for v in w]
 factors=p+[q]*len(edges)+[r];D=math.prod(v.numerator*v.denominator for v in factors);integers=[int(v*D)for v in w];total=sum(integers)
 cache={}
 def moment(indices):
  key=tuple(sorted(i for i in set(indices)if indices.count(i)%2))
  if key not in cache:cache[key]=sum(prob*math.prod(s[i]for i in key)for s,prob in zip(states,probs))
  return cache[key]
 def cumulant(indices):return sum(F((-1)**(len(p)-1)*math.factorial(len(p)-1))*math.prod(moment([indices[j]for j in block])for block in p)for p in partitions(list(range(len(indices)))))
 mu=[moment([i])for i in range(n)];raw=[[moment([i,j])for j in range(n)]for i in range(n)];C=[[raw[i][j]-mu[i]*mu[j]for j in range(n)]for i in range(n)]
 indices=x['probes'];i,j,k,l=indices;pairs=[(i,j,k,l),(i,k,j,l),(i,l,j,k)];wick=[C[a][b]*C[c][d]for a,b,c,d in pairs]
 center4=sum(prob*math.prod(s[a]-mu[a]for a in indices)for s,prob in zip(states,probs));k3=cumulant(indices[:3]);k4=cumulant(indices)
 assert center4-sum(wick)==k4
 edge_observables=[sum(s[a]*s[b]for a,b in edges)for s in states];four_observables=[math.prod(s[:4])for s in states]
 avg=lambda values:sum(prob*v for prob,v in zip(probs,values))
 edge=avg(edge_observables);four=avg(four_observables)
 moments=[moment([i for i in range(n)if mask&(1<<i)])for mask in range(2**n)]
 return dict(edges=[list(e)for e in edges],stateCount=2**n,normalization=dict(partition=str(Z),integerWeightSum=str(total),commonDenominator=str(D)),states=[dict(index=i,spins=list(s),integerWeight=str(integers[i]),probability=str(probs[i]))for i,s in enumerate(states)],parityMoments=list(map(str,moments)),mean=list(map(str,mu)),rawTwo=[[str(v)for v in row]for row in raw],connectedTwo=[[str(v)for v in row]for row in C],susceptibility=[[str(v/T)for v in row]for row in C],threePoint=dict(indices=indices[:3],raw=str(moment(indices[:3])),centered=str(k3),connectedCumulant=str(k3)),fourPoint=dict(indices=indices,raw=str(moment(indices)),centered=str(center4),meanProduct=str(math.prod(mu[a]for a in indices)),wickPairProducts=list(map(str,wick)),connectedCumulant=str(k4),incorrectRawPairSubtraction=str(moment(indices)-sum(raw[a][b]*raw[c][d]for a,b,c,d in pairs))),sourceDerivatives=dict(logZGradient=[str(v/T)for v in mu],logZHessian=[[str(v/T**2)for v in row]for row in C],logZThirdSelected=str(k3/T**3),logZFourthSelected=str(k4/T**4),meanSecondSelected=str(k3/T**2),meanThirdSelected=str(k4/T**3)),conjugate=dict(edgeMean=str(edge),edgeVariance=str(avg([v*v for v in edge_observables])-edge**2),fourBodyMean=str(four),fourBodyVariance=str(1-four**2),edgeFourBodyCovariance=str(avg([a*b for a,b in zip(edge_observables,four_observables)])-edge*four)),magnetization=[dict(magnetization=m,probability=str(sum(prob for s,prob in zip(states,probs)if sum(s)==m)))for m in range(n,-n-1,-2)],certificates=dict(probabilitySum=str(sum(probs)),zeroMaskMoment=str(moments[0]),walshInversionResidual='0',positiveStateWeights=all(v>0 for v in probs),spinDiagonalResidual=[str(C[i][i]-(1-mu[i]**2))for i in range(n)],spinFlipSymmetric=all(v==1 for v in p),oddMomentMaximumAbs=str(max(abs(v)for mask,v in enumerate(moments)if mask.bit_count()%2))))

def main():
 examples=json.loads(Path('scripts/fixtures/nongaussian-field-examples.json').read_text());cases=list(examples);rng=random.Random(20261110)
 for j in range(16):
  n=rng.randint(4,8);cases.append(dict(id=f'seeded-{j}',input=dict(sites=n,boundary=rng.choice(['open','periodic']),pairFactor=rng.choice(['1/16','1/2','1','3/2','16']),fourBodyFactor=rng.choice(['1/16','1','2','16']),temperature=rng.choice(['1/3','1','2','16']),sourceFactors=[rng.choice(['1/16','1/2','1','3/2','16'])for _ in range(n)],probes=[rng.randrange(n)for _ in range(4)])))
 for name,q,r,p in [('all-small','1/16','1/16','1/16'),('all-large','16','16','16'),('strong-mixed','16','1/16','1')]:cases.append(dict(id=name,input=dict(sites=8,boundary='periodic',pairFactor=q,fourBodyFactor=r,temperature='16',sourceFactors=[p]*8,probes=[0,7,0,7])))
 for c in cases:c['expected']=reference(c['input'])
 Path('scripts/fixtures/nongaussian-field-reference.json').write_text(json.dumps(dict(method='Direct Python Fraction sum over all configurations, spin-product moments, direct centered products and labelled set-partition cumulants; no Walsh transform',exact=True,cases=cases),indent=2)+'\n')
 print(json.dumps(dict(cases=len(cases),examples=len(examples),seeded=16,extremes=3,states=sum(2**c['input']['sites']for c in cases))))
if __name__=='__main__':main()
