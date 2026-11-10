from fractions import Fraction as F
from pathlib import Path
import json, math

def moments(weights, values):
 total=sum(weights);p=[w/total for w in weights];mean=sum(a*x for a,x in zip(p,values));var=sum(a*(x-mean)**2 for a,x in zip(p,values));third=sum(a*(x-mean)**3 for a,x in zip(p,values));return p,mean,var,third
weights=[F(1),F(1,2),F(1,4)];p,mean,var,third=moments(weights,[F(0),F(1),F(2)])
assert mean==F(4,7) and var==F(26,49) and third==F(114,343)
reverse=moments(list(reversed(weights)),[F(0),F(1),F(2)])
assert reverse[1]==2-mean and reverse[2]==var and reverse[3]==-third
shift=F(2**60);shifted=moments(weights,[shift+i for i in range(3)])
assert shifted[1]==shift+mean and shifted[2:]==(var,third)
states=[(-1,-1),(-1,1),(1,-1),(1,1)];spinw=[F(2),F(1,2),F(1,2),F(2)];spinp=[w/sum(spinw) for w in spinw]
a=[F(s*t) for s,t in states];m=[F(s+t) for s,t in states]
_,ea,vara,_=moments(spinw,a);_,em,varm,_=moments(spinw,m)
cov=sum(w*(x-ea)*(y-em) for w,x,y in zip(spinp,a,m));assert ea==F(3,5) and vara==F(16,25) and em==0 and varm==F(16,5) and cov==0
mi=sum(float(w)*math.log(float(4*w)) for w in spinp);defect=math.log(5/4);assert abs(mi-defect)>.01
q=F(1,2);L=2;tail=q**(L+1);kept=1-tail;full=[(1-q)*q**n for n in range(L+1)];conditional=[w/kept for w in full];assert conditional==p
assert (sum(abs(a-b) for a,b in zip(full,conditional))+tail)/2==tail
infinite_mean=q/(1-q);infinite_var=q/(1-q)**2;assert infinite_mean==1 and infinite_var==2
result={'contract':'Ideal algebraic logarithm-of-two hand cases. Exact rational weights are stipulated; this is not an oracle for rounded binary64 energies, temperatures or exp(log(2)).','threeLevel':{'weights':[str(w) for w in weights],'Z':str(sum(weights)),'probabilities':[str(x) for x in p],'meanIndex':str(mean),'varianceIndex':str(var),'thirdCumulantIndex':str(third),'invertedMeanIndex':str(reverse[1]),'invertedThirdCumulantIndex':str(reverse[3]),'offsetInGapUnits':str(shift),'shiftedMeanIndex':str(shifted[1])},'coupledSpins':{'states':states,'weights':[str(w) for w in spinw],'Z':str(sum(spinw)),'probabilities':[str(x) for x in spinp],'meanAlignment':str(ea),'varianceAlignment':str(vara),'meanMagnetization':str(em),'varianceMagnetization':str(varm),'alignmentMagnetizationCovariance':str(cov),'mutualInformationApprox':mi,'logFactorizationDefectApprox':defect},'ladderCutoff':{'q':str(q),'cutoff':L,'omittedMass':str(tail),'keptMass':str(kept),'conditionedProbabilities':[str(x) for x in conditional],'infiniteMeanIndex':str(infinite_mean),'infiniteVarianceIndex':str(infinite_var),'cutoffMeanIndex':str(mean),'cutoffVarianceIndex':str(var),'infiniteMeanEnergyInGapUnits':str(F(1,2)+infinite_mean),'cutoffMeanEnergyInGapUnits':str(F(1,2)+mean),'forwardKlApprox':-math.log(float(kept)),'reverseKlStatus':'infinite'}}
Path('scripts/fixtures/partition-function-ideal-hand-cases.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print('Exact rational checks passed: three-level cumulants, energy shift, spin covariance and ladder truncation.')
