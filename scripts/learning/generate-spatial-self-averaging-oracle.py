"""Independent exact spatial-mixture reference: direct binomial powers,
Fraction moments, exact lattice events and named microstate enumeration.
No TypeScript imports or recurrence from the runtime model.
"""
import argparse
from fractions import Fraction as F
from itertools import product
from math import comb
from pathlib import Path
import json


def point(q):
    q=F(q)
    return {'numerator':str(q.numerator),'denominator':str(q.denominator)}


def evaluate(p):
    n=p['siteCount'];a=p['occupiedWeight'];b=p['emptyWeight'];prob=F(a,a+b)
    power={'fixed':0,'inverse-size':1,'inverse-size-squared':2}[p['commonScaling']]
    c=F(p['commonNumerator'],p['commonDenominator']*n**power);epsilon=F(p['epsilonNumerator'],p['epsilonDenominator'])
    iid=[comb(n,k)*prob**k*(1-prob)**(n-k)for k in range(n+1)]
    shared=[(1-prob if k==0 else prob if k==n else F(0))for k in range(n+1)]
    mixed=[(1-c)*x+c*y for x,y in zip(iid,shared)]
    included=[k for k in range(n+1)if abs(F(k,n)-prob)>=epsilon]
    mean=sum(F(k,n)*mixed[k]for k in range(n+1));second=sum(F(k*k,n*n)*mixed[k]for k in range(n+1));variance=second-mean*mean
    independent_variance=sum((F(k,n)-prob)**2*iid[k]for k in range(n+1))
    tail=sum((mixed[k]for k in included),F(0));iid_tail=sum((iid[k]for k in included),F(0));shared_tail=sum((shared[k]for k in included),F(0))
    bound=min(F(1),variance/epsilon**2);iid_bound=min(F(1),independent_variance/epsilon**2);mixture_bound=(1-c)*iid_bound+c*shared_tail
    v=prob*(1-prob)
    pair_cov=sum(F(k*(k-1),n*(n-1))*mixed[k]for k in range(n+1))-prob**2 if n>1 else None
    pair_corr=pair_cov/v if n>1 and v else None
    factor=variance/independent_variance if independent_variance else None
    equivalent=v/variance if v else None
    observations={
      'current.summary.commonWeight':c,'current.summary.meanDensity':mean,'current.summary.densityVariance':variance,'current.summary.scaledDensityVariance':n*variance,
      'current.summary.independentDensityVariance':independent_variance,'current.summary.eventProbability':tail,'current.summary.independentEventProbability':iid_tail,'current.summary.collectiveEventProbability':shared_tail,
      'current.summary.chebyshevBound':bound,'current.summary.mixtureAwareBound':mixture_bound,'current.summary.independentReferenceBound':iid_bound,
      'current.siteVariance':v,'current.distinctPairCovariance':pair_cov,'current.distinctPairCorrelation':pair_corr,
      'current.meanCount':n*mean,'current.countVariance':n*n*variance,'current.densitySecondMoment':second,
      'current.diagonalCovarianceSum':n*v,'current.orderedCrossCovarianceSum':n*n*variance-n*v,'current.covarianceSumDensityVariance':variance,
      'current.designFactor':1+(n-1)*c,'current.varianceRatioToIndependent':factor,'current.summary.varianceEquivalentCount':equivalent,
      'current.distributionNormalization':sum(mixed),'current.densityMeanFromDistribution':mean,'current.densityVarianceFromDistribution':variance,'current.varianceIdentityResidual':F(0),
      'current.independentEventContribution':(1-c)*iid_tail,'current.collectiveEventContribution':c*shared_tail,'current.eventMixtureIdentityResidual':F(0),
      'current.chebyshevSlack':bound-tail,'current.mixtureAwareSlack':mixture_bound-tail,'current.eventProbabilityMinusIndependent':tail-iid_tail,
      'current.eventTailRatioToIndependent':tail/iid_tail if iid_tail else None,
    }
    assert sum(iid)==sum(shared)==sum(mixed)==1
    assert mean==prob and bound>=mixture_bound>=tail>=0
    # For small systems, independently assign a probability to each named
    # configuration and sum pair covariances directly over ordered site pairs.
    enumeration=None
    if n<=8:
        total=F(0);m=F(0);vsum=F(0);cross=F(0)
        for state in product([0,1],repeat=n):
            k=sum(state);weight=(1-c)*prob**k*(1-prob)**(n-k)+c*(1-prob if k==0 else prob if k==n else F(0))
            total+=weight;m+=F(k,n)*weight;vsum+=(F(k,n)-prob)**2*weight
            cross+=sum((F(x)-prob)*(F(y)-prob)for i,x in enumerate(state)for j,y in enumerate(state)if i!=j)*weight
        assert total==1 and m==mean and vsum==variance and cross==n*n*variance-n*v
        enumeration={'namedStates':2**n,'orderedPairProducts':2**n*n*(n-1)}
    selected=sorted({0,n,1 if n>1 else 0,n//2,max(0,n-1),*included[:2],*included[-2:]})
    rows=[{'count':k,'density':point(F(k,n)),'deviation':point(F(k,n)-prob),'independentProbability':point(iid[k]),'collectiveProbability':point(shared[k]),'probability':point(mixed[k]),'inDeviationEvent':k in included}for k in selected]
    c_inf=F(p['commonNumerator'],p['commonDenominator']) if power==0 else F(0)
    atoms={}
    for x,weight in [(F(0),c_inf*(1-prob)),(prob,1-c_inf),(F(1),c_inf*prob)]:
        if weight: atoms[x]=atoms.get(x,F(0))+weight
    atom_variance=sum((x-prob)**2*weight for x,weight in atoms.items())
    atom_event=sum((weight for x,weight in atoms.items()if abs(x-prob)>=epsilon),F(0))
    def scaled_variance(size):
        common=F(p['commonNumerator'],p['commonDenominator']*size**power)
        law=[(1-common)*comb(size,k)*prob**k*(1-prob)**(size-k)+common*(1-prob if k==0 else prob if k==size else F(0))for k in range(size+1)]
        return size*sum((F(k,size)-prob)**2*law[k]for k in range(size+1))
    # In the vanishing-common-mode families N Var(rho) is a polynomial of
    # degree <=2 in 1/N. Exact Lagrange evaluation at 0 uses independently
    # summed laws at N=1,2,4 instead of copying the runtime's limit constants.
    scaled_limit=None if atom_variance else scaled_variance(1)/3-2*scaled_variance(2)+F(8,3)*scaled_variance(4)
    limit_observations={'limits.commonWeight':c_inf,'limits.densityVariance':atom_variance,'limits.deviationEventProbability':atom_event,'limits.distinctPairCorrelation':atom_variance/v if v else None}
    tags={}
    for name,value,kind in [('scaledDensityVariance',scaled_limit,'positive-infinity'),('varianceEquivalentCount',v/atom_variance if atom_variance else None,'positive-infinity'),('varianceEquivalentFraction',F(0)if atom_variance else v/scaled_limit if v else None,'not-applicable')]:
        if not v and name!='scaledDensityVariance': value=None;kind='not-applicable'
        tags[name]='finite'if value is not None else kind
        if value is not None:limit_observations['limits.'+name+'.value']=value
    limit_atoms=[{'density':point(x),'probability':point(weight)}for x,weight in sorted(atoms.items())]
    return {'observations':{key:point(value)if value is not None else None for key,value in observations.items()},'limitObservations':{key:point(value)if value is not None else None for key,value in limit_observations.items()},'limitTags':tags,'limitAtoms':limit_atoms,'limitClassifications':{'selfAveragesInProbabilityAlongPath':not atom_variance,'commonModeVanishesAlongPath':not c_inf,'countVarianceGrowth':'zero'if not v else 'quadratic-in-site-count'if atom_variance else 'linear-in-site-count'},'selectedRows':rows,'eventCounts':included,
      'flags':{'distributionMomentsMatch':True,'independentReferenceBoundFailsForMixture':tail>iid_bound,'eventContainsPositiveMass':tail>0,'eventIsWholeGrid':len(included)==n+1,'positiveMassCount':sum(q>0 for q in mixed),'distributionIsProductForThisSize':n==1 or c==0 or v==0},'enumeration':enumeration}


def generate():
    base=dict(siteCount=4,occupiedWeight=1,emptyWeight=1,commonNumerator=1,commonDenominator=2,commonScaling='fixed',epsilonNumerator=1,epsilonDenominator=2)
    variants=[('half-fixed',{}),('half-iid',{'commonNumerator':0}),('half-collective',{'commonNumerator':1,'commonDenominator':1}),('half-inverse',{'commonScaling':'inverse-size'}),('half-inverse-squared',{'commonScaling':'inverse-size-squared'}),
      ('one-fixed',{'siteCount':1,'commonNumerator':1,'commonDenominator':1}),('one-inverse',{'siteCount':1,'commonNumerator':1,'commonDenominator':1,'commonScaling':'inverse-size'}),
      ('zero-occupancy',{'occupiedWeight':0,'siteCount':8}),('certain-occupancy',{'emptyWeight':0,'siteCount':8}),('deterministic-event-grid',{'occupiedWeight':0,'epsilonNumerator':1,'epsilonDenominator':1}),
      ('smaller-tail-larger-variance',{'siteCount':2,'occupiedWeight':1,'emptyWeight':9,'epsilonNumerator':3,'epsilonDenominator':20}),('smaller-tail-mirror',{'siteCount':2,'occupiedWeight':9,'emptyWeight':1,'epsilonNumerator':3,'epsilonDenominator':20}),
      ('third-boundary-inclusive',{'siteCount':3,'occupiedWeight':1,'emptyWeight':2,'epsilonNumerator':1,'epsilonDenominator':3}),('third-boundary-above',{'siteCount':3,'occupiedWeight':1,'emptyWeight':2,'epsilonNumerator':333334,'epsilonDenominator':1000000}),
      ('whole-lattice-event',{'siteCount':2,'occupiedWeight':1,'emptyWeight':2,'epsilonNumerator':1,'epsilonDenominator':100}),('empty-lattice-event',{'epsilonNumerator':2,'epsilonDenominator':1}),
      ('non-power-size',{'siteCount':7,'commonScaling':'inverse-size','occupiedWeight':2,'emptyWeight':5,'epsilonNumerator':2,'epsilonDenominator':7}),
      ('sixty-four-fixed',{'siteCount':64}),('sixty-four-inverse',{'siteCount':64,'commonScaling':'inverse-size'}),('sixty-four-inverse-squared',{'siteCount':64,'commonScaling':'inverse-size-squared'}),
      ('max-size-fixed',{'siteCount':512}),('max-size-inverse',{'siteCount':512,'commonScaling':'inverse-size'}),('max-size-inverse-squared',{'siteCount':512,'commonScaling':'inverse-size-squared'}),
      ('rare-iid-underflow',{'siteCount':128,'occupiedWeight':1,'emptyWeight':999999,'commonNumerator':0,'epsilonNumerator':999999,'epsilonDenominator':1000000}),
      ('rare-iid-max',{'siteCount':512,'occupiedWeight':1,'emptyWeight':999999,'commonNumerator':0,'epsilonNumerator':999999,'epsilonDenominator':1000000}),
      ('rare-inverse-squared',{'siteCount':512,'occupiedWeight':1,'emptyWeight':999999,'commonNumerator':1,'commonDenominator':1000000,'commonScaling':'inverse-size-squared','epsilonNumerator':999999,'epsilonDenominator':1000000}),
      ('large-raw-weights',{'siteCount':128,'occupiedWeight':1000000,'emptyWeight':999999,'commonNumerator':999999,'commonDenominator':1000000,'epsilonNumerator':1,'epsilonDenominator':1000000}),
      ('rescaled-representation',{'occupiedWeight':2,'emptyWeight':2,'commonNumerator':2,'commonDenominator':4,'epsilonNumerator':2,'epsilonDenominator':4}),
      ('small-common-fixed',{'siteCount':8,'occupiedWeight':1,'emptyWeight':7,'commonNumerator':1,'commonDenominator':1000000,'epsilonNumerator':1,'epsilonDenominator':8}),
      ('endpoint-shared-empty-event',{'siteCount':8,'commonNumerator':1,'commonDenominator':1,'epsilonNumerator':3,'epsilonDenominator':4}),
      ('unit-weight-deterministic',{'siteCount':1,'occupiedWeight':0,'emptyWeight':1,'commonNumerator':0,'epsilonNumerator':1000000,'epsilonDenominator':1}),
      ('two-thirds-inverse-squared',{'siteCount':8,'occupiedWeight':2,'emptyWeight':1,'commonScaling':'inverse-size-squared','epsilonNumerator':1,'epsilonDenominator':3})]
    cases=[]
    for name,changes in variants:
        p={**base,**changes};cases.append({'id':name,'parameters':p,**evaluate(p)})
    return {'schemaVersion':1,'scope':'Independent Fraction direct-binomial mixture, moments, inclusive events and named-state enumeration','caseCount':len(cases),'exactObservationComparisons':sum(sum(v is not None for v in c['observations'].values())+sum(v is not None for v in c['limitObservations'].values())+2*len(c['limitAtoms'])+5*len(c['selectedRows'])for c in cases),'nullObservations':sum(sum(v is None for v in c['observations'].values())for c in cases),'selectedRows':sum(len(c['selectedRows'])for c in cases),'enumerationCases':sum(c['enumeration']is not None for c in cases),'enumeratedMicrostates':sum(c['enumeration']['namedStates']for c in cases if c['enumeration']),'cases':cases}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='scripts/fixtures/spatial-self-averaging-oracle.json');args=parser.parse_args();result=generate();Path(args.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k!='cases'}))
