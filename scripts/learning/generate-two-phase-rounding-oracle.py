"""Independent two-named-state sums and 180-digit mpmath references.
Requires mpmath; no TypeScript source or runtime series is imported.
"""
from fractions import Fraction as F
from pathlib import Path
import argparse
import json
import sys
import mpmath as mp
sys.set_int_max_str_digits(32768)
mp.mp.dps = 180


def point(q):
    q=F(q)
    return {'numerator':str(q.numerator),'denominator':str(q.denominator)}


def real(q):
    return mp.mpf(q.numerator)/q.denominator if isinstance(q,F) else mp.mpf(q)


def decimal(q):
    return mp.nstr(q,155)


def evaluate(p):
    n=p['siteCount'];c=F(p['offsetNumerator'],p['offsetDenominator']);path=p['sizePath'];side=p['approachSide']
    z=F(p['activityNumerator'],p['activityDenominator']) if path=='fixed' else 1+c/F(n if path=='inverse-size' else n*n)
    if path!='fixed' and side<0:z=1/z
    named=[(0,F(1)),(n,z**n)];normalizer=sum(w for _,w in named)
    probabilities=[(k,w/normalizer)for k,w in named]
    mean=sum(k*q for k,q in probabilities);density=mean/n
    variance=sum((k-mean)**2*q for k,q in probabilities)
    density_variance=sum((F(k,n)-density)**2*q for k,q in probabilities)
    minority=min(q for _,q in probabilities);ratio=min(w for _,w in named)/max(w for _,w in named)
    observations={'activity':z,'activityPower':z**n,'partitionFunction':normalizer,'occupiedProbability':probabilities[1][1],'emptyProbability':probabilities[0][1],'meanDensity':density,'densityVariance':density_variance,'meanCount':mean,'countVariance':variance,'densityResponseToField':variance/n,'countResponseToField':variance,'minorityProbability':minority,'minorityToMajorityWeight':ratio,'exponentialPressureCorrection':1+ratio,'densityVarianceFromNamedStates':density_variance}
    g=mp.log(real(z));correction=mp.log1p(real(ratio))/n
    logs={'field':g,'scaledField':n*g,'limitingPressureAtCurrentField':max(mp.mpf(0),g),'pressure':max(mp.mpf(0),g)+correction,'pressureCorrection':correction,'pressureCorrectionUpperBound':mp.log(2)/n,'tenToNinetyFieldWidth':2*mp.log(9)/n}
    if path=='fixed':
        limit_p=mp.mpf(0)if z<1 else mp.mpf(1)if z>1 else mp.mpf('.5')
        scaled_kind='negative-infinity'if z<1 else 'positive-infinity'if z>1 else 'finite';scaled=mp.mpf(0)
    else:
        scaled=side*real(c)if path=='inverse-size'else mp.mpf(0);scaled_kind='finite';limit_p=1/(1+mp.exp(-scaled))
    limit_var=limit_p*(1-limit_p)
    limits={'pressure':max(mp.mpf(0),g)if path=='fixed'else mp.mpf(0),'meanDensity':limit_p,'densityVariance':limit_var}
    if scaled_kind=='finite':limits['scaledField.value']=scaled
    response_kind='finite'if limit_var==0 else 'positive-infinity'
    if response_kind=='finite':limits['densityResponseToField.value']=mp.mpf(0)
    roots=[]
    for j in sorted({0,n-1,n//2,n//4}):
        angle=mp.pi*(2*j+1)/n;root=mp.exp(mp.j*angle)
        assert abs(1+root**n)<mp.mpf('1e-170')
        roots.append({'index':j,'turns':point(F(2*j+1,2*n)),'real':decimal(mp.re(root)),'imaginary':decimal(mp.im(root))})
    return {'parameters':p,'observations':{k:point(v)for k,v in observations.items()},'logs':{k:decimal(v)for k,v in logs.items()},'limits':{k:decimal(v)for k,v in limits.items()},'limitTags':{'scaledField':scaled_kind,'densityResponseToField':response_kind},'roots':roots,'classification':{'selfAveragesAlongPath':limit_var==0,'nonzeroLimitingDensityVariance':limit_var!=0,'responseDivergesAlongPath':limit_var!=0}}


def generate():
    base=dict(siteCount=4,activityNumerator=2,activityDenominator=1,sizePath='fixed',approachSide=1,offsetNumerator=1,offsetDenominator=2)
    variants=[('four-up',{}),('four-down',{'activityNumerator':1,'activityDenominator':2}),('four-coexist',{'activityNumerator':1}),('one',{'siteCount':1}),('two',{'siteCount':2}),('three',{'siteCount':3}),('seven',{'siteCount':7}),('sixty-four',{'siteCount':64}),('max-up',{'siteCount':512}),('max-down',{'siteCount':512,'activityNumerator':1,'activityDenominator':2}),('max-coexist',{'siteCount':512,'activityNumerator':1}),('extreme-up',{'siteCount':512,'activityNumerator':1000000}),('extreme-down',{'siteCount':512,'activityNumerator':1,'activityDenominator':1000000}),('close-up',{'siteCount':512,'activityNumerator':1000000,'activityDenominator':999999}),('close-down',{'siteCount':512,'activityNumerator':999999,'activityDenominator':1000000}),('rescaled',{'activityNumerator':4,'activityDenominator':2,'offsetNumerator':2,'offsetDenominator':4})]
    for path in ['inverse-size','inverse-size-squared']:
        for side in [1,-1]:
            for n in [1,4,512]:variants.append((f'{path}-{side}-{n}',{'sizePath':path,'approachSide':side,'siteCount':n}))
        variants.append((f'{path}-zero',{'sizePath':path,'offsetNumerator':0,'siteCount':512}))
        variants.append((f'{path}-tiny',{'sizePath':path,'offsetNumerator':1,'offsetDenominator':1000000,'siteCount':512}))
        variants.append((f'{path}-full',{'sizePath':path,'offsetNumerator':1,'offsetDenominator':1,'siteCount':512}))
    cases=[{'id':name,**evaluate({**base,**delta})}for name,delta in variants]
    return {'schemaVersion':1,'scope':'Direct sums over the two named configurations; independent 180-digit mpmath logarithms, limits and complex roots','referenceDecimalDigits':155,'caseCount':len(cases),'exactObservationComparisons':sum(len(v['observations'])for v in cases),'logIntervalComparisons':sum(len(v['logs'])for v in cases),'limitIntervalComparisons':sum(len(v['limits'])for v in cases),'selectedComplexRoots':sum(len(v['roots'])for v in cases),'cases':cases}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='scripts/fixtures/two-phase-rounding-oracle.json');args=parser.parse_args();result=generate();Path(args.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k!='cases'}))
