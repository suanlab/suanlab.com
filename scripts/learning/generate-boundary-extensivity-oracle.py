"""Independent finite geometry / Fraction reference. Never imports TypeScript.
Small cases enumerate lattice vertices/edges; large cases use full-rectangle
counts and the total-spin identity rather than the runtime's pair decomposition.
"""
import argparse
from fractions import Fraction as F
from itertools import product, combinations
from pathlib import Path
import json


def frac(value):
    q = F(value)
    return {'numerator': str(q.numerator), 'denominator': str(q.denominator)}


def calculate(p):
    d, scale = p['dimension'], p['scale']
    a = p['lengthOne'] * (scale if p['growthMode'] == 'uniform-dilation' else 1)
    b = p['lengthTwo'] * (scale if p['growthMode'] == 'uniform-dilation' else 1)
    w = 1 if d == 1 else p['width'] * scale
    A, B = a*w**(d-1), b*w**(d-1)
    N, J, s = A+B, F(p['couplingNumerator'],p['couplingDenominator']), p['spinOne']*p['spinTwo']
    local_total = (a+b-1)*w**(d-1) + (d-1)*(a+b)*(w-1)*(w**(d-2) if d>1 else 0)
    interface = w**(d-1)
    inside_a = (a-1)*w**(d-1) + (d-1)*a*(w-1)*(w**(d-2) if d>1 else 0)
    inside_b = local_total-inside_a-interface
    S = A*p['spinOne']+B*p['spinTwo']
    near_energy = -J*(local_total-(1-s)*interface)
    complete_energy = -J*F(S*S-N,2)
    enumerated = None
    if N <= 100:
        vertices = list(product(range(a+b), *([range(w)]*(d-1))))
        spin = lambda v: p['spinOne'] if v[0]<a else p['spinTwo']
        cuts, groups = [], [[],[]]
        energies = [F(0), F(0)]
        local = F(0)
        complete = F(0)
        for left,right in combinations(vertices,2):
            contribution = -J*spin(left)*spin(right)
            complete += contribution
            if sum(abs(x-y) for x,y in zip(left,right)) != 1:
                continue
            local += contribution
            if (left[0]<a) != (right[0]<a):
                cuts.append((left,right))
            else:
                group = 0 if left[0]<a else 1
                groups[group].append((left,right)); energies[group] += contribution
        assert local==near_energy and complete==complete_energy
        assert len(cuts)==interface and len(groups[0])==inside_a and len(groups[1])==inside_b
        assert local-energies[0]-energies[1] == -J*s*len(cuts)
        enumerated={'vertices':N,'allPairs':N*(N-1)//2,'nearestEdges':len(cuts)+sum(map(len,groups)),'interfaceEdges':len(cuts)}
    faces=2*w**(d-1)+(2*(d-1)*(a+b)*w**(d-2) if d>1 else 0)
    reference={
        'current.geometry.externalFacesPerSite':F(faces,N),
        'current.geometry.interfaceFacesPerSite':F(interface,N),
        'current.geometry.blockFractionOne':F(A,N),
        'current.geometry.magnetizationDensity':F(S,N),
    }
    for name,energy,den,cross,ma,mb in [
        ('nearest-neighbor',near_energy,1,interface,inside_a,inside_b),
        ('complete-unscaled',complete_energy,1,A*B,A*(A-1)//2,B*(B-1)//2),
        ('complete-kac',complete_energy/N,N,A*B,A*(A-1)//2,B*(B-1)//2),
    ]:
        root='current.modes.'+name+'.'
        reference.update({root+'restrictedEnergyOne':-J*ma/den,root+'restrictedEnergyTwo':-J*mb/den,root+'combinedEnergy':energy,root+'crossEnergy':-J*s*cross/den,
                          root+'cutDifference':-J*s*cross/den,root+'reconstructionResidual':F(0),root+'energyPerSite':energy/N,root+'crossEnergyPerSite':-J*s*cross/(den*N),root+'crossMagnitudePerSite':J*cross/(den*N),root+'maximumAbsoluteEnergy':J*(ma+mb+cross)/den,root+'maximumAbsoluteEnergyPerSite':J*(ma+mb+cross)/(den*N)})
    independent_one=-J*F(A-1,2);independent_two=-J*F(B-1,2)
    restricted_sum=-J*F(A*(A-1)+B*(B-1),2*N)
    separate=independent_one+independent_two
    reference.update({'current.separatelyNormalizedKac.energyOne':independent_one,'current.separatelyNormalizedKac.energyTwo':independent_two,'current.separatelyNormalizedKac.energySum':separate,
                      'current.separatelyNormalizedKac.combinedMinusSeparate':complete_energy/N-separate,'current.separatelyNormalizedKac.combinedMinusSeparatePerSite':(complete_energy/N-separate)/N,
                      'current.separatelyNormalizedKac.normalizationCounterterm':restricted_sum-separate,'current.separatelyNormalizedKac.reconstructionResidual':F(0)})
    counts={'siteCountOne':str(A),'siteCountTwo':str(B),'siteCount':str(N),'externalFaceCount':str(faces),'interfaceFaceCount':str(interface)}
    return {'counts':counts,'observations':{key:frac(value)for key,value in reference.items()},'enumeration':enumerated}


def generate():
    base=dict(dimension=2,lengthOne=2,lengthTwo=2,width=3,growthMode='uniform-dilation',scale=1,spinOne=1,spinTwo=-1,couplingNumerator=1,couplingDenominator=1)
    variants=[('square-antiparallel',{}),('square-parallel',{'spinTwo':1}),('cube-asymmetric',{'dimension':3,'lengthOne':1,'lengthTwo':2,'width':2}),('chain-antiparallel',{'dimension':1,'width':1}),('chain-parallel',{'dimension':1,'width':1,'spinTwo':1}),
              ('two-sites-anti',{'dimension':1,'width':1,'lengthOne':1,'lengthTwo':1}),('two-sites-plus',{'dimension':1,'width':1,'lengthOne':1,'lengthTwo':1,'spinTwo':1}),('empty-internal-blocks',{'dimension':3,'lengthOne':1,'lengthTwo':1,'width':1}),
              ('thin-sheet',{'growthMode':'fixed-thickness','lengthOne':1,'lengthTwo':1,'width':1,'scale':8}),('thin-cube',{'dimension':3,'growthMode':'fixed-thickness','lengthOne':1,'lengthTwo':1,'width':1,'scale':3}),
              ('unequal-slabs',{'growthMode':'fixed-thickness','lengthOne':1,'lengthTwo':4,'width':1,'scale':2}),('unequal-spin-reversed',{'lengthOne':1,'lengthTwo':3,'width':2,'spinOne':-1,'spinTwo':1}),
              ('zero-coupling',{'couplingNumerator':0,'scale':16}),('fractional-coupling',{'couplingNumerator':7,'couplingDenominator':11,'width':2,'scale':2}),('non-power-scale',{'scale':7}),('uniform-16',{'scale':16}),
              ('fixed-16',{'growthMode':'fixed-thickness','scale':16}),('uniform-1024',{'dimension':3,'scale':1024}),('slab-1024',{'dimension':3,'growthMode':'fixed-thickness','scale':1024}),
              ('huge-uniform-balanced',{'dimension':3,'lengthOne':1000000,'lengthTwo':1000000,'width':1000000,'scale':1000000,'couplingNumerator':1000000}),
              ('huge-uniform-biased',{'dimension':3,'lengthOne':999999,'lengthTwo':1000000,'width':1000000,'scale':1000000,'couplingNumerator':1,'couplingDenominator':1000000}),
              ('huge-slab',{'dimension':3,'lengthOne':999999,'lengthTwo':1000000,'width':1000000,'scale':1000000,'growthMode':'fixed-thickness'}),
              ('global-flip',{'spinOne':-1,'spinTwo':1}),('block-swap',{'lengthOne':3,'lengthTwo':1,'width':2}),('coupling-rescaled',{'couplingNumerator':14,'couplingDenominator':22,'width':2,'scale':2}),
              ('parallel-slab',{'growthMode':'fixed-thickness','spinTwo':1,'scale':4}),('narrow-two-layers',{'growthMode':'fixed-thickness','lengthOne':1,'lengthTwo':1,'width':1,'scale':1})]
    cases=[]
    for name,changes in variants:
        p={**base,**changes};cases.append({'id':name,'parameters':p,**calculate(p)})
    return {'schemaVersion':1,'scope':'Independent Fraction energies and direct tiny-lattice edge enumeration; no TypeScript imports','caseCount':len(cases),'exactObservationComparisons':sum(len(c['observations'])for c in cases),'enumeratedCases':sum(c['enumeration'] is not None for c in cases),'cases':cases}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='scripts/fixtures/boundary-extensivity-oracle.json');args=parser.parse_args()
    result=generate();Path(args.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k!='cases'}))
