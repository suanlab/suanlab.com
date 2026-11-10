#!/usr/bin/env python3
"""Offline independent mpmath diagnostics for full physical fold inputs.

No production output or arithmetic is imported. Two precisions must give the
same reference cells. These are numerical diagnostics,not certified intervals.
Node tests read the checked-in fixture without installing mpmath.
"""
import json
import sys
from pathlib import Path
import mpmath as mp
sys.dont_write_bytecode=True
sys.set_int_max_str_digits(131072)
BASE=Path(__file__).resolve().parents[2]
REF_BITS=704
cases=[]
for u,h in [('0','1'),('0','1/999999999989'),('0','16'),('1','1'),('-1','1'),('1/999999999989','16'),('-1/999999999989','16'),
            ('1/999999999989','1/999999999989'),('-1/999999999989','1/999999999989'),
            ('16','1/999999999989'),('-16','1/999999999989'),('16','16'),('-16','16'),
            ('233810741045/100000000000','1'),('233810741046/100000000000','1'),
            ('408794944413/100000000000','1'),('408794944414/100000000000','1'),
            ('1','195312499/100000000000'),('1','195312501/100000000000'),
            ('-1','195312499/100000000000'),('-1','195312501/100000000000'),
            ('999999999983/999999999989','999999999979/999999999961')]:
    cases.append({'id':f'u{u}-h{h}','input':{'mode':'fold','control':u,'hbar':h},'strategy':'auto'})
for u in ['4','17/4','9/2','-4','-17/4','-9/2']:
    for strategy in ['auto','central','asymptotic']:
        cases.append({'id':f'overlap-u{u}-{strategy}','input':{'mode':'fold','control':u,'hbar':'1/64'},'strategy':strategy})

def rational(s):
    n,_,d=s.partition('/');return mp.mpf(n)/(mp.mpf(d)if d else 1)
def cell(value):
    scaled=value*mp.mpf(2)**REF_BITS
    return {'lower':str(int(mp.floor(scaled))),'upper':str(int(mp.ceil(scaled)))}
def calculate(p):
    u,h=rational(p['control']),rational(p['hbar']);third=mp.root(h,3)
    z=-u/(third*third);zeta=2*abs(u)**mp.mpf('1.5')/(3*h)
    ai=mp.airyai(z);decay=zeta if u<0 else mp.mpf(0)
    scaled_ai=ai*mp.exp(decay)
    integral=2*mp.pi*third*ai;scaled_integral=2*mp.pi*third*scaled_ai
    fields={'argument':z,'phaseScale':zeta,'airy':ai,'scaledAiry':scaled_ai,'integral':integral,'scaledIntegral':scaled_integral,'negativeExponential':decay,'logMagnitude':mp.log(abs(scaled_integral))-decay}
    result={key:cell(value)for key,value in fields.items()}
    if u>0:
        result['stationaryApproximation']=cell(2*mp.sqrt(mp.pi*h/mp.sqrt(u))*mp.cos(zeta-mp.pi/4))
        root=mp.sqrt(u);action=2*u*root/3
        result['stationaryPoints']=[{'position':cell(-root),'action':cell(action),'curvature':cell(-2*root)},
                                   {'position':cell(root),'action':cell(-action),'curvature':cell(2*root)}]
    else:result['stationaryApproximation']=None
    result['complexStationaryPoints']=[]
    if u<0:
        root=mp.sqrt(-u);action=2*(-u)*root/3
        for sign in [1,-1]:
            result['complexStationaryPoints'].append({'position':{'real':cell(0),'imaginary':cell(sign*root)},'action':{'real':cell(0),'imaginary':cell(sign*action)},'curvature':{'real':cell(0),'imaginary':cell(sign*2*root)},'contributesToRealContour':sign==1})
    return result
rows=[]
for case in cases:
    results=[]
    for bits in [2304,2560]:
        with mp.workprec(bits):results.append(calculate(case['input']))
    assert results[0]==results[1],case['id']
    rows.append({**case,'expected':results[1]});print(case['id'],flush=True)
result={'source':'mpmath independent full Ai andstationary-phase formula at2304/2560working bits;diagnostic cells,not certified reference intervals','mpmathVersion':mp.__version__,'certifiedReference':False,'referenceBits':REF_BITS,'cases':rows}
out=BASE/'scripts/fixtures/real-time-airy-references.json';out.write_text(json.dumps(result,indent=2)+'\n')
print('wrote',len(rows),'cases',flush=True)
