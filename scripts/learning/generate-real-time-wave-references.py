#!/usr/bin/env python3
"""Independent wave diagnostics from the dense exact reference fixture.

Regulated Gaussian phases use products over analytic Hessian eigenvalues,
not production pivots. Packet propagation uses one complex Gaussian formula,
not production interval half angles. All65packet samples are covered.
"""
from pathlib import Path
import json
import sys
import mpmath as mp
sys.dont_write_bytecode=True
sys.set_int_max_str_digits(131072)
BASE=Path(__file__).resolve().parents[2]
ROOT=BASE/'.runtime/learning/real-time-phase-numerics';ROOT.mkdir(parents=True,exist_ok=True)
REFERENCE_BITS=704

def q(value):
    n,_,d=value.partition('/');return mp.mpf(n)/(mp.mpf(d)if d else 1)
def cell(value,bits=REFERENCE_BITS):
    scaled=value*mp.mpf(2)**bits
    result={'lower':str(int(mp.floor(scaled))),'upper':str(int(mp.ceil(scaled)))}
    if bits!=REFERENCE_BITS:result['referenceBits']=bits
    return result
def complex_cell(value,bits=REFERENCE_BITS):return{'real':cell(value.real,bits),'imaginary':cell(value.imag,bits)}
def calculate(case):
    p,e=case['input'],case['expected'];n=p['intervals'];h=q(p['hbar']);mass=q(p['mass']);delta=q(p['time'])/n;omega=q(p['omegaSquared']);s=mass/delta
    nu=e['inertia'][1]
    if p['mode']=='chain':
        if e['inverseHash']:
            amplitude=mp.sqrt(s/(2*mp.pi*h*abs(q(e['determinant'])/s**(n-1))))
            unit=mp.exp(1j*(q(e['stationaryAction'])/h-mp.pi/4-mp.pi*nu/2))
            unregulated={'amplitude':cell(amplitude),'unitPhase':complex_cell(unit),'value':complex_cell(amplitude*unit),'logMagnitude':cell(mp.log(amplitude))}
        else:unregulated={'maslovIndex':nu+1,'parity':(-1)**(nu+1),'unitPhase':complex_cell((-1j)**(nu+1))}
        regulated=None
        if e['regulator']:
            eps=q(p['damping']);v=omega*delta*delta
            eigenvalues=[s*(2-v-2*mp.cos(mp.pi*k/n))for k in range(1,n)]
            root_product=mp.mpc(1)
            if v==2:
                # Exact spectral pairing lambda,-lambda. This preserves the
                # analytically real root product instead of a rounding residue.
                for eigenvalue in eigenvalues[:(n-1)//2]:root_product*=mp.sqrt(eps*eps+eigenvalue*eigenvalue)
                if (n-1)%2:root_product*=mp.sqrt(eps)
            else:
                for eigenvalue in eigenvalues:root_product*=mp.sqrt(eps-1j*eigenvalue)
            half=mp.sqrt(mp.mpf(1)/2)
            free=[mp.mpc(1),half-1j*half,mp.mpc(0,-1),-half-1j*half,mp.mpc(-1),-half+1j*half,mp.mpc(0,1),half+1j*half][n%8]
            base=mp.sqrt(s/(2*mp.pi*h))**n*free*(2*mp.pi*h)**(mp.mpf(n-1)/2)/root_product
            exponent=q(e['regulator']['exponent']['real'])+1j*q(e['regulator']['exponent']['imaginary'])
            scaled=base*mp.exp(1j*exponent.imag);decay=-exponent.real
            regulated={'scaledAmplitude':cell(abs(base)),'unitPhase':complex_cell(scaled/abs(base)),'scaledValue':complex_cell(scaled),'value':complex_cell(scaled*mp.exp(-decay),9216),
                       'negativeExponential':cell(decay),'logMagnitude':cell(mp.log(abs(base))-decay)}
        return{'unregulated':unregulated,'regulated':regulated}
    variance=q(e['variance']);final_variance=q(e['finalCovariance'][0][0]);position,momentum=map(q,e['center']);chi=q(e['finalWidth']['real']);action=q(e['centralAction']);A,B=map(q,e['map'][0]);chirp=q(p['chirp'])
    if B:
        cg=h/(2*variance)-1j*(A/B+chirp)
        G=mp.exp(-1j*mp.pi/4-1j*mp.pi*nu/2)/mp.sqrt(abs(B)*cg)
        metaplectic=G/abs(G)
    else:metaplectic=(-1j)**(nu+1)
    normalization=(2*mp.pi*final_variance)**(-mp.mpf(1)/4)
    center_phase=metaplectic*mp.exp(1j*action/h);standard_deviation=mp.sqrt(final_variance)
    samples=[]
    for index in range(65):
        offset=mp.mpf(index-32)/8;dx=offset*standard_deviation;unit=center_phase*mp.exp(1j*(momentum*dx+chi*dx*dx/2)/h)
        wave=normalization*mp.exp(-offset*offset/4)*unit
        samples.append({'position':cell(position+dx),'unitPhase':complex_cell(unit),'value':complex_cell(wave),'density':cell(abs(wave)**2)})
    return{'normalization':cell(normalization),'logNormalization':cell(mp.log(normalization)),'metaplecticPhase':complex_cell(metaplectic),'centerPhase':complex_cell(center_phase),
           'centerValue':complex_cell(normalization*center_phase),'standardDeviation':cell(standard_deviation),'samples':samples}
source=json.loads((BASE/'scripts/fixtures/real-time-phase-exact-references.json').read_text());rows=[]
for case in source['cases']:
    results=[]
    precisions=[10240,10496]if case['expected'].get('regulator')else[2304,2560]
    for bits in precisions:
        with mp.workprec(bits):results.append(calculate(case))
    assert results[0]==results[1],case['id']
    rows.append({'id':case['id'],'input':case['input'],'expected':results[1]});print(case['id'],flush=True)
    (ROOT/'wave-reference-progress.json').write_text(json.dumps({'completed':len(rows),'lastId':case['id']})+'\n')
result={'source':'Independent mpmath wave diagnostics at2304/2560bits (regulated cases10240/10496);raw regulated cells9216bits;exact spectral pairing whenv=2;regulated eigenvalue-root product;closed Gaussian packet formula andall65samples','certifiedReference':False,'referenceBits':REFERENCE_BITS,'mpmathVersion':mp.__version__,'cases':rows}
out=BASE/'scripts/fixtures/real-time-wave-references.json';out.write_text(json.dumps(result,indent=2)+'\n');print('wrote',len(rows),'cases',out.stat().st_size,'bytes',flush=True)
