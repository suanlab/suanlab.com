#!/usr/bin/env python3
"""Certified Airy origin constants from positive Gamma integrals.

Standard library only. No production arithmetic/constants are imported.
Integrate exp(-t^3) and t*exp(-t^3) on[0,11] by an integrated Taylor
polynomial with a signed first-omitted bound; bound the positive infinite
integral tail analytically. All finite operations round outwards as integers.
"""
from fractions import Fraction as F
from math import factorial
from pathlib import Path
import hashlib
import json
import sys
sys.dont_write_bytecode=True
sys.set_int_max_str_digits(131072)
BITS=4096
S=1<<BITS
R=11
T=R**3
TERMS=5000
TAIL_BITS=1862
BASE=Path(__file__).resolve().parents[2]
ROOT=BASE/'.runtime/learning/real-time-airy-origin'
ROOT.mkdir(parents=True,exist_ok=True)

def ceil(n,d):return -((-n)//d)
def multiply(a,b):
    values=[a[0]*b[0],a[0]*b[1],a[1]*b[0],a[1]*b[1]]
    return min(values)//S,ceil(max(values),S)
def reciprocal(a):
    assert a[0]>0
    return S*S//a[1],ceil(S*S,a[0])
def cube_root(n):
    x=1<<ceil(n.bit_length(),3)
    while True:
        y=(2*x+n//(x*x))//3
        if y>=x:break
        x=y
    while x**3>n:x-=1
    while (x+1)**3<=n:x+=1
    return x

def gamma_thirds():
    lower=[0,0];upper=[0,0];power=(S,S)
    for k in range(TERMS):
        for i,r in enumerate([1,2]):
            lo=power[0]//(3*k+r);hi=ceil(power[1],3*k+r)
            if k%2:lower[i]-=hi;upper[i]-=lo
            else:lower[i]+=lo;upper[i]+=hi
        power=(power[0]*T//(k+1),ceil(power[1]*T,k+1))
    result=[]
    for i,r in enumerate([1,2]):
        factor=3*R**r
        finite_tail=ceil(factor*power[1],3*TERMS+r)
        infinite_tail=1<<(BITS-TAIL_BITS)
        # On[0,R] Taylor's integral remainder has positive sign at this
        # even first omitted power, with the first omitted absolute bound.
        bounds=(factor*lower[i],factor*upper[i]+finite_tail+infinite_tail)
        assert bounds[0]>0
        assert bounds[1]-bounds[0] < 1<<(BITS-1800)
        exact_n=factor*T**TERMS
        exact_d=(3*TERMS+r)*factorial(TERMS)
        assert exact_n*(1<<2200)<exact_d
        result.append(bounds)
    return result

# e^5 > sum_{k=0}^7 5^k/k! >128=2^7, so e^-1331 <2^-1862.
assert sum(F(5**k,factorial(k))for k in range(8))>128
# With y=t^3, Gamma(r/3)=3 integral_0^infty t^(r-1)e^-t^3 dt.
# Its tail is integral_T^infty y^(r/3-1)e^-y dy <=e^-T,
# because T>=1 and r=1,2. No quadrature or floating constants are used.
g1,g2=gamma_thirds()
root=cube_root(3*S**3)
assert root**3<=3*S**3<(root+1)**3
root3=(root,root+1)
a0=reciprocal(multiply(multiply(root3,root3),g2))
pos_derivative=reciprocal(multiply(root3,g1))
a1=(-pos_derivative[1],-pos_derivative[0])
assert a0[0]>0 and a1[1]<0
for value in [a0,a1]:assert value[1]-value[0] < 1<<(BITS-1800)
OUT_BITS=1536
factor=1<<(BITS-OUT_BITS)
def observe(value):return {'lower':str(value[0]//factor),'upper':str(ceil(value[1],factor))}
result={'source':'Positive Gamma integrals with integrated exponential Taylor bounds andanalytic infinite tails;stdlib exact directed integers',
        'certified':True,'workBits':BITS,'referenceBits':OUT_BITS,'radius':R,'terms':TERMS,'infiniteTailUpperPowerOfTwo':-TAIL_BITS,
        'gammaOneThird':observe(g1),'gammaTwoThirds':observe(g2),'cubeRootThree':observe(root3),
        'airyAtZero':observe(a0),'airyDerivativeAtZero':observe(a1),
        'scope':'Origin constants only;no full Airy evaluator,asymptotic error orproduction calculator certified here.'}
out=BASE/'scripts/fixtures/real-time-airy-origin-references.json'
out.write_text(json.dumps(result,indent=2)+'\n')
(ROOT/'verification.json').write_text(json.dumps({'originConstantsCertified':True,'workBits':BITS,'referenceBits':OUT_BITS,'integratedTaylorTerms':TERMS,'infiniteTailUpperPowerOfTwo':-TAIL_BITS,'finiteTaylorTailBelowPowerOfTwo':-2200,'widthBelowPowerOfTwo':-1800,'sourceSha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'fixtureSha256':hashlib.sha256(out.read_bytes()).hexdigest(),'productionAiryImplemented':False,'wholeGoalComplete':False},indent=2)+'\n')
print(json.dumps({'certifiedOriginConstants':True,'referenceBits':OUT_BITS,'widthBelowPowerOfTwo':-1800,'productionAiryImplemented':False}))
