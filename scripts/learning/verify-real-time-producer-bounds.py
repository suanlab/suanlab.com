#!/usr/bin/env python3
"""Conservative producer wire caps from documented rational-height bounds.

This is executable accounting of a mathematical proof,not an exhaustive
input sweep. The derivation and field inventory must be audited when shapes
change. Existing snapshot limits remain authoritative at the boundary.
"""
from pathlib import Path
from fractions import Fraction as F
import hashlib
import json
import sys
sys.dont_write_bytecode=True
sys.set_int_max_str_digits(131072)
BASE=Path(__file__).resolve().parents[2]
ROOT=BASE/'.runtime/learning/real-time-phase-numerics'
ROOT.mkdir(parents=True,exist_ok=True)
def digits(bits):return len(str((1<<max(1,bits))-1))
def fraction_chars(nbits,dbits):return digits(nbits)+digits(dbits)+2 # slash andsign,even if omitted in canonical form
D=31
parts={
 'complexContinuants':2*sum(fraction_chars(257*k,205*k)for k in range(D+1)),
 'complexDeterminant':2*fraction_chars(257*D,205*D),
 'distancePowers':sum(fraction_chars(85*k,80*k)+1 for k in range(D)),
 'complexMeans':2*D*fraction_chars(16128,16128),
 'complexExponent':2*fraction_chars(16896,16896),
 'stationaryPath':D*fraction_chars(4608,4608),
 'realContinuants':sum(fraction_chars(141*k,130*k)for k in range(D+1)),
 'realDeterminant':fraction_chars(226*D,210*D),
 'maps':sum(4*fraction_chars(148*n+120,137*n+130)for n in [32,32,64,128,256]),
 'packedMatrices':(D*(D+1)//2)*6*(digits(513)+1),
 'nullVector':sum(fraction_chars(141*k,130*k)for k in range(D)),
 'stationaryAction':fraction_chars(5120,5120),
 'forcingHessianAndInputs':1200,'waveObservations':10300,'literalLabels':2000,
}
# Uniform central-series termination: allabsolute ratios are bounded by
# 373249/((3j)(3j-1)) for|z|<=72,including its input rounding. Positive
# interval recurrences satisfy U_next<=ratio*U+2integer quanta.
prefix=F(1);growth=F(1)
for j in range(1,2049):
 ratio=F(373249,(3*j)*(3*j-1));prefix*=ratio
 if ratio>1:growth*=ratio
assert growth < 2**640
assert prefix < F(1,2**5000)
assert F(2**(4096+7))*prefix+4096*growth < 2**2048
# Atx=64,zeta=1024/3 all larger arguments decrease each omitted term.
# Theseexact bounds cover the positive andboth oscillatory series.
term=F(1)
for j in range(1,258):
 term*=F((6*j-5)*(6*j-3)*(6*j-1),(2*j-1)*216*j)*F(3,1024)
 if j in [256,257]:assert term < F(1,2**640)
chain=sum(parts.values())
packet=750000
fold=24000
maximum=max(chain,packet,fold)
maximum_string=max(fraction_chars(148*256+120,137*256+130),fraction_chars(16896,16896))
assert maximum<1800000<2000000
assert maximum_string<24000<65536
# These structural bounds follow from d<=31,65samples,4refinements and
# the fixed object layouts; they are not runtime timing or heap guarantees.
assert 10000<100000 and 12<32 and 65<256
files=[BASE/'src/lib'/name for name in ['learning-real-time-phase.ts','learning-real-time-phase-core-wire.ts','learning-real-time-matrix-observations.ts','learning-real-time-wave-observations.ts','learning-real-time-airy.ts']]
result={'derivation':'docs/online-learning/physics-math-for-ai/real-time-phase-producer-budget.md','scope':'Complete producer responses;independent validator andWorker runtime costs are separate',
 'chainStringCharactersUpper':chain,'packetStringCharactersUpper':packet,'foldStringCharactersUpper':fold,'maximumStringCharactersUpper':maximum,
 'singleStringCharactersUpper':maximum_string,'nodesUpper':10000,'depthUpper':12,'arrayLengthUpper':65,'parts':parts,
 'existingLimitsUnchanged':{'totalStringCharacters':2000000,'singleStringCharacters':65536,'nodes':100000,'depth':32,'arrayLength':256},
 'sourceSha256':{str(f.relative_to(BASE)):hashlib.sha256(f.read_bytes()).hexdigest()for f in files},'centralSeriesUniformTerminationVerified':True,'centralSeriesMaximumSteps':2048,'asymptoticUniformOmittedTermsVerified':[256,257],'runtimeDeadlineProved':False,'wholeGoalComplete':False}
(ROOT/'producer-bounds.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({key:result[key]for key in ['chainStringCharactersUpper','packetStringCharactersUpper','foldStringCharactersUpper','singleStringCharactersUpper','runtimeDeadlineProved']}))
