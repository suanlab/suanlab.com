"""Independent mpmath logarithms; no site TypeScript arithmetic imports."""
import json
from pathlib import Path
from fractions import Fraction
import mpmath as mp
mp.mp.dps = 1800
values = [Fraction(1), Fraction(2), Fraction(1,2), Fraction(1,3), Fraction(3,7), Fraction(7,3), Fraction(999999,1000000), Fraction(1000001,1000000)]
values += [Fraction(2)**k for k in [-4096,-2048,-512,-1,0,1,512,2048,4096]]
values += [Fraction(10)**k for k in [-1000,-200,-20,20,200,1000]]
values += [Fraction(1)+Fraction(sign,10**power) for power in [20,100,400] for sign in [-1,1]]
values += [Fraction(100000000000000000003+17*i,100000000000000000019+29*i) for i in range(8)]
values = list(dict.fromkeys(values))
cases=[dict(numerator=str(x.numerator),denominator=str(x.denominator),logarithm=mp.nstr(mp.log(mp.mpf(x.numerator)/x.denominator),520)) for x in values]
p=Path('scripts/fixtures/schmidt-wire-log-reference.json')
p.write_text(json.dumps(dict(method='mpmath 1800 decimal working digits;520 significant digit natural logarithms;exact Fraction inputs',cases=cases),indent=2)+'\n')
print(json.dumps(dict(cases=len(cases),file=str(p))))
