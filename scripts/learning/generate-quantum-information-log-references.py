"""High-precision corroboration for the independent analytic log enclosure."""
import json
from fractions import Fraction as F
from pathlib import Path
import mpmath as m
m.mp.dps=130
D=10**31
values={F(n,d) for n in range(1,8) for d in range(1,8)}
values.update([F(D-1,D),F(D+1,D),F(3*D-1,2*D),F(3*D+1,2*D),F(1,D),F(D),F(D-1,2*D),F(D+1,2*D)])
cases=[dict(value=str(x),log2=m.nstr(m.log(m.mpf(x.numerator)/x.denominator,2),120)) for x in sorted(values)]
output=Path('scripts/fixtures/quantum-information-reference-log-independent.json')
output.write_text(json.dumps(dict(method='mpmath130-decimal log2 corroboration, not formal interval reference',allowance='1e-110',cases=cases),indent=2)+'\n')
print(json.dumps(dict(cases=len(cases),file=str(output))))
