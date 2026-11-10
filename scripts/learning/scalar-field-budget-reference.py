"""Exact independent energy values at the advertised input and grid-size limits."""
from fractions import Fraction as Q
from pathlib import Path
import json
import sympy as sp
primes=[int(sp.nextprime(10**17+1000*i)) for i in range(70)]
position=[Q(10**18-1,primes[i]) for i in range(32)]
velocity=[Q(-(10**18-1),primes[i+32]) for i in range(32)]
h=Q(1,primes[64]);c=Q(10**18-1);a=Q(-(10**18-1),primes[65]);b=Q(10**18-1,primes[66])
cases=[]
for boundary in ['periodic','dirichlet-zero','neumann-zero']:
    edges=[(position[i],position[i+1]) for i in range(31)]
    if boundary=='periodic':edges.append((position[-1],position[0]))
    if boundary=='dirichlet-zero':edges=[(Q(0),position[0])]+edges+[(position[-1],Q(0))]
    kinetic=h*sum(v*v for v in velocity)/2
    gradient=c*c*sum((right-left)**2 for left,right in edges)/(2*h)
    onsite=h*sum(a*x*x/2+b*x**4/4 for x in position)
    cases.append({'label':boundary+'/maximum-input-budget','input':{'boundary':boundary,'spacing':str(h),'speed':str(c),'quadratic':str(a),'quartic':str(b),'position':list(map(str,position)),'velocity':list(map(str,velocity))},'expectedEnergy':str(kinetic+gradient+onsite)})
p=Path('scripts/fixtures/scalar-field-budget-reference.json');p.write_text(json.dumps({'generator':'Python fractions, independently constructed finite energies','cases':cases},indent=2)+'\n')
print(json.dumps({'cases':3,'sites':32,'maximumEnergyStringLength':max(len(c['expectedEnergy'])for c in cases)}))
