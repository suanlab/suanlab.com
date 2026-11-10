"""SymPy energy differentiation oracle; no product code or numerical kernels."""
import json
from pathlib import Path
import sympy as sp

cases=[]
for boundary in ['periodic','dirichlet-zero','neumann-zero']:
    for n in [3,4,7,16,32]:
        q=sp.symbols(f'q0:{n}')
        h=sp.Rational(1,2) if n%2==0 else sp.Rational(2,3)
        c=sp.Rational(3,2)
        a=sp.Rational(-2,3)
        b=sp.Rational(5,7)
        pairs=[(i,i+1) for i in range(n-1)]
        if boundary=='periodic': pairs.append((n-1,0))
        if boundary=='dirichlet-zero': pairs=[(None,0)]+pairs+[(n-1,None)]
        gradients=[(0 if right is None else q[right])-(0 if left is None else q[left]) for left,right in pairs]
        edge_energy=[c*c*x*x/(2*h) for x in gradients]
        onsite=[h*(a*x*x/2+b*x**4/4) for x in q]
        U=sum(edge_energy)+sum(onsite)
        derivatives=[sp.diff(U,x) for x in q]
        H=[[sp.diff(g,x) for x in q] for g in derivatives]
        for variant in ['mixed','uniform']:
            values=[sp.Rational(((i*7)%11)-5,6) for i in range(n)] if variant=='mixed' else [sp.Rational(1,3)]*n
            velocities=[sp.Rational(((i*5)%7)-3,5) for i in range(n)]
            subs=dict(zip(q,values))
            evaluate=lambda x:sp.cancel(sp.sympify(x).subs(subs))
            grad=list(map(evaluate,derivatives))
            acc=[-g/h for g in grad]
            kinetic=[h*x*x/2 for x in velocities]
            potential_power=sum(g*v for g,v in zip(grad,velocities))
            kinetic_power=h*sum(v*dv for v,dv in zip(velocities,acc))
            momentum_rate=h*sum(acc)
            local_force=-h*sum(a*x+b*x**3 for x in values)
            expected={
                'massPerSite':str(h),
                'length':str(h*(n+1 if boundary=='dirichlet-zero' else n)),
                'positions':[str(h*(i+(sp.Rational(1,2) if boundary=='neumann-zero' else 1 if boundary=='dirichlet-zero' else 0))) for i in range(n)],
                'potentialGradient':list(map(str,grad)),
                'acceleration':list(map(str,acc)),
                'potentialHessian':[[str(evaluate(x)) for x in row] for row in H],
                'kineticBySite':list(map(str,kinetic)),
                'localPotentialBySite':[str(evaluate(x)) for x in onsite],
                'kineticEnergy':str(sum(kinetic)),
                'gradientEnergy':str(evaluate(sum(edge_energy))),
                'localPotentialEnergy':str(evaluate(sum(onsite))),
                'energy':str(evaluate(U)+sum(kinetic)),
                'potentialPower':str(potential_power),
                'kineticPower':str(kinetic_power),
                'energyDerivative':str(potential_power+kinetic_power),
                'totalMomentum':str(h*sum(velocities)),
                'momentumDerivative':str(momentum_rate),
                'localPotentialForce':str(local_force),
                'boundaryForce':str(momentum_rate-local_force),
                'edges':[{'left':left,'right':right,'difference':str(evaluate(d)),'energy':str(evaluate(e))} for (left,right),d,e in zip(pairs,gradients,edge_energy)],
                'localPotentialLowerBound':str(-a*a/(4*b)),
            }
            assert potential_power+kinetic_power==0
            if boundary!='dirichlet-zero': assert momentum_rate-local_force==0
            cases.append({'label':f'{boundary}/{n}/{variant}','input':{'boundary':boundary,'spacing':str(h),'speed':str(c),'quadratic':str(a),'quartic':str(b),'position':list(map(str,values)),'velocity':list(map(str,velocities))},'expected':expected})
report={'generator':'SymPy differentiation of finite variational energy','sympyVersion':sp.__version__,'independentOfProduct':True,'cases':cases}
out=Path('scripts/fixtures/scalar-field-lattice-reference.json')
out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'cases':len(cases),'path':str(out),'maximumSites':32,'boundaries':3,'hessianCoefficients':sum(len(c['expected']['potentialHessian'])**2 for c in cases)}))
