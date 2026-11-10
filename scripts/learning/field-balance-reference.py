"""Independent SymPy chain-rule reference for allocated energies and Noether balances."""
import json
from pathlib import Path
import sympy as s

source=json.loads(Path('scripts/fixtures/scalar-field-lattice-reference.json').read_text())['cases']
cases=[{'label':c['label']+'/share='+share,'input':{'field':c['input'],'share':share}} for c in source for share in ['0','1/3','1/2','1']]
base={'boundary':'periodic','spacing':'1','speed':'1','quadratic':'0','quartic':'0','position':['1','0','-1'],'velocity':['1','0','0']}
extras=[('hand-share-zero',{},'0'),('hand-share-half',{},'1/2'),('hand-share-third',{},'1/3'),('uniform-velocity',{'velocity':['1','1','1']},'1/2'),('hand-share-one',{},'1'),('fixed-instant-zero',{'boundary':'dirichlet-zero'},'1/2'),('massive-shift-breaking',{'quadratic':'1','position':['1','0','0']},'1/2'),('quartic-cyclic-counterexample',{'quartic':'1','position':['1','2','0'],'velocity':['0','0','0']},'1/2'),('neumann-shift',{'boundary':'neumann-zero'},'1/2'),('potential-equilibrium-not-symmetry',{'quadratic':'-1','quartic':'1','position':['1','1','1'],'velocity':['0','0','0']},'1/2')]
cases += [{'label':label,'input':{'field':{**base,**change},'share':share}} for label,change,share in extras]
cache={}
def model(field):
    n=len(field['position']);boundary=field['boundary'];h,c,a,b=map(s.Rational,[field[k]for k in ['spacing','speed','quadratic','quartic']]);key=(n,boundary,h,c,a,b)
    if key in cache:return cache[key]
    q=s.symbols(f'q0:{n}');v=s.symbols(f'v0:{n}');alpha=s.Symbol('alpha')
    pairs=[(i,i+1)for i in range(n-1)]
    if boundary=='periodic':pairs.append((n-1,0))
    if boundary=='dirichlet-zero':pairs=[(None,0)]+pairs+[(n-1,None)]
    edge=[c*c*((q[r]if r is not None else 0)-(q[l]if l is not None else 0))**2/(2*h)for l,r in pairs]
    onsite=[h*(a*x*x/2+b*x**4/4)for x in q];kinetic=[h*y*y/2 for y in v];U=sum(edge)+sum(onsite);H=U+sum(kinetic)
    acc=[-s.diff(U,x)/h for x in q]
    def derivative(expr):
        return s.expand(sum(s.diff(expr,x)*y+s.diff(expr,y)*z for x,y,z in zip(q,v,acc)))
    dotedge=list(map(derivative,edge));Ei=[kinetic[i]+onsite[i]for i in range(n)];J=[];C=[]
    for ei,((l,r),energy)in enumerate(zip(pairs,edge)):
        if l is not None and r is not None:
            Ei[l]+=alpha*energy;Ei[r]+=(1-alpha)*energy
            left_force=-s.diff(energy,q[l]);J.append(s.expand(-left_force*v[l]-alpha*dotedge[ei]))
        else:Ei[l if l is not None else r]+=energy;J.append(s.Integer(0))
        C.append(s.diff(energy,q[l]) if l is not None else -s.diff(energy,q[r]))
    Edot=list(map(derivative,Ei));Q=[h*y for y in v];Qdot=[h*z for z in acc];src=[-s.diff(onsite[i],q[i])for i in range(n)]
    def inflow(i,flows):return s.expand(sum(flow for (l,r),flow in zip(pairs,flows)if r==i)-sum(flow for (l,r),flow in zip(pairs,flows)if l==i))
    Ein=[inflow(i,J)for i in range(n)];Qin=[inflow(i,C)for i in range(n)]
    assert all(s.expand(Edot[i]-Ein[i])==0 for i in range(n))
    assert all(s.expand(Qdot[i]-Qin[i]-src[i])==0 for i in range(n))
    assert s.expand(sum(Ei)-H)==0
    spatial=None
    if boundary=='periodic':
        Dq=[(q[(i+1)%n]-q[(i-1)%n])/(2*h)for i in range(n)];Dv=[(v[(i+1)%n]-v[(i-1)%n])/(2*h)for i in range(n)]
        P=h*sum(y*dx for y,dx in zip(v,Dq));Pdot=derivative(P);vv=s.expand(h*sum(y*dy for y,dy in zip(v,Dv)));defect=-h*b*sum(x**3*dx for x,dx in zip(q,Dq));assert vv==0;assert s.expand(Pdot-defect)==0
        spatial=(P,Pdot,vv,defect)
    result=locals();cache[key]=result;return result

for row in cases:
    inp=row['input'];field=inp['field'];m=model(field);n=m['n'];subs={**dict(zip(m['q'],map(s.Rational,field['position']))),**dict(zip(m['v'],map(s.Rational,field['velocity']))),m['alpha']:s.Rational(inp['share'])}
    ev=lambda expr:s.cancel(s.sympify(expr).xreplace(subs))
    values=lambda name:list(map(ev,m[name]))
    energy=values('Ei');edot=values('Edot');ein=values('Ein');charge=values('Q');qdot=values('Qdot');qin=values('Qin');src=values('src');eflow=values('J');qflow=values('C');pairs=m['pairs']
    edges=[{'left':l,'right':r,'energy':str(ev(e)),'energyDerivative':str(ev(de)),'energyCurrent':str(j),'shiftCurrent':str(c)}for(l,r),e,de,j,c in zip(pairs,m['edge'],m['dotedge'],eflow,qflow)]
    cells=[{'site':i,'energy':str(energy[i]),'energyDerivative':str(edot[i]),'energyInflow':str(ein[i]),'energyResidual':str(edot[i]-ein[i]),'shiftCharge':str(charge[i]),'shiftDerivative':str(qdot[i]),'shiftInflow':str(qin[i]),'shiftSource':str(src[i]),'shiftResidual':str(qdot[i]-qin[i]-src[i])}for i in range(n)]
    prefixes=[]
    for last in range(n):
        inside=lambda i:i is not None and i<=last
        cut=lambda flows:sum((flow if inside(r)else -flow)for(l,r),flow in zip(pairs,flows)if inside(l)!=inside(r))
        e=sum(energy[:last+1]);de=sum(edot[:last+1]);je=cut(eflow);Q=sum(charge[:last+1]);dQ=sum(qdot[:last+1]);jq=cut(qflow);sq=sum(src[:last+1]);assert de-je==0 and dQ-jq-sq==0
        prefixes.append({'lastSite':last,'energy':str(e),'energyDerivative':str(de),'energyInflow':str(je),'energyResidual':str(de-je),'shiftCharge':str(Q),'shiftDerivative':str(dQ),'shiftInflow':str(jq),'shiftSource':str(sq),'shiftResidual':str(dQ-jq-sq)})
    spatial=None
    if m['spatial'] is not None:
        P,dP,vv,defect=map(ev,m['spatial']);spatial={'value':str(P),'derivative':str(dP),'velocityTerm':str(vv),'nonlinearDefect':str(defect),'balanceResidual':str(dP-defect),'quadraticSkewGeneratorSymmetry':m['b']==0,'cyclicPermutationSymmetry':True,'literalContinuousSpatialTranslationClaimed':False,'convention':'delta-phi=+epsilon*Dphi'}
    total=ev(m['H']);row['expected']={'input':inp,'sites':n,'edges':edges,'cells':cells,'prefixes':prefixes,'totalEnergy':str(total),'energyPartitionResidual':str(sum(energy)-total),'energyDerivative':str(sum(edot)),'shiftCharge':str(sum(charge)),'shiftDerivative':str(sum(qdot)),'shiftSecondDerivative':str(ev(m['derivative'](sum(m['Qdot'])))),'shiftSource':str(sum(src)),'shiftBoundaryInflow':str(sum(qin)),'uniformFieldShiftSymmetry':m['a']==0 and m['b']==0 and field['boundary']!='dirichlet-zero','instantaneousShiftDerivativeZero':sum(qdot)==0,'boundaryEnergyAllocation':'full-to-adjacent-active-site','timeDiscretized':False,'spatialGenerator':spatial}

x,t=s.symbols('x t');phi=s.Function('phi')(x,t);c,a,b=s.symbols('c a b');V=a*phi**2/2+b*phi**4/4;pi=s.diff(phi,t);grad=s.diff(phi,x);L=pi**2/2-c*c*grad**2/2-V;R=s.diff(phi,t,2)-c*c*s.diff(phi,x,2)+s.diff(V,phi)
assert s.simplify(s.diff(pi**2/2+c*c*grad**2/2+V,t)+s.diff(-c*c*pi*grad,x)-pi*R)==0
assert s.simplify(s.diff(pi,t)+s.diff(-c*c*grad,x)+s.diff(V,phi)-R)==0
assert s.simplify(s.diff(pi*grad,t)+s.diff(-c*c*grad**2-L,x)-grad*R)==0
out=Path('scripts/fixtures/field-balance-independent.json');out.write_text(json.dumps({'generator':'Independent symbolic variational energy and chain-rule derivatives','sympyVersion':s.__version__,'continuumOffShellIdentitiesChecked':3,'cases':cases},indent=2)+'\n')
print(json.dumps({'cases':len(cases),'symbolicModels':len(cache),'cellBalances':sum(2*r['expected']['sites']for r in cases),'prefixBalances':sum(2*r['expected']['sites']for r in cases),'path':str(out)}))
