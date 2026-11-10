"""Independent mapped-vector rational strains and 1600-digit spectral references."""
import argparse,json,math,random
from fractions import Fraction as F
from pathlib import Path
import mpmath as m
p=argparse.ArgumentParser();p.add_argument('--output',default='scripts/fixtures/continuum-stress-audit-independent.json');out=Path(p.parse_args().output)
if out.exists():raise SystemExit('Refusing to overwrite references')
m.mp.dps=1600
rng=random.Random(20261106);tiny=float.fromhex('0x0.0000000000001p-1022')
def mp(x):
 if isinstance(x,F):return m.mpf(x.numerator)/x.denominator
 return m.mpf(x)
def obs(x):
 x=mp(x);v=float(x)
 return dict(value=v if x==0 or v!=0 else None,sign=int(m.sign(x)))
def dot(a,b):return sum(x*y for x,y in zip(a,b))
strain=[]
params=[(1.,1.,k*tiny,angle) for k in [0,1,2,4,16] for angle in [0.,math.pi/4,.7,math.pi/2]]
params += [(a,b,g,angle) for a,b in [(1.,1.5),(.5,1.),(2.,1.)] for g in [tiny,1e-200] for angle in [0.,.7]]
params += [(rng.uniform(.5,2),rng.uniform(.5,2),rng.uniform(-1,1),rng.uniform(-3,3)) for _ in range(24)]
for a,b,g,angle in params:
 nx,ny=math.cos(angle),math.sin(angle);n=list(map(F,[nx,ny]));A,B,G=map(F,[a,b,g]);e1=[A,F(0)];e2=[G,B]
 green=[(dot(e1,e1)-1)/2,dot(e1,e2)/2,(dot(e2,e2)-1)/2]
 mapped=[A*n[0]+G*n[1],B*n[1]];q=(dot(mapped,mapped)-dot(n,n))/2
 engineering=m.sqrt(1+2*mp(q))-1
 strain.append(dict(model=dict(stretchX=a,stretchY=b,shear=g,rotation=0,translationX=0,translationY=0),angle=angle,normal=[nx,ny],expected=dict(green=[obs(v)for v in green],directionalGreen=obs(q),engineering=obs(engineering))))
principal=[]
matrices=[(tiny,0.,0.),(-tiny,0.,0.),(tiny,tiny,0.),(0.,0.,tiny),(tiny,0.,tiny),(1.,tiny,0.),(-1.,-tiny,0.),(1.,0.,tiny),(1.,tiny,tiny),(.25,1.,.5),(-.25,-1.,.5),(0.,0.,0.),(3.,-1.,1.)]
matrices += [(rng.uniform(-8,8),rng.uniform(-8,8),rng.uniform(-8,8)) for _ in range(24)]
for a,d,b in matrices:
 A,D,B=map(mp,[a,d,b]);mean=(A+D)/2;radius=m.sqrt(((A-D)/2)**2+B*B)
 principal.append(dict(stress=dict(normalX=a,normalY=d,shear=b,gradientX=0,gradientY=0,crossGradient=0),expected=dict(values=[obs(mean+radius),obs(mean-radius)],direction=None if a==d and b==0 else float(m.atan2(2*B,A-D)/2))))
out.write_text(json.dumps(dict(method='Fraction mapped-vector inner products; mpmath1600-digit eigenvalues and engineering strains; no product imports',strainCases=strain,principalCases=principal),indent=2)+'\n');print(f'Wrote {len(strain)} strains and {len(principal)} principal-stress references')
