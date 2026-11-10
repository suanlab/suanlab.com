"""Independent physical 2x2 matrix exponential and entropy state differences.

No product imports. All supplied binary64 numbers are converted exactly.
Physical-time integrals additionally check three ordinary cases. Limit cases use
1000 digits to resolve state changes far below binary64 temperature resolution.
"""
import argparse
import json
import math
import random
from pathlib import Path
import mpmath as mp

parser = argparse.ArgumentParser()
parser.add_argument('--output', default='scripts/fixtures/thermal-contact-independent.json')
output = Path(parser.parse_args().output)
if output.exists():
    raise SystemExit('Refusing to overwrite an existing reference')
rng = random.Random(26110617)
base = dict(capacity1=2., capacity2=3., initialTemperature1=330., initialTemperature2=270., conductance=.4)

def observation(value):
    return dict(value=str(value), sign=int(mp.sign(value)), logMagnitude=None if value == 0 else str(mp.log(abs(value))))

def case(parameters, start, end, digits, integrals=False):
    with mp.workdps(digits):
        c1, c2, t1, t2, k = [mp.mpf(parameters[key]) for key in ['capacity1', 'capacity2', 'initialTemperature1', 'initialTemperature2', 'conductance']]
        a, b = mp.mpf(start), mp.mpf(end)
        matrix = mp.matrix([[-k/c1, k/c1], [k/c2, -k/c2]])
        initial = mp.matrix([t1, t2])
        def state(time):
            return initial.copy() if k == 0 or t1 == t2 or time == 0 else mp.expm(matrix*time)*initial
        before, after = state(a), state(b)
        q = c2*(after[1]-before[1])
        s1 = c1*mp.log(after[0]/before[0])
        s2 = c2*mp.log(after[1]/before[1])
        difference = after[0]-after[1]
        expected = dict(temperature1=after[0], temperature2=after[1], equilibriumTemperature=(c1*t1+c2*t2)/(c1+c2),
                        temperatureDifference=difference, heatRate=k*difference,
                        entropyProductionRate=k*difference**2/(after[0]*after[1]),
                        heat=q, energyChange1=c1*(after[0]-before[0]), energyChange2=q,
                        temperatureChange1=after[0]-before[0], temperatureChange2=after[1]-before[1],
                        entropyChange1=s1, entropyChange2=s2, entropyChange=s1+s2,
                        relaxationRate=k*(1/c1+1/c2))
        if k != 0:
            expected['relaxationTime'] = 1/(k*(1/c1+1/c2))
        row = dict(parameters=parameters, start=start, end=end, digits=digits,
                   expected={key:observation(value) for key,value in expected.items()})
        if integrals:
            # Rescale physical time to [0,1]; this integrates the actual rate,
            # rather than the production entropy quadrature's transferred fraction.
            duration = b-a
            def rate(u, entropy=False):
                x,y=state(a+duration*u)
                return duration*k*((x-y)**2/(x*y) if entropy else x-y)
            qi = mp.quad(lambda u:rate(u), [0,mp.mpf('.5'),1])
            si = mp.quad(lambda u:rate(u,True), [0,mp.mpf('.5'),1])
            assert mp.almosteq(qi,q,rel_eps=mp.mpf('1e-100'),abs_eps=mp.mpf('1e-100'))
            assert mp.almosteq(si,s1+s2,rel_eps=mp.mpf('1e-100'),abs_eps=mp.mpf('1e-100'))
            row['physicalTimeIntegrals']=dict(heat=observation(qi),entropy=observation(si))
        return row

ordinary=[]
for i in range(24):
    p=base.copy() if i==0 else dict(capacity1=rng.uniform(.5,4),capacity2=rng.uniform(.5,4),
        initialTemperature1=rng.uniform(250,350),initialTemperature2=rng.uniform(250,350),conductance=rng.uniform(.01,2))
    end=3. if i==0 else rng.uniform(.05,8)
    start=0. if i%2==0 else end*rng.uniform(.05,.8)
    ordinary.append(case(p,start,end,120,integrals=i in [0,1,2]))
    print(f'ordinary {i+1}/24',flush=True)
small=math.nextafter(0.,1.)
limits=[
    (base,0.,small),
    (dict(base,conductance=small),0.,8.),
    (dict(base,conductance=small),0.,small),
    (dict(base,initialTemperature1=math.nextafter(300.,math.inf),initialTemperature2=300.),0.,3.),
    (dict(base,initialTemperature1=math.nextafter(300.,math.inf),initialTemperature2=300.,conductance=small),0.,small),
    (dict(base,capacity1=.5,capacity2=.5,initialTemperature1=350.,initialTemperature2=250.,conductance=2.),0.,8.),
    (dict(base,conductance=0.),0.,8.),
    (dict(base,initialTemperature1=300.,initialTemperature2=300.),0.,8.),
    (base,0.,0.),
    (base,3.,3.),
    (base,3.,math.nextafter(3.,math.inf)),
    (dict(base,capacity1=4.,capacity2=.5,initialTemperature1=250.,initialTemperature2=350.,conductance=2.),7.,8.),
]
edge=[]
for i,(p,start,end) in enumerate(limits):
    edge.append(case(p,start,end,1000));print(f'limit {i+1}/12',flush=True)
output.parent.mkdir(parents=True,exist_ok=True)
output.write_text(json.dumps(dict(method='physical-variable matrix exponential and state entropy; three physical-time integrals',ordinary=ordinary,limits=edge),indent=2)+'\n')
print(f'Saved24 ordinary and12 limit references to {output}',flush=True)
