import test from 'node:test';
import assert from 'node:assert/strict';
import refs from './fixtures/central-force-independent.json';
import {inverseSquareReduction as reduce, ellipticKeplerMotion as orbit, solveEllipticKepler as solve} from '../src/lib/learning-central-force';
const close=(a:number,b:number,tol=2e-11)=>assert.ok(Math.abs(a-b)<=tol*Math.max(1,Math.abs(a),Math.abs(b)),`${a} versus ${b}`);
const cross=(a:number[],b:number[])=>a[0]*b[1]-a[1]*b[0];
test('700 independent Fraction Cartesian initial invariants verify exact radial reduction and admissibility',()=>{
  assert.equal(refs.radial.length,700);
  for(const c of refs.radial){const r=reduce(c.values);
    for(const key of Object.keys(c.exact) as (keyof typeof c.exact)[]) assert.equal(r[key].label,c.exact[key]);
    assert.deepEqual(r.eccentricityVector.map(v=>v.label),c.eVector);
    for(const key of Object.keys(c.query) as (keyof typeof c.query)[]) assert.equal(r.query[key].label,c.query[key]);
    assert.equal(r.query.allowed,c.allowed);assert.equal(r.eccentricityIdentityResidual.label,'0');
    if(!c.allowed){assert.equal(r.query.radialSpeed,null);assert.deepEqual(r.query.radialVelocityBranches,[]);}
    else{close(r.query.radialSpeed!**2/2,r.query.radialKineticPerMass.value);}
  }
});
test('220 independent 70-digit bisection roots verify signed, small and multirevolution Kepler anomalies',()=>{
  assert.equal(refs.roots.length,220);
  for(const c of refs.roots){const r=solve(c.M,c.e);close(r.eccentricAnomaly,c.E,2e-12);assert.ok(r.iterations<=100);assert.ok(r.eccentricAnomaly>=r.bracket[0]-1e-12 && r.eccentricAnomaly<=r.bracket[1]+1e-12);close(r.eccentricAnomaly-c.e*Math.sin(r.eccentricAnomaly),c.M,1e-14);}
});
test('160 high-precision time solutions and independent Cartesian ODE/area integrations verify actual orbital motion',()=>{
  assert.equal(refs.elliptic.length,160);
  for(const c of refs.elliptic){const r=orbit(c.values),q=r.query,h=c.root;
    close(q.eccentricAnomaly,h.E);close(q.meanAnomaly,h.M);close(q.radius,h.radius);close(q.totalEnergy,h.energy);close(q.angularMomentum,h.angularMomentum);close(q.sweptArea,h.area);close(q.geometricSweptArea,h.area);
    q.position.forEach((v,j)=>{close(v,h.position[j]);close(v,c.odePosition[j],5e-9);});
    q.velocity.forEach((v,j)=>{close(v,h.velocity[j]);close(v,c.odeVelocity[j],5e-9);});
    close(q.sweptArea,c.odeArea,2e-8);
  }
});
test('circular equality gives a single allowed radius at the effective-potential minimum',()=>{
  const r=reduce([1,2,0,10,10,1]);assert.equal(r.conic,'circle');assert.equal(r.eccentricitySquared.label,'0');assert.equal(r.specificEnergy.label,'-1/2');assert.equal(r.periapsis,1);assert.equal(r.apoapsis,1);assert.equal(r.circularRadius,1);assert.equal(r.circularStabilityCurvature,1);assert.equal(r.query.turning,true);assert.deepEqual(r.query.radialVelocityBranches,[0]);
  const forbidden=reduce([1,2,0,10,20,1]);assert.equal(forbidden.query.allowed,false);assert.equal(forbidden.query.radialKineticPerMass.label,'-1/8');
});
test('nonzero angular momentum distinguishes bound, parabolic and hyperbolic conics by exact energy',()=>{
  const ellipse=reduce([1,2,0,5,10,1]),parabola=reduce([2,2,0,20,10,1]),hyperbola=reduce([1,2,10,20,30,1]);
  assert.equal(ellipse.conic,'ellipse');close(ellipse.periapsis!,1/7);close(ellipse.apoapsis!,1);
  assert.equal(parabola.conic,'parabola');assert.equal(parabola.specificEnergy.label,'0');assert.equal(parabola.eccentricitySquared.label,'1');assert.equal(parabola.apoapsis,null);
  assert.equal(hyperbola.conic,'hyperbola');assert.ok(hyperbola.specificEnergy.value>0);assert.ok(hyperbola.eccentricity>1);assert.equal(hyperbola.apoapsis,null);
});
test('zero angular momentum is a radial collision branch, never an invented parabolic spatial curve',()=>{
  for(const vr of [0,10,20,-30]){const r=reduce([1,2,vr,0,30,1]);assert.equal(r.eccentricitySquared.label,'1');assert.equal(r.radial,true);assert.equal(r.conic,null);assert.deepEqual(r.spatialGeometry,[]);assert.equal(r.periapsis,null);assert.equal(r.semiLatusRectum,null);assert.equal(r.circularRadius,null);assert.equal(r.collisionContinuation,false);}
  const bound=reduce([1,2,0,0,30,1]);assert.equal(bound.radialClass,'bound-collision');assert.equal(bound.apoapsis,1);assert.equal(bound.query.allowed,false);
  assert.equal(reduce([1,2,20,0,30,1]).radialClass,'positive-energy-collision-or-escape');
});
test('energy reduction supplies both radial speed branches and does not fabricate a time-selected query state',()=>{
  const r=reduce([1,2,10,10,10,1]);assert.equal(r.query.allowed,true);assert.deepEqual(r.query.radialVelocityBranches,[-1,1]);assert.equal(r.geometryIsTimeTrajectory,false);
  const opposite=reduce([1,2,-10,10,10,1]);assert.deepEqual(opposite.query,r.query);assert.equal(opposite.eccentricityVector[1].value,-r.eccentricityVector[1].value);
});
test('mass scales total energy and angular momentum without changing the prescribed gravitational acceleration model',()=>{
  const a=reduce([3,4,10,10,20,1]),b=reduce([3,4,10,10,20,4]);assert.deepEqual(a.spatialGeometry,b.spatialGeometry);assert.deepEqual(a.query,b.query);assert.equal(b.angularMomentum.value,4*a.angularMomentum.value);assert.equal(b.totalEnergy.value,4*a.totalEnergy.value);
});
test('all conic geometry samples stay on the positive-radius branch inside the declared clipping circle',()=>{
  for(const c of refs.radial.filter((_,j)=>j%7===0)){const r=reduce(c.values);if(r.radial)continue;assert.ok(r.spatialGeometry.length>700);
    for(const [x,y]of r.spatialGeometry){const radius=Math.hypot(x,y);assert.ok(radius>0 && radius<=r.spatialRadius*(1+1e-10));const [ex,ey]=r.eccentricityVector.map(v=>v.value);close(radius+ex*x+ey*y,r.semiLatusRectum!.value,2e-10);}
    const nearest=Math.min(...r.spatialGeometry.map(([x,y])=>Math.hypot(x-r.initialRadius.value,y)));close(nearest,0,2e-9);
    assert.ok(r.effectiveCurves.flat(2).every(Number.isFinite));
  }
});
test('near-radial unbound geometry retains its tiny periapsis without cancellation or asymptote connections',()=>{
  const r=reduce([2,1,-30,1,120,4]);assert.ok(r.periapsis!>0);assert.ok(r.periapsis!<.001);assert.equal(r.clipped,true);assert.ok(r.anomalyLimit!<Math.PI);assert.ok(r.spatialGeometry.every(p=>Math.hypot(...p)<=12+1e-7));
  const z=reduce([1,8,30,30,120,1]);assert.equal(z.conic,'hyperbola');assert.ok(z.anomalyLimit!<Math.acos(-1/z.eccentricity));
});
test('effective potential has a stable circular minimum and its stationary condition matches the inverse-square force',()=>{
  for(const c of refs.radial.filter((_,j)=>j%11===0)){const r=reduce(c.values);if(r.radial)continue;const rc=r.circularRadius!,h=r.specificAngularMomentum.value;close(r.mu/rc**2-h*h/rc**3,0,1e-8);assert.ok(r.circularStabilityCurvature!>0);close(-r.mu/rc+h*h/(2*rc**2),r.minimumSpecificEnergy!);assert.ok(r.specificEnergy.value>=r.minimumSpecificEnergy!-1e-10);}
});
test('Kepler circular motion uses the focus as center and the circular limit needs no periapsis direction',()=>{
  const r=orbit([1,2,0,0,0,10,1]);close(r.query.position[0],Math.cos(1));close(r.query.position[1],Math.sin(1));close(r.query.speed,1);close(r.query.radius,1);close(r.query.radialVelocity,0);close(r.period,2*Math.PI);assert.deepEqual(r.focus,[0,0]);close(Math.hypot(...r.ellipseCenter),0);
});
test('elliptic initial periapsis, focus-centered force, energy and angular momentum match compatible data',()=>{
  const r=orbit([2,4,10,0,0,0,3]);close(r.periapsis,1);close(r.apoapsis,3);assert.deepEqual(r.initial.position,[1,0]);close(r.initial.speed,Math.sqrt(3));close(r.initial.acceleration[0],-2);close(r.initial.totalEnergy,-1.5);close(r.initial.angularMomentum,3*Math.sqrt(3));assert.deepEqual(r.ellipseCenter,[-1,0]);
});
test('orientation rotates vectors while retrograde reverses signed angular momentum and swept area',()=>{
  const a=orbit([1,2,10,0,0,10,1]),rot=orbit([1,2,10,6,0,10,1]),back=orbit([1,2,10,0,1,10,1]);
  close(rot.query.position[0],-a.query.position[1]);close(rot.query.position[1],a.query.position[0]);close(rot.query.velocity[0],-a.query.velocity[1]);close(rot.query.velocity[1],a.query.velocity[0]);close(rot.query.sweptArea,a.query.sweptArea);close(back.query.position[0],a.query.position[0]);close(back.query.position[1],-a.query.position[1]);close(back.query.sweptArea,-a.query.sweptArea);close(back.query.angularMomentum,-a.query.angularMomentum);close(back.query.totalEnergy,a.query.totalEnergy);
});
test('accumulated area and eccentric anomaly retain repeated revolutions',()=>{
  const r=orbit([6,1,19,0,0,200,1]);assert.ok(r.query.eccentricAnomaly>40*Math.PI);assert.ok(r.query.sweptArea>10*Math.PI*r.semiMajorAxis*r.semiMinorAxis);close(r.query.sweptArea,r.query.geometricSweptArea);assert.ok(Math.abs(r.query.rootResidual)<1e-11);
});
test('time sampling explicitly includes periapsides, apoapsides and query, resolving sharp eccentric passages',()=>{
  const r=orbit([6,1,19,0,0,13,1]),curve=r.curves[0];
  assert.equal(curve[0][0],0);assert.equal(curve[curve.length-1][0],20);assert.ok(curve.some(([t])=>t===r.time));
  for(let k=1;k*Math.PI<Math.abs(solve(r.meanMotion*20,r.eccentricity).eccentricAnomaly);k++){const t=k*Math.PI/r.meanMotion;assert.ok(curve.some(([x,y])=>Math.abs(x-t)<1e-12 && Math.abs(y-(k%2?r.apoapsis:r.periapsis))<1e-12));}
  close(Math.hypot(...r.actualPath[r.actualPath.length-1].map((v,j)=>v-r.query.position[j]) as [number,number]),0);
  assert.ok(curve.every(([t])=>t>=0&&t<=20));
});
test('query-zero path contains only the initial point, while full ellipse is explicitly separate geometry',()=>{
  const r=orbit([6,1,19,-6,1,0,4]);assert.ok(r.actualPath.length<=2);r.actualPath.forEach(p=>p.forEach((v,j)=>close(v,r.initial.position[j])));assert.ok(r.geometry.length>500);assert.equal(r.scope,'bound-elliptic');assert.equal(r.collisionContinuation,false);assert.ok(r.periapsis>0);
});
test('orbital speed and period follow vis-viva and Kepler scaling, not uniform ellipse angle',()=>{
  const a=orbit([1,2,10,0,0,10,1]),b=orbit([1,4,10,0,0,10,1]),c=orbit([4,2,10,0,0,10,1]);
  close(b.period/a.period,2**1.5);close(c.period/a.period,.5);close(a.query.speed**2,a.mu*(2/a.query.radius-1/a.semiMajorAxis));close(a.period**2/a.semiMajorAxis**3,4*Math.PI**2/a.mu);
  assert.notEqual(a.query.eccentricAnomaly,a.query.meanAnomaly);
});
test('finite difference velocity and acceleration verify Kepler time parametrization and inverse-square dynamics',()=>{
  // Controls have tenth-second times; use sufficiently slow orbit and centered
  // differences, with the truncation error expected for that physical step.
  const states=[99,100,101].map(t=>orbit([1,8,10,3,0,t,1]).query),q=states[1];
  for(let j=0;j<2;j++){close((states[2].position[j]-states[0].position[j])/.2,q.velocity[j],2e-5);close((states[2].velocity[j]-states[0].velocity[j])/.2,q.acceleration[j],2e-5);}
});
test('all new outputs are JSON-safe, finite and fresh; no mutation leaks into later evaluations',()=>{
  for(const [f,p]of [[reduce,[1,2,0,10,10,1]],[orbit,[6,1,19,0,0,200,4]]] as const){const r=f([...p]),s=JSON.stringify(r);assert.doesNotThrow(()=>JSON.parse(s));assert.ok(!s.includes('null,null'));if('geometry'in r)r.geometry[0][0]=999;else r.spatialGeometry[0][0]=999;assert.equal(JSON.stringify(f([...p])),s);}
  const r=orbit([6,1,19,0,0,200,4]);assert.ok(r.curves.flat(2).every(Number.isFinite));assert.ok(r.actualPath.flat().every(Number.isFinite));close(cross(r.query.position,r.query.velocity),r.specificAngularMomentum);
});
test('invalid codes, singular radii and nonelliptic solver inputs reject explicitly',()=>{
  for(const [f,p]of [[reduce,[1,2,0,10,10,1]],[orbit,[1,2,10,0,0,10,1]]] as const){assert.throws(()=>f([]));for(let j=0;j<p.length;j++){const q:number[]=[...p];q[j]=NaN;assert.throws(()=>f(q));q[j]=p[j]+.5;assert.throws(()=>f(q));q[j]=Infinity;assert.throws(()=>f(q));}assert.throws(()=>f(new Array(p.length)));}
  assert.throws(()=>reduce([1,0,0,10,10,1]));assert.throws(()=>reduce([1,2,0,10,0,1]));assert.throws(()=>orbit([1,2,20,0,0,10,1]));assert.throws(()=>orbit([1,2,10,0,0,10,0]));
  for(const [M,e]of [[NaN,.5],[0,1],[0,-1],[Infinity,0],[1e7,.1]])assert.throws(()=>solve(M,e));
});

for(const id of ['inverse-square-effective-potential','elliptic-kepler-time-area'])for(const language of ['ko','en']as const)test(`central-force manuscript renders complete equations and explained exercises: ${id}.${language}`,async()=>{
  const fs=await import('node:fs/promises'),{getLearningContent}=await import('../src/lib/learning-content');
  const source=await fs.readFile(`content/learning/physics-math-for-ai/${id}.${language}.md`,'utf8'),r=await getLearningContent(id);
  assert.ok((source.match(/^## /gm)||[]).length>=10);assert.ok(source.includes('https://ocw.mit.edu/'));
  const heading=language==='en'?'## Worked examples and exercises with solutions':'## 예제와 해설이 있는 연습문제';
  const start=source.indexOf(heading);assert.ok(start>=0,'The original exercise section must remain');
  const end=source.indexOf('\n## ',start+heading.length),exercises=source.slice(start,end<0?undefined:end);
  assert.deepEqual([...exercises.matchAll(/^(\d)\. /gm)].map(m=>Number(m[1])),[1,2,3,4,5,6]);
  if(language==='en'){assert.ok(source.split(/\s+/).length>1000);assert.ok(!/[가-힣]/.test(source));assert.equal((exercises.match(/\*\*Solution:/g)||[]).length,6);}
  else {assert.ok((source.match(/[가-힣]/g)||[]).length>1000);assert.equal((exercises.match(/\*\*해설:/g)||[]).length,6);}
  assert.ok(r[language].includes('katex-display'));assert.ok(!r[language].includes('katex-error'));assert.ok(r.outline[language].length>=10);assert.equal(new Set(r.outline[language].map(x=>x.id)).size,r.outline[language].length);
});

test('registered adapters preserve finite JSON-safe stats and bound all spatial geometry including radial degeneracy',async()=>{
  const {evaluateClassicalMechanics:evaluate}=await import('../src/lib/learning-classical-mechanics');
  for(const [id,cases]of [['inverse-square-effective-potential',refs.radial],['elliptic-kepler-time-area',refs.elliptic]]as const)for(const c of cases){const r=evaluate(id,c.values),[xmin,xmax,ymin,ymax]=r.bounds!;assert.doesNotThrow(()=>JSON.stringify(r));assert.ok(Object.values(r.stats).every(v=>v===null||typeof v==='number'&&Number.isFinite(v)));for(const [x,y]of r.curves.flat()){assert.ok(Number.isFinite(x)&&Number.isFinite(y));assert.ok(x>=xmin&&x<=xmax&&y>=ymin&&y<=ymax);}}
});
test('six registered guided examples agree with independent circular, parabolic, radial and elliptic calculations',async()=>{
  const {classicalMechanicsLabs:labs,learningGuides}=await import('../src/data/interactive-book');
  const reduced=labs['inverse-square-effective-potential'].presets.map(p=>reduce(p.values));assert.deepEqual(reduced.map(r=>r.conic),['circle','parabola',null]);assert.deepEqual(reduced.map(r=>r.query.allowed),[true,true,false]);assert.deepEqual(reduced.map(r=>r.eccentricitySquared.label),['0','1','1']);
  const kepler=labs['elliptic-kepler-time-area'].presets.map(p=>orbit(p.values));close(kepler[0].query.sweptArea,.5);close(kepler[1].query.totalEnergy,-1.5);assert.ok(kepler[2].query.eccentricAnomaly<-40*Math.PI);assert.ok(kepler[2].query.sweptArea<0);
  for(const id of ['inverse-square-effective-potential','elliptic-kepler-time-area']as const){assert.equal(learningGuides[id].experiments.length,3);assert.equal(learningGuides[id].steps.length,4);}
});
