import test from 'node:test';
import assert from 'node:assert/strict';
import refs from './fixtures/lagrange-equations-independent.json';
import {polarLagrangeNewtonEquivalence as polar,doublePendulumLagrangeDynamics as double} from '../src/lib/learning-lagrange-equations';
const close=(a:number,b:number,tol=5e-10)=>assert.ok(Math.abs(a-b)<=tol*Math.max(1,Math.abs(a),Math.abs(b)),`${a} vs ${b}`);
const vector=(a:number[],b:number[],tol=5e-10)=>{assert.equal(a.length,b.length);a.forEach((x,j)=>close(x,b[j],tol));};
const fp=[1,2,2,0,0,2,20],dp=[1,1,2,2,6,0,10,-10,20,2];

test('240 independent60digit flow/radius derivatives verify polar Lagrange kinematics and regularity',()=>{
  assert.equal(refs.polar.length,240);
  for(const row of refs.polar){const r=polar(row.p),q=r.query;vector(r.analyticOriginTimes,row.originTimes.map(Number));vector(q.position,row.position.map(Number));vector(q.velocity,row.velocity.map(Number));vector(q.acceleration,row.acceleration.map(Number));close(q.radius,Number(row.radius));close(q.kinetic,Number(row.kinetic));close(q.potential,Number(row.potential));close(q.angularMomentum,Number(row.angularMomentum));assert.equal(q.analyticOrigin,row.analyticOrigin);assert.equal(q.chartRegular,!row.analyticOrigin);if(row.polar){assert.ok(q.polar);for(const k of ['radialRate','angularRate','radialAcceleration','angularAcceleration']as const)close(q.polar[k],Number(row.polar[k]));}else assert.equal(q.polar,null);}
});
test('120 independently differentiated Cartesian-energy models verify double-pendulum initial dynamics and reactions',()=>{
  assert.equal(refs.instantaneous.length,120);
  for(const row of refs.instantaneous){const q=double(row.p).initial,x=row.initial;vector(q.accelerations,x.accelerations);vector(q.accelerations,x.highPrecisionAccelerations.map(Number));q.matrix.forEach((r,j)=>vector(r,x.matrix[j]));vector([...q.pose.p1,...q.pose.p2],x.pose);vector(q.velocities.flat(),x.velocities);vector(q.cartesianAccelerations.flat(),x.cartesianAccelerations);vector(q.rod1Force,x.rod1Force);vector(q.rod2Force,x.rod2Force);close(q.kinetic,x.kinetic);close(q.potential,x.potential);close(q.energy,x.energy);}
});
test('finest RK4 trajectories agree with18 independently refined DOP853 paths',()=>{
  assert.equal(refs.trajectories.length,18);
  for(const row of refs.trajectories){assert.ok(row.solverRefinementGap<1e-9);const p=[...row.p];p[9]=4;const r=double(p);for(const[t,z]of row.times.map((t,j)=>[t,row.states[j]]as const)){const j=Math.round(t*100);vector([r.curves[0][j][1],r.curves[1][j][1]],z.slice(0,2),2e-6);vector([r.curves[2][j][1],r.curves[3][j][1]],z.slice(2),2e-6);}}
});
test('free motion has nonzero coordinate accelerations while physical acceleration vanishes',()=>{
  const r=polar([1,0,2,0,2,2,0]),q=r.query;vector(q.position,[1,0]);vector(q.velocity,[1,1]);assert.ok(q.polar);close(q.polar.radialAcceleration,1);close(q.polar.angularAcceleration,-2);vector(q.polar.reconstructedAcceleration,[0,0]);assert.equal(q.newtonResidual,0);assert.equal(r.freeLimit,true);
});
test('circular central motion has zero radial coordinate acceleration and nonzero centripetal acceleration',()=>{
  const r=polar(fp),q=r.query;assert.equal(r.circularByInitialData,true);close(q.radius,1);assert.ok(q.polar);close(q.polar.radialAcceleration,0);assert.ok(Math.hypot(...q.acceleration)>2);vector(q.position,[0,1]);close(q.polar.angularRate,Math.PI/2);close(q.polar.physicalRadialAcceleration,-(Math.PI**2)/4);
});
test('harmonic origin crossing keeps Cartesian motion finite and polar inverse null without epsilon clamps',()=>{
  const r=polar([1,2,2,0,0,0,20]),q=r.query;assert.equal(q.analyticOrigin,true);assert.equal(q.radius,0);vector(q.position,[0,0]);vector(q.velocity,[-Math.PI/2,0]);vector(q.acceleration,[0,0]);assert.equal(q.polar,null);assert.equal(q.chartRegular,false);assert.deepEqual(r.analyticOriginTimes,[1]);for(const groups of r.polarSegments){assert.equal(groups.length,2);assert.ok(groups[0].every(([t])=>t<1));assert.ok(groups[1].every(([t])=>t>1));}
});
test('free radial crossing and principal-angle branch changes do not break physical evolution',()=>{
  const r=polar([2,0,2,0,-2,0,20]);assert.equal(r.query.polar,null);vector(r.final.position,[-1,0]);vector(r.final.velocity,[-1,0]);assert.equal(r.final.chartRegular,true);assert.ok(r.final.polar);close(Math.abs(r.final.polar.angle),Math.PI);assert.equal(r.maximumEnergyResidual,0);
});
test('near-origin regular states remain defined and all displayed polar segments have real gaps only at origins',()=>{
  for(const tc of [19,21]){const q=polar([1,2,2,0,0,0,tc]).query;assert.ok(q.radius>0);assert.ok(q.polar);assert.equal(q.chartRegular,true);assert.equal(q.analyticOrigin,false);close(q.polar.accelerationDiscrepancy,0);}
});
test('mass changes stiffness at fixed frequency while preserving central trajectory',()=>{
  const a=polar(fp),p=[...fp];p[0]=4;const b=polar(p);vector(a.query.position,b.query.position);vector(a.query.velocity,b.query.velocity);close(b.stiffness,4*a.stiffness);close(b.initialEnergy,4*a.initialEnergy);close(b.initialAngularMomentum,4*a.initialAngularMomentum);
});
test('polar energy and momentum remain conserved and both EL equations reconstruct Newton acceleration',()=>{
  for(let j=0;j<100;j++){const r=polar([1+j%4,j%5,1+j%6,-4+j%9,-4+(j*3)%9,-4+(j*7)%9,j%41]);assert.ok(r.maximumEnergyResidual<1e-10);assert.ok(r.maximumAngularMomentumResidual<1e-10);if(r.query.polar){close(r.query.polar.radialELResidual,0);close(r.query.polar.angularELResidual,0);close(r.query.polar.angularMomentumRate,0);close(r.query.polar.accelerationDiscrepancy,0);}}
});
test('downward and upright equilibria remain stationary with exact cardinal identities',()=>{
  for(const code of [0,12,-12]){const r=double([1,1,2,2,code,code,0,0,40,2]);vector(r.final.angles,r.initial.angles);vector(r.final.rates,[0,0]);vector(r.final.accelerations,[0,0]);assert.equal(r.maxEnergyError,0);assert.equal(r.maxRefinedEnergyError,0);assert.equal(r.maxAngleRefinementGap,0);}
});
test('resting downward rods transmit the two masses weight with no constraint power',()=>{
  const q=double([2,3,2,4,0,0,0,0,0,1]).initial;vector(q.rod1Force,[0,50]);vector(q.rod2Force,[0,30]);close(q.rod1SignedTension,50);close(q.rod2SignedTension,30);assert.equal(q.constraintPower,0);assert.equal(q.potential,0);
});
test('positive mass matrix, full EL balance and independently reconstructed Cartesian rod forces agree',()=>{
  for(const row of refs.instantaneous){const r=double(row.p),q=r.query;assert.ok(q.determinant>0);assert.ok(q.matrix[0][0]>0);vector(q.eulerLagrangeResidual,[0,0],3e-9);vector(q.rodTangentialResiduals,[0,0],3e-9);vector(q.rodLengthSquaredResiduals,[0,0]);close(q.constraintPower,0,3e-9);close(q.energyRate,0,3e-9);close(q.kineticDiscrepancy,0);close(q.determinantDiscrepancy,0);assert.ok(q.kinetic>=-1e-12);assert.ok(q.potential>=0);}
});
test('omitting velocity-dependent inertial terms changes coupled accelerations',()=>{
  const r=double(dp),q=r.initial;assert.ok(q.naiveAccelerationError>.5);assert.ok(Math.hypot(...q.coriolis)>.5);assert.ok(r.maxEnergyError>0);assert.ok(r.maxRefinedEnergyError>0);
});
test('step refinement exposes numerical errors instead of relabeling RK4 as exact dynamics',()=>{
  const row=refs.trajectories[2],p=[...row.p];p[9]=1;const a=double(p);p[9]=2;const b=double(p);
  const truth=row.states[row.states.length-1],error=(r:ReturnType<typeof double>)=>Math.hypot(...r.final.angles.map((v,j)=>v-truth[j]));
  assert.ok(error(b)<error(a));const ratio=a.maxEnergyError/a.maxRefinedEnergyError;assert.ok(ratio>12&&ratio<20,`observed energy refinement ratio ${ratio}`);assert.ok(a.maxAngleRefinementGap>0);assert.ok(a.maxRateRefinementGap>0);assert.equal(a.stepSize,.0025);assert.equal(a.refinedStepSize,.00125);assert.equal(b.stepSize,.00125);assert.equal(a.integrationSteps,800);assert.equal(a.refinedIntegrationSteps,1600);assert.equal(a.method,'RK4');
});
test('query time selects the same physical trajectory and is not an integration step-count control',()=>{
  const a=double(dp),p=[...dp];p[8]=40;const b=double(p);assert.deepEqual(a.curves,b.curves);assert.deepEqual(a.final,b.final);assert.equal(a.time,1);assert.equal(b.time,2);assert.notDeepEqual(a.query.angles,b.query.angles);
});
for(const[name,fn,p,ranges]of [['polar',polar,fp,[[1,4],[0,4],[1,6],[-4,4],[-4,4],[-4,4],[0,40]]],['double',double,dp,[[1,4],[1,4],[1,6],[1,6],[-12,12],[-12,12],[-10,10],[-10,10],[0,40],[1,4]]]]as const){
 test(`${name}: malformed arity, sparse controls and every invalid control boundary are rejected`,()=>{assert.throws(()=>fn([]));assert.throws(()=>fn(new Array(p.length)));assert.throws(()=>fn([...p,0]));ranges.forEach(([lo,hi],i)=>{for(const v of [lo-1,hi+1,NaN,Infinity,p[i]+.5]){const q:number[]=[...p];q[i]=v;assert.throws(()=>fn(q));}});});
 test(`${name}: results are fresh, finite, serializable and do not mutate inputs`,()=>{const input:number[]=[...p],before=[...input],a=fn(input),b=fn(input);assert.deepEqual(input,before);assert.deepEqual(a,b);const walk=(v:unknown):void=>{if(typeof v==='number')assert.ok(Number.isFinite(v));else if(v&&typeof v==='object')Object.values(v).forEach(walk);};walk(a);a.curves[0][0][1]=123456;assert.notDeepEqual(a,b);assert.doesNotThrow(()=>JSON.stringify(b));});
}

import {readFile} from 'node:fs/promises';
import {getLearningContent} from '../src/lib/learning-content';
for(const id of ['polar-lagrange-newton-equivalence','double-pendulum-lagrange-dynamics'])for(const locale of ['ko','en']as const)test(`${id}.${locale}: detailed derivations, intuition and six explained exercises render strict math`,async()=>{
  const source=await readFile(`content/learning/physics-math-for-ai/${id}.${locale}.md`,'utf8'),content=await getLearningContent(id);
  assert.ok(content.outline[locale].length>=10);assert.equal(new Set(content.outline[locale].map(h=>h.id)).size,content.outline[locale].length);assert.ok(content[locale].includes('katex-display'));assert.ok(!content[locale].includes('katex-error'));assert.ok(source.includes('https://underactuated.mit.edu/multibody.html'));assert.equal((source.match(/^\d\. /gm)||[]).length,6);
  if(locale==='en'){assert.doesNotMatch(source,/[가-힣]/);assert.ok(source.split(/\s+/).length>1000);assert.equal((source.match(/\*\*Solution:\*\*/g)||[]).length,6);}else{assert.ok((source.match(/[가-힣]/g)||[]).length>1000);assert.equal((source.match(/\*\*해설:\*\*/g)||[]).length,6);}
});

test('analytic crossings between output grid points are inserted and break polar curves',()=>{
  for(const p of [[1,0,1,0,-3,0,0],[1,2,2,0,2,0,0],[1,3,2,0,0,0,0]]){
    const r=polar(p);assert.ok(r.analyticOriginTimes.length>0);for(const time of r.analyticOriginTimes){for(const segments of r.polarSegments)assert.ok(segments.every(segment=>!segment.some(([t])=>t<time)||!segment.some(([t])=>t>time)),'No segment bridges an undefined origin');const eventIndex=r.curves[12].findIndex(([t,rad])=>Math.abs(t-time)<1e-14&&rad===0);assert.ok(eventIndex>=0,'An actual zero-radius event is sampled');vector(r.phase[eventIndex],[0,0]);}
    for(const c of r.curves)assert.ok(c.every(([t,v],j)=>Number.isFinite(v)&&(j===0||t>c[j-1][0])));
  }
  const free=polar([1,0,1,0,-3,0,0]);close(free.analyticOriginTimes[0],1/3);assert.equal(free.polarSegments[0].length,2);
});

import {analyticalMechanicsLabs,learningUnits,learningGuides} from '../src/data/interactive-book';
import {evaluateAnalyticalMechanics} from '../src/lib/learning-analytical-mechanics';
for(const id of ['polar-lagrange-newton-equivalence','double-pendulum-lagrange-dynamics']as const)test(`${id}: registered guide presets preserve physical meanings and units`,()=>{
 const spec=analyticalMechanicsLabs[id],guide=learningGuides[id],unit=learningUnits.find(u=>u.id===id)!;
 assert.equal(guide.steps.length,4);assert.equal(guide.experiments.length,3);assert.equal(unit.questions.length,2);guide.experiments.forEach((x,j)=>assert.equal(x.preset,j));
 const rows=spec.presets.map(x=>evaluateAnalyticalMechanics(id,x.values));
 if(id==='polar-lagrange-newton-equivalence'){
  close(rows[0].stats['query radial coordinate acceleration m per s squared']as number,1);close(rows[0].stats['query angular acceleration rad per s squared']as number,-2);close(rows[0].stats['query Cartesian ax m per s squared']as number,0);assert.equal(rows[1].stats['1{circular initial data}'],1);close(rows[1].stats['query radial coordinate acceleration m per s squared']as number,0);assert.equal(rows[2].stats['1{polar chart regular}'],0);assert.equal(rows[2].stats['query angular rate rad per s'],null);close(rows[2].stats['query Cartesian vx m per s']as number,-Math.PI/2);
 }else{
  vector(rows[0].doubleLagrange!.query.accelerations,[-10,0]);vector(rows[1].doubleLagrange!.query.accelerations,[-10.5,1]);close(rows[1].stats['query mechanical energy J']as number,21.5);assert.equal(rows[2].stats['query time s'],2);assert.equal(rows[2].stats['RK4 step s'],.00125);assert.ok((rows[2].stats['maximum sampled angle refinement gap rad']as number)>0);
 }
});
