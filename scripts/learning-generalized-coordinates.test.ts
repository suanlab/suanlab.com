import test from 'node:test';
import assert from 'node:assert/strict';
import refs from './fixtures/generalized-coordinates-independent.json';
import {polarGeneralizedCoordinates as polar,doublePendulumGeneralizedMetric as double} from '../src/lib/learning-generalized-coordinates';
import {analyticalMechanicsKinds,isAnalyticalMechanicsKind,evaluateAnalyticalMechanics} from '../src/lib/learning-analytical-mechanics';
import {analyticalMechanicsLabs,learningUnits,learningGuides} from '../src/data/interactive-book';
const close=(a:number,b:number,tol=3e-10)=>assert.ok(Math.abs(a-b)<=tol*Math.max(1,Math.abs(a),Math.abs(b)),`${a} vs ${b}`);
const vector=(a:number[],b:number[],tol=3e-10)=>{assert.equal(a.length,b.length);a.forEach((x,j)=>close(x,b[j],tol));};
const matrix=(a:number[][],b:number[][])=>{assert.equal(a.length,b.length);a.forEach((r,j)=>vector(r,b[j]));};

test('analytical registry retains generalized coordinates and extends to action without absorbing prior Newton or geometry units',()=>{
  assert.deepEqual(Object.keys(analyticalMechanicsLabs).sort(),[...analyticalMechanicsKinds].sort());
  assert.ok(analyticalMechanicsKinds.includes('polar-generalized-velocity-force-metric'));
  assert.ok(analyticalMechanicsKinds.includes('double-pendulum-generalized-mass-metric'));
  for(const id of analyticalMechanicsKinds)assert.equal(isAnalyticalMechanicsKind(id),true);
  for(const id of ['mechanics-energy','circle-manifold-charts','inverted-spring-initial-state-sensitivity','unknown',''])assert.equal(isAnalyticalMechanicsKind(id),false);
});
for(const id of analyticalMechanicsKinds)test(`analytical controls, localized guides and null-aware finite readouts: ${id}`,()=>{
  const spec=analyticalMechanicsLabs[id],g=learningGuides[id],u=learningUnits.find(u=>u.id===id)!;
  assert.equal(g.steps.length,4);assert.equal(g.experiments.length,3);assert.equal(u.questions.length,2);
  const checkEnglish=(v:unknown)=>{if(!v||typeof v!=='object')return;for(const[k,x]of Object.entries(v)){if(k==='en'){assert.equal(typeof x,'string');assert.ok(!/[가-힣]/.test(String(x)));}else checkEnglish(x);}};for(const v of [spec,g,u])checkEnglish(v);
  const base=spec.controls.map(c=>c.initial);
  for(const p of [base,...spec.presets.map(p=>p.values),...spec.controls.flatMap((c,j)=>[c.min,c.max].map(x=>{const v=base.slice();v[j]=x;return v;}))]){
    const r=evaluateAnalyticalMechanics(id,p),[x0,x1,y0,y1]=r.bounds!;assert.ok(Object.values(r.stats).every(v=>v===null||typeof v==='number'&&Number.isFinite(v)));assert.ok(r.curves.flat().every(([x,y])=>x>=x0&&x<=x1&&y>=y0&&y<=y1));assert.doesNotThrow(()=>JSON.stringify(r));
  }
});
test('polar registered presets expose force-free curvature, work duality and an undefined inverse at the origin',()=>{
  const rows=analyticalMechanicsLabs[analyticalMechanicsKinds[0]].presets.map(p=>evaluateAnalyticalMechanics(analyticalMechanicsKinds[0],p.values));
  assert.deepEqual(rows.map(r=>r.stats['kinetic energy joules']),[5,5,1]);assert.equal(rows[0].stats['radial coordinate acceleration m per s squared'],4);assert.equal(rows[0].stats['angular coordinate acceleration rad per s squared'],-4);assert.equal(rows[1].stats['Cartesian power watts'],11);assert.equal(rows[2].stats['radial coordinate acceleration m per s squared'],null);assert.equal(rows[2].stats['local Jacobian rank'],1);assert.equal(rows[2].stats['physical configuration DOF'],2);
});
test('double-pendulum registered presets preserve point energy and power under the actual rate transformation',()=>{
  const rows=analyticalMechanicsLabs[analyticalMechanicsKinds[1]].presets.map(p=>evaluateAnalyticalMechanics(analyticalMechanicsKinds[1],p.values));
  vector(rows.map(r=>r.stats['Cartesian kinetic energy joules']as number),[.5,1.5,.5]);vector(rows.map(r=>r.stats['Cartesian power watts']as number),[0,-10,-10]);assert.equal(rows[0].stats['tip velocity x m per s'],0);assert.equal(rows[0].stats['relative angular rate2 rad per s'],-2);assert.equal(rows[0].stats['finite tangent constraint1 m squared'],.25);assert.ok(Math.abs(rows[0].stats['finite mapped constraint1 m squared']as number)<1e-12);
});

test('900 independent symbolic-Jacobian/Hessian and linear-solve references verify polar kinematics, force and domain probes',()=>{
  assert.equal(refs.polar.length,900);for(const f of refs.polar){const r=polar(f.values);vector(r.position,f.position);matrix(r.jacobian,f.jacobian);vector(r.velocity,f.velocity);vector(r.curvatureAcceleration,f.curvature);vector(r.cartesianAcceleration,f.physicalAcceleration);matrix(r.metric,f.metric);vector(r.generalizedForce,f.generalizedForce);close(r.kineticCartesian,f.kinetic);close(r.cartesianPower,f.power);vector(r.virtualDisplacement,f.virtualDisplacement);close(r.virtualWorkCartesian,f.virtualWork);assert.equal(r.queryProbe.inParameterDomain,f.probeValid);assert.equal(r.queryProbe.regularTarget,f.probeRegular);vector(r.queryProbe.linearizedPoint,f.probeLinear);
    if(f.generalizedAcceleration){vector(r.generalizedAcceleration!,f.generalizedAcceleration);vector(r.naiveAcceleration!,f.naiveAcceleration!);assert.ok(r.accelerationResidual!<1e-11);}else{assert.equal(r.generalizedAcceleration,null);assert.equal(r.naiveAcceleration,null);assert.equal(r.accelerationResidual,null);}
    if(f.probeExact){vector(r.queryProbe.exactPoint!,f.probeExact);close(r.queryProbe.linearizationError!,f.probeError!);close(r.queryProbe.finiteWork!,f.probeWork!);}else{assert.equal(r.queryProbe.exactPoint,null);assert.equal(r.queryProbe.linearizationError,null);assert.equal(r.queryProbe.finiteWork,null);}
    close(r.metricResidual,0);close(r.kineticResidual,0);close(r.powerResidual,0);close(r.virtualWorkResidual,0);
  }
});
test('900 symbolic stacked-point and weighted-SVD references verify double-pendulum metrics, constraints and chart invariance',()=>{
  assert.equal(refs.double.length,900);for(const f of refs.double){const r=double(f.values);vector(r.pose.stack,f.position);matrix(r.jacobian,f.jacobian);vector(r.velocityStack,f.velocities);matrix(r.metricAbsolute,f.metric);matrix(r.metricRelative,f.metricRelative);vector(r.eigenvaluesAbsolute,f.eigenvalues);vector(r.eigenvaluesRelative,f.eigenvaluesRelative);close(r.numericalDeterminant,f.determinant);close(r.kineticCartesian,f.kinetic);close(r.potential,f.potential);vector(r.generalizedForceAbsolute,f.force);vector(r.generalizedForceRelative,f.forceRelative);close(r.cartesianPower,f.power);vector(r.relativeRates,f.relativeRates);matrix(r.constraintJacobian,f.constraintJacobian);vector(r.constraintVelocity,f.constraintVelocity);vector(r.virtualDisplacement,f.virtualDisplacement);close(r.virtualWorkCartesian,f.virtualWork);vector(r.queryProbe.exactPose.stack,f.probeExact);vector(r.queryProbe.linearPose,f.probeLinear);vector(r.queryProbe.exactConstraints,f.probeExactConstraints);vector(r.queryProbe.linearConstraints,f.probeLinearConstraints);close(r.queryProbe.linearizationError,f.probeError);close(r.queryProbe.finiteWork,f.probeWork);
    for(const x of [r.metricResidual,r.kineticAbsoluteResidual,r.kineticRelativeResidual,r.generalizedForceResidual,r.powerAbsoluteResidual,r.powerRelativeResidual,r.virtualWorkAbsoluteResidual,r.virtualWorkRelativeResidual,...r.constraintValues,...r.constraintVelocity,...r.constraintTangentResidual.flat()])close(x,0);assert.ok(r.eigenvaluesAbsolute[0]>0);assert.ok(r.eigenvaluesRelative[0]>0);close(r.numericalDeterminant,r.determinantFormula);
  }
});
test('180 independent 50-digit polar derivatives recover velocity and Newton acceleration from a local coordinate polynomial',()=>{
  assert.equal(refs.polarDifferential.length,180);for(const f of refs.polarDifferential){const r=polar(f.values);vector(r.velocity,f.velocity);vector(r.reconstructedAcceleration!,f.acceleration);vector(r.cartesianAcceleration,f.acceleration);}
});
test('180 independent 50-digit double-pendulum derivatives verify physical rates and virtual tangent directions',()=>{
  assert.equal(refs.doubleDifferential.length,180);for(const f of refs.doubleDifferential){const r=double(f.values);vector(r.velocityStack,f.velocity);vector(r.virtualDisplacement,f.variation);}
});
test('a force-free particle can have nonzero polar accelerations that cancel curvature terms',()=>{
  const r=polar([2,2,0,10,20,0,0,10,10,50]);vector(r.velocity,[1,2]);vector(r.cartesianAcceleration,[0,0]);vector(r.generalizedAcceleration!,[4,-4]);vector(r.curvatureAcceleration,[-4,4]);vector(r.naiveAcceleration!,[4,-4]);assert.equal(r.kineticCartesian,5);close(r.naiveAccelerationError!,Math.sqrt(32));close(r.accelerationResidual!,0);assert.equal(r.dynamicsIntegrated,false);
});
test('radial force and angular generalized torque are work-conjugate quantities with different units',()=>{
  const r=polar([2,2,0,10,20,30,40,10,10,50]);vector(r.generalizedForce,[3,4]);vector(r.generalizedAcceleration!,[5.5,-2]);vector(r.cartesianAcceleration,[1.5,2]);assert.equal(r.cartesianPower,11);assert.equal(r.generalizedPower,11);const q=polar([2,4,0,10,20,30,40,10,10,50]);assert.equal(q.tangentialForce,4);assert.equal(q.generalizedForce[1],8);assert.equal(q.radialForce,3);
});
test('a singular polar origin keeps Cartesian acceleration finite while inverse coordinate acceleration remains undefined',()=>{
  const r=polar([2,0,0,10,20,30,40,10,10,50]);assert.equal(r.localChartRegular,false);assert.equal(r.localJacobianRank,1);assert.equal(r.physicalDOF,2);assert.equal(r.analyticMetricDeterminant,0);assert.equal(r.generalizedAcceleration,null);assert.equal(r.reconstructedAcceleration,null);assert.equal(r.naiveAccelerationError,null);vector(r.cartesianAcceleration,[1.5,2]);close(r.kineticResidual,0);close(r.powerResidual,0);assert.equal(r.kineticCartesian,1);
});
test('angle periodicity is a global representation issue distinct from a local rank failure',()=>{
  const a=polar([1,2,0,10,10,10,20,0,0,0]),b=polar([1,2,24,10,10,10,20,0,0,0]);vector(a.position,b.position,1e-14);vector(a.velocity,b.velocity,1e-14);close(a.kineticCartesian,b.kineticCartesian,1e-14);assert.notEqual(a.angle,b.angle);assert.equal(a.localChartRegular,true);assert.equal(b.localChartRegular,true);assert.equal(a.globallyOneToOne,false);
});
test('an invalid finite polar probe does not invalidate its regular base point or invent a negative-radius position',()=>{
  const r=polar([2,1,0,0,0,10,20,-10,5,100]);assert.equal(r.localChartRegular,true);assert.equal(r.queryProbe.inParameterDomain,false);assert.equal(r.queryProbe.exactPoint,null);assert.equal(r.queryProbe.linearizationError,null);assert.equal(r.queryProbe.finiteWork,null);assert.equal(r.domainBoundaryEpsilon,.5);assert.ok(r.linearizationCurve.every(([e])=>e<=.5));assert.deepEqual(r.actualProbeCurve[r.actualProbeCurve.length-1],[0,0]);assert.equal(r.linearizationCurve[r.linearizationCurve.length-1][0],.5);
});
test('a non-grid polar domain boundary is explicitly sampled as radius zero without crossing into invalid probes',()=>{
  const r=polar([1,1,3,0,0,0,0,-6,4,50]),edge=5/6;close(r.domainBoundaryEpsilon!,edge);assert.ok(r.linearizationCurve.some(([e])=>e===edge));assert.ok(r.linearizationCurve.every(([e])=>e<=edge));vector(r.actualProbeCurve[r.actualProbeCurve.length-1],[0,0]);
});
test('zero virtual variation and zero epsilon give zero linearization error without changing chart regularity',()=>{
  for(const p of [[2,2,3,10,20,30,40,0,0,100],[2,2,3,10,20,30,40,10,10,0]]){const r=polar(p);assert.equal(r.queryProbe.linearizationError,0);assert.equal(r.queryProbe.workLinearizationError,0);}const r=double([1,1,2,2,3,-4,10,10,0,0,0,0,100]);assert.equal(r.queryProbe.linearizationError,0);assert.equal(r.queryProbe.workLinearizationError,0);
});
test('the metric-weighted kinetic value is invariant under angle representation and mass scaling',()=>{
  const a=polar([1,3,4,7,-9,-20,10,0,0,0]),b=polar([4,3,4,7,-9,-20,10,0,0,0]);vector(a.velocity,b.velocity);close(b.kineticCartesian,4*a.kineticCartesian);assert.equal(b.localChartRegular,a.localChartRegular);close(a.kineticFormula,a.kineticMetric);
});
test('opposite absolute angular rates can keep the tip instantaneously still with nonzero generalized forces',()=>{
  const r=double([1,1,2,2,0,0,10,-10,30,0,10,-10,50]);vector(r.velocities[0],[1,0]);vector(r.velocities[1],[0,0]);matrix(r.metricAbsolute,[[2,1],[1,1]]);matrix(r.metricRelative,[[5,2],[2,1]]);vector(r.relativeRates,[1,-2]);vector(r.generalizedForceAbsolute,[3,3]);vector(r.generalizedForceRelative,[6,3]);assert.equal(r.cartesianPower,0);assert.equal(r.kineticCartesian,.5);assert.equal(r.couplingKinetic,-1);assert.equal(r.diagonalKinetic,1.5);
});
test('perpendicular links expose different absolute/relative metric spectra while preserving physical kinetic energy',()=>{
  const r=double([1,1,2,2,0,6,10,10,0,0,10,10,50]);vector(r.velocities[0],[1,0]);vector(r.velocities[1],[1,1]);matrix(r.metricAbsolute,[[2,0],[0,1]]);matrix(r.metricRelative,[[3,1],[1,1]]);assert.equal(r.kineticCartesian,1.5);vector(r.generalizedForceAbsolute,[0,-10]);vector(r.generalizedForceRelative,[-10,-10]);close(r.cartesianPower,-10);assert.ok(Math.abs(r.eigenvaluesRelative[0]-r.eigenvaluesAbsolute[0])>.1);close(r.eigenvaluesRelative[0],2-Math.SQRT2);close(r.eigenvaluesRelative[1],2+Math.SQRT2);
});
test('unequal masses and lengths retain a strictly positive metric at aligned links',()=>{
  const r=double([2,1,2,1,6,6,0,20,0,0,10,0,50]);matrix(r.metricAbsolute,[[3,.5],[.5,.25]]);matrix(r.metricRelative,[[4.25,.75],[.75,.25]]);close(r.determinantFormula,.5);close(r.kineticCartesian,.5);vector(r.gravityGeneralized,[-30,-5]);close(r.cartesianPower,-10);assert.equal(r.positiveDefiniteByAssumptions,true);assert.equal(r.constraintRankByAssumptions,2);assert.equal(r.physicalDOF,2);
});
test('virtual tangent constraints vanish to first order while a finite tangent step violates both link lengths',()=>{
  const r=double([1,1,2,2,0,0,10,-10,30,0,10,-10,50]);vector(r.constraintVelocity,[0,0]);matrix(r.constraintTangentResidual,[[0,0],[0,0]]);vector(r.queryProbe.linearConstraints,[.25,.25]);vector(r.queryProbe.exactConstraints,[0,0]);assert.equal(r.virtualWorkCartesian,0);assert.ok(r.queryProbe.finiteWork<0);assert.ok(r.queryProbe.linearizationError>0);assert.equal(r.constraintReactionsSolved,false);
});
test('halving a small finite virtual step reduces first-order configuration error quadratically',()=>{
  const p=[1,1,2,2,0,0,10,-10,30,0,10,-10,2],a=double(p);p[12]=1;const b=double(p);close(a.queryProbe.linearizationError/b.queryProbe.linearizationError,4,.001);assert.ok(a.queryProbe.linearConstraints.every(x=>x>=-1e-12));
});
test('relative angular rates are unwrapped differences and chart changes preserve power rather than component values',()=>{
  const r=double([1,1,2,2,12,-12,20,-20,30,30,10,-10,50]);close(r.relativeAngles[1],-2*Math.PI);vector(r.relativeRates,[2,-4]);assert.notDeepEqual(r.generalizedForceAbsolute,r.generalizedForceRelative);close(r.cartesianPower,r.relativePower);close(r.kineticCartesian,r.kineticRelative);
});
test('all numerical probe curves have increasing dimensionless epsilon and finite values without claiming a time trajectory',()=>{
  for(const r of [polar([4,6,-24,-20,20,-40,40,-10,10,100]),double([4,4,6,6,-12,12,-20,20,-30,30,-10,10,100])]){assert.equal(r.dynamicsIntegrated,false);assert.equal(r.instantaneousOnly,true);const curves='probeCurves'in r?r.probeCurves:[r.linearizationCurve,r.finiteWorkCurve];for(const c of curves){assert.ok(c.flat().every(Number.isFinite));assert.ok(c.every(([e],j)=>e>=0&&e<=1&&(j===0||e>c[j-1][0])));}}
});
test('invalid arity, sparse controls, zero links/masses, fractional and nonfinite codes are rejected',()=>{
  for(const[fn,p,ranges]of [[polar,[2,2,0,10,20,30,40,10,10,50],[[1,4],[0,6],[-24,24],[-20,20],[-20,20],[-40,40],[-40,40],[-10,10],[-10,10],[0,100]]],[double,[1,1,2,2,0,0,10,-10,30,0,10,-10,50],[[1,4],[1,4],[1,6],[1,6],[-12,12],[-12,12],[-20,20],[-20,20],[-30,30],[-30,30],[-10,10],[-10,10],[0,100]]]]as const){assert.throws(()=>fn([]));assert.throws(()=>fn(new Array(p.length)));assert.throws(()=>fn([...p,0]));for(let j=0;j<p.length;j++)for(const v of [NaN,Infinity,p[j]+.5,ranges[j][0]-1,ranges[j][1]+1]){const q:number[]=[...p];q[j]=v;assert.throws(()=>fn(q));}}
});
test('results are fresh and JSON-safe without leaking mutable geometry or matrix state',()=>{
  for(const[fn,p]of [[polar,[2,2,0,10,20,30,40,10,10,50]],[double,[1,1,2,2,0,0,10,-10,30,0,10,-10,50]]]as const){const r=fn([...p]),serialized=JSON.stringify(r);assert.doesNotThrow(()=>JSON.parse(serialized));r.jacobian[0][0]=999;assert.equal(JSON.stringify(fn([...p])),serialized);}
});

for(const id of ['polar-generalized-velocity-force-metric','double-pendulum-generalized-mass-metric'])for(const language of ['ko','en']as const)test(`generalized-coordinate manuscript renders derivations, intuition and six explained exercises: ${id}.${language}`,async()=>{
  const fs=await import('node:fs/promises'),{getLearningContent}=await import('../src/lib/learning-content');
  const source=await fs.readFile(`content/learning/physics-math-for-ai/${id}.${language}.md`,'utf8'),content=await getLearningContent(id);
  assert.ok((source.match(/^## /gm)||[]).length>=10);assert.equal((source.match(/^\d\. /gm)||[]).length,6);assert.ok(source.includes('https://'));
  if(language==='en'){assert.ok(source.split(/\s+/).length>1000);assert.ok(!/[가-힣]/.test(source));assert.equal((source.match(/\*\*Solution:/g)||[]).length,6);}else{assert.ok((source.match(/[가-힣]/g)||[]).length>1000);assert.equal((source.match(/\*\*해설:/g)||[]).length,6);}
  assert.ok(content[language].includes('katex-display'));assert.ok(!content[language].includes('katex-error'));assert.ok(content.outline[language].length>=10);assert.equal(new Set(content.outline[language].map(x=>x.id)).size,content.outline[language].length);
});
