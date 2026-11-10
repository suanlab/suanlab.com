import test from 'node:test';
import assert from 'node:assert/strict';
import refs from './fixtures/initial-state-role-independent.json';
import {invertedSpringInitialState as spring,delayedNewtonRelease as release} from '../src/lib/learning-initial-state-role';
import {evaluateClassicalMechanics} from '../src/lib/learning-classical-mechanics';
import {classicalMechanicsLabs} from '../src/data/interactive-book';
const close=(a:number,b:number,tol=2e-10)=>assert.ok(Math.abs(a-b)<=tol*Math.max(1,Math.abs(a),Math.abs(b)),`${a} vs ${b}`);
const cache=new Map<string,ReturnType<typeof release>>();
const released=(p:number[])=>{const k=JSON.stringify(p);let r=cache.get(k);if(!r){r=release(p);cache.set(k,r);}return r;};

test('registered spring examples separate equal-energy branches, growing perturbations and the free shear limit',()=>{
  const rows=classicalMechanicsLabs['inverted-spring-initial-state-sensitivity'].presets.map(p=>evaluateClassicalMechanics('inverted-spring-initial-state-sensitivity',p.values));
  assert.deepEqual(rows.map(r=>r.initialStateSpring!.initialEnergyA.label),['0','0','0']);
  close(rows[0].stats['A query position meters']as number,Math.exp(-2));close(rows[0].stats['B query position meters']as number,Math.exp(2));
  close(rows[1].stats['actual amplification']as number,Math.exp(3));close(rows[2].stats['actual amplification']as number,Math.sqrt(5));assert.equal(rows[2].stats['1{free particle limit}'],1);
  for(const r of rows){const[x0,x1,y0,y1]=r.bounds!;assert.ok(r.curves.flat().every(([x,y])=>x>=x0&&x<=x1&&y>=y0&&y<=y1));}
});
test('registered release examples distinguish nonunique law, currently identical states and smooth zero coefficient',()=>{
  const rows=classicalMechanicsLabs['nonlipschitz-newton-delayed-release'].presets.map(p=>evaluateClassicalMechanics('nonlipschitz-newton-delayed-release',p.values));
  assert.deepEqual(rows.map(r=>r.stats['1{nonunique model constructed}']),[1,1,0]);assert.deepEqual(rows.map(r=>r.stats['1{distinct query states}']),[1,0,0]);
  assert.equal(rows[0].delayedRelease!.queryA.kinetic.label,'512');assert.equal(rows[0].delayedRelease!.queryB.kinetic.label,'8');
  for(const r of rows){assert.equal(r.stats['A exact Newton residual N'],0);assert.equal(r.stats['B exact energy residual joules'],0);const[x0,x1,y0,y1]=r.bounds!;assert.ok(r.curves.flat().every(([x,y])=>x>=x0&&x<=x1&&y>=y0&&y<=y1));}
});

test('800 independent matrix exponential and SVD cases verify states, variational flow and singular bounds',()=>{
  assert.equal(refs.linear.length,800);
  for(const f of refs.linear){const r=spring(f.values);assert.equal(r.initialEnergyA.label,f.energyA);assert.equal(r.initialEnergyB.label,f.energyB);for(let j=0;j<2;j++){for(let k=0;k<2;k++)close(r.matrix[j][k],f.matrix[j][k]);close([r.queryA.position,r.queryA.velocity][j],f.queryA[j]);close([r.queryB.position,r.queryB.velocity][j],f.queryB[j]);close([r.sigmaMax,r.sigmaMin][j],f.sigma[j]);close(r.predictedDifference[j],f.delta[j]);}assert.ok(r.queryDistance<=r.operatorBound+2e-10*Math.max(1,r.operatorBound));assert.ok(Math.abs(r.queryA.scaledEnergyResidual)<1e-12);assert.ok(Math.abs(r.queryB.scaledEnergyResidual)<1e-12);close(r.sigmaMax*r.sigmaMin,1,1e-14);close(r.numericalDeterminant,1,3e-7);}
});
test('160 independently integrated two-state Newton IVPs agree with the closed-form modal evaluation',()=>{
  assert.equal(refs.linearODE.length,160);for(const f of refs.linearODE){const r=spring(f.values);for(const[j,x]of [r.queryA.position,r.queryA.velocity,r.queryB.position,r.queryB.velocity].entries())close(x,f.state[j],4e-10);}
});
test('same initial position and zero energy do not specify growing versus decaying complete states',()=>{
  const r=spring([1,1,10,-10,10,10,20]);assert.equal(r.sameInitialPosition,true);assert.equal(r.sameInitialEnergy,true);assert.equal(r.sameCompleteInitialState,false);assert.equal(r.initialEnergyA.label,'0');close(r.queryA.position,Math.exp(-2));close(r.queryB.position,Math.exp(2));assert.equal(r.modeA!.growing,0);assert.equal(r.modeB!.decaying,0);
});
test('the exact decaying mode stays numerically resolved at the largest lambda-time product',()=>{
  const r=spring([1,4,10,-20,0,0,50]);close(r.queryA.position,Math.exp(-10),1e-15);close(r.queryA.velocity,-2*Math.exp(-10),1e-15);assert.equal(r.modeA!.growing,0);assert.equal(r.queryA.energyResidual,0);
});
test('a complete repeated state has identical futures while a projected position crossing can carry different velocities',()=>{
  const same=spring([4,3,13,-7,13,-7,33]);assert.equal(same.sameCompleteInitialState,true);assert.equal(same.initialDistance,0);assert.equal(same.queryDistance,0);assert.equal(same.amplification,null);assert.deepEqual(same.queryA,same.queryB);
  const cross=spring([1,1,0,10,0,-10,0]);assert.equal(cross.sameInitialPosition,true);assert.equal(cross.sameCompleteInitialState,false);assert.equal(cross.queryA.position,cross.queryB.position);assert.notEqual(cross.queryA.velocity,cross.queryB.velocity);
});
test('perturbation direction determines growth or contraction within the same area-preserving flow',()=>{
  const grow=spring([1,1,0,0,10,10,30]),decay=spring([1,1,0,0,10,-10,30]);close(grow.amplification!,Math.exp(3));close(decay.amplification!,Math.exp(-3));close(grow.sigmaMax,Math.exp(3));close(grow.sigmaMin,Math.exp(-3));assert.equal(grow.analyticDeterminant,1);assert.equal(grow.locallyLipschitz,true);
});
test('free-particle limit preserves full state and has a shear bound rather than exp(lambda*t)=1',()=>{
  const r=spring([2,0,10,0,10,10,20]);assert.equal(r.freeLimit,true);assert.equal(r.modeA,null);assert.deepEqual(r.matrix,[[1,2],[0,1]]);assert.equal(r.queryA.position,1);assert.equal(r.queryB.position,3);assert.equal(r.queryB.velocity,1);close(r.sigmaMax,1+Math.SQRT2);close(r.amplification!,Math.sqrt(5));assert.ok(r.amplification!<r.sigmaMax);assert.equal(r.initialEnergyB.label,'1');
});
test('rounding observed state can destroy exact inverse recovery while obeying the inverse operator error bound',()=>{
  for(const f of refs.linear.filter((_,j)=>j%19===0)){const r=spring(f.values);assert.ok(r.roundedReconstructionErrorNorm<=r.roundedRecoveryBound+2e-7*Math.max(1,r.roundedRecoveryBound));assert.ok(r.roundedRecoveryBound<=r.roundingBoxBound+1e-7);}
  const r=spring([1,1,10,-10,10,10,50]);assert.deepEqual(r.roundedObservationA,[.007,-.007]);assert.ok(r.roundedReconstructionErrorNorm>.05);assert.ok(r.reconstructionErrorNorm<1e-12);
});
test('700 independent translated-polynomial derivative and potential-integral references verify all exact release states',()=>{
  assert.equal(refs.delayed.length,700);for(const f of refs.delayed){const r=released(f.values);for(const key of ['queryA','queryB']as const)for(const k of ['position','velocity','acceleration','force','kinetic','potential','energy']as const)assert.equal(r[key][k].label,f[key][k]);for(const s of [r.queryA,r.queryB,r.stationary]){assert.equal(s.newtonResidual.label,'0');assert.equal(s.energyResidual.label,'0');assert.ok(s.position.value>=0);assert.ok(s.kinetic.value>=0);}}
});
test('180 independent positive-state Newton integrations validate the moving branch without selecting a wait from the zero state',()=>{
  assert.equal(refs.postReleaseODE.length,180);for(const f of refs.postReleaseODE){const r=release([f.mass,f.coefficientCode,0,0,Math.round(f.finish*10)]);close(r.queryA.position.value,f.state[0]);close(r.queryA.velocity.value,f.state[1]);assert.ok(f.scaledError<2e-10);}
});
test('different finite waits and a genuine stationary branch solve exactly the same complete zero initial state',()=>{
  const r=release([1,2,0,10,20]);assert.equal(r.queryA.position.label,'16');assert.equal(r.queryB.position.label,'1');assert.equal(r.queryA.velocity.label,'32');assert.equal(r.queryB.velocity.label,'4');assert.equal(r.queryA.kinetic.label,'512');assert.equal(r.queryB.kinetic.label,'8');assert.equal(r.stationary.position.label,'0');assert.equal(r.sameCompleteInitialState,true);assert.equal(r.nonuniqueConstructed,true);assert.equal(r.finiteWaitFamiliesDistinct,true);assert.equal(r.distinctAtQuery,true);assert.equal(r.stationaryIsInfiniteWait,true);
});
test('at and before release the position, velocity and acceleration match the stationary branch with no impulse',()=>{
  for(const t of [0,9,10]){const r=release([4,4,10,50,t]);for(const key of ['position','velocity','acceleration','force','kinetic','potential']as const)assert.equal(r.queryA[key].value,0);assert.equal(r.queryA.nonzeroMotion,false);assert.equal(r.distinctAtQuery,false);}const later=release([4,4,10,50,11]);assert.equal(later.queryA.position.label,'1/2500');assert.equal(later.queryA.velocity.label,'2/125');assert.equal(later.queryA.acceleration.label,'12/25');
});
test('zero coefficient restores smooth free force and collapses every finite-wait family to one stationary solution',()=>{
  const r=release([3,0,0,50,100]);assert.equal(r.locallyLipschitzAtOrigin,true);assert.equal(r.nonuniqueConstructed,false);assert.equal(r.finiteWaitFamiliesDistinct,false);assert.equal(r.distinctAtQuery,false);assert.ok(r.curves.flat().every(([,y])=>y===0));assert.ok(r.originSlopeProbes.every(p=>p.forcePerPosition.value===0));
});
test('equal displayed waits can coincide without removing the other members of the nonunique solution family',()=>{
  const r=release([1,1,10,10,20]);assert.deepEqual(r.queryA,r.queryB);assert.equal(r.finiteWaitFamiliesDistinct,false);assert.equal(r.distinctAtQuery,false);assert.equal(r.nonuniqueConstructed,true);assert.ok(r.queryA.position.value>0);assert.equal(r.stationary.position.value,0);
});
test('sample times include both release times and query with exact energy cancellations',()=>{
  const r=release([2,3,13,27,19]);for(const curve of r.curves){assert.ok(curve.some(([t])=>t===1.3));assert.ok(curve.some(([t])=>t===2.7));assert.ok(curve.some(([t])=>t===1.9));assert.ok(curve.every((p,j)=>j===0||p[0]>curve[j-1][0]));}for(const j of [14,17])assert.ok(r.curves[j].every(([,e])=>e===0));assert.ok(r.curves[2].every(([,x])=>x===0));
});
test('force-per-position probes grow toward zero for positive coefficient but finite samples are not a uniqueness test',()=>{
  const r=release([2,2,0,10,20]);assert.deepEqual(r.originSlopeProbes.map(p=>p.position.label),['1','1/4','1/16','1/64','1/256']);assert.deepEqual(r.originSlopeProbes.map(p=>p.forcePerPosition.label),['24','48','96','192','384']);assert.equal(r.locallyLipschitzAtOrigin,false);
});
test('mass scales force and energy while leaving the prescribed release kinematics unchanged',()=>{
  const a=release([1,2,0,10,20]),b=release([4,2,0,10,20]);for(const k of ['position','velocity','acceleration']as const)assert.deepEqual(a.queryA[k],b.queryA[k]);for(const k of ['force','kinetic','potential']as const)close(b.queryA[k].value,4*a.queryA[k].value);
});
test('all models reject missing, sparse, fractional, nonfinite and out-of-range controls',()=>{
  for(const[fn,p,ranges]of [[spring,[1,1,10,-10,10,10,20],[[1,4],[0,4],[-20,20],[-20,20],[-20,20],[-20,20],[0,50]]],[release,[1,2,0,10,20],[[1,4],[0,4],[0,50],[0,50],[0,100]]]]as const){assert.throws(()=>fn([]));assert.throws(()=>fn(new Array(p.length)));for(let j=0;j<p.length;j++)for(const v of [NaN,Infinity,p[j]+.5,ranges[j][0]-1,ranges[j][1]+1]){const q:number[]=[...p];q[j]=v;assert.throws(()=>fn(q));}}
});
test('outputs are fresh and JSON-safe with finite numeric curves at maximal controls',()=>{
  for(const[fn,p]of [[spring,[1,4,20,20,-20,-20,50]],[release,[4,4,0,50,100]]]as const){const r=fn([...p]),serialized=JSON.stringify(r);assert.ok(r.curves.flat(2).every(Number.isFinite));assert.doesNotThrow(()=>JSON.parse(serialized));r.curves[0][0][1]=999;assert.equal(JSON.stringify(fn([...p])),serialized);}
});

for(const id of ['inverted-spring-initial-state-sensitivity','nonlipschitz-newton-delayed-release'])for(const language of ['ko','en']as const)test(`initial-state manuscript renders complete derivations, intuition and six explained exercises: ${id}.${language}`,async()=>{
  const fs=await import('node:fs/promises'),{getLearningContent}=await import('../src/lib/learning-content');
  const source=await fs.readFile(`content/learning/physics-math-for-ai/${id}.${language}.md`,'utf8'),content=await getLearningContent(id);
  assert.ok((source.match(/^## /gm)||[]).length>=10);assert.equal((source.match(/^\d\. /gm)||[]).length,6);assert.ok(source.includes('https://'));
  if(language==='en'){assert.ok(source.split(/\s+/).length>1000);assert.ok(!/[가-힣]/.test(source));assert.equal((source.match(/\*\*Solution:/g)||[]).length,6);}else{assert.ok((source.match(/[가-힣]/g)||[]).length>1000);assert.equal((source.match(/\*\*해설:/g)||[]).length,6);}
  assert.ok(content[language].includes('katex-display'));assert.ok(!content[language].includes('katex-error'));assert.ok(content.outline[language].length>=10);assert.equal(new Set(content.outline[language].map(x=>x.id)).size,content.outline[language].length);
});
