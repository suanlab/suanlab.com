import test from 'node:test';
import assert from 'node:assert/strict';
import refs from './fixtures/hamiltonian-canonical-independent.json';
import {polarHamiltonianLegendreMap as polar,timeGaugeHamiltonianEnergy as gauge} from '../src/lib/learning-hamiltonian-canonical';
const close=(a:number,b:number,tol=5e-10)=>assert.ok(Math.abs(a-b)<=tol*Math.max(1,Math.abs(a),Math.abs(b)),`${a} vs ${b}`);
const vector=(a:number[],b:number[],tol=5e-10)=>{assert.equal(a.length,b.length);a.forEach((x,j)=>close(x,b[j],tol));};
const fp=[1,2,2,0,0,2,20],gp=[1,0,0,2,4,20];
const rat=(x:string)=>{const[n,d='1']=x.split('/');return Number(n)/Number(d);};

test('240 independent60digit Cartesian/Legendre references verify momenta and canonical equations',()=>{
 assert.equal(refs.polar.length,240);for(const row of refs.polar){const r=polar(row.p),q=r.query;close(q.physicalEnergy,Number(row.physicalEnergy));close(q.physicalAngularMomentum,Number(row.physicalAngularMomentum));assert.equal(q.analyticOrigin,row.origin);if(row.canonical){assert.ok(q.canonical);for(const key of Object.keys(row.canonical)as(keyof typeof row.canonical)[])close(q.canonical[key],Number(row.canonical[key]));}else{assert.equal(q.canonical,null);assert.equal(q.principalAngle,null);}}
});
test('180 independently differentiated and quadrature-checked time-gauge trajectories agree',()=>{
 assert.equal(refs.gauge.length,180);for(const row of refs.gauge){const q=gauge(row.p).query;for(const key of Object.keys(row.query)as(keyof typeof row.query)[])close(q[key],Number(row.query[key]));}
});
test('64 independent exact Fraction free-path quadratures verify action, momentum and Hamiltonian shifts',()=>{
 assert.equal(refs.free.length,64);for(const row of refs.free){const q=gauge(row.p).query;for(const key of Object.keys(row.exact)as(keyof typeof row.exact)[])close(q[key],rat(row.exact[key]),2e-12);}
});
test('free polar motion has changing radial momentum with constant angular momentum and Hamiltonian',()=>{
 const r=polar([1,0,2,0,2,2,0]),q=r.query.canonical!;assert.equal(q.radialMomentum,1);assert.equal(q.angularMomentum,1);assert.equal(q.hamiltonian,1);assert.equal(q.radialMomentumRate,1);assert.equal(q.angularMomentumRate,0);assert.equal(q.radialVelocity,1);assert.equal(q.angularVelocity,1);close(q.angularAccelerationFromHamilton,-2);assert.ok(r.final.canonical);close(r.final.canonical.hamiltonian,1);close(r.final.canonical.angularMomentum,1);assert.notEqual(r.final.canonical.radialMomentum,q.radialMomentum);
});
test('circular momenta invert to the correct rates; clockwise angular momentum retains its sign',()=>{
 for(const W of [-2,2]){const q=polar([1,2,2,0,0,W,20]).query.canonical!;close(q.radialMomentum,0);close(q.radialMomentumRate,0);close(q.angularMomentum,W*Math.PI/4);close(q.angularVelocity,W*Math.PI/4);close(q.hamiltonian,Math.PI**2/4);close(q.radialKinetic,0);close(q.angularKinetic,Math.PI**2/8);close(q.potential,Math.PI**2/8);}
});
test('canonical polar state is null at origins while Cartesian energy and angular momentum stay finite',()=>{
 const r=polar([1,2,2,0,0,0,20]);assert.equal(r.query.canonical,null);assert.equal(r.query.principalAngle,null);assert.equal(r.query.chartRegular,false);vector(r.query.position,[0,0]);vector(r.query.velocity,[-Math.PI/2,0]);close(r.query.physicalEnergy,Math.PI**2/8);assert.equal(r.query.physicalAngularMomentum,0);assert.ok(r.initial.canonical);assert.ok(r.final.canonical);
});
test('every canonical segment and radial phase curve splits at on-grid and off-grid origins',()=>{
 for(const p of [[1,0,1,0,-3,0,0],[1,2,2,0,2,0,0],[1,3,2,0,0,0,40]]){const r=polar(p);assert.ok(r.analyticOriginTimes.length);for(const root of r.analyticOriginTimes)for(const groups of r.canonicalSegments)for(const segment of groups)assert.ok(!segment.some(([t])=>t<root)||!segment.some(([t])=>t>root));assert.equal(r.radialPhaseSegments.length,r.canonicalSegments[0].length);for(const phase of r.radialPhaseSegments)assert.ok(phase.every(([radius])=>radius>0));}
});
test('Legendre velocity Hessian and inverse are positive at regular radii with separate coordinate units',()=>{
 for(const row of refs.polar){const q=polar(row.p).query.canonical;if(!q)continue;assert.ok(q.velocityHessianDiagonal.every(x=>x>0));assert.ok(q.velocityHessianDeterminant>0);q.velocityHessianDiagonal.forEach((x,j)=>close(x*q.inverseVelocityHessianDiagonal[j],1));close(q.velocityHessianDiagonal[0]*q.velocityHessianDiagonal[1],q.velocityHessianDeterminant);close(q.radialVelocityDiscrepancy,0);close(q.angularVelocityDiscrepancy,0);close(q.radialMomentumRateDiscrepancy,0);close(q.angularMomentumRateDiscrepancy,0);close(q.radialAccelerationDiscrepancy,0);close(q.angularAccelerationDiscrepancy,0);close(q.legendreDiscrepancy,0);close(q.physicalEnergyDiscrepancy,0);}
});
test('mass scaling preserves the same central flow and scales canonical momenta and energy',()=>{
 const a=polar(fp),p=[...fp];p[0]=4;const b=polar(p);vector(a.query.position,b.query.position);close(b.stiffness,4*a.stiffness);close(b.query.canonical!.angularMomentum,4*a.query.canonical!.angularMomentum);close(b.query.canonical!.hamiltonian,4*a.query.canonical!.hamiltonian);close(b.query.canonical!.velocityHessianDeterminant,16*a.query.canonical!.velocityHessianDeterminant);
});
test('ordinary autonomous oscillator has canonical momentum equal to mechanical momentum and H=E',()=>{
 const r=gauge([1,2,2,0,0,20]);assert.equal(r.explicitlyTimeDependent,false);assert.equal(r.selectedOrbitHasConstantHamiltonian,true);for(let j=0;j<=200;j++){close(r.curves[2][j][1],r.curves[3][j][1]);close(r.curves[6][j][1],r.curves[7][j][1]);close(r.curves[7][j][1],r.initialEnergy);}
});
test('time-dependent gauge changes H sign without changing restoring physical motion or energy',()=>{
 const a=gauge([1,2,2,0,8,0]),b=gauge([1,2,2,0,8,20]);assert.equal(a.explicitlyTimeDependent,true);assert.equal(a.selectedOrbitHasConstantHamiltonian,false);assert.ok(a.query.hamiltonian<0);assert.ok(b.query.hamiltonian>0);close(a.query.physicalEnergy,Math.PI**2/8);close(b.query.physicalEnergy,Math.PI**2/8);close(b.query.hamiltonian,Math.PI**2/8);assert.ok(a.stiffness>0);assert.ok(a.stiffness-a.gamma<0);assert.ok(a.hamiltonianRange[0]<0&&a.hamiltonianRange[1]>0);
});
test('rational free example distinguishes all momenta, Hamiltonian, physical force and power',()=>{
 const r=gauge(gp),q=r.query;assert.equal(q.position,1);assert.equal(q.velocity,1);assert.equal(q.mechanicalMomentum,1);assert.equal(q.canonicalMomentum,3);assert.equal(q.canonicalMomentumRate,4);close(q.mechanicalForce,0);assert.equal(q.physicalEnergy,.5);assert.equal(q.hamiltonian,-.5);assert.equal(q.hamiltonianTotalTime,-2);assert.equal(q.partialTime,-2);assert.equal(q.physicalEnergyRate,0);assert.equal(q.baseCumulativeAction,.5);assert.equal(q.gaugeCumulativeAction,1.5);
});
test('explicit time dependence is a function property; selected rest orbits may still have constant H',()=>{
 for(const p of [[1,0,2,0,4,40],[1,2,0,0,8,40]]){const r=gauge(p);assert.equal(r.explicitlyTimeDependent,true);assert.equal(r.selectedOrbitHasConstantHamiltonian,true);close(r.hamiltonianRange[0],r.hamiltonianRange[1]);close(r.query.hamiltonianTotalTime,0);}
});
test('gauge variation preserves every physical trajectory sample and changes only its canonical representation',()=>{
 for(const f of [0,1,2,4]){const a=gauge([3,f,2,-2,0,13]),b=gauge([3,f,2,-2,8,13]);for(const j of [0,1,4,5,6,11,14,17])assert.deepEqual(a.curves[j],b.curves[j]);assert.notDeepEqual(a.curves[3],b.curves[3]);assert.notDeepEqual(a.curves[7],b.curves[7]);}
});
test('partial derivatives at fixed canonical P reproduce time evolution and action one-form coefficients',()=>{
 for(let j=0;j<180;j++){const r=gauge([1+j%4,j%5,-4+j%9,-4+(j*3)%9,-8+(j*7)%17,j%41]),q=r.query;for(const key of ['hamiltonXResidual','hamiltonPResidual','eulerLagrangeResidual','newtonResidual','physicalEnergyRate','energyRelationDiscrepancy','legendreDiscrepancy','totalMinusExplicitTime','oneFormPositionDiscrepancy','oneFormTimeDiscrepancy','oneFormAlongPathDiscrepancy']as const)close(q[key],0);close(q.hamiltonianTotalFromCanonical,q.hamiltonianTotalTime);assert.ok(r.maximumSampledEnergyResidual<1e-10);assert.ok(r.maximumSampledPositionEquationResidual<1e-10);assert.ok(r.maximumSampledMomentumEquationResidual<1e-10);}
});
test('query time selects the same analytic trajectory and canonical representation history',()=>{
 const a=gauge(gp),p=[...gp];p[5]=0;const b=gauge(p);assert.deepEqual(a.curves,b.curves);assert.deepEqual(a.phaseCurves,b.phaseCurves);assert.notEqual(a.query.canonicalMomentum,b.query.canonicalMomentum);
});
for(const[name,fn,p,ranges]of [['polar',polar,fp,[[1,4],[0,4],[1,6],[-4,4],[-4,4],[-4,4],[0,40]]],['gauge',gauge,gp,[[1,4],[0,4],[-4,4],[-4,4],[-8,8],[0,40]]]]as const){
 test(`${name}: invalid arity,sparse,nonfinite,fractional and out-of-range controls are rejected`,()=>{assert.throws(()=>fn([]));assert.throws(()=>fn([...p,0]));assert.throws(()=>fn(new Array(p.length)));for(let i=0;i<p.length;i++)for(const v of [NaN,Infinity,p[i]+.5,ranges[i][0]-1,ranges[i][1]+1]){const q:number[]=[...p];q[i]=v;assert.throws(()=>fn(q));}});
 test(`${name}: results remain fresh,finite,serializable and do not mutate control arrays`,()=>{const input:number[]=[...p],before=[...input],a=fn(input),b=fn(input);assert.deepEqual(input,before);assert.deepEqual(a,b);const walk=(v:unknown):void=>{if(typeof v==='number')assert.ok(Number.isFinite(v));else if(v&&typeof v==='object')Object.values(v).forEach(walk);};walk(a);a.query.time=123456;assert.notDeepEqual(a,b);assert.doesNotThrow(()=>JSON.stringify(b));});
}

import {readFile} from 'node:fs/promises';
import {getLearningContent} from '../src/lib/learning-content';
for(const id of ['polar-hamiltonian-legendre-map','time-gauge-hamiltonian-energy'])for(const locale of ['ko','en']as const)test(`${id}.${locale}: derivations,worked examples and six explained exercises render strict math`,async()=>{
 const source=await readFile(`content/learning/physics-math-for-ai/${id}.${locale}.md`,'utf8'),content=await getLearningContent(id);assert.ok(content.outline[locale].length>=10);assert.equal(new Set(content.outline[locale].map(h=>h.id)).size,content.outline[locale].length);assert.ok(content[locale].includes('katex-display'));assert.ok(!content[locale].includes('katex-error'));assert.equal((source.match(/^\d\. /gm)||[]).length,6);assert.ok(source.includes('https://davidtong.org/teaching/classical-dynamics/dynhtml/S4'));
 if(locale==='en'){assert.doesNotMatch(source,/[가-힣]/);assert.ok(source.split(/\s+/).length>1000);assert.equal((source.match(/\*\*Solution:\*\*/g)||[]).length,6);}else{assert.ok((source.match(/[가-힣]/g)||[]).length>1000);assert.equal((source.match(/\*\*해설:\*\*/g)||[]).length,6);}
});

import {learningUnits,learningGuides,analyticalMechanicsLabs} from '../src/data/interactive-book';
import {curriculumDomains} from '../src/data/learning-curriculum';
import {evaluateAnalyticalMechanics,isAnalyticalMechanicsKind} from '../src/lib/learning-analytical-mechanics';
test('registered canonical lessons link prerequisites, bilingual guides and all three live presets',()=>{
 for(const id of ['polar-hamiltonian-legendre-map','time-gauge-hamiltonian-energy'] as const){
  assert.ok(isAnalyticalMechanicsKind(id));const unit=learningUnits.find(u=>u.id===id)!;assert.ok(unit);assert.ok(unit.prerequisiteIds.every(p=>learningUnits.some(u=>u.id===p)));assert.ok(curriculumDomains.find(d=>d.id==='analytical-mechanics')!.unitIds.includes(id));
  const guide=learningGuides[id],lab=analyticalMechanicsLabs[id];assert.equal(guide.steps.length,4);assert.equal(guide.experiments.length,3);assert.equal(unit.questions.length,2);
  for(const [j,preset]of lab.presets.entries()){const result=evaluateAnalyticalMechanics(id,preset.values);assert.ok(result.table.rows.length>=20);assert.equal(guide.experiments[j].preset,j);assert.equal(preset.values.length,lab.controls.length);assert.ok(result.polarHamiltonian||result.timeGaugeHamiltonian);for(const [i,value]of preset.values.entries())assert.ok(value>=lab.controls[i].min&&value<=lab.controls[i].max);}
  for(const q of unit.questions)assert.ok(q.answer>=0&&q.answer<q.options.length);
  assert.doesNotMatch(JSON.stringify([unit.title.en,unit.subtitle.en,guide.intuition.en,...guide.steps.map(s=>[s.title.en,s.explanation.en]),...guide.experiments.map(e=>[e.title.en,e.predict.en,e.action.en,e.explain.en]),...lab.controls.map(c=>c.label.en),lab.axis.en,lab.note.en]),/[가-힣]/);
 }
});
test('live evaluator keeps origin canonical readouts null while physical energy survives',()=>{
 const r=evaluateAnalyticalMechanics('polar-hamiltonian-legendre-map',[1,2,2,0,0,0,20]);assert.equal(r.stats['query Hamiltonian J'],null);assert.equal(r.stats['1{canonical chart regular}'],0);close(r.stats['query physical energy J'] as number,Math.PI**2/8);assert.ok(r.table.rows.some(row=>row[0]==='Hamiltonian'&&row[1]==='undefined at origin'));
});
test('live time-gauge readout distinguishes forces and momenta in the rational free experiment',()=>{
 const r=evaluateAnalyticalMechanics('time-gauge-hamiltonian-energy',[1,0,0,2,4,20]);assert.equal(r.stats['query canonical momentum kg m per s'],3);assert.equal(r.stats['query canonical momentum rate N'],4);close(r.stats['query physical force N'] as number,0);assert.equal(r.stats['query gauge action Js'],1.5);
});
