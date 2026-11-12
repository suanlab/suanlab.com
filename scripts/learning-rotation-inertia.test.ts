import test from 'node:test';
import assert from 'node:assert/strict';
import tensors from './fixtures/rotation-inertia-tensor-independent.json';
import eulers from './fixtures/rotation-inertia-euler-independent.json';
import {pointMassInertia as tensor,torqueFreeEuler as euler,bodyToLabRotation as rotation} from '../src/lib/learning-rotation-inertia';
const close=(a:number,b:number,tol=1e-8)=>assert.ok(Math.abs(a-b)<=tol*Math.max(1,Math.abs(a),Math.abs(b)),`${a} versus ${b}`);
const base=[1,1,1,2,4,6,0,10,10,0,0,0];

test('600 independent Fraction point sums verify exact inertia, momenta, kinetic energies and COM decomposition',()=>{
  assert.equal(tensors.length,600);
  for(const c of tensors){const r=tensor(c.values),e=c.exact;
    for(const key of ['mass','kineticCOM','kineticPivot','translationKinetic']as const)assert.equal(r[key].label,e[key]);
    assert.equal(r.axisInertia?.label??null,e.axisInertia);
    for(const key of ['tensorCOM','tensorPivot']as const)assert.deepEqual(r[key].map(row=>row.map(x=>x.label)),e[key]);
    for(const key of ['angularMomentumCOM','angularMomentumPivot','linearMomentum','centerVelocity']as const)assert.deepEqual(r[key].map(x=>x.label),e[key]);
    for(const key of ['kineticDecompositionResidual','pointKineticResidual']as const)assert.equal(r[key].label,'0');
    for(const key of ['pointMomentumResidual','pointAngularResidual','originAngularResidual']as const)assert.ok(r[key].every(x=>x.label==='0'));
    assert.ok(r.parallelAxisResidual.flat().every(x=>x.label==='0'));assert.ok(r.centerOfMass.every(x=>x.label==='0'));
  }
});
test('600 independent NumPy spectral and rotated-coordinate checks verify the full tensor map',()=>{
  for(const c of tensors){const r=tensor(c.values);r.spectrum.eigen.forEach((e,j)=>{assert.ok(e.value>0);close(e.value,c.eigenvalues[j],5e-11);});r.spectrum.residuals.forEach(x=>close(x,0,5e-11));
    for(const key of ['labOmega','labAngularMomentum']as const)r[key].forEach((x,j)=>close(x,c[key][j],5e-11));
    r.labTensor.forEach((row,i)=>row.forEach((x,j)=>close(x,c.labTensor[i][j],5e-11)));
    for(const [index,value]of c.directional)close(r.directionalInertia[index][1],value,5e-11);
  }
});
test('anisotropic diagonal inertia makes angular momentum nonparallel to angular velocity',()=>{
  const r=tensor(base);assert.deepEqual(r.tensorCOM.map(row=>row.map(x=>x.label)),[['26','0','0'],['0','20','0'],['0','0','10']]);assert.deepEqual(r.angularMomentumPivot.map(x=>x.label),['26','20','0']);assert.equal(r.kineticPivot.label,'23');assert.equal(r.nonparallel,true);assert.ok(r.angleBetweenAngularAndOmega!>0);
});
test('moving the selected pivot changes the velocity field and introduces COM translation',()=>{
  const p=base.slice();p[10]=10;const r=tensor(p);assert.equal(r.pivotChangesVelocityField,true);assert.equal(r.timeEvolutionSolved,false);assert.equal(r.kineticCOM.label,'23');assert.equal(r.kineticPivot.label,'26');assert.equal(r.translationKinetic.label,'3');assert.deepEqual(r.centerVelocity.map(x=>x.label),['0','0','1']);assert.deepEqual(r.linearMomentum.map(x=>x.label),['0','0','6']);assert.deepEqual(r.angularMomentumPivot.map(x=>x.label),['26','26','0']);assert.deepEqual(r.angularMomentumCOM.map(x=>x.label),['26','20','0']);assert.equal(r.nonparallel,false);
});
test('negative offdiagonal tensor entries coexist with strictly positive kinetic energy and principal moments',()=>{
  const p=base.slice();p[6]=1;p[9]=10;const r=tensor(p);assert.equal(r.tensorCOM[0][2].label,'-3');assert.equal(r.tensorCOM[2][0].label,'-3');assert.ok(r.spectrum.eigen.every(e=>e.value>0));assert.ok(r.kineticPivot.value>0);assert.equal(r.pointKineticResidual.label,'0');
});
test('exact repeated-spectrum detection does not pretend an isotropic principal basis is unique',()=>{
  const r=tensor([1,1,1,2,2,2,0,10,10,0,0,0]);assert.equal(r.spectrum.repeatedSpectrum,true);assert.equal(r.spectrum.blockDegenerate,true);r.spectrum.eigen.forEach(e=>close(e.value,4));assert.equal(r.nonparallel,false);assert.equal(tensor(base).spectrum.repeatedSpectrum,false);
});
test('zero angular velocity leaves directional axis inertia and momentum-angle query undefined',()=>{
  const r=tensor([4,4,4,1,1,1,-4,0,0,0,-20,-12]);assert.equal(r.zeroOmega,true);assert.equal(r.axisInertia,null);assert.equal(r.normalizedAxis,null);assert.equal(r.angleBetweenAngularAndOmega,null);assert.equal(r.kineticPivot.label,'0');assert.ok(r.angularMomentumPivot.every(x=>x.label==='0'));
});
test('a common coordinate rotation transforms tensor and vectors while preserving exact physical scalars',()=>{
  const p=base.slice(),a=tensor(p);p[11]=6;const b=tensor(p);close(b.labOmega[0],-a.labOmega[1]);close(b.labOmega[1],a.labOmega[0]);close(b.labAngularMomentum[0],-a.labAngularMomentum[1]);close(b.labAngularMomentum[1],a.labAngularMomentum[0]);assert.deepEqual(a.kineticPivot,b.kineticPivot);assert.deepEqual(a.spectrum,b.spectrum);b.points.forEach((v,j)=>{close(v.labPosition[0],-a.points[j].labPosition[1]);close(v.labPosition[1],a.points[j].labPosition[0]);});
});
test('queried axis inertia lies in the complete3D spectral bounds, distinct from the displayed xz direction sweep',()=>{
  for(const c of tensors.filter((_,j)=>j%9===0)){const r=tensor(c.values);if(r.axisInertia){assert.ok(r.axisInertia.value>=r.spectrum.eigen[0].value-1e-10);assert.ok(r.axisInertia.value<=r.spectrum.eigen[2].value+1e-10);}assert.equal(r.directionalInertia.length,181);close(r.directionalInertia[0][0],0);close(r.directionalInertia[180][0],Math.PI);close(r.directionalInertia[0][1],r.tensorPivot[0][0].value);}
});
test('180 independent DOP853 body-transport and rotation-matrix integrations verify actual Euler motion',()=>{
  assert.equal(eulers.length,180);
  for(const c of eulers){const r=euler(c.values),q=r.query;assert.equal(r.initialEnergy.label,c.energy);assert.equal(r.initialAngularMomentumNormSquared.label,c.angularNormSquared);q.bodyOmega.forEach((x,j)=>close(x,c.bodyOmega[j]));q.rotation.forEach((row,i)=>row.forEach((x,j)=>close(x,c.rotation[i][j])));q.labAngularMomentum.forEach((x,j)=>close(x,c.labAngularMomentum[j]));assert.ok(r.strictTriangleInequalities);}
});
test('all numerical sample states retain energy, angular-norm and inertial vector-momentum invariants',()=>{
  for(const c of eulers.filter((_,j)=>j%7===0)){const r=euler(c.values);for(const q of r.states){close(q.energyResidual,0);close(q.angularNormResidual,0);q.labAngularResidual.forEach(x=>close(x,0));close(q.rawQuaternionNormResidual,0);close(q.orthogonalityResidual,0);close(q.determinantResidual,0);}}
});
test('axisymmetric analytic body precession verifies the sign of the Euler transport term',()=>{
  for(const t of [0,1,10]){const r=euler([1,1,2,10,5,10,0,t*10]),nu=-1/3;assert.deepEqual(r.principalMoments,[3,3,2]);assert.equal(r.initialEnergy.label,'23/8');assert.equal(r.initialAngularMomentumNormSquared.label,'61/4');close(r.query.bodyOmega[0],Math.cos(nu*t)-.5*Math.sin(nu*t));close(r.query.bodyOmega[1],Math.sin(nu*t)+.5*Math.cos(nu*t));close(r.query.bodyOmega[2],1);assert.equal(r.axisymmetric,true);assert.equal(r.spherical,false);}
});
test('a spherical rotor keeps body angular velocity constant while its orientation follows the physical spin',()=>{
  const r=euler([1,1,1,10,10,10,0,100]),norm=Math.sqrt(3),half=norm*10/2,expected=rotation([Math.cos(half),...Array(3).fill(Math.sin(half)/norm)]);r.query.bodyOmega.forEach(x=>close(x,1));r.query.rotation.forEach((row,i)=>row.forEach((x,j)=>close(x,expected[i][j])));assert.equal(r.spherical,true);assert.equal(r.initialEnergy.label,'3');assert.equal(r.initialAngularMomentumNormSquared.label,'12');
});
test('pure principal-axis spin gives the elementary initial-orientation plus physical-time rotation',()=>{
  const r=euler([3,2,1,0,0,10,6,100]),theta=Math.PI/2+10;close(r.query.rotation[0][0],Math.cos(theta));close(r.query.rotation[1][0],Math.sin(theta));close(r.query.rotation[2][2],1);assert.deepEqual(r.query.bodyOmega,[0,0,1]);
});
test('zero spin is stationary with a valid nonzero initial orientation and no artificial numerical torque',()=>{
  const r=euler([5,5,5,0,0,0,6,100]);assert.equal(r.stationary,true);assert.equal(r.initialEnergy.label,'0');for(const s of r.states){assert.deepEqual(s.bodyOmega,[0,0,0]);assert.equal(s.energy,0);assert.deepEqual(s.rawQuaternion,r.initialQuaternion);}close(r.query.rotation[1][0],1);
});
test('intermediate-axis linear instability is distinguished from extreme-axis oscillations and repeated moments',()=>{
  const r=euler([3,2,1,0,10,0,0,100]);assert.deepEqual(r.principalMoments,[3,4,5]);assert.deepEqual(r.stability.map(s=>s.linearClass),['oscillatory','unstable','oscillatory']);close(r.stability[1].unitSpinGrowthRateSquared,1/15);assert.deepEqual(r.query.bodyOmega,[0,1,0]);assert.ok(euler([1,1,1,10,0,0,0,100]).stability.every(s=>s.linearClass==='degenerate'));const perturbed=euler([3,2,1,1,10,0,0,100]);assert.ok(Math.abs(perturbed.query.bodyOmega[0])>.1);close(perturbed.query.energyResidual,0);
});
test('body momentum can change under zero external torque while the inertial momentum stays fixed',()=>{
  const r=euler([1,1,2,10,5,10,6,100]);assert.deepEqual(r.externalTorque,[0,0,0]);assert.ok(Math.abs(r.query.bodyAngularMomentum[0]-r.initial.bodyAngularMomentum[0])>.1);r.query.labAngularResidual.forEach(x=>close(x,0));assert.equal(r.solver.quaternionProjectedDuringIntegration,false);assert.equal(r.query.orientationNormalizedForDisplay,true);
});
test('Hamilton quaternion sign equivalence and explicit invalid-norm rejection protect orientation calculations',()=>{
  assert.deepEqual(rotation([1,2,3,4]),rotation([-1,-2,-3,-4]));for(const q of [[0,0,0,0],[NaN,0,0,0],[Infinity,0,0,0],[1,0],new Array(4),[1e308,1e308,1e308,1e308]])assert.throws(()=>rotation(q));
});
test('query time is an exact sample stop and finite plot arrays stay in the declared0to10second interval',()=>{
  const r=euler([5,1,4,20,-20,20,-6,37]);assert.equal(r.query.time,3.7);assert.equal(r.initial.time,0);assert.equal(r.states[r.states.length-1].time,10);assert.ok(r.states.some(s=>s.time===3.7));assert.ok(r.curves.flat().every(([t,x])=>t>=0&&t<=10&&Number.isFinite(x)));assert.ok(r.solver.acceptedSteps>0);assert.ok(r.solver.acceptedSteps+r.solver.rejectedSteps<40000);
});
test('all models reject invalid, fractional, missing and nonfinite discrete controls',()=>{
  for(const [fn,p,ranges]of [[tensor,base,[[1,4],[1,4],[1,4],[1,6],[1,6],[1,6],[-4,4],[-20,20],[-20,20],[-20,20],[-20,20],[-12,12]]],[euler,[1,1,2,10,5,10,0,10],[[1,5],[1,5],[1,5],[-20,20],[-20,20],[-20,20],[-12,12],[0,100]]]]as const){assert.throws(()=>fn([]));assert.throws(()=>fn(new Array(p.length)));for(let j=0;j<p.length;j++)for(const value of [NaN,Infinity,p[j]+.5,ranges[j][0]-1,ranges[j][1]+1]){const q:number[]=[...p];q[j]=value;assert.throws(()=>fn(q));}}
});
test('serialized model results are finite and fresh without mutable shared state',()=>{
  for(const [fn,p]of [[tensor,base],[euler,[1,1,2,10,5,10,0,10]]]as const){const r=fn([...p]),text=JSON.stringify(r);assert.doesNotThrow(()=>JSON.parse(text));if('points'in r)r.points[0].labPosition[0]=999;else r.query.rotation[0][0]=999;assert.equal(JSON.stringify(fn([...p])),text);}
});

for(const id of ['point-mass-inertia-tensor','torque-free-euler-body-lab'])for(const language of ['ko','en']as const)test(`rotation manuscript has rendered derivations, intuitive examples and six explained exercises: ${id}.${language}`,async()=>{
  const fs=await import('node:fs/promises'),{getLearningContent}=await import('../src/lib/learning-content');
  const source=await fs.readFile(`content/learning/physics-math-for-ai/${id}.${language}.md`,'utf8'),rendered=await getLearningContent(id);
  assert.ok((source.match(/^## /gm)||[]).length>=10);assert.ok(source.includes('https://ocw.mit.edu/'));
  const heading=language==='en'?'## Worked examples and exercises with solutions':'## 예제와 해설이 있는 연습문제';
  const start=source.indexOf(heading);assert.ok(start>=0,'The original exercise section must remain');
  const end=source.indexOf('\n## ',start+heading.length),exercises=source.slice(start,end<0?undefined:end);
  assert.deepEqual([...exercises.matchAll(/^(\d)\. /gm)].map(m=>Number(m[1])),[1,2,3,4,5,6]);
  if(language==='en'){assert.ok(source.split(/\s+/).length>1000);assert.ok(!/[가-힣]/.test(source));assert.equal((exercises.match(/\*\*Solution:/g)||[]).length,6);}else {assert.ok((source.match(/[가-힣]/g)||[]).length>1000);assert.equal((exercises.match(/\*\*해설:/g)||[]).length,6);}
  assert.ok(rendered[language].includes('katex-display'));assert.ok(!rendered[language].includes('katex-error'));assert.ok(rendered.outline[language].length>=10);assert.equal(new Set(rendered.outline[language].map(x=>x.id)).size,rendered.outline[language].length);
});
test('new worked tensor examples independently verify isotropic energy and the mixed xz energy cross term',()=>{
  const isotropic=tensor([1,1,1,2,2,2,0,10,10,0,0,0]);assert.equal(isotropic.kineticPivot.label,'4');assert.deepEqual(isotropic.angularMomentumPivot.map(x=>x.label),['4','4','0']);const shear=tensor([1,1,1,2,4,6,1,10,10,10,0,0]);assert.equal(shear.tensorCOM[0][2].label,'-3');assert.equal(shear.kineticPivot.label,'51/2');
});

test('orthographic rotation projection contracts3D norms and leaves view-direction components unobservable',async()=>{
  const {projectRotationVector:project}=await import('../src/lib/learning-rotation-inertia');close(Math.hypot(...project([1,1,1])),0);close(Math.hypot(...project([1,-1,0])),Math.sqrt(2));for(const c of tensors){const v=c.labAngularMomentum;assert.ok(Math.hypot(...project(v))<=Math.hypot(...v)+1e-10);}for(const v of [[],[1,2],[1,2,NaN],new Array(3)])assert.throws(()=>project(v));
});
test('registered rotation adapters have bounded finite projected geometry and JSON-safe physical readouts',async()=>{
  const {evaluateClassicalMechanics:evaluate}=await import('../src/lib/learning-classical-mechanics');
  for(const [id,cases]of [['point-mass-inertia-tensor',tensors],['torque-free-euler-body-lab',eulers]]as const)for(const c of cases){const r=evaluate(id,c.values),[xmin,xmax,ymin,ymax]=r.bounds!;assert.doesNotThrow(()=>JSON.stringify(r));assert.ok(Object.values(r.stats).every(v=>v===null||typeof v==='number'&&Number.isFinite(v)));for(const[x,y]of r.curves.flat()){assert.ok(Number.isFinite(x)&&Number.isFinite(y));assert.ok(x>=xmin&&x<=xmax&&y>=ymin&&y<=ymax);}}
});
test('six registered guided examples preserve the worked inertia and torque-free analytic initial quantities',async()=>{
  const {classicalMechanicsLabs:labs,learningGuides}=await import('../src/data/interactive-book');const a=labs['point-mass-inertia-tensor'].presets.map(p=>tensor(p.values)),b=labs['torque-free-euler-body-lab'].presets.map(p=>euler(p.values));assert.deepEqual(a.map(x=>x.kineticPivot.label),['4','23','26']);assert.deepEqual(a.map(x=>x.translationKinetic.label),['0','0','3']);assert.deepEqual(a.map(x=>x.spectrum.repeatedSpectrum),[true,false,true]);assert.deepEqual(b.map(x=>x.initialEnergy.label),['3','23/8','403/200']);assert.deepEqual(b.map(x=>x.initialAngularMomentumNormSquared.label),['12','61/4','1609/100']);for(const id of ['point-mass-inertia-tensor','torque-free-euler-body-lab']as const){assert.equal(learningGuides[id].steps.length,4);assert.equal(learningGuides[id].experiments.length,3);}
});
