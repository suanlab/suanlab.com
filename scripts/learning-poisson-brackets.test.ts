import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import refs from './fixtures/poisson-brackets-independent.json';
import {canonicalPoissonBracket as bracket,poissonObservableCanonicalFlow as observable,angularMomentumPoissonAlgebra as angular} from '../src/lib/learning-poisson-brackets';
import {getLearningContent} from '../src/lib/learning-content';
const close=(a:number,b:number,tol=5e-10)=>assert.ok(Math.abs(a-b)<=tol*Math.max(1,Math.abs(a),Math.abs(b)),`${a} vs ${b}`);
const vector=(a:number[],b:number[],tol=5e-10)=>{assert.equal(a.length,b.length);a.forEach((v,j)=>close(v,b[j],tol));};
const matrix=(a:number[][],b:number[][])=>{assert.equal(a.length,b.length);a.forEach((v,j)=>vector(v,b[j]));};
const rational=(x:string)=>{const[n,d='1']=x.split('/');return Number(n)/Number(d);};
const op=[1,0,2,2,2,0,-2,20],ap=[1,2,2,2,1,1,2,0];

test('canonical fundamental brackets preserve pair ordering and cross-pair zeros in1D and3D',()=>{
 assert.equal(bracket([1],[0],[0],[1]),1);assert.equal(bracket([0],[1],[1],[0]),-1);assert.equal(bracket([1],[0],[1],[0]),0);
 for(let i=0;i<3;i++)for(let j=0;j<3;j++){const e=(a:number)=>[0,1,2].map(k=>Number(k===a)),z=[0,0,0];assert.equal(bracket(e(i),z,z,e(j)),Number(i===j));assert.equal(bracket(z,e(i),e(j),z),-Number(i===j)||0);}
});
test('gradient operator rejects empty,mismatched,sparse,nonfinite and overflowing inputs',()=>{
 for(const args of [[[],[],[],[]],[[1],[0,0],[0],[1]],[new Array(1),[0],[0],[1]],[[NaN],[0],[0],[1]],[[1],[0],[0],[Infinity]],[[1e308],[0],[0],[1e308]],[[1,0,0,0],[0,0,0,0],[0,0,0,0],[1,0,0,0]]])assert.throws(()=>bracket(...args as [number[],number[],number[],number[]]));
});
test('240independent symbolic and60digit trajectory derivatives verify observable rates and coordinate brackets',()=>{
 assert.equal(refs.observables.length,240);
 for(const row of refs.observables){const r=observable(row.p),q=r.query;for(const key of ['position','momentum','observable','gradientQ','gradientP','poisson']as const)close(q[key],Number(row[key]));close(q.explicitTimeDerivative,Number(row.partialTime));close(q.totalTimeDerivative,Number(row.totalTime));close(q.chainTimeDerivative,Number(row.totalTime));close(q.transformedBracket,Number(row.canonicalBracket));close(q.badNaiveBracket,Number(row.badNaiveBracket));close(q.reversePoisson,-q.poisson);}
});
test('64exact Fraction free cases independently verify explicit clock cancellation and bad scaling',()=>{
 assert.equal(refs.exactFree.length,64);for(const row of refs.exactFree){const q=observable(row.p).query;for(const key of ['position','momentum','energy','observable','poisson','badNaiveBracket']as const)close(q[key],rational(row[key]),1e-12);close(q.explicitTimeDerivative,rational(row.explicitTime),1e-12);close(q.totalTimeDerivative,rational(row.totalTime),1e-12);}
});
test('180independent3Dsymbolic brackets and60digit actual trajectory derivatives verify algebra and torque',()=>{
 assert.equal(refs.angular.length,180);for(const row of refs.angular){const q=angular(row.p).query;vector(q.position,row.position.map(Number));vector(q.momentum,row.momentum.map(Number));vector(q.angularMomentum,row.angularMomentum.map(Number));matrix(q.bracketMatrix,row.bracketMatrix.map(r=>r.map(Number)));vector(q.torque,row.torque.map(Number));vector(q.hamiltonBrackets,row.directDerivative.map(Number));vector(q.directAngularMomentumRate,row.directDerivative.map(Number));}
});
test('an explicitly time-dependent free invariant has nonzero bracket canceled by its partial derivative',()=>{
 const r=observable(op),q=r.query;assert.equal(r.hasExplicitTimeDependence,true);assert.equal(r.globallyConservedObservable,true);assert.equal(r.selectedOrbitHasConstantObservable,true);assert.equal(q.observable,1);assert.equal(q.baseObservable,2);assert.equal(q.explicitTimeContribution,-1);assert.equal(q.poisson,1);assert.equal(q.explicitTimeDerivative,-1);assert.equal(q.totalTimeDerivative,0);for(const[,a]of r.curves[4])close(a,1);
});
test('the rational free scaling example retains physical rates only with the correct induced bracket',()=>{
 const q=observable(op).query;assert.equal(q.position,2);assert.equal(q.momentum,1);assert.equal(q.canonicalPosition,4);assert.equal(q.canonicalMomentum,.5);assert.equal(q.badMomentum,2);assert.equal(q.canonicalFundamentalBracket,1);assert.equal(q.badFundamentalBracket,4);assert.equal(q.transformedBracket,1);assert.equal(q.badNaiveBracket,.25);assert.equal(q.badTensorCorrectedBracket,1);assert.equal(q.physicalQRate,2);assert.equal(q.badNaiveQRate,.5);
});
test('zero instantaneous oscillator derivative does not mark a nonconstant observable conserved',()=>{
 const r=observable([1,2,2,0,1,0,0,10]);assert.equal(r.globallyConservedObservable,false);assert.equal(r.selectedOrbitHasConstantObservable,false);close(r.query.totalTimeDerivative,0);close(r.query.observable,-Math.PI/4);close(r.initial.observable,0);assert.ok(r.curves[4].some(([,a])=>Math.abs(a)>.5));
});
test('all four action-valued choices retain the correct base derivatives and free degeneracy',()=>{
 for(const f of [0,2])for(let code=0;code<4;code++){const r=observable([2,f,2,-2,3,code,0,13]),q=r.query,expected=[q.momentum*q.momentum/r.mass-r.stiffness*q.position*q.position,r.referenceFrequency*q.position*q.momentum,-r.stiffness*q.position*q.momentum/(r.mass*r.referenceFrequency),0][code];close(q.poisson,expected);close(q.explicitTimeDerivative,0);}
 const a=observable([2,0,2,-2,3,2,0,13]),b=observable([2,0,2,-2,3,3,0,13]);assert.deepEqual(a.curves[4],b.curves[4]);assert.equal(a.referenceFrequency,1);
});
test('global conservation and constant selected zero-energy orbits are distinct exact classifications',()=>{
 const rest=observable([1,0,2,0,2,1,1,40]);assert.equal(rest.hasExplicitTimeDependence,true);assert.equal(rest.globallyConservedObservable,false);assert.equal(rest.selectedOrbitHasConstantObservable,true);assert.equal(rest.query.observable,.5);close(rest.query.totalTimeDerivative,0);
 const zero=observable([1,2,0,0,2,0,1,40]);assert.equal(zero.globallyConservedObservable,false);assert.equal(zero.selectedOrbitHasConstantObservable,true);assert.equal(zero.query.observable,0);
 for(const f of [0,1,4])assert.equal(observable([1,f,2,2,2,3,0,17]).globallyConservedObservable,true);
});
test('canonical scale changes coordinates,not physical history,and rescues the bad bracket with its tensor',()=>{
 const baseline=observable(op);for(let a=1;a<=4;a++){const p=[...op];p[4]=a;const r=observable(p),q=r.query;assert.deepEqual(r.curves[0],baseline.curves[0]);assert.deepEqual(r.curves[1],baseline.curves[1]);assert.deepEqual(r.curves[4],baseline.curves[4]);close(q.transformedBracket,q.poisson);close(q.badTensorCorrectedBracket,q.poisson);close(q.badNaiveQRate*a*a,q.physicalQRate);close(q.badNaivePRate*a*a,q.physicalBadPRate);}
});
test('quadratic action algebra and its nontrivial Jacobi cancellation retain action units',()=>{
 const q=observable(op).query;vector(q.quadraticBasis,[2,.5,2]);close(q.quadraticFGBracket,2);vector(q.jacobiTerms,[-4,4,0]);close(q.jacobiResidual,0);
});
test('mass rescales momenta,actions and rates while fixed frequency keeps physical position unchanged',()=>{
 const a=observable([1,2,2,2,2,0,1,9]),b=observable([4,2,2,2,2,0,1,9]);assert.deepEqual(a.curves[0],b.curves[0]);close(b.stiffness,4*a.stiffness);for(const key of ['momentum','energy','observable','poisson','explicitTimeDerivative','totalTimeDerivative']as const)close(b.query[key],4*a.query[key]);
});
test('query time selects a point without modifying either complete observable or spatial history',()=>{
 const a=observable(op),p=[...op];p[7]=0;const b=observable(p);assert.deepEqual(a.curves,b.curves);assert.deepEqual(a.phaseCurves,b.phaseCurves);assert.notEqual(a.query.position,b.query.position);
 const aa=angular(ap),pp=[...ap];pp[7]=20;const bb=angular(pp);assert.deepEqual(aa.curves,bb.curves);assert.deepEqual(aa.spatialProjections,bb.spatialProjections);assert.notDeepEqual(aa.query.position,bb.query.position);
});
test('isotropic angular momenta are conserved even with a nonzero mutual bracket',()=>{
 const r=angular(ap);assert.equal(r.isotropic,true);assert.deepEqual(r.globallyConservedComponents,[true,true,true]);vector(r.initial.angularMomentum,[1,0,-1]);assert.equal(r.initial.bracketMatrix[0][1],-1);for(const q of [r.initial,r.query,r.final])vector(q.torque,[0,0,0]);for(let i=0;i<3;i++)for(const[,v]of r.curves[i])close(v,r.initial.angularMomentum[i]);
});
test('anisotropic zero initial angular momentum can have immediate nonzero torque at constant energy',()=>{
 const r=angular([1,0,1,2,1,3,0,0]);vector(r.query.angularMomentum,[0,0,0]);vector(r.query.torque,[-3*Math.PI**2/16,Math.PI**2/4,-(Math.PI**2)/16]);close(r.query.energy,5*Math.PI**2/32);assert.deepEqual(r.globallyConservedComponents,[false,false,false]);assert.deepEqual(r.selectedOrbitHasConstantComponents,[false,false,false]);assert.ok(r.curves[0].some(([,v])=>Math.abs(v)>.1));assert.ok(r.maximumSampledEnergyResidual<1e-10);
});
test('partial rotational symmetry conserves exactly the indicated component globally',()=>{
 const r=angular([1,4,1,1,1,1,0,0]);assert.equal(r.isotropic,false);assert.deepEqual(r.globallyConservedComponents,[true,false,false]);for(const[,v]of r.curves[0])close(v,1);assert.ok(r.curves[1].some(([,v])=>Math.abs(v)>.1));
});
test('a planar orbit preserves two zero selected components without erasing the full3Dbracket',()=>{
 const r=angular([1,1,2,0,0,0,2,0]),later=angular([1,1,2,0,0,0,2,2]);assert.deepEqual(r.globallyConservedComponents,[false,false,false]);assert.deepEqual(r.selectedOrbitHasConstantComponents,[true,true,false]);vector(r.query.angularMomentum,[0,0,1]);assert.equal(r.query.bracketMatrix[0][1],1);close(r.query.torque[2],0);assert.ok(later.query.torque[2]<0);for(const[,z]of r.curves[19])assert.equal(z,0);
});
test('the Cartesian origin remains regular and zero axes imply selected conservation without global symmetry',()=>{
 const r=angular([1,1,2,3,3,0,1,0]);vector(r.query.position,[0,0,0]);vector(r.query.angularMomentum,[0,0,0]);assert.deepEqual(r.globallyConservedComponents,[false,false,false]);assert.deepEqual(r.selectedOrbitHasConstantComponents,[true,true,true]);assert.ok(r.query.momentum[1]!==0);
});
test('rotation generators give position and momentum variations,not physical time rates',()=>{
 const r=angular([1,1,2,0,0,0,2,0]);vector(r.query.rotationPosition,[0,1,0]);vector(r.query.rotationMomentum,[-1,0,0]);vector(r.query.jacobiRotationTerms,[0,0,0]);
 for(let axis=0;axis<3;axis++){const z=angular([1,2,2,2,1,1,axis,0]);vector(z.query.rotationPosition,[[0,-1,1],[1,0,-1],[-1,1,0]][axis]);vector(z.query.rotationMomentum,[[0,-1,0],[1,0,-1],[0,1,0]][axis]);}const s=angular([1,2,2,2,1,1,2,0]);vector(s.query.rotationPosition,[-1,1,0]);vector(s.query.jacobiRotationTerms,[-1,1,0]);close(s.query.jacobiRotationResidual,0);
});
test('displayed rotation axis leaves physical dynamics unchanged and selects the corresponding bracket row',()=>{
 const r=angular(ap);for(let axis=0;axis<3;axis++){const p=[...ap];p[6]=axis;const a=angular(p);for(const j of [0,1,2,3,4,5,9,10,17,18,19,20,21,22])assert.deepEqual(a.curves[j],r.curves[j]);assert.deepEqual(a.spatialProjections,r.spatialProjections);for(let j=0;j<3;j++)close(a.curves[6+j][0][1],a.initial.bracketMatrix[axis][j]);}
});
test('orthographic projections retain true Cartesian coordinates and equal spatial units',()=>{
 const r=angular([2,1,2,3,2,2,1,20]);const q=r.query;vector(r.spatialProjections[0][100],[q.position[0],q.position[1]]);vector(r.spatialProjections[1][100],[q.position[1],q.position[2]]);vector(r.spatialProjections[2][100],[q.position[2],q.position[0]]);assert.equal(r.curves.length,23);
});
test('3Dmass scaling retains geometry and scales angular momenta,torque,bracket values and energy once',()=>{
 const a=angular([1,1,2,3,1,2,0,17]),b=angular([4,1,2,3,1,2,0,17]);assert.deepEqual(a.spatialProjections,b.spatialProjections);for(const key of ['momentum','angularMomentum','torque','hamiltonBrackets']as const)vector(b.query[key],a.query[key].map(v=>4*v));matrix(b.query.bracketMatrix,a.query.bracketMatrix.map(r=>r.map(v=>4*v)));close(b.query.energy,4*a.query.energy);
});
for(const[name,fn,p,ranges]of [['observable',observable,op,[[1,4],[0,4],[-4,4],[-4,4],[1,4],[0,3],[-2,2],[0,40]]],['angular',angular,ap,[[1,4],[0,4],[0,4],[0,4],[0,4],[0,4],[0,2],[0,40]]]]as const){
 test(`${name}: all control boundaries reject sparse,nonfinite,fractional and out-of-domain codes`,()=>{assert.throws(()=>fn([]));assert.throws(()=>fn([...p,0]));assert.throws(()=>fn(new Array(p.length)));for(let j=0;j<p.length;j++)for(const v of [NaN,Infinity,p[j]+.5,ranges[j][0]-1,ranges[j][1]+1]){const q:number[]=[...p];q[j]=v;assert.throws(()=>fn(q));}});
 test(`${name}: results are fresh,finite and JSON-safe without mutating controls or shared initial presets`,()=>{const input=[...p],before=[...p],a=fn(input),b=fn(input);assert.deepEqual(input,before);assert.deepEqual(a,b);const walk=(v:unknown):void=>{if(typeof v==='number')assert.ok(Number.isFinite(v));else if(v&&typeof v==='object')Object.values(v).forEach(walk);};walk(a);for(let j=0;j<p.length;j++)for(const bound of ranges[j]){const row:number[]=[...p];row[j]=bound;walk(fn(row));}a.query.time=1234;assert.deepEqual(fn(input),b);assert.doesNotThrow(()=>JSON.stringify(b));if(name==='angular'){const c=angular(ap);c.initialPosition[0]=999;assert.equal(angular(ap).initialPosition[0],1);}});
}
for(const id of ['poisson-observable-canonical-flow','angular-momentum-poisson-algebra'])for(const locale of ['ko','en']as const)test(`${id}.${locale}: substantive derivation,worked examples and six explained exercises render strict math`,async()=>{
 const source=await readFile(`content/learning/physics-math-for-ai/${id}.${locale}.md`,'utf8'),c=await getLearningContent(id);assert.ok(c.outline[locale].length>=10);assert.equal(c.outline.ko.length,c.outline.en.length);assert.equal(new Set(c.outline[locale].map(h=>h.id)).size,c.outline[locale].length);assert.ok(c[locale].includes('katex-display'));assert.ok(!c[locale].includes('katex-error'));assert.equal((source.match(/^\d\. /gm)||[]).length,6);assert.ok(source.includes('https://davidtong.org/teaching/classical-dynamics/dynhtml/S4'));if(locale==='en'){assert.ok(source.split(/\s+/).length>1400);assert.doesNotMatch(source,/[가-힣]/);assert.equal((source.match(/\*\*Solution:\*\*/g)||[]).length,6);}else{assert.ok((source.match(/[가-힣]/g)||[]).length>3000);assert.equal((source.match(/\*\*해설:\*\*/g)||[]).length,6);}
});

import {learningUnits,learningGuides,analyticalMechanicsLabs} from '../src/data/interactive-book';
import {curriculumDomains} from '../src/data/learning-curriculum';
import {evaluateAnalyticalMechanics,isAnalyticalMechanicsKind} from '../src/lib/learning-analytical-mechanics';
test('registered Poisson lessons retain ordered prerequisites and three valid bilingual guided experiments',()=>{
 const allIds=learningUnits.map(u=>u.id);
 for(const id of ['poisson-observable-canonical-flow','angular-momentum-poisson-algebra']as const){
  assert.ok(isAnalyticalMechanicsKind(id));const unit=learningUnits.find(u=>u.id===id)!;assert.ok(unit);assert.ok(unit.prerequisiteIds.every(p=>allIds.indexOf(p)>=0&&allIds.indexOf(p)<allIds.indexOf(id)));assert.ok(curriculumDomains.find(d=>d.id==='analytical-mechanics')!.unitIds.includes(id));
  const guide=learningGuides[id],lab=analyticalMechanicsLabs[id];assert.equal(guide.steps.length,4);assert.equal(guide.experiments.length,3);assert.equal(unit.questions.length,2);
  for(const[j,preset]of lab.presets.entries()){const r=evaluateAnalyticalMechanics(id,preset.values);assert.ok(r.poissonObservable||r.angularPoisson);assert.equal(guide.experiments[j].preset,j);assert.equal(preset.values.length,8);for(const[i,v]of preset.values.entries())assert.ok(v>=lab.controls[i].min&&v<=lab.controls[i].max);assert.ok(Object.values(r.stats).every(v=>typeof v==='number'&&Number.isFinite(v)));}
  assert.doesNotMatch(JSON.stringify([unit.title.en,unit.subtitle.en,guide.intuition.en,...guide.steps.map(s=>[s.title.en,s.explanation.en]),...guide.experiments.map(e=>[e.title.en,e.predict.en,e.action.en,e.explain.en]),...lab.controls.map(c=>c.label.en),lab.note.en,lab.axis.en]),/[가-힣]/);
 }
});
test('live observable adapter distinguishes nonzero bracket from conserved total observable',()=>{
 const r=evaluateAnalyticalMechanics('poisson-observable-canonical-flow',[1,0,2,2,2,0,-2,20]);assert.equal(r.stats['query observable Js'],1);assert.equal(r.stats['query Poisson rate J'],1);assert.equal(r.stats['query explicit time rate J'],-1);assert.equal(r.stats['query total observable rate J'],0);assert.equal(r.stats['1{globally conserved observable}'],1);assert.equal(r.stats['bad fundamental bracket'],4);assert.equal(r.stats['query naive Q rate m per s'],.5);
});
test('live angular adapter keeps global and selected component flags separate from mutual brackets',()=>{
 const r=evaluateAnalyticalMechanics('angular-momentum-poisson-algebra',[1,1,2,0,0,0,2,0]);assert.equal(r.stats['query bracket Lx Ly Js'],1);assert.equal(r.stats['1{globally conserved Lx}'],0);assert.equal(r.stats['1{constant Lx on selected orbit}'],1);assert.equal(r.stats['1{constant Lz on selected orbit}'],0);assert.ok(r.table.rows.some(row=>row[0]==='initial vy'&&row[1]===1));
});
