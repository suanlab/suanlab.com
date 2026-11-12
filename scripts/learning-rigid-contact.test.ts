import test from 'node:test';
import assert from 'node:assert/strict';
import refs from './fixtures/rigid-contact-independent.json';
import {frictionlessRodImpact as rod,drivenWheelContact as wheel} from '../src/lib/learning-rigid-contact';
import {evaluateClassicalMechanics} from '../src/lib/learning-classical-mechanics';
import {classicalMechanicsLabs} from '../src/data/interactive-book';
const close=(a:number,b:number,tol=5e-9)=>assert.ok(Math.abs(a-b)<=tol*Math.max(1,Math.abs(a),Math.abs(b)),`${a} versus ${b}`);
const wheelCache=new Map<string,ReturnType<typeof wheel>>();
const wheelCase=(p:number[])=>{const key=JSON.stringify(p);let r=wheelCache.get(key);if(!r){r=wheel(p);wheelCache.set(key,r);}return r;};

test('registered impact examples expose the independently worked impulse, loss and endpoint restitution',()=>{
  const results=classicalMechanicsLabs['frictionless-rod-contact-impulse'].presets.map(p=>evaluateClassicalMechanics('frictionless-rod-contact-impulse',p.values));
  assert.deepEqual(results.map(r=>r.rodImpact!.normalImpulse.label),['3/10','4/5','2/5']);
  assert.deepEqual(results.map(r=>r.rodImpact!.energyLoss.label),['3/80','0','1/5']);
  for(const r of results){assert.equal(r.stats['exact contact angular residual'],0);assert.equal(r.stats['active restitution residual'],0);const[x0,x1,y0,y1]=r.bounds!;assert.ok(r.curves.flat().every(([x,y])=>x>=x0&&x<=x1&&y>=y0&&y<=y1));}
});
test('registered wheel examples preserve opposite contact works and relative-slip heat',()=>{
  const results=classicalMechanicsLabs['driven-wheel-rolling-sliding'].presets.map(p=>evaluateClassicalMechanics('driven-wheel-rolling-sliding',p.values));
  assert.deepEqual(results.map(r=>r.wheelContact!.query.kinetic.label),['3','1/3','3']);
  assert.deepEqual(results.map(r=>r.wheelContact!.query.heat.label),['0','1/6','0']);
  assert.deepEqual(results.filter((_,i)=>i!==1).map(r=>r.wheelContact!.query.frictionTranslationWork.label),['-1','2']);
  for(const r of results){assert.equal(r.stats['exact energy budget residual'],0);assert.equal(r.stats['exact contact heat residual'],0);const[x0,x1,y0,y1]=r.bounds!;assert.ok(r.curves.flat().every(([x,y])=>x>=x0&&x<=x1&&y>=y0&&y<=y1));}
});

test('900 independent rational linear-system and uniform-rod quadrature cases verify impact states and losses',()=>{
  assert.equal(refs.rod.length,900);
  for(const c of refs.rod){const r=rod(c.values);assert.equal(r.active,c.active);assert.equal(r.normalImpulse.label,c.impulse);assert.equal(r.energyLoss.label,c.loss);
    for(const key of ['before','after']as const){for(const k of ['velocity','contactVelocity','momentum']as const)assert.deepEqual(r[key][k].map(x=>x.label),c[key][k]);for(const k of ['omega','spinAngularMomentum','orbitalAngularMomentumAboutContact','angularMomentumAboutContact','kineticTranslation','kineticRotation','kinetic']as const)assert.equal(r[key][k].label,c[key][k]);}
    for(const k of ['energyLossResidual','normalMomentumResidual','spinImpulseResidual','contactAngularResidual']as const)assert.equal(r[k].label,'0');assert.equal(r.restitutionResidual?.label??null,r.active?'0':null);assert.ok(r.normalImpulse.value>=0);assert.ok(r.energyLoss.value>=0);assert.ok(r.after.contactVelocity[1].value>=0);
  }
});
test('a rising COM can have a closing endpoint whose impact changes both translation and spin',()=>{
  const r=rod([1,2,2,0,5,-10,5]);assert.equal(r.active,true);assert.equal(r.before.velocity[1].label,'1/2');assert.equal(r.before.contactVelocity[1].label,'-1/2');assert.equal(r.normalImpulse.label,'3/10');assert.equal(r.after.velocity[1].label,'4/5');assert.equal(r.after.omega.label,'-11/20');assert.equal(r.after.contactVelocity[1].label,'1/4');assert.equal(r.energyLoss.label,'3/80');
});
test('elastic rod impact preserves energy while contact opens and the COM can still move downward',()=>{
  const r=rod([1,2,2,10,-10,0,10]);assert.equal(r.inertia.label,'2/3');assert.equal(r.normalImpulse.label,'4/5');assert.equal(r.after.velocity[1].label,'-1/5');assert.equal(r.after.omega.label,'6/5');assert.equal(r.after.contactVelocity[1].label,'1');assert.equal(r.before.kinetic.label,'1');assert.equal(r.after.kinetic.label,'1');assert.equal(r.energyLoss.label,'0');
});
test('plastic normal contact speed0 does not imply a motionless rod or a solved persistent contact',()=>{
  const r=rod([1,2,2,10,-10,0,0]);assert.equal(r.after.contactVelocity[1].label,'0');assert.equal(r.after.velocity[0].label,'1');assert.equal(r.after.velocity[1].label,'-3/5');assert.equal(r.after.omega.label,'3/5');assert.equal(r.after.kinetic.label,'4/5');assert.equal(r.energyLoss.label,'1/5');assert.equal(r.instantaneousOnly,true);assert.equal(r.persistentContactSolved,false);
});
test('nonclosing and exactly stationary normal contacts apply no attractive or artificial impulse',()=>{
  for(const p of [[1,2,2,0,5,10,5],[1,2,2,0,-10,10,5],[1,0,1,0,0,0,0]]){const r=rod(p);assert.equal(r.active,false);assert.equal(r.normalImpulse.label,'0');assert.equal(r.restitutionResidual,null);assert.deepEqual(r.after,r.before);assert.equal(r.energyLoss.label,'0');}
});
test('a vertical rod has positive inertia but the normal impulse has no spin torque',()=>{
  const r=rod([4,0,6,-20,-20,20,10]);assert.equal(r.inertia.label,'12');assert.equal(r.normalImpulse.label,'16');assert.equal(r.after.velocity[1].label,'2');assert.equal(r.after.omega.label,r.before.omega.label);assert.ok(r.geometry.upperEndpoint[1]>r.geometry.contact[1]);assert.equal(r.geometry.contact[1],0);
});
test('frictionless impulse leaves COM tangential momentum unchanged while contact tangential speed can change',()=>{
  const r=rod([1,2,2,10,-10,0,10]);assert.equal(r.tangentialImpulse.label,'0');assert.equal(r.after.momentum[0].label,r.before.momentum[0].label);assert.equal(r.before.contactVelocity[0].label,'1');assert.equal(r.after.contactVelocity[0].label,'11/5');assert.equal(r.before.angularMomentumAboutContact.label,r.after.angularMomentumAboutContact.label);
});
test('1000 independent Fraction rolling constraints, branch integrations and contact-work sums verify wheel states',()=>{
  assert.equal(refs.wheel.length,1000);
  for(const c of refs.wheel){const r=wheelCase(c.values);for(const key of Object.keys(c.query)as(keyof typeof c.query)[])assert.equal(r.query[key].label,c.query[key],key);assert.equal(r.requiredRollingFriction.label,c.required);assert.equal(r.zeroSlipEventTime?.label??null,c.event);assert.equal(r.query.rolling,c.rolling);assert.equal(r.eventReached,c.eventReached);assert.equal(r.rollingFeasible,c.feasible);for(const key of ['energyResidual','contactHeatResidual','linearImpulseResidual','angularImpulseResidual']as const)assert.equal(r.query[key].label,'0');assert.ok(r.query.heat.value>=0);assert.ok(r.query.contactPower.value<=0);}
});
test('180 independent event-detecting Cartesian branch ODEs verify actual translation, spin, work and relative slip',()=>{
  assert.equal(refs.ode.length,180);
  for(const c of refs.ode){const r=wheelCase(c.values),q=r.query;for(const[key,value]of [['position',c.position],['clockwiseAngle',c.angle],['velocity',c.velocity],['clockwiseOmega',c.omega],['externalWork',c.externalWork],['frictionTranslationWork',c.frictionTranslationWork],['frictionRotationWork',c.frictionRotationWork],['slipDistance',c.slipDistance],['COMdistance',c.COMdistance]]as const)close(q[key].value,value);if(c.event!==null)close(r.zeroSlipEventTime!.value,c.event);}
});
test('force-driven pure rolling from rest uses backward static friction with zero total contact work',()=>{
  const r=wheel([1,2,2,2,5,30,0,0,0,10]),q=r.query;assert.equal(r.requiredRollingFriction.label,'-1');assert.equal(q.rolling,true);assert.equal(q.velocity.label,'2');assert.equal(q.clockwiseOmega.label,'2');assert.equal(q.position.label,'1');assert.equal(q.clockwiseAngle.label,'1');assert.equal(q.kineticTranslation.label,'2');assert.equal(q.kineticRotation.label,'1');assert.equal(q.heat.label,'0');assert.equal(q.externalWork.label,'3');assert.equal(q.frictionTranslationWork.label,'-1');assert.equal(q.frictionRotationWork.label,'1');assert.equal(q.contactWork.label,'0');
});
test('a freely sliding disk reaches rolling at1/6s and dissipates relative slip rather than COM travel',()=>{
  const r=wheel([1,2,2,2,5,0,0,10,0,10]),q=r.query;assert.equal(r.zeroSlipEventTime!.label,'1/6');assert.equal(r.eventVelocity!.label,'2/3');assert.equal(r.eventClockwiseOmega!.label,'2/3');assert.equal(r.eventPosition!.label,'5/36');assert.equal(r.eventClockwiseAngle!.label,'1/18');assert.equal(q.position.label,'25/36');assert.equal(q.clockwiseAngle.label,'11/18');assert.equal(q.slipDistance.label,'1/12');assert.equal(q.COMdistance.label,'25/36');assert.equal(q.heat.label,'1/6');assert.equal(q.kinetic.label,'1/3');assert.equal(q.contactWork.label,'-1/6');assert.equal(q.rolling,true);
});
test('torque-driven static rolling can have forward friction and opposite translation/spin contact works',()=>{
  const r=wheel([1,2,2,2,5,0,30,0,0,10]),q=r.query;assert.equal(q.friction.label,'2');assert.equal(q.velocity.label,'2');assert.equal(q.clockwiseOmega.label,'2');assert.equal(q.frictionTranslationWork.label,'2');assert.equal(q.frictionRotationWork.label,'-2');assert.equal(q.contactWork.label,'0');assert.equal(q.kinetic.label,'3');assert.equal(q.externalWork.label,'3');
});
test('insufficient static friction releases rotational slip with nonnegative heat and correct torque work',()=>{
  const r=wheel([1,2,1,0,5,0,30,0,0,10]),q=r.query;assert.equal(r.rollingFeasible,false);assert.equal(q.rolling,false);assert.equal(q.friction.label,'1');assert.equal(q.velocity.label,'1');assert.equal(q.clockwiseOmega.label,'4');assert.equal(q.slipVelocity.label,'-3');assert.equal(q.position.label,'1/2');assert.equal(q.clockwiseAngle.label,'2');assert.equal(q.kinetic.label,'9/2');assert.equal(q.heat.label,'3/2');assert.equal(q.externalWork.label,'6');
});
test('static threshold equality selects rolling but a drive excess selects incipient slip',()=>{
  const a=wheel([1,2,1,0,5,30,0,0,0,0]),b=wheel([1,2,1,0,5,31,0,0,0,0]);assert.equal(a.requiredRollingFriction.label,'-1');assert.equal(a.query.rolling,true);assert.equal(a.query.friction.label,'-1');assert.equal(b.query.rolling,false);assert.equal(b.query.incipientSliding,true);assert.equal(b.query.slipVelocity.label,'0');assert.equal(b.query.friction.label,'-1');assert.ok(b.query.slipAcceleration.value>0);
});
test('relative-slip reversal requires the two traveled branches while COM can reverse at another instant',()=>{
  const r=wheel([1,2,2,0,5,100,0,-10,0,10]),q=r.query;assert.equal(r.zeroSlipEventTime!.label,'1/16');assert.equal(r.rollsAfterEvent,false);assert.equal(q.slipDistance.label,'229/128');assert.equal(q.signedSlipDisplacement.label,'221/128');assert.equal(q.heat.label,'229/64');assert.equal(q.position.label,'415/128');assert.equal(q.clockwiseAngle.label,'97/64');assert.equal(q.velocity.label,'29/4');assert.equal(q.clockwiseOmega.label,'7/2');assert.ok(q.COMdistance.value>Math.abs(q.position.value));assert.ok(q.slipDistance.value>Math.abs(q.signedSlipDisplacement.value));
});
test('non-grid rolling events are explicitly sampled with force jumps and continuous velocity/spin',()=>{
  const r=wheel([1,2,2,2,5,0,0,10,0,10]),t=1/6;assert.deepEqual(r.curves[4].filter(([time])=>time===t).map(([,f])=>f),[-2,0]);close(r.curves[1].find(([time])=>time===t)![1],2/3,1e-14);close(r.curves[2].find(([time])=>time===t)![1],2/3,1e-14);assert.equal(r.curves[3].find(([time])=>time===t)![1],0);const at=wheel([1,2,2,2,5,0,0,12,0,2]);assert.equal(at.zeroSlipEventTime!.label,'1/5');assert.equal(at.query.rolling,true);assert.equal(at.query.friction.label,'0');assert.equal(at.query.velocity.label,'4/5');assert.equal(at.query.clockwiseOmega.label,'4/5');
});
test('a rolling event beyond the visible horizon remains in the table without an out-of-range plot vertex',()=>{
  const r=wheel([4,2,1,0,10,79,0,20,0,100]);assert.equal(r.zeroSlipEventTime!.label,'80');assert.equal(r.eventReached,false);assert.equal(r.query.rolling,false);assert.equal(r.query.slipVelocity.label,'7/4');assert.equal(r.query.heat.label,'75');assert.ok(r.curves.flat().every(([t])=>t>=0&&t<=10));
});
test('zero friction supports independent translation/spin without invented heat or forced no-slip constraints',()=>{
  const r=wheel([1,2,0,0,5,0,0,10,0,100]);assert.equal(r.query.velocity.label,'1');assert.equal(r.query.clockwiseOmega.label,'0');assert.equal(r.query.slipVelocity.label,'1');assert.equal(r.query.rolling,false);assert.equal(r.query.heat.label,'0');assert.equal(r.query.contactWork.label,'0');assert.equal(r.zeroSlipEventTime,null);
});
test('all sampled heat curves are nondecreasing and close the force-plus-torque energy budget',()=>{
  for(const c of refs.wheel.filter((_,j)=>j%37===0)){const r=wheelCase(c.values);let previous=0;for(let j=0;j<r.curves[7].length;j++){const heat=r.curves[8][j][1];assert.ok(heat>=previous-1e-10);previous=heat;close(r.curves[7][j][1]+heat,r.curves[9][j][1],1e-11);close(r.curves[1][j][1]-r.curves[2][j][1],r.curves[3][j][1],1e-11);}}
});
test('material-mark geometry follows clockwise angle while work retains the unwrapped rotation count',()=>{
  const r=wheel([1,1,0,0,1,0,100,0,0,100]),q=r.query,g=r.queryGeometry;assert.ok(q.clockwiseAngle.value>1000);assert.equal(q.torqueWork.label,'200000');close(Math.hypot(g.materialMark[0]-g.center[0],g.materialMark[1]-g.center[1]),r.radius.value,1e-12);assert.equal(g.contact[1],0);assert.equal(r.normalImpactSolved,false);assert.equal(r.clockwisePositive,true);
});
test('simultaneous sign reversal mirrors signed wheel motion while preserving energy, heat and traveled distances',()=>{
  const p=[1,2,2,0,5,100,0,-10,0,10],a=wheel(p);for(const j of [5,6,7,8])p[j]=-p[j];const b=wheel(p);for(const k of ['position','clockwiseAngle','velocity','clockwiseOmega','slipVelocity','friction']as const)close(a.query[k].value,-b.query[k].value,1e-13);for(const k of ['kinetic','heat','externalWork','COMdistance','slipDistance']as const)assert.deepEqual(a.query[k],b.query[k]);
});
test('models reject nonphysical lengths, inertia ratios, restitution, fractional, sparse and nonfinite controls',()=>{
  for(const [fn,p,ranges]of [[rod,[1,2,2,10,-10,0,10],[[1,4],[0,6],[1,6],[-20,20],[-20,20],[-20,20],[0,10]]],[wheel,[1,2,2,2,5,0,0,10,0,10],[[1,4],[1,6],[0,10],[0,10],[1,10],[-100,100],[-100,100],[-20,20],[-20,20],[0,100]]]]as const){assert.throws(()=>fn([]));assert.throws(()=>fn(new Array(p.length)));for(let j=0;j<p.length;j++)for(const value of [NaN,Infinity,p[j]+.5,ranges[j][0]-1,ranges[j][1]+1]){const q:number[]=[...p];q[j]=value;assert.throws(()=>fn(q));}}
});
test('results are fresh, finite and JSON-safe without mutable exact arithmetic leaking to callers',()=>{
  for(const [fn,p]of [[rod,[1,2,2,10,-10,0,10]],[wheel,[1,2,2,2,5,0,0,10,0,10]]]as const){const r=fn([...p]),serialized=JSON.stringify(r);assert.doesNotThrow(()=>JSON.parse(serialized));if('before'in r)r.geometry.center[0]=999;else r.curves[0][0][0]=999;assert.equal(JSON.stringify(fn([...p])),serialized);}
});

for(const id of ['frictionless-rod-contact-impulse','driven-wheel-rolling-sliding'])for(const language of ['ko','en']as const)test(`rigid-contact manuscript renders derivations, intuition and six explained exercises: ${id}.${language}`,async()=>{
  const fs=await import('node:fs/promises'),{getLearningContent}=await import('../src/lib/learning-content');
  const source=await fs.readFile(`content/learning/physics-math-for-ai/${id}.${language}.md`,'utf8'),content=await getLearningContent(id);
  assert.ok((source.match(/^## /gm)||[]).length>=10);assert.ok(source.includes('https://'));
  const heading=language==='en'?'## Worked examples and exercises with solutions':'## 예제와 해설이 있는 연습문제';
  const start=source.indexOf(heading);assert.ok(start>=0,'The original exercise section must remain');
  const end=source.indexOf('\n## ',start+heading.length),exercises=source.slice(start,end<0?undefined:end);
  assert.deepEqual([...exercises.matchAll(/^(\d)\. /gm)].map(m=>Number(m[1])),[1,2,3,4,5,6]);
  if(language==='en'){assert.ok(source.split(/\s+/).length>1000);assert.ok(!/[가-힣]/.test(source));assert.equal((exercises.match(/\*\*Solution:/g)||[]).length,6);}else {assert.ok((source.match(/[가-힣]/g)||[]).length>1000);assert.equal((exercises.match(/\*\*해설:/g)||[]).length,6);}
  assert.ok(content[language].includes('katex-display'));assert.ok(!content[language].includes('katex-error'));assert.ok(content.outline[language].length>=10);assert.equal(new Set(content.outline[language].map(x=>x.id)).size,content.outline[language].length);
});
