import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { binarySCM, binaryConfounding, colliderSelection, observationPolicy, causalKinds, evaluateCausal } from '../src/lib/learning-causality';
import { causalLabs, learningGuides, learningUnits } from '../src/data/interactive-book';
import { getLearningContent } from '../src/lib/learning-content';
const close = (a: number, b: number) => assert.ok(Math.abs(a-b)<1e-11, `${a} != ${b}`);
const sum = (x: number[]) => x.reduce((a,b)=>a+b,0);

test('binary SCM joint laws satisfy independent closed-form chain, fork and XOR calculations',()=>{
 for(const pi of [0,.2,.5,1])for(const ex of [0,.1,.5,1])for(const ey of [0,.1,.5,1])for(const mode of [0,1,2]) {
  const r=binarySCM(pi,ex,ey,mode,-1),px=ex+(1-2*ex)*pi;
  close(sum(r.observational),1);assert.deepEqual(r.observational,r.manipulated);
  close(r.observational[2]+r.observational[3],px);
  close(r.observationalMean,ey+(1-2*ey)*(mode===0?px:mode===1?pi:ex));
  for(const x of [0,1])for(const y of [0,1]) {
   let expected=0;
   for(const z of [0,1]) {
    const pz=z?pi:1-pi,probX=x===z?1-ex:ex,source=mode===0?x:mode===1?z:Number(x!==z),probY=y===source?1-ey:ey;
    expected+=pz*probX*probY;
   }
   close(r.observational[2*x+y],expected);
  }
 }
});
test('surgical intervention retains exogenous atom weights, replaces only X and recomputes descendants',()=>{
 for(const mode of [0,1,2])for(const pi of [0,.3,1])for(const force of [0,1]) {
  const r=binarySCM(pi,.2,.1,mode,force),natural=binarySCM(pi,.2,.1,mode,-1);
  assert.deepEqual(r.atoms.map(a=>a.mass),natural.atoms.map(a=>a.mass));assert.ok(r.atoms.every(a=>a.x===force));
  close(sum(r.manipulated),1);close(r.manipulated[2*(1-force)]+r.manipulated[2*(1-force)+1],0);
  const signal=mode===0?force:mode===1?pi:force===0?pi:1-pi;close(r.manipulatedMean,.1+.8*signal);
 }
 const xor=binarySCM(.5,0,0,2,1);close(xor.observationalMean,0);close(xor.manipulatedMean,.5);
});
test('fork observation updates its common cause while forcing X does not',()=>{
 const r=binarySCM(.5,.1,.1,1,1);close(r.conditionalOne!,.82);close(r.manipulatedMean,.5);
 assert.equal(binarySCM(0,0,.1,1,1).conditionalOne,null);
});
test('backdoor adjustment uses independent observational cell ratios and equals model interventions under support',()=>{
 for(const pi of [0,.3,.5,1])for(const e0 of [.1,.5,.9])for(const e1 of [.1,.5,.9])for(const tau of [-.2,0,.2]) {
  const r=binaryConfounding(pi,e0,e1,tau);close(sum(r.cells.map(c=>c.mass)),1);close(sum(r.joint),1);close(r.effect,tau);close(r.adjustedDifference!,tau);
  for(const x of [0,1]) {
   let adjusted=0;
   for(const z of [0,1])if(r.population[z]>0){const cells=r.cells.filter(c=>c.z===z&&c.x===x);adjusted+=r.population[z]*sum(cells.filter(c=>c.y===1).map(c=>c.mass))/sum(cells.map(c=>c.mass));}
   close(adjusted,r.adjusted[x]!);close(r.interventional[x],.2+.5*pi+tau*x);
  }
 }
});
test('common-cause mixture causes a strict sign reversal and random assignment removes it',()=>{
 const r=binaryConfounding(.5,.1,.9,-.1);close(r.observed[0]!,.25);close(r.observed[1]!,.55);close(r.crudeDifference!,.3);close(r.effect,-.1);
 const randomized=binaryConfounding(.5,.5,.5,-.1);close(randomized.crudeDifference!,-.1);
});
test('missing treatment strata stay unavailable while zero-mass target strata need no support',()=>{
 const r=binaryConfounding(.5,0,1,-.1);assert.deepEqual(r.adjusted,[null,null]);close(r.effect,-.1);
 const zero=binaryConfounding(0,.5,0,-.1);close(zero.adjustedDifference!,-.1);
 const all=binaryConfounding(1,0,.5,-.1);close(all.adjustedDifference!,-.1);
 assert.equal(binaryConfounding(.5,0,0,-.1).observed[1],null);
});
test('collider conditioning agrees with a separately enumerated selection event',()=>{
 for(const rho of [0,.1,.5,1]) {
  const r=colliderSelection(rho),weights=[.25*(1-rho),.25,.25,.25],mass=1-rho/4;
  close(r.selectedMass,mass);r.joint.forEach((p,i)=>close(p,weights[i]/mass));close(sum(r.joint),1);
  close(r.observed[0],1/(2-rho));close(r.observed[1],.5);close(r.causalEffect,0);
 }
 close(colliderSelection(0).difference,0);close(colliderSelection(1).difference,-.5);
});
test('policy means equal independent integration over Z, policy X and Bernoulli Y',()=>{
 for(const pi of [0,.2,1])for(const r of [0,.3,1]) {
  const result=observationPolicy(pi,.1,.9,1,r);let expected=0;
  for(const z of [0,1])for(const x of [0,1])expected+=(z?pi:1-pi)*(x?r:1-r)*(.1+.6*z+.2*x);
  close(result.policyMean,expected);close(result.policyMean,.1+.6*pi+.2*r);close(result.doMeans[1]-result.doMeans[0],.2);
 }
});
test('observation and intervention differ through posterior population weights, not changed outcome equations',()=>{
 const r=observationPolicy(.5,.1,.9,1,1);close(r.posterior![1],.9);close(r.observed!,.84);close(r.intervention,.6);
 for(const x of [0,1]) { const randomized=observationPolicy(.3,.4,.4,x,.5);close(randomized.observed!,randomized.intervention); }
});
test('impossible observational queries remain null while specified interventions and policies remain defined',()=>{
 const r=observationPolicy(1,1,1,0,0);assert.equal(r.queryMass,0);assert.equal(r.posterior,null);assert.equal(r.observed,null);assert.equal(r.adjusted,null);close(r.intervention,.7);close(r.policyMean,.7);
 assert.ok(observationPolicy(0,.5,0,1,.5).supported);
 const missing=observationPolicy(.5,0,1,1,.5);assert.equal(missing.adjusted,null);assert.ok(missing.observed!==null);
});
for(const id of causalKinds)test(`causal controls, presets, invalid inputs and finite results: ${id}`,()=>{
 const spec=causalLabs[id],guide=learningGuides[id],unit=learningUnits.find(u=>u.id===id)!;
 assert.equal(guide.steps.length,4);assert.equal(guide.experiments.length,3);assert.equal(unit.questions.length,2);
 const initial=spec.controls.map(c=>c.initial),vectors=[initial,...spec.presets.map(p=>p.values)];
 for(const [i,c] of spec.controls.entries())for(const v of [c.min,c.max]){const p=initial.slice();p[i]=v;vectors.push(p);}
 for(const p of vectors){const r=evaluateCausal(id,p);assert.ok(r.curves.flat(2).every(Number.isFinite));assert.ok(!/[가-힣]/.test(r.statement!));for(const v of Object.values(r.stats).flat())assert.ok(v===null||Number.isFinite(v));const [lx,hx,ly,hy]=r.bounds!;assert.ok(hx>lx&&hy>ly);assert.ok(r.curves.flat().every(([x,y])=>x>=lx&&x<=hx&&y>=ly&&y<=hy));}
 assert.throws(()=>evaluateCausal(id,[]));for(let i=0;i<initial.length;i++){const p=initial.slice();p[i]=NaN;assert.throws(()=>evaluateCausal(id,p));}
 for(const [i,c] of spec.controls.entries())if(c.step===1){const p=initial.slice();p[i]=c.min+.5;assert.throws(()=>evaluateCausal(id,p));}
});
for(const id of causalKinds)for(const lang of ['ko','en'] as const)test(`causal manuscript, worked solution and KaTeX rendering: ${id}.${lang}`,async()=>{
 const raw=await fs.readFile(`content/learning/physics-math-for-ai/${id}.${lang}.md`,'utf8'),content=await getLearningContent(id);
 assert.ok((raw.match(/^## /gm)||[]).length>=6);assert.ok(raw.includes('$$'));assert.ok(raw.includes('https://'));assert.ok(lang==='en'?/exercise/i.test(raw)&&/solution|answer/i.test(raw):/문제/.test(raw)&&/풀이|해설/.test(raw));
 if(lang==='en'){assert.ok(raw.split(/\s+/).length>400);assert.ok(!/[가-힣]/.test(raw));}else assert.ok((raw.match(/[가-힣]/g)||[]).length>700);
 assert.ok(content[lang].includes('katex-display'));assert.ok(!content[lang].includes('katex-error'));
});
