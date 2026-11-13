import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { getLearningContent } from '../src/lib/learning-content';
import { curieWeissVariationalModel as model } from '../src/lib/learning-curie-weiss-variational';
const id='curie-weiss-variational-stability-limits';
const source=(lang:'ko'|'en')=>readFileSync(`content/learning/physics-math-for-ai/${id}.${lang}.md`,'utf8');
for(const lang of ['ko','en'] as const)test(`${lang}: detailed variational theory renders strict equations and worked explanations`,async()=>{
 const raw=source(lang),r=await getLearningContent(id),html=r[lang];assert.equal(r.outline[lang].length,16);assert.equal(new Set(r.outline[lang].map(x=>x.id)).size,16);
 assert.equal((raw.match(/^### [1-6]\. /gm)||[]).length,6);assert.equal((raw.match(/\*\*(?:Explanation\.|해설\.)\*\*/g)||[]).length,6);
 assert.equal((html.match(/class="katex-display"/g)||[]).length,23);assert.equal((html.match(/aria-label="(?:Equation|수식) /g)||[]).length,23);
 assert.equal((html.match(/<table[^>]*tabindex="0"/g)||[]).length,3);assert.doesNotMatch(html,/katex-error|\$\$/);assert.doesNotMatch(raw,/^# |^---/m);
 if(lang==='en'){assert.ok(raw.split(/\s+/).length>2800);assert.doesNotMatch(html,/[가-힣]/);}else assert.ok((raw.match(/[가-힣]/g)||[]).length>5000);
});
test('translations share all equations and explicitly distinguish preparations and evidence limits',()=>{
 const eq=(s:string)=>[...s.matchAll(/\$\$([\s\S]*?)\$\$/g)].map(x=>x[1].replace(/\s+/g,''));assert.deepEqual(eq(source('ko')),eq(source('en')));
 for(const p of [/type-counting bound/,/not provide a uniform error bound/,/actual binary64 rational input/,/off-shell surrogate/,/positive infinity/,/within either pure phase/,/does not authorize replacing the mean/,/combined finite-versus-limit view is a proposed teaching interaction/,/requires its own derivation/])assert.match(source('en'),p);
});
const near=(x:number|null,y:number,t=2e-12)=>{assert.notEqual(x,null);assert.ok(Math.abs(x!-y)<t*Math.max(1,Math.abs(y)),`${x} vs ${y}`);};
test('noninteracting worked example agrees with three independent analytic formulas',()=>{
 const r=model({couplingK:0,fieldB:.2});near(r.roots[0].magnetization.value,Math.tanh(.2));near(r.thermodynamicLimitPressure!.value,Math.log(2*Math.cosh(.2)));near(r.roots[0].branchFieldDerivative.value,1/Math.cosh(.2)**2);
});
test('the stipulated ln 3 ordered root has positive curvature and lower free energy than zero',()=>{
 assert.ok(4/3-Math.log(3)>0);assert.ok(5*Math.log(3)/8-Math.log(2)<0);
 const r=model({couplingK:Math.log(3),fieldB:0});assert.deepEqual(r.globalMinimumIndices,[0,2]);near(r.roots[2].magnetization.value,.5);near(r.roots[2].freeEnergyDifferenceToZero.value,5*Math.log(3)/8-Math.log(2));
});
test('spinodal example has the stated rational turning point and logarithmic threshold',()=>{
 const r=model({couplingK:9/8,fieldB:0});assert.ok(r.spinodal);near(r.spinodal.turningMagnetization.value,1/3);near(r.spinodal.fieldMagnitude.value,3/8-Math.log(2)/2);assert.equal(r.rootCount,3);
});
test('critical and ordered response examples preserve their distinct local classifications',()=>{
 const c=model({couplingK:1,fieldB:0});assert.equal(c.roots[0].role,'critical-minimum');assert.equal(c.roots[0].branchFieldDerivative.resolution,'mathematical-infinity');
 const r=model({couplingK:2,fieldB:0});near(r.roots[2].magnetization.value,.9575040240772688);near(r.roots[2].curvature.value,10.02124723580547);near(r.roots[2].branchFieldDerivative.value,1/10.02124723580547);assert.equal(r.roots[1].branchFieldDerivative.sign,-1);
});
