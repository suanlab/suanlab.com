import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { getLearningContent } from '../src/lib/learning-content';
import { landauSixthOrderModel as model } from '../src/lib/learning-landau-sixth-order';
const id='landau-sixth-order-coexistence-stability';
const source=(lang:'ko'|'en')=>readFileSync(`content/learning/physics-math-for-ai/${id}.${lang}.md`,'utf8');
for(const lang of ['ko','en'] as const)test(`${lang}: detailed sixth-order theory renders strict equations and worked explanations`,async()=>{
 const raw=source(lang),r=await getLearningContent(id),html=r[lang];assert.equal(r.outline[lang].length,16);assert.equal(new Set(r.outline[lang].map(x=>x.id)).size,16);
 assert.equal((raw.match(/^### [1-6]\. /gm)||[]).length,6);assert.equal((raw.match(/\*\*(?:Explanation\.|해설\.)\*\*/g)||[]).length,6);
 assert.equal((html.match(/class="katex-display"/g)||[]).length,21);assert.equal((html.match(/aria-label="(?:Equation|수식) /g)||[]).length,21);
 assert.equal((html.match(/<table[^>]*tabindex="0"/g)||[]).length,3);assert.doesNotMatch(html,/katex-error|\$\$/);assert.doesNotMatch(raw,/^# |^---/m);
 if(lang==='en'){assert.ok(raw.split(/\s+/).length>2500);assert.doesNotMatch(html,/[가-힣]/);}else assert.ok((raw.match(/[가-힣]/g)||[]).length>4500);
});

test('translations share exact equations and preserve domain, thermodynamic and integration limits',()=>{
 const eq=(s:string)=>[...s.matchAll(/\$\$([\s\S]*?)\$\$/g)].map(x=>x[1].replace(/\s+/g,''));assert.deepEqual(eq(source('ko')),eq(source('en')));
 for(const pattern of [/whole real line/,/not the external field b/,/quintuple zero/,/two different positive local minima/,/Sturm/,/not automatically temperature/,/free-energy density/,/no single-valued smooth branch/,/those require integration checks/])assert.match(source('en'),pattern);
});
const near=(x:number|null,y:number,t=2e-12)=>{assert.notEqual(x,null);assert.ok(Math.abs(x!-y)<t*Math.max(1,Math.abs(y)),`${x} vs ${y}`);};
const base={quadraticA:3/16,quarticB:-1,sexticC:1,fieldH:0};
test('coexistence exercise has three exact zero minima and four barriers of 1/96',()=>{
 const r=model(base);assert.deepEqual(r.globalMinimumIndices,[0,2,4]);assert.equal(r.barriers.length,4);r.barriers.forEach(x=>near(x.barrier.value,1/96));for(const i of r.globalMinimumIndices)assert.equal(r.roots[i].constrainedFreeEnergy.resolution,'structural-zero');near(r.roots[0].magnetization.value,-Math.sqrt(3)/2);
});
test('ordered fold and central flat maximum exercises use different local multiplicities',()=>{
 const f=model({...base,quadraticA:1/4});assert.equal(f.rootCount,3);assert.equal(f.roots[0].stationaryMultiplicity,2);assert.equal(f.roots[0].localKind,'stationary-inflection');assert.equal(f.roots[0].branchFieldDerivative,null);
 const z=model({...base,quadraticA:0}).roots.find(x=>x.magnetization.sign===0)!;assert.equal(z.stationaryMultiplicity,3);assert.equal(z.localKind,'maximum');assert.equal(z.branchFieldDerivative!.sign,-1);
});
test('tilted fold and tilted quartic-minimum examples satisfy exact derivative conditions',()=>{
 const fold=model({quadraticA:-5/16,quarticB:0,sexticC:1,fieldH:-1/8}),f=fold.roots.find(x=>x.stationaryMultiplicity===2)!;near(f.magnetization.value,.5);near(f.leadingStationaryDerivative.value,5/2);
 const minimum=model({quadraticA:15/16,quarticB:-2.5,sexticC:3,fieldH:.25}).roots[0];assert.equal(minimum.magnetization.value,.5);assert.equal(minimum.stationaryMultiplicity,3);assert.equal(minimum.leadingStationaryDerivative.value,30);assert.equal(minimum.constrainedFreeEnergy.value,-5/128);
});
test('tricritical field law and thermal preparation formulas remain separate checks',()=>{
 const r=model({quadraticA:0,quarticB:0,sexticC:1,fieldH:.00001});near(r.roots[0].magnetization.value,.1);near(r.roots[0].constrainedFreeEnergy.value,-5e-6/6);
 const alpha=2,f0=3,T0=10,b=-1,c=1,ystar=-3*b/(4*c),Tstar=T0+3*b*b/(16*alpha*c);const ds=f0*alpha*ystar/2;near(ds,2.25);near(Tstar*ds,22.7109375);
});
