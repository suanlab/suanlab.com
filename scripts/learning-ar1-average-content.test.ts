import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { getLearningContent } from '../src/lib/learning-content';
import { ar1FiniteAverageModel as model } from '../src/lib/learning-ar1-average';
const id='finite-correlated-average-ar1-uncertainty';
const source=(language:'ko'|'en')=>readFileSync(`content/learning/physics-math-for-ai/${id}.${language}.md`,'utf8');
const near=(a:number|null,b:number)=>{assert.notEqual(a,null);assert.ok(Math.abs(a!-b)<=2e-12*Math.max(1,Math.abs(b)),`${a} vs ${b}`);};
for(const language of ['ko','en'] as const)test(`${language}: finite average theory, stationary preparation and endpoint distinctions render accessibly`,async()=>{
 const raw=source(language),rendered=await getLearningContent(id),html=rendered[language];
 assert.equal(rendered.outline[language].length,17);assert.equal(new Set(rendered.outline[language].map(h=>h.id)).size,17);
 assert.equal((raw.match(/^### [1-6]\. /gm)??[]).length,6);assert.equal((raw.match(/\*\*(?:Explanation\.|해설\.)\*\*/g)??[]).length,6);
 assert.equal((html.match(/class="katex-display"/g)??[]).length,19);assert.equal((html.match(/aria-label="(?:Equation|수식) /g)??[]).length,19);
 assert.equal((html.match(/<table[^>]*tabindex="0"/g)??[]).length,3);assert.equal((html.match(/<pre[^>]*tabindex="0"/g)??[]).length,2);
 assert.doesNotMatch(raw,/^# |^---|%%/m);assert.doesNotMatch(html,/katex-error|\$\$/);
 if(language==='en'){assert.ok(raw.split(/\s+/).length>2700);assert.doesNotMatch(html,/[가-힣]/);}else assert.ok((raw.match(/[가-힣]/g)??[]).length>5000);
});
test('both manuscripts share equations and runnable example with explicit finite/asymptotic and preparation contracts',()=>{
 const equations=(s:string)=>[...s.matchAll(/\$\$([\s\S]*?)\$\$/g)].map(m=>m[1].replace(/\s+/g,''));assert.deepEqual(equations(source('ko')),equations(source('en')));
 const code=(s:string)=>[...s.matchAll(/```python([\s\S]*?)```/g)].map(m=>m[1]);assert.deepEqual(code(source('ko')),code(source('en')));
 for(const pattern of [/initial fluctuation and all innovations are independent/,/coefficient is specified, not estimated/,/Starting instead from Y1=0/,/fixed \|ρ\|<1/,/not uniformly bounded/,/not ordinary mixing-time estimates/,/ordinary infinite autocovariance series alternates without converging/,/integer square roots rounded outward/,/not reused as the reference/,/actual reading interactions need integration evidence/])assert.match(source('en'),pattern);
});
test('four-observation rational examples and variance-equivalent counts match the model',()=>{
 for(const [rho,variance,count] of [[0,1/4,4],[.5,33/64,64/33],[-.5,7/64,64/7]]){
  const r=model({sampleCount:4,correlationRho:rho});near(r.meanVariance.value,variance);near(r.varianceInflation.value,4*variance);near(r.varianceEquivalentSampleCount.value,count);near(r.standardError.value,Math.sqrt(variance));near(r.totalSumVariance.value,16*variance);
 }
});
test('short positive dependence violates a naive use of the large-sample variance',()=>{
 const r=model({sampleCount:2,correlationRho:.5});near(r.meanVariance.value,.75);near(r.asymptoticMeanVariance!.value,1.5);near(r.finiteToAsymptoticVarianceRatio!.value,.5);
});
test('stationary and deterministic-zero starts have different first and later variances',()=>{
 const rho=.5,innovation=1-rho*rho;
 assert.equal(innovation,.75);assert.equal(innovation*(1+rho*rho),.9375);
 const r=model({sampleCount:3,correlationRho:rho});assert.equal(r.autocovariances[0].covariance.value,1);
 assert.notEqual(0,r.autocovariances[0].covariance.value);assert.notEqual(innovation,1);
});
test('perfect cancellation of a linear mean does not cancel a squared-observable mean',()=>{
 const r=model({sampleCount:2,correlationRho:-1});assert.equal(r.meanVariance.resolution,'structural-zero');
 const squares=[];for(const initial of [1,2]){const pair=[initial,-initial];assert.equal((pair[0]+pair[1])/2,0);squares.push((pair[0]**2+pair[1]**2)/2);}
 assert.deepEqual(squares,[1,4]);assert.notEqual(squares[0],squares[1]);
});
test('digit-bearing AR1 IDs render while path traversal, separators and encoded paths are rejected',async()=>{
 for(const unsafe of ['../ar1','/ar1','ar1/../other','ar1\\other','ar1%2fother','ar1%2e%2e','ar1.ko','ar1\0','ar1--mean','-ar1','ar1-','1ar','AR1','ar1 mean'])await assert.rejects(getLearningContent(unsafe),/Invalid learning unit ID/);
 const rendered=await getLearningContent(id);assert.ok(rendered.en.includes('covariance'));assert.equal(rendered.outline.en.length,17);
});
