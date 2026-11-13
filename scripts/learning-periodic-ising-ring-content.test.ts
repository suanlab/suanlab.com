import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { getLearningContent } from '../src/lib/learning-content';
import { periodicIsingRingModel as model } from '../src/lib/learning-periodic-ising-ring';
const id='periodic-ising-ring-correlation-length';
const source=(language:'ko'|'en')=>readFileSync(`content/learning/physics-math-for-ai/${id}.${language}.md`,'utf8');
const near=(a:number|null,b:number,tolerance=2e-12)=>{assert.notEqual(a,null);assert.ok(Math.abs(a!-b)<=tolerance*Math.max(1,Math.abs(b)),`${a} vs ${b}`);};
for(const language of ['ko','en'] as const)test(`${language}: detailed two-path ring theory, accessible equations, table and code render`,async()=>{
 const raw=source(language),rendered=await getLearningContent(id),html=rendered[language];
 assert.equal(rendered.outline[language].length,16);assert.equal(new Set(rendered.outline[language].map(h=>h.id)).size,16);
 assert.equal((raw.match(/^### [1-6]\. /gm)??[]).length,6);assert.equal((raw.match(/\*\*(?:Explanation\.|해설\.)\*\*/g)??[]).length,6);
 assert.equal((html.match(/class="katex-display"/g)??[]).length,19);assert.equal((html.match(/aria-label="(?:Equation|수식) /g)??[]).length,19);
 assert.equal((html.match(/<table[^>]*tabindex="0"/g)??[]).length,3);assert.equal((html.match(/<pre[^>]*tabindex="0"/g)??[]).length,2);
 assert.doesNotMatch(raw,/^# |^---|%%/m);assert.doesNotMatch(html,/katex-error|\$\$/);
 if(language==='en'){assert.ok(raw.split(/\s+/).length>2700);assert.doesNotMatch(html,/[가-힣]/);}else assert.ok((raw.match(/[가-힣]/g)??[]).length>5000);
});
test('translations share all equations and executable example; limit, boundary, source and reference contracts remain explicit',()=>{
 const equations=(s:string)=>[...s.matchAll(/\$\$([\s\S]*?)\$\$/g)].map(m=>m[1].replace(/\s+/g,''));assert.deepEqual(equations(source('ko')),equations(source('en')));
 const code=(s:string)=>[...s.matchAll(/```python([\s\S]*?)```/g)].map(m=>m[1]);assert.deepEqual(code(source('ko')),code(source('en')));
 for(const pattern of [/N=2 is excluded/,/two paths/,/all local fields/i,/not automatically a stochastic transition kernel/,/fixed r, then N large/,/zero-range limit/,/not a derivative with respect to K/,/thermodynamic-limit pressure/,/not claimed as transfer checks/,/does not by itself verify the implementation/])assert.match(source('en'),pattern);
});
test('four ideal hand examples agree with the actual rounded-log model within numerical tolerance',()=>{
 for(const [N,sign,c1,c2,chi] of [[3,1,2/3,2/3,7/3],[3,-1,-2/7,-2/7,3/7],[4,1,10/17,8/17,45/17],[4,-1,-10/17,8/17,5/17]]){
  const r=model({spinCount:N,couplingK:sign*Math.log(3)/2});near(r.correlations[1].correlation.value,c1);near(r.correlations[2].correlation.value,c2);near(r.commonFieldResponsePerSpin.value,chi);
  near(r.totalSpinVariance.value,N*chi);near(r.magnetizationMeanVariance.value,chi/N);
 }
});
test('common-field response is the derivative of the spin average in the explicitly extended finite family',()=>{
 for(const N of [3,4,7])for(const K of [-.7,.7]){
  const states=Array.from({length:2**N},(_,i)=>Array.from({length:N},(_,j)=>((i>>j)&1)?1:-1));
  const mean=(b:number)=>{let W=0,M=0;for(const s of states){const m=s.reduce((a,x)=>a+x,0),w=Math.exp(K*s.reduce((a,x,i)=>a+x*s[(i+1)%N],0)+b*m);W+=w;M+=w*m/N;}return M/W;};
  const step=2e-5,r=model({spinCount:N,couplingK:K});near((mean(step)-mean(-step))/(2*step),r.commonFieldResponsePerSpin.value!,1e-8);
 }
});
test('odd and even cold-limit explanations follow exact named ground-state counts and total spins',()=>{
 for(const N of [3,4,5,6,7,8]){
  const states=Array.from({length:2**N},(_,i)=>Array.from({length:N},(_,j)=>((i>>j)&1)?1:-1));
  const bond=(s:number[])=>s.reduce((a,x,i)=>a+x*s[(i+1)%N],0),minimum=Math.min(...states.map(bond)),ground=states.filter(s=>bond(s)===minimum);
  assert.equal(ground.length,N%2?2*N:2);for(const s of ground)assert.equal(Math.abs(s.reduce((a,x)=>a+x,0)),N%2?1:0);
  const chi=ground.reduce((a,s)=>a+s.reduce((a,x)=>a+x,0)**2,0)/ground.length/N;assert.equal(chi,N%2?1/N:0);
 }
});
test('a large negative-K envelope and a small uniform response describe different observables',()=>{
 const r=model({spinCount:128,couplingK:-8});assert.ok(r.infiniteChainCorrelationLength.value!>1e6);assert.ok(r.infiniteChainCommonFieldResponse.value!<1e-6);
 const t=r.infiniteChainOneStepCorrelation.value!;assert.ok(t<0);near((1+t)/(1-t),r.infiniteChainCommonFieldResponse.value!);
});
