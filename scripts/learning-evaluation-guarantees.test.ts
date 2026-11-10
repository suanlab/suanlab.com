import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {upperCP,selectionCoverage,stratifiedBounds} from '../src/lib/learning-evaluation-guarantees';
const near=(a:number,b:number,tol=1e-11)=>assert.ok(Math.abs(a-b)<=tol,`${a} != ${b}`);
test('420 coverage designs match independent rational binomial event calculations',()=>{
 const fixture=JSON.parse(fs.readFileSync('scripts/fixtures/evaluation-selection-independent.json','utf8'));assert.equal(fixture.rows.length,420);
 for(const [n,p,alpha,m,naive,adjusted,fresh,naiveFlag,adjustedFlag]of fixture.rows){const r=selectionCoverage(n,p,alpha,m,0);near(r.naiveSelectedCoverage,naive);near(r.adjustedSelectedCoverage,adjusted);near(r.freshHoldoutCoverage,fresh);assert.equal(r.naiveMeetsAdvertised,naiveFlag);assert.equal(r.adjustedMeetsAdvertised,adjustedFlag);assert.equal(r.adjustedMeetsAdvertised,true);assert.equal(r.freshMeetsAdvertised,true);}
});
test('60 numerical confidence endpoints match independently computed beta quantiles',()=>{
 const fixture=JSON.parse(fs.readFileSync('scripts/fixtures/evaluation-upper-independent.json','utf8'));assert.equal(fixture.rows.length,60);for(const [n,k,a,u]of fixture.rows)near(upperCP(n,k,a),u);
});
test('strict CDF equality is covered, rather than turned into a false failure by a rounded comparison',()=>{
 const r=selectionCoverage(1,90,10,1,0);assert.equal(r.naiveSelectedCoverage,1);assert.equal(r.naiveMeetsAdvertised,true);near(r.naiveUpper,.9);assert.equal(r.adjustedUpper,r.naiveUpper);
});
test('the selection guide quantifies winner optimism and its correction under the stated independent design',()=>{
 const r=selectionCoverage(40,10,5,20,0);near(r.naiveSelectedCoverage,.7424318673550774);near(r.freshHoldoutCoverage,.9852191170585655);near(r.naiveUpper,.07215752450551458);near(r.adjustedUpper,.13910834066826522);near(r.expectedWinnerErrorRate!,.023391829069476046);assert.equal(r.naiveMeetsAdvertised,false);assert.equal(r.adjustedMeetsAdvertised,true);assert.equal(r.adjustedSelectedCoverage,1);
});
test('winner count probabilities normalize without negative clipping and single-model selection is unchanged',()=>{
 for(const n of [0,1,40,100])for(const p of [0,1,10,50,99,100])for(const m of [1,2,20,50]){const r=selectionCoverage(n,p,5,m,100);near(r.pmf.reduce((a,b)=>a+b,0),1);near(r.minPMF.reduce((a,b)=>a+b,0),1);assert.ok(r.minPMF.every(v=>v>=0&&Number.isFinite(v)));if(m===1){r.pmf.forEach((v,i)=>near(v,r.minPMF[i]));near(r.naiveSelectedCoverage,r.freshHoldoutCoverage);}if(n>0)assert.ok(r.expectedWinnerErrorRate!<=r.p+1e-12);}
});
test('no observations provide a trivial upper bound and an unavailable empirical error rate',()=>{
 const r=selectionCoverage(0,10,5,50,40);assert.equal(r.k,0);assert.equal(r.queryClamped,true);assert.equal(r.naiveUpper,1);assert.equal(r.adjustedUpper,1);assert.equal(r.expectedWinnerErrorRate,null);assert.equal(r.naiveSelectedCoverage,1);assert.deepEqual(r.pmf,[1]);
 const both=stratifiedBounds(0,100,0,100,5,50,100,1);assert.deepEqual(both.upper,[1,1]);assert.equal(both.weightedPoint,null);assert.equal(both.pooledObservedError,null);assert.equal(both.knownMixtureUpper,1);
});
test('confidence endpoints are ordered by error count and widen under smaller allocated alpha',()=>{
 for(const n of [1,10,40,100]){let last=0;for(let k=0;k<=n;k++){const u=upperCP(n,k,.05),adjusted=upperCP(n,k,.001);assert.ok(u>=last-1e-12&&u<=1);assert.ok(adjusted>=u-1e-12);last=u;}}
});
test('same observable law in the tested stratum permits distinct deployment risks when another stratum is absent',()=>{
 const a=stratifiedBounds(40,0,0,0,5,50,30,0),b=stratifiedBounds(40,0,0,0,5,50,30,1);assert.deepEqual(a.k,b.k);assert.deepEqual(a.upper,b.upper);assert.equal(a.conditionalTruth[0],b.conditionalTruth[0]);assert.notEqual(a.conditionalTruth[1],b.conditionalTruth[1]);assert.equal(a.pooledObservedError,0);assert.equal(a.weightedPoint,null);near(a.targetRisk,.01);near(b.targetRisk,.205);near(a.knownMixtureUpper,.5440486514394012);near(a.ambiguousMixtureUpper,.8176194605757605);
});
test('192 stratified query/mix settings preserve valid numerical ranges, explicit query clamping and interval endpoint maximization',()=>{
 let cases=0;for(const n0 of [0,1,40,100])for(const n1 of [0,1,40,100])for(const q of [0,25,50,100])for(const tv of [0,20,100]){const r=stratifiedBounds(n0,20,n1,40,5,q,tv,1);assert.ok(r.upper.every(v=>v>=0&&v<=1));assert.deepEqual(r.k,[Math.min(20,n0),Math.min(40,n1)]);assert.deepEqual(r.countQueryClamped,[20>n0,40>n1]);assert.ok(r.ambiguousMixtureUpper>=r.knownMixtureUpper-1e-12&&r.ambiguousMixtureUpper<=1);for(let i=0;i<=20;i++){const w=r.weightRange[0]+i/20*(r.weightRange[1]-r.weightRange[0]);assert.ok(w*r.upper[0]+(1-w)*r.upper[1]<=r.ambiguousMixtureUpper+1e-12);}cases++;}assert.equal(cases,192);
});
test('deployment weighting handles an absent zero-weight stratum without division by zero',()=>{
 const a=stratifiedBounds(0,0,40,16,5,0,0,1),b=stratifiedBounds(40,0,0,0,5,100,0,1);near(a.weightedPoint!,.4);assert.equal(b.weightedPoint,0);near(a.knownMixtureUpper,a.upper[1]);near(b.knownMixtureUpper,b.upper[0]);
});
test('query changes do not change the repeated-experiment coverage calculation',()=>{
 const a=selectionCoverage(40,10,5,20,0),b=selectionCoverage(40,10,5,20,100);assert.equal(a.naiveSelectedCoverage,b.naiveSelectedCoverage);assert.deepEqual(a.pmf,b.pmf);assert.deepEqual(a.minPMF,b.minPMF);assert.notEqual(a.naiveUpper,b.naiveUpper);assert.equal(b.queryClamped,true);assert.equal(b.naiveUpper,1);assert.doesNotThrow(()=>JSON.stringify(b));
});
test('all supported inputs are validated before expensive probability calculations and arrays do not alias',()=>{
 const vector=[40,10,5,20,0];for(let i=0;i<vector.length;i++)for(const bad of [NaN,Infinity,-1,.5,1000]){const v=vector.slice();v[i]=bad;assert.throws(()=>Reflect.apply(selectionCoverage,null,v));}
 const params=[40,0,40,16,5,50,30,1];for(let i=0;i<params.length;i++)for(const bad of [NaN,Infinity,-1,.5,1000]){const v=params.slice();v[i]=bad;assert.throws(()=>Reflect.apply(stratifiedBounds,null,v));}
 for(const a of [0,1,NaN,Infinity,1e-8])assert.throws(()=>upperCP(40,0,a));assert.throws(()=>upperCP(40,41,.05));const a=selectionCoverage(40,10,5,20,0);a.pmf[0]=99;a.minPMF[0]=99;const b=selectionCoverage(40,10,5,20,0);assert.ok(b.pmf[0]<1);assert.ok(b.minPMF[0]<1);const c=stratifiedBounds(40,0,0,0,5,50,30,1);c.n[0]=99;c.upper[0]=99;assert.equal(stratifiedBounds(40,0,0,0,5,50,30,1).n[0],40);
});

test('stratified adapter preserves missing empirical values and restricts the robust segment to the weight interval',async()=>{
 const {evaluatePrivacy}=await import('../src/lib/learning-privacy'),r=evaluatePrivacy('stratified-evaluation-shift-bounds',[40,0,0,0,5,50,30,1]);assert.equal(r.stats['target weighted empirical error'],null);assert.equal(r.stats['pooled observed error'],0);assert.deepEqual(r.curves[2].map(p=>p[0]),[.2,.8]);assert.ok(r.curves[2].every(p=>p[1]===r.evaluationStratified!.ambiguousMixtureUpper));assert.equal(r.table!.rows.length,2);
});
test('changing the unobserved truth world leaves all stratified numerical bounds unchanged',async()=>{
 const {evaluatePrivacy}=await import('../src/lib/learning-privacy'),a=evaluatePrivacy('stratified-evaluation-shift-bounds',[40,0,0,0,5,50,30,0]),b=evaluatePrivacy('stratified-evaluation-shift-bounds',[40,0,0,0,5,50,30,1]);assert.deepEqual(a.curves[0],b.curves[0]);assert.deepEqual(a.curves[2],b.curves[2]);assert.notDeepEqual(a.curves[1],b.curves[1]);assert.equal(a.stats['known mixture numerical upper'],b.stats['known mixture numerical upper']);
});
test('selection adapter reports impossible queries and exact boundary coverage separately from numerical endpoints',async()=>{
 const {evaluatePrivacy}=await import('../src/lib/learning-privacy'),boundary=evaluatePrivacy('selection-aware-benchmark-coverage',[1,90,10,1,0]),impossible=evaluatePrivacy('selection-aware-benchmark-coverage',[40,0,5,20,1]);assert.equal(boundary.stats['query naive endpoint covers truth exact event'],1);assert.equal(impossible.stats['query count possible'],0);assert.equal(impossible.stats['single candidate query probability'],0);assert.equal(impossible.table!.rows.length,41);
});
test('both evaluation adapters reject missing controls and expose finite JSON without invented no-data rates',async()=>{
 const {evaluatePrivacy}=await import('../src/lib/learning-privacy');for(const[id,v]of [['stratified-evaluation-shift-bounds',[0,100,0,100,5,50,100,0]],['selection-aware-benchmark-coverage',[0,100,5,50,100]]] as const){const r=evaluatePrivacy(id,[...v]);assert.ok(Object.values(r.stats).every(v=>v===null||Number.isFinite(v)));assert.ok(r.curves.flat(2).every(Number.isFinite));assert.ok(!/NaN|Infinity/.test(JSON.stringify(r)));assert.throws(()=>evaluatePrivacy(id,[]));}const r=evaluatePrivacy('selection-aware-benchmark-coverage',[0,100,5,50,100]);assert.equal(r.stats['expected selected empirical error'],null);assert.equal(r.stats['effective query count'],0);
});
