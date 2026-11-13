import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {getLearningContent} from '../src/lib/learning-content';
import {learningUnits,learningGuides} from '../src/data/interactive-book';
import {reversibleTwoStateMixingModel as model} from '../src/lib/learning-two-state-mixing';
const id='reversible-two-state-periodicity-mixing',base={forwardNumerator:1,backwardNumerator:2,transitionDenominator:4,initialOneNumerator:0,initialDenominator:1,stepCount:3};
const raw=(lang:'ko'|'en')=>readFileSync(`content/learning/physics-math-for-ai/${id}.${lang}.md`,'utf8');
for(const lang of ['ko','en'] as const)test(`${lang}: detailed two-state explanation, accessible equations and six answered exercises render`,async()=>{
 const s=raw(lang),r=await getLearningContent(id),html=r[lang];assert.equal(r.outline[lang].length,18);assert.equal(new Set(r.outline[lang].map(h=>h.id)).size,18);assert.equal((s.match(/^### [1-6]\. /gm)??[]).length,6);assert.equal((s.match(/\*\*(?:Explanation\.|해설\.)\*\*/g)??[]).length,6);assert.equal((html.match(/class="katex-display"/g)??[]).length,24);assert.equal((html.match(/aria-label="(?:Equation|수식) /g)??[]).length,24);assert.equal((html.match(/<table[^>]*tabindex="0"/g)??[]).length,5);assert.doesNotMatch(html,/katex-error|\$\$|@@E\d/);assert.doesNotMatch(s,/^# |^---/m);if(lang==='en'){assert.ok(s.split(/\s+/).length>2500);assert.doesNotMatch(html,/[가-힣]/);}else assert.ok((s.match(/[가-힣]/g)??[]).length>5000);
});
test('translations share equations and keep identity, support, worst-initial mixing and averaging meanings intact',()=>{
 const equations=(s:string)=>[...s.matchAll(/\$\$([\s\S]*?)\$\$/g)].map(m=>m[1].replace(/\s+/g,''));assert.deepEqual(equations(raw('ko')),equations(raw('en')));
 for(const term of [/stationary family/,/outside the observed stationary support/,/not reached within the recorded range/,/absolute gap/,/expected empirical occupancy/,/does not give the variance/,/not a unit-variance Gaussian/]){
  assert.match(raw('en'),term);
 }
});
test('the worked asymmetric probabilities, selected distance and worst-initial distance agree with the model',()=>{
 const r=model(base);assert.deepEqual(r.history.map(row=>row.probabilities[1].value),[0,1/4,5/16,21/64]);assert.equal(r.current.totalVariationFromStationary!.value,1/192);assert.equal(r.current.worstCaseTotalVariation!.value,1/96);assert.equal(r.stationaryIndicatorVariance!.value,2/9);assert.equal(r.stationaryLagOneCovariance!.value,1/18);assert.equal(r.mixingAtQuarter.firstStep,1);
});
test('the exercises retain distinct periodic, absorbing, identity and underflow conclusions',()=>{
 const flip=model({...base,forwardNumerator:1,backwardNumerator:1,transitionDenominator:1});assert.equal(flip.current.probabilities[1].value,1);assert.equal(flip.cesaroAverage[1].value,1/2);assert.equal(flip.limitDistribution,null);assert.equal(flip.signedSpectralGap.value,2);assert.equal(flip.absoluteSpectralGap.value,0);
 const absorbing=model({...base,forwardNumerator:0,backwardNumerator:1,transitionDenominator:2,initialOneNumerator:1});assert.equal(absorbing.mixingAtQuarter.firstStep,2);assert.equal(absorbing.classification.irreducible,false);assert.equal(absorbing.stationaryLagOneCorrelation,null);assert.equal(absorbing.stationaryTimeReversal![1],null);
 const identity=model({...base,forwardNumerator:0,backwardNumerator:0,initialOneNumerator:1,initialDenominator:3});assert.equal(identity.stationaryDistribution,null);assert.deepEqual(identity.limitDistribution,identity.initial);
 const tiny=model({...base,forwardNumerator:500000,backwardNumerator:499999,transitionDenominator:1000000,stepCount:64});assert.equal(tiny.current.totalVariationFromStationary!.value,null);assert.equal(tiny.current.totalVariationFromStationary!.sign,1);assert.equal(tiny.classification.currentDistributionIsStationary,false);
});
test('registered manuscript keeps its dedicated six-step guide and three experiments',()=>{
 assert.ok(learningUnits.length>=530);assert.ok(learningUnits.some(u=>u.id===id));assert.equal(learningGuides[id].steps.length,6);assert.equal(learningGuides[id].experiments.length,3);
});
