import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {getLearningContent} from '../src/lib/learning-content';
import {learningUnits,learningGuides} from '../src/data/interactive-book';
import {stationaryThreeStateCycleModel as model} from '../src/lib/learning-stationary-cycle';
const id='stationary-flux-three-state-cycle';
const raw=(language:'ko'|'en')=>readFileSync(`content/learning/physics-math-for-ai/${id}.${language}.md`,'utf8');
for(const language of ['ko','en'] as const)test(`${language}: full cycle explanation, accessible math and six explained exercises render`,async()=>{
 const source=raw(language),content=await getLearningContent(id),html=content[language];assert.equal(content.outline[language].length,18);assert.equal(new Set(content.outline[language].map(h=>h.id)).size,18);assert.equal((source.match(/^### [1-6]\. /gm)??[]).length,6);assert.equal((source.match(/\*\*(?:Explanation\.|해설\.)\*\*/g)??[]).length,6);
 assert.equal((html.match(/class="katex-display"/g)??[]).length,25);assert.equal((html.match(/aria-label="(?:Equation|수식) /g)??[]).length,25);assert.equal((html.match(/<table[^>]*tabindex="0"/g)??[]).length,4);assert.doesNotMatch(html,/katex-error|\$\$|@@E\d/);assert.doesNotMatch(source,/^# |^---/m);
 if(language==='en'){assert.ok(source.split(/\s+/).length>2500);assert.doesNotMatch(html,/[가-힣]/);}else assert.ok((source.match(/[가-힣]/g)??[]).length>5500);
});
test('both translations share equations and retain the distinctions needed for detailed balance',()=>{
 const equations=(s:string)=>[...s.matchAll(/\$\$([\s\S]*?)\$\$/g)].map(m=>m[1].replace(/\s+/g,''));assert.deepEqual(equations(raw('ko')),equations(raw('en')));
 for(const phrase of [/clockwise/,/Row vector/i,/stationary traffic/,/every initial distribution/,/identity/,/period three/,/upper bound/,/not frequencies/,/outside-display/,/physical entropy production/])assert.match(raw('en'),phrase);
});
test('the worked ledger and the three-step distribution match the numerical model',()=>{
 const r=model({forwardWeight:2,backwardWeight:1,stayWeight:1,initialWeights:[1,0,0],stepCount:3});assert.deepEqual(r.current.probabilities.map(q=>q.value),[22/64,21/64,21/64]);assert.equal(r.stationaryFlux[0][1].value,1/6);assert.equal(r.stationaryFlux[1][0].value,1/12);assert.equal(r.clockwiseCurrent.value,1/12);assert.equal(r.conditionalForwardCycleProbability.value,1/8);assert.equal(r.conditionalBackwardCycleProbability.value,1/64);
});
test('all three manuscript endpoint exercises preserve their distinct conclusions',()=>{
 const base={forwardWeight:1,backwardWeight:0,stayWeight:0,initialWeights:[1,0,0],stepCount:8};const cycle=model(base);assert.deepEqual(cycle.current.probabilities.map(q=>q.value),[0,0,1]);assert.equal(cycle.classification.uniqueClassPeriod,3);
 const identity=model({...base,forwardWeight:0,stayWeight:1,initialWeights:[1,2,7]});assert.equal(identity.classification.stationaryReferenceIsUnique,false);assert.equal(identity.current.totalVariationFromUniform.value,11/30);
 const tiny=model({...base,forwardWeight:1000000,backwardWeight:999999,stayWeight:1000000,stepCount:64});assert.equal(tiny.current.totalVariationFromUniform.value,null);assert.equal(tiny.current.totalVariationFromUniform.sign,1);assert.equal(tiny.classification.currentDistributionIsStationary,false);
});
test('registered manuscript keeps its dedicated six-step guide and three experiments',()=>{
 assert.ok(learningUnits.length>=530);assert.ok(learningUnits.some(u=>u.id===id));assert.equal(learningGuides[id].steps.length,6);assert.equal(learningGuides[id].experiments.length,3);
});
