import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {getLearningContent} from '../src/lib/learning-content';
import {learningUnits,learningGuides} from '../src/data/interactive-book';
import {metropolisHastingsProposalModel as model} from '../src/lib/learning-hastings-proposal';
const id='metropolis-hastings-proposal-balance',base={targetWeights:[1,2,4],forwardWeight:2,backwardWeight:1,stayWeight:1};
const raw=(lang:'ko'|'en')=>readFileSync(`content/learning/physics-math-for-ai/${id}.${lang}.md`,'utf8');
for(const lang of ['ko','en'] as const)test(`${lang}: complete proposal explanation and six answered exercises render accessible strict mathematics`,async()=>{
 const s=raw(lang),r=await getLearningContent(id),html=r[lang];assert.equal(r.outline[lang].length,17);assert.equal(new Set(r.outline[lang].map(h=>h.id)).size,17);assert.equal((s.match(/^### [1-6]\. /gm)??[]).length,6);assert.equal((s.match(/\*\*(?:Explanation\.|해설\.)\*\*/g)??[]).length,6);assert.equal((html.match(/class="katex-display"/g)??[]).length,19);assert.equal((html.match(/aria-label="(?:Equation|수식) /g)??[]).length,19);assert.equal((html.match(/<table[^>]*tabindex="0"/g)??[]).length,5);assert.doesNotMatch(html,/katex-error|\$\$|@@E\d/);assert.doesNotMatch(s,/^# |^---/m);
 if(lang==='en'){assert.ok(s.split(/\s+/).length>2700);assert.doesNotMatch(html,/[가-힣]/);}else assert.ok((s.match(/[가-힣]/g)??[]).length>5500);
});
test('translations retain identical equations and distinguish conditional absence, accidental invariance and the sampling clock',()=>{
 const equations=(s:string)=>[...s.matchAll(/\$\$([\s\S]*?)\$\$/g)].map(m=>m[1].replace(/\s+/g,''));assert.deepEqual(equations(raw('ko')),equations(raw('en')));
 for(const term of [/not provided/,/accepted self proposals/,/stationary distribution/,/minimum target weight occurring at least twice/,/not the same jump chain/,/does not make aperiodicity/,/not a completion claim/,/six-decimal/])assert.match(raw('en'),term);
});
test('the worked accept/reject and intended/actual preparation calculations agree with exact runtime values',()=>{
 const r=model(base);assert.equal(r.corrected.acceptance[2][0]!.value,1/8);assert.equal(r.corrected.transition[2][0].value,1/16);assert.equal(r.corrected.totalHoldingByState[2].value,11/16);assert.equal(r.corrected.meanProposalAcceptanceAtTarget.value,3/4);assert.equal(r.corrected.meanMovementAtTarget.value,1/2);assert.deepEqual(r.uncorrected.targetResidual.map(q=>q.value),[0,-1/28,1/28]);assert.deepEqual(r.uncorrected.stationaryDistribution!.map(q=>q.value),[1/7,5/21,13/21]);assert.equal(r.uncorrected.stationaryTargetDistance!.value,1/21);assert.equal(r.uncorrected.meanMovementAtTarget.value,3/7);assert.equal(r.uncorrected.meanMovementAtStationary!.value,23/56);
});
test('exercise conclusions preserve missing reverse events, the accidental target, and the changed jump law',()=>{
 const oneWay=model({...base,forwardWeight:1,backwardWeight:0,stayWeight:0}).corrected;assert.equal(oneWay.acceptance[0][1]!.value,0);assert.equal(oneWay.acceptance[0][2],null);assert.equal(oneWay.stationaryDistribution,null);assert.equal(oneWay.classification.stationaryIsUnique,false);
 const accidental=model({...base,targetWeights:[1,1,4]}).uncorrected;assert.equal(accidental.targetInvariant,true);assert.equal(accidental.targetDetailedBalance,false);assert.equal(accidental.targetCurrent[0][1].value,1/24);
 const jump=model(base).corrected;assert.deepEqual(jump.jumpStationaryDistribution!.map(q=>q.value),[3/14,3/7,5/14]);assert.equal(jump.jumpTargetDistance!.value,3/14);
});
test('registered manuscript keeps its dedicated six-step guide and three experiments',()=>{
 assert.ok(learningUnits.length>=530);assert.ok(learningUnits.some(u=>u.id===id));assert.equal(learningGuides[id].steps.length,6);assert.equal(learningGuides[id].experiments.length,3);
});
