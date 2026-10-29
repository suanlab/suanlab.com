import assert from 'node:assert/strict';
import test from 'node:test';
import { evaluateLogic as evaluate, implies, isPrime, smallestDivisor, relationProperties, logicKinds } from '../src/lib/learning-logic';
import { logicLabs, learningGuides, learningUnits } from '../src/data/interactive-book';
for(const p of [false,true])for(const q of [false,true])test(`implication, contrapositive and converse: ${p},${q}`,()=>{
 assert.equal(implies(p,q),!p||q);assert.equal(implies(p,q),implies(!q,!p));const r=evaluate('propositions-implications',[+p,+q]);assert.equal(r.stats['P⇒Q'],!(p&&!q));assert.equal(r.stats['Q⇒P'],!q||p);assert.equal(r.stats['P⇔Q'],p===q);
});
test('all truth-table valuations are present and implication has one forbidden row',()=>{
 const r=evaluate('propositions-implications',[1,0]);assert.equal(r.rows.length,4);assert.deepEqual(r.rows.filter(v=>!v[4]).map(v=>v.slice(0,2)),[[true,false]]);r.rows.forEach(v=>assert.equal(v[4],v[5]));assert.ok(r.rows.some(v=>v[4]!==v[6]));
});
test('input-dependent witnesses differ from a common witness',()=>{
 const r=evaluate('quantifiers-counterexamples',[3,0]);assert.equal(r.stats['∀x∃y R(x,y)'],true);assert.equal(r.stats['∃y∀x R(x,y)'],false);assert.equal(r.stats['global witness y'],null);assert.deepEqual(r.rows.map(v=>v[2]),[0,1,2]);assert.deepEqual(r.matrix,[[true,false,false],[false,true,false],[false,false,true]]);
});
test('strictly larger finite witnesses fail at the maximum, while weak inequality has a common witness',()=>{
 const strict=evaluate('quantifiers-counterexamples',[3,1]);assert.equal(strict.stats['∀x∃y R(x,y)'],false);assert.deepEqual(strict.counterexamples,[[2]]);const weak=evaluate('quantifiers-counterexamples',[3,2]);assert.equal(weak.stats['∃y∀x R(x,y)'],true);assert.deepEqual(weak.stats['global witness y'],[2]);
});
test('set-bounded quantification on empty and singleton domains has explicit boundary semantics',()=>{
 const empty=evaluate('quantifiers-counterexamples',[0,0]);assert.equal(empty.stats['∀x∃y R(x,y)'],true);assert.equal(empty.stats['∃y∀x R(x,y)'],false);assert.equal(empty.stats['∀x∀y R(x,y)'],true);assert.equal(empty.stats['∃x∃y R(x,y)'],false);assert.equal(empty.checked,0);const single=evaluate('quantifiers-counterexamples',[1,0]);assert.equal(single.stats['∃y∀x R(x,y)'],true);
});
test('set operations and identity mapping match independent reference values',()=>{
 const r=evaluate('sets-functions',[3,3,1,0,1]);assert.deepEqual(r.stats.A,[0,2]);assert.deepEqual(r.stats.B,[1,2]);assert.deepEqual(r.stats['A∩B'],[2]);assert.deepEqual(r.stats['A∪B'],[0,1,2]);assert.deepEqual(r.mapping,[0,1,2]);assert.equal(r.stats.bijective,true);assert.deepEqual(r.stats['f⁻¹(even)'],[0,2]);
});
test('surjective is not injective and a preimage does not require an inverse',()=>{
 const r=evaluate('sets-functions',[4,2,1,0,2]);assert.deepEqual(r.mapping,[0,1,0,1]);assert.equal(r.stats.injective,false);assert.equal(r.stats.surjective,true);assert.deepEqual(r.stats['f⁻¹(even)'],[0,2]);assert.deepEqual(r.counterexamples,[[0,2],[1,3]]);
});
test('declared codomain changes surjectivity without invalidating a function',()=>{
 const r=evaluate('sets-functions',[2,4,1,0,1]);assert.equal(r.stats.injective,true);assert.equal(r.stats.surjective,false);assert.deepEqual(r.stats['f(D)'],[0,1]);assert.deepEqual(r.targets,[0,1,2,3]);
});
test('congruence is equivalence but not antisymmetric on the reference domain',()=>{
 const r=evaluate('relations-orders',[4,0,2]);assert.equal(r.stats.equivalence,true);assert.equal(r.stats.antisymmetric,false);assert.deepEqual(r.propertyFailures?.antisymmetric,[[1,3],[2,4],[3,1],[4,2]]);assert.equal(r.stats['partial order'],false);
});
test('divisibility on positive integers gives a partial but not total order',()=>{
 const r=evaluate('relations-orders',[4,1,2]);assert.equal(r.stats['partial order'],true);assert.equal(r.stats['total order'],false);assert.equal(r.matrix?.[1][2],false);assert.equal(r.matrix?.[2][1],false);assert.equal(r.stats.symmetric,false);
});
test('strict order is not a non-strict partial order; equality is symmetric and antisymmetric',()=>{
 const strict=evaluate('relations-orders',[4,2,2]);assert.equal(strict.stats.reflexive,false);assert.equal(strict.stats.antisymmetric,true);assert.equal(strict.stats.transitive,true);assert.equal(strict.stats['partial order'],false);const equality=relationProperties([0,1,2],(x,y)=>x===y);assert.equal(equality.symmetric,true);assert.equal(equality.antisymmetric,true);assert.equal(equality.totalOrder,false);
});
test('a real transitivity failure includes a triple satisfying both links but not the shortcut',()=>{
 const relation=(x:number,y:number)=>Math.abs(x-y)<=1;const r=relationProperties([0,1,2],relation);assert.equal(r.symmetric,true);assert.equal(r.transitive,false);assert.ok(r.failures.transitive.some(v=>v.join(',')==='0,1,2'));r.failures.transitive.forEach(([x,y,z])=>assert.ok(relation(x,y)&&relation(y,z)&&!relation(x,z)));
});
test('negative integer parity is handled and stronger false conclusions return actual counterexamples',()=>{
 const reference=evaluate('proof-strategies',[0,8]);assert.equal(reference.stats['finite check'],true);assert.equal(reference.stats['global proof available'],true);const stronger=evaluate('proof-strategies',[1,3]);assert.equal(stronger.stats['first counterexample'],-2);const wrong=evaluate('proof-strategies',[2,3]);assert.equal(wrong.stats['first counterexample'],-3);
});
test('sum cases include the empty sum and agree at n=5 and n=45',()=>{
 const r=evaluate('mathematical-induction',[0,5]);assert.deepEqual(r.rows.map(v=>v[1]),[0,1,3,6,10,15]);assert.equal(r.stats['checked cases'],6);assert.equal(r.stats['global proof available'],true);assert.equal(evaluate('mathematical-induction',[0,45]).rows.at(-1)?.[1],1035);
});
test('a successful base does not establish the induction step',()=>{
 const r=evaluate('mathematical-induction',[1,4]);assert.deepEqual(r.rows[0],[1,2,1,true]);assert.equal(r.stats['first counterexample'],2);assert.equal(r.stats['global proof available'],false);assert.deepEqual(r.counterexamples,[[2],[3],[4]]);
});
test('forty prime-looking cases do not justify a universal claim',()=>{
 const first=evaluate('mathematical-induction',[2,39]);assert.equal(first.stats['finite check'],true);assert.equal(first.stats['global proof available'],false);const later=evaluate('mathematical-induction',[2,40]);assert.equal(later.stats['first counterexample'],40);assert.deepEqual(later.rows.at(-1),[40,1681,41,false]);
});
for(const [n,prime,divisor] of [[-1,false,null],[0,false,null],[1,false,null],[2,true,2],[41,true,41],[49,false,7],[1681,false,41]] as const)test(`small integer primality: ${n}`,()=>{assert.equal(isPrime(n),prime);assert.equal(smallestDivisor(n),divisor);});
test('an empty finite check does not claim a global proof',()=>{
 const r=evaluate('mathematical-induction',[1,0]);assert.equal(r.checked,0);assert.equal(r.stats['finite check'],true);assert.equal(r.stats['global proof available'],false);
});
for(const kind of logicKinds)test(`logic controls, presets and guides are consistent: ${kind}`,()=>{
 const spec=logicLabs[kind];assert.equal(spec.presets.length,3);assert.equal(learningGuides[kind].steps.length,4);assert.equal(learningGuides[kind].experiments.length,3);assert.ok(learningUnits.some(u=>u.id===kind));for(const preset of spec.presets){assert.equal(preset.values.length,spec.controls.length);preset.values.forEach((x,i)=>assert.ok(Number.isInteger(x)&&x>=spec.controls[i].min&&x<=spec.controls[i].max));const r=evaluate(kind,preset.values);assert.ok(r.checked>=0);r.rows.flat().forEach(cell=>{if(typeof cell==='number')assert.ok(Number.isFinite(cell));});}learningGuides[kind].experiments.forEach(g=>assert.ok(spec.presets[g.preset]));
});
test('logic parameter validation rejects malformed and fractional states',()=>{
 assert.throws(()=>evaluate('propositions-implications',[1]));assert.throws(()=>evaluate('propositions-implications',[1,.5]));assert.throws(()=>evaluate('quantifiers-counterexamples',[-1,0]));assert.throws(()=>evaluate('sets-functions',[3,0,1,0,1]));assert.throws(()=>evaluate('relations-orders',[4,0,1]));assert.throws(()=>evaluate('mathematical-induction',[2,46]));
});
