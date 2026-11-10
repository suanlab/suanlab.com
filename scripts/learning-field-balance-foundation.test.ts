import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {calculateFieldBalance} from '../src/lib/learning-field-balance';
import {readFieldBalanceInput,fieldBalanceInputKey,type FieldBalanceInput} from '../src/lib/learning-field-balance-input';
import {fieldBalanceExamples} from '../src/lib/learning-field-balance-examples';
import {getLearningContent} from '../src/lib/learning-content';
const refs=JSON.parse(readFileSync('scripts/fixtures/field-balance-independent.json','utf8'))as{cases:{label:string;input:FieldBalanceInput;expected:unknown}[]};
for(const row of refs.cases)test(`independent chain-rule and region balances: ${row.label}`,()=>{
 assert.deepEqual(calculateFieldBalance(row.input),row.expected);
});
const preset=(id:string)=>fieldBalanceExamples.find(x=>x.id===id)!;
test('symmetric three-site currents explain each local change and the selected subregion',()=>{
 const r=calculateFieldBalance(preset('symmetric-flow').input);assert.equal(r.totalEnergy,'7/2');assert.deepEqual(r.cells.map(c=>c.energy),['7/4','1/2','5/4']);assert.deepEqual(r.edges.map(e=>e.energyCurrent),['1/2','0','-1']);assert.deepEqual(r.cells.map(c=>c.energyDerivative),['-3/2','1/2','1']);assert.equal(r.prefixes[1].energy,'9/4');assert.equal(r.prefixes[1].energyDerivative,'-1');assert.equal(r.prefixes[1].energyInflow,'-1');assert.equal(r.shiftCharge,'1');assert.equal(r.spatialGenerator?.value,'1/2');
});
test('allocation changes local energies and matching currents while the global energy is unchanged',()=>{
 const left=calculateFieldBalance(preset('left-share-zero').input),right=calculateFieldBalance(preset('left-share-one').input);
 assert.equal(left.totalEnergy,right.totalEnergy);assert.deepEqual(left.cells.map(c=>c.energy),['5/2','1/2','1/2']);assert.deepEqual(right.cells.map(c=>c.energy),['1','1/2','2']);assert.deepEqual(left.edges.map(e=>e.energyCurrent),['1','0','0']);assert.deepEqual(right.edges.map(e=>e.energyCurrent),['0','0','-2']);assert.deepEqual(left.cells.map(c=>c.energyDerivative),['-1','1','0']);assert.deepEqual(right.cells.map(c=>c.energyDerivative),['-2','0','2']);
 const uniform=calculateFieldBalance(preset('uniform-velocity').input);assert(uniform.edges.every(e=>e.energyDerivative==='0'));assert(uniform.edges.some(e=>e.energyCurrent!=='0'));
});
test('boundary and equilibrium counterexamples separate instantaneous derivatives from symmetry',()=>{
 const fixed=calculateFieldBalance(preset('fixed-instant-zero').input);assert.equal(fixed.uniformFieldShiftSymmetry,false);assert.equal(fixed.instantaneousShiftDerivativeZero,true);assert.equal(fixed.shiftSecondDerivative,'-1');assert.deepEqual(fixed.cells.map(c=>c.energy),['5/4','1/2','3/4']);assert.deepEqual(fixed.cells.map(c=>c.energyDerivative),['-1/2','1/2','0']);assert.equal(fixed.spatialGenerator,null);assert.equal(fixed.edges[0].energyCurrent,'0');assert.equal(fixed.edges.at(-1)!.energyCurrent,'0');
 const eq=calculateFieldBalance(preset('equilibrium-not-symmetry').input);assert.equal(eq.shiftDerivative,'0');assert.equal(eq.shiftSecondDerivative,'0');assert.equal(eq.uniformFieldShiftSymmetry,false);
 const free=calculateFieldBalance(preset('neumann-shift').input);assert.equal(free.uniformFieldShiftSymmetry,true);assert.equal(free.shiftDerivative,'0');assert.equal(free.spatialGenerator,null);
});
test('cyclic invariance is distinct from a quadratic skew-generator symmetry',()=>{
 const quartic=calculateFieldBalance(preset('quartic-cyclic').input);assert.equal(quartic.totalEnergy,'29/4');assert.equal(quartic.spatialGenerator?.value,'0');assert.equal(quartic.spatialGenerator?.derivative,'3');assert.equal(quartic.spatialGenerator?.nonlinearDefect,'3');assert.equal(quartic.spatialGenerator?.quadraticSkewGeneratorSymmetry,false);assert.equal(quartic.spatialGenerator?.cyclicPermutationSymmetry,true);
 const input=preset('quartic-cyclic').input,rotated=calculateFieldBalance({...input,field:{...input.field,position:['2','0','1']}});assert.equal(rotated.totalEnergy,quartic.totalEnergy);
 const massive=calculateFieldBalance(preset('massive-shift-breaking').input);assert.equal(massive.shiftDerivative,'-1');assert.equal(massive.uniformFieldShiftSymmetry,false);assert.equal(massive.spatialGenerator?.derivative,'0');assert.equal(massive.spatialGenerator?.quadraticSkewGeneratorSymmetry,true);
});
test('all twelve reader-safe examples have an independent mathematical reference',()=>{
 const labels=['hand-share-half','hand-share-zero','hand-share-one','hand-share-third','fixed-instant-zero','neumann-shift','massive-shift-breaking','quartic-cyclic-counterexample','potential-equilibrium-not-symmetry','uniform-velocity','periodic/32/mixed/share=1/2','periodic/32/mixed/share=0'];assert.equal(fieldBalanceExamples.length,labels.length);
 fieldBalanceExamples.forEach((p,i)=>{const expected=refs.cases.find(c=>c.label===labels[i])!.expected;assert(expected);const result=calculateFieldBalance(p.input);assert.deepEqual({...result,input:(expected as{input:unknown}).input},expected);assert(!/[가-힣]/.test(p.label.en));assert.doesNotThrow(()=>JSON.stringify(result));});
});
test('input ownership bounds the allocation and preserves exact rational spelling',()=>{
 const first=preset('symmetric-flow').input;for(const share of ['-1','2','1/0','0.5','01','1/01'])assert.equal(readFieldBalanceInput({...first,share}),null);
 const owned=readFieldBalanceInput({...first,share:'2/2'});assert(owned);assert(Object.isFrozen(owned));assert(Object.isFrozen(owned.field.position));assert.notEqual(fieldBalanceInputKey(owned),fieldBalanceInputKey({...first,share:'1'}));
 let reads=0;const bad={...first};Object.defineProperty(bad,'share',{get(){reads++;throw Error('Getter invoked');}});assert.equal(readFieldBalanceInput(bad),null);assert.equal(reads,0);assert.equal(readFieldBalanceInput({...first,extra:1}),null);assert.equal(readFieldBalanceInput({...first,field:{...first.field,spacing:'0'}}),null);assert.throws(()=>calculateFieldBalance({...first,share:'2'}),RangeError);
});
test('bilingual Noether draft has sixteen explained sections and strict equations',async()=>{
 const c=await getLearningContent('field-noether-current-boundary-balance');for(const locale of ['ko','en']as const){assert.equal(c.outline[locale].length,16);assert(c[locale].includes('katex-display'));assert(!c[locale].includes('katex-error'));assert(!/<h1/.test(c[locale]));}assert(!/[가-힣]/.test(c.en));
});
