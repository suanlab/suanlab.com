import assert from 'node:assert/strict';
import test from 'node:test';
import {readFileSync} from 'node:fs';
import {calculateScalarField} from '../src/lib/learning-scalar-field';
import {readScalarFieldInput,type ScalarFieldInput} from '../src/lib/learning-scalar-field-input';
import {getLearningContent} from '../src/lib/learning-content';
const base:ScalarFieldInput={boundary:'periodic',spacing:'1',speed:'1',quadratic:'0',quartic:'0',position:['1','0','-1'],velocity:['1','0','0']};
const fixtures=JSON.parse(readFileSync('scripts/fixtures/scalar-field-lattice-reference.json','utf8'))as{cases:{label:string;input:ScalarFieldInput;expected:Record<string,unknown>}[]};
for(const row of fixtures.cases)test(`independent symbolic energy,forces and Hessian: ${row.label}`,()=>{
 const result=calculateScalarField(row.input);for(const[field,value]of Object.entries(row.expected))assert.deepEqual(result[field as keyof typeof result],value,field);
 assert.equal(result.timeDiscretized,false);assert.equal(result.energyDerivative,'0');assert.equal(result.sites,row.input.position.length);
});
test('three-site worked example separates exact energy,power and initial state',()=>{
 const r=calculateScalarField(base);assert.equal(r.energy,'7/2');assert.equal(r.gradientEnergy,'3');assert.equal(r.kineticEnergy,'1/2');assert.deepEqual(r.acceleration,['-3','0','3']);assert.equal(r.potentialPower,'3');assert.equal(r.kineticPower,'-3');assert.equal(r.totalMomentum,'1');assert.equal(r.momentumDerivative,'0');
 const changed=calculateScalarField({...base,velocity:['0','0','0']});assert.deepEqual(changed.acceleration,r.acceleration);assert.equal(changed.energy,'3');assert.notEqual(changed.totalMomentum,r.totalMomentum);
});
test('three boundary problems retain their geometry and distinct endpoint forces',()=>{
 const constant={...base,position:['1','1','1'],velocity:['0','0','0']};
 const p=calculateScalarField(constant),n=calculateScalarField({...constant,boundary:'neumann-zero'}),d=calculateScalarField({...constant,boundary:'dirichlet-zero'});
 assert.deepEqual(p.acceleration,['0','0','0']);assert.deepEqual(n.acceleration,p.acceleration);assert.deepEqual(d.acceleration,['-1','0','-1']);assert.equal(d.boundaryForce,'-2');
 assert.deepEqual([p.edges.length,n.edges.length,d.edges.length],[3,2,4]);assert.deepEqual([p.length,n.length,d.length],['3','3','4']);assert.deepEqual(n.positions,['1/2','3/2','5/2']);assert.deepEqual(d.positions,['1','2','3']);assert.equal(d.energy,'1');
});
test('uniform shift and quartic equilibria are conditional statements',()=>{
 const shifted=calculateScalarField({...base,position:['6','5','4']});assert.deepEqual(shifted.acceleration,calculateScalarField(base).acceleration);assert.equal(shifted.energy,'7/2');
 for(const boundary of ['periodic','neumann-zero']as const){const r=calculateScalarField({...base,boundary,quadratic:'-1',quartic:'1',position:['1','1','1']});assert.deepEqual(r.acceleration,['0','0','0']);assert.equal(r.localPotentialLowerBound,'-1/4');}
 const local=calculateScalarField({...base,quadratic:'1',position:['1','1','1']});assert.deepEqual(local.acceleration,['-1','-1','-1']);assert.equal(local.localPotentialForce,'-3');
});
test('a local unbounded potential does not imply an unbounded finite Dirichlet energy',()=>{
 const r=calculateScalarField({...base,boundary:'dirichlet-zero',speed:'2',quadratic:'-1',position:['1','1','1'],velocity:['0','0','0']});
 assert.equal(r.localPotentialLowerBound,null);assert.equal(r.energy,'5/2');assert.deepEqual(r.potentialHessian,[['7','-4','0'],['-4','7','-4'],['0','-4','7']]);
 // Sylvester leading minors7,33,119 prove this finite quadratic form is positive definite.
 assert.equal(7*7-4*4,33);assert.equal(7*33-16*7,119);
});
test('spacing changes the mass and force metric; exact tiny values remain nonzero',()=>{
 const r=calculateScalarField({...base,spacing:'1/2'});assert.equal(r.massPerSite,'1/2');assert.equal(r.gradientEnergy,'6');assert.deepEqual(r.acceleration,['-12','0','12']);
 const tiny=calculateScalarField({...base,quadratic:'0',quartic:'1',position:Array(3).fill('1/100000000000000000'),velocity:['0','0','0']});assert.notEqual(tiny.localPotentialEnergy,'0');assert(tiny.acceleration.every(x=>x.startsWith('-1/')));assert.equal(tiny.energyDerivative,'0');assert.doesNotThrow(()=>JSON.stringify(tiny));
});
test('inputs own exact bounded fields and reject malformed data without executing getters',()=>{
 const raw={...base,position:[...base.position],velocity:[...base.velocity]},owned=readScalarFieldInput(raw);assert(owned);raw.position[0]='9';assert.equal(owned.position[0],'1');assert(Object.isFrozen(owned));assert(Object.isFrozen(owned.position));
 for(const changed of [{spacing:'0'},{spacing:'-1'},{speed:'0/2'},{quartic:'-1'},{quadratic:'0.1'},{speed:'1/0'},{position:['1','2']},{velocity:['1','0','0','0']},{position:Array(33).fill('0'),velocity:Array(33).fill('0')},{extra:true}]){assert.equal(readScalarFieldInput({...base,...changed}),null);assert.throws(()=>calculateScalarField({...base,...changed}),RangeError);}
 let reads=0;const bad={...base};Object.defineProperty(bad,'spacing',{get(){reads++;throw Error('Getter invoked');}});assert.equal(readScalarFieldInput(bad),null);const entries=[...base.position];Object.defineProperty(entries,'0',{get(){reads++;throw Error('Entry getter invoked');}});assert.equal(readScalarFieldInput({...base,position:entries}),null);assert.equal(reads,0);
 const sparse=Array(3);sparse[0]='1';assert.equal(readScalarFieldInput({...base,position:sparse}),null);
});
test('first scalar-field manuscripts keep explained examples,14 sections and strict equations',async()=>{
 const content=await getLearningContent('scalar-field-action-boundary-variation');for(const language of ['ko','en']as const){assert.equal(content.outline[language].length,14);assert(content[language].includes('katex-display'));assert(!content[language].includes('katex-error'));assert(!/<h1/.test(content[language]));}assert(!/[가-힣]/.test(content.en));
});
