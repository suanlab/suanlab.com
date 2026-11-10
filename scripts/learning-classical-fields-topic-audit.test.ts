/** Cross-unit evidence for the planned introductory topic;not a whole-domain audit. */
import test from 'node:test';
import assert from 'node:assert/strict';
import {learningUnits,learningGuides} from '../src/data/interactive-book';
import {getLearningContent} from '../src/lib/learning-content';
import {calculateScalarField} from '../src/lib/learning-scalar-field';
import {calculateFieldBalance} from '../src/lib/learning-field-balance';
import {calculateFieldModes} from '../src/lib/learning-field-modes';
import {fieldModesExamples} from '../src/lib/learning-field-modes-examples';
const ids=['scalar-field-action-boundary-variation','field-noether-current-boundary-balance','field-fourier-modes-lattice-continuum']as const;
test('the planned three-unit classical-field path has actual bilingual theory,guides and explained questions',async()=>{
 const sections=[14,16,18],steps=[7,7,8],guides=[4,5,6],questions=[5,6,7];
 for(const[i,id]of ids.entries()){const unit=learningUnits.find(u=>String(u.id)===id);assert(unit,id);assert.equal(learningUnits.indexOf(unit),584+i);assert.equal(unit.questions.length,questions[i]);assert.equal(learningGuides[unit.id].steps.length,steps[i]);assert.equal(learningGuides[unit.id].experiments.length,guides[i]);const content=await getLearningContent(id);for(const language of ['ko','en']as const){assert.equal(content.outline[language].length,sections[i]);assert(content[language].length>6000);assert(content[language].includes('<math'));assert(!content[language].includes('katex-error'));}assert(!/[가-힣]/.test(content.en));}
 const last=learningUnits.find(u=>String(u.id)===ids[2])!;assert.deepEqual(last.prerequisiteIds,ids.slice(0,2));
});
test('action,local balance and spectral decomposition agree on one exact finite model',()=>{
 const input=fieldModesExamples.find(e=>e.id==='three-site-pair')!.input,modes=calculateFieldModes(input),field={boundary:input.boundary,spacing:modes.spacing,speed:input.speed,quadratic:input.quadratic,quartic:input.quartic,position:input.position,velocity:input.velocity},scalar=calculateScalarField(field),balance=calculateFieldBalance({field,share:'1/2'});
 assert.equal(scalar.energy,'7/2');assert.equal(balance.totalEnergy,scalar.energy);assert.equal(modes.exact.fullEnergy,scalar.energy);assert.equal(modes.exact.quadraticEnergy,'7/2');assert.equal(balance.energyDerivative,'0');assert(balance.cells.every(c=>c.energyResidual==='0'&&c.shiftResidual==='0'));
});
test('the same fixed-length boundary geometry and mass scaling survive all three representations',()=>{
 for(const id of ['fixed-length-dirichlet','cell-centered-neumann']){const input=fieldModesExamples.find(e=>e.id===id)!.input,modes=calculateFieldModes(input),field={boundary:input.boundary,spacing:modes.spacing,speed:input.speed,quadratic:input.quadratic,quartic:input.quartic,position:input.position,velocity:input.velocity},scalar=calculateScalarField(field),balance=calculateFieldBalance({field,share:'1/2'});assert.deepEqual(scalar.positions,modes.selected.positions);assert.equal(scalar.massPerSite,modes.spacing);assert.equal(scalar.length,modes.length);assert.equal(scalar.energy,modes.exact.fullEnergy);assert.equal(balance.totalEnergy,scalar.energy);}
});
test('a nonlinear remainder changes the quadratic approximation without changing the physical energy definition',()=>{
 const input=fieldModesExamples.find(e=>e.id==='double-well-reference')!.input,modes=calculateFieldModes(input),field={boundary:input.boundary,spacing:modes.spacing,speed:input.speed,quadratic:input.quadratic,quartic:input.quartic,position:input.position,velocity:input.velocity};assert.equal(calculateScalarField(field).energy,'4');assert.equal(calculateFieldBalance({field,share:'1/2'}).totalEnergy,'4');assert.equal(modes.exact.fullEnergy,'4');assert.equal(modes.exact.quadraticEnergy,'9/2');assert.equal(modes.exact.referencePotential,'-1');assert.equal(modes.exact.nonlinearRemainder,'1/2');assert.equal(modes.exact.expansionResidual,'0');
});
