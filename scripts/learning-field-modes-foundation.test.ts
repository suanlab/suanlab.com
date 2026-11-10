import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {build} from 'esbuild';
import {calculateFieldModes,type FieldModeRange} from '../src/lib/learning-field-modes';
import {readFieldModesInput,fieldModesInputKey,type FieldModesInput} from '../src/lib/learning-field-modes-input';
import {fieldModesExamples} from '../src/lib/learning-field-modes-examples';
import {snapshotQubitWire} from '../src/lib/learning-qubit-wire';
type F={n:bigint;d:bigint};
function f(n:bigint,d=BigInt(1)):F{assert(d>BigInt(0));return{n,d};}
function rational(s:string):F{const[n,d='1']=s.split('/');return f(BigInt(n),BigInt(d));}
function decimal(s:string):F{const[coefficient,exp='0']=s.toLowerCase().split('e'),negative=coefficient.startsWith('-'),[integer,fraction='']=coefficient.replace(/^[-+]/,'').split('.'),n=BigInt(integer+fraction)*(negative?-BigInt(1):BigInt(1)),power=Number(exp)-fraction.length;return power>=0?f(n*BigInt(10)**BigInt(power)):f(n,BigInt(10)**BigInt(-power));}
const le=(a:F,b:F)=>a.n*b.d<=b.n*a.d,add=(a:F,b:F)=>f(a.n*b.d+b.n*a.d,a.d*b.d),neg=(a:F)=>f(-a.n,a.d),mul=(a:F,b:F)=>f(a.n*b.n,a.d*b.d),abs=(a:F)=>f(a.n<BigInt(0)?-a.n:a.n,a.d);
function close(actual:FieldModeRange,reference:string,label:string){const lo=rational(actual.lower),hi=rational(actual.upper),ref=decimal(reference),magnitude=add(f(BigInt(1)),abs(ref)),allowance=mul(decimal('1e-320'),magnitude);assert(le(lo,hi),label);assert(le(lo,add(ref,allowance))&&le(add(ref,neg(allowance)),hi),label+' reference misses enclosure');assert(le(add(hi,neg(lo)),mul(decimal('1e-90'),magnitude)),label+' enclosure is too wide');}
function containsZero(x:FieldModeRange){assert(le(rational(x.lower),f(BigInt(0))));assert(le(f(BigInt(0)),rational(x.upper)));}
type Reference={label:string;input:FieldModesInput;expected:{exact:Record<string,string>;spacing:string;effectiveCurvature:string;referenceIsEquilibrium:boolean;referenceAcceleration:string[];interpretation:string;modes:{index:number;wave:number;family:string;classification:string;eigenvalue:string;coordinate:string;velocity:string;quadraticEnergy:string;massNorm:string}[];selectedShape:string[];refinements:{level:number;sites:number;spacing:string;discreteEigenvalue:string;continuumEigenvalue:string;waveOperatorGap:string;relativeWaveOperatorGap:string;secondOrderUpperBound:string;masslessPhaseSpeedRatio:string|null}[]}};
const refs=JSON.parse(readFileSync('scripts/fixtures/field-modes-independent.json','utf8'))as{cases:Reference[];symbolicHessianModels:unknown[]};
for(const c of refs.cases)test(`independent energy matrix spectrum and mass projection: ${c.label}`,()=>{
 const r=calculateFieldModes(c.input),e=c.expected;assert.deepEqual(r.exact,e.exact);assert.equal(r.spacing,e.spacing);assert.equal(r.effectiveCurvature,e.effectiveCurvature);assert.equal(r.referenceIsEquilibrium,e.referenceIsEquilibrium);assert.deepEqual(r.referenceAcceleration,e.referenceAcceleration);assert.equal(r.interpretation,e.interpretation);assert.equal(r.modes.length,e.modes.length);
 r.modes.forEach((row,i)=>{const expected=e.modes[i];for(const key of ['index','wave','family','classification']as const)assert.equal(row[key],expected[key],key);for(const key of ['eigenvalue','coordinate','velocity','quadraticEnergy','massNorm']as const)close(row[key],expected[key],`${c.label}/${i}/${key}`);});
 r.selected.shape.forEach((x,i)=>close(x,e.selectedShape[i],`${c.label}/shape/${i}`));r.selected.operatorResidual.forEach(containsZero);
 for(const key of ['positionParsevalResidual','velocityParsevalResidual','modalEnergyResidual']as const)containsZero(r.checks[key]);r.checks.positionReconstructionResidual.forEach(containsZero);r.checks.velocityReconstructionResidual.forEach(containsZero);assert.equal(r.checks.orthogonalityResidualMaximumUpper.lower,'0');
 r.refinements.forEach((row,i)=>{const expected=e.refinements[i];for(const key of ['level','sites','spacing']as const)assert.equal(row[key],expected[key]);for(const key of ['discreteEigenvalue','continuumEigenvalue','waveOperatorGap','relativeWaveOperatorGap','secondOrderUpperBound']as const)close(row[key],expected[key],`${c.label}/refinement/${i}/${key}`);if(expected.masslessPhaseSpeedRatio===null)assert.equal(row.masslessPhaseSpeedRatio,null);else close(row.masslessPhaseSpeedRatio!,expected.masslessPhaseSpeedRatio,`${c.label}/phase-ratio`);});
});
const example=(id:string)=>fieldModesExamples.find(x=>x.id===id)!;
test('boundary coordinates,degeneracies and zero modes follow the declared grids',()=>{
 const periodic=calculateFieldModes(example('four-site-cosine').input);assert.deepEqual(periodic.modes.map(m=>m.exactEigenvalue),['0','2','2','4']);assert.deepEqual(periodic.selected.positions,['0','1','2','3']);assert.equal(periodic.exact.quadraticEnergy,'5/2');assert.equal(periodic.exact.perturbationMassNorm,'2');assert.deepEqual(periodic.modes.map(m=>m.family),['constant','cosine','sine','nyquist']);
 const fixed=calculateFieldModes(example('fixed-length-dirichlet').input);assert.equal(fixed.spacing,'4/5');assert.deepEqual(fixed.selected.positions,['4/5','8/5','12/5','16/5']);assert.equal(fixed.exact.fullEnergy,'29/10');assert.equal(fixed.modes.some(m=>m.classification==='zero-curvature'),false);
 const neumann=calculateFieldModes(example('cell-centered-neumann').input);assert.deepEqual(neumann.selected.positions,['1/2','3/2','5/2','7/2']);assert.equal(neumann.modes[0].exactEigenvalue,'0');
 const middle=calculateFieldModes(example('zero-middle-mode').input);assert.equal(middle.modes[1].classification,'zero-curvature');assert.equal(middle.modes[1].exactEigenvalue,'0');assert.equal(middle.selected.phaseSpeed,null);assert.equal(middle.modes[0].classification,'unstable');
});
test('Nyquist stops the symbol group speed while fixed-wave refinement differs from tracking Nyquist',()=>{
 const nyquist=calculateFieldModes(example('periodic-nyquist').input);assert.deepEqual(nyquist.selected.periodicSymbolGroupSpeed,{lower:'0',upper:'0'});assert.equal(nyquist.selected.aliasPartner,2);assert.equal(nyquist.refinements[1].sites,8);assert(nyquist.refinements.every(row=>row.physicalWave===2&&row.fixedLength==='4'));
 const fixed=calculateFieldModes(example('boundary-stabilized').input);assert.deepEqual(fixed.refinements.map(r=>r.sites),[3,7,15,31]);assert(fixed.refinements.every(row=>row.fixedLength==='4'));
 const longer=calculateFieldModes(example('long-interval-instability').input);assert.equal(fixed.modes[0].classification,'oscillatory');assert.equal(longer.modes[0].classification,'unstable');
});
test('equilibrium,wall forces and nonlinear Taylor terms are distinct contracts',()=>{
 const well=calculateFieldModes(example('double-well-reference').input);assert.equal(well.referenceIsEquilibrium,true);assert.equal(well.referenceHasBoundaryJump,false);assert.equal(well.effectiveCurvature,'2');assert.equal(well.exact.referencePotential,'-1');assert.equal(well.exact.quadraticEnergy,'9/2');assert.equal(well.exact.nonlinearRemainder,'1/2');assert.equal(well.exact.fullEnergy,'4');
 const drifting=calculateFieldModes(example('nonstationary-reference').input);assert.equal(drifting.referenceIsEquilibrium,false);assert.deepEqual(drifting.referenceAcceleration,['-1','-1','-1','-1']);assert.equal(drifting.interpretation,'frozen-hessian-nonstationary-reference');
 const wall=calculateFieldModes(example('fixed-reference-force').input);assert.deepEqual(wall.referenceAcceleration,['-1','0','0','-1']);assert.equal(wall.referenceIsEquilibrium,false);assert.equal(wall.referenceHasBoundaryJump,true);assert.equal(wall.continuumRefinementScope,'frozen-operator-eigenvalues-only');assert.equal(wall.exact.expansionResidual,'0');
});
test('owned input bounds do not invoke accessors or silently coerce Fourier indices',async()=>{
 const p=example('four-site-cosine').input,owned=readFieldModesInput({...p});assert(owned&&Object.isFrozen(owned.position));assert.equal(fieldModesInputKey(owned),fieldModesInputKey(p));assert.notEqual(fieldModesInputKey({...p,length:'4/1'}),fieldModesInputKey(p));
 for(const x of [{...p,selectedBasis:4},{...p,selectedBasis:1.5},{...p,selectedBasis:-1},{...p,length:'0'},{...p,speed:'-1'},{...p,quartic:'-1'},{...p,background:'0.1'},{...p,position:['1','2']},{...p,velocity:['0','0','0']},{...p,extra:true}])assert.equal(readFieldModesInput(x),null);
 let reads=0;const bad={...p};Object.defineProperty(bad,'length',{get(){reads++;throw Error('Getter invoked');}});assert.equal(readFieldModesInput(bad),null);assert.equal(reads,0);assert.throws(()=>calculateFieldModes(bad),RangeError);
 const bundle=await build({entryPoints:['src/lib/learning-field-modes-examples.ts'],bundle:true,write:false,platform:'browser',format:'esm',metafile:true,logLevel:'silent'});assert(!Object.keys(bundle.metafile!.inputs).some(p=>/learning-field-modes\.ts|spin-angle|evolution-interval/.test(p)));assert(!bundle.outputFiles[0].text.includes('BigInt('));
});
test('all teaching presets have an independently generated preparation and reference',()=>{
 assert.equal(fieldModesExamples.length,15);assert.equal(refs.symbolicHessianModels.length,9);for(const e of fieldModesExamples){const ref=refs.cases.find(c=>c.label===e.id);assert(ref,e.id);assert.deepEqual(ref.input,e.input);assert(!/[가-힣]/.test(e.label.en));}
});
for(const boundary of ['periodic','dirichlet-zero','neumann-zero']as const)test(`large rational32-site result fits the existing owned-wire budget: ${boundary}`,()=>{
 const input={...example('thirty-two-components').input,boundary,length:'1/999999999999999999',speed:'999999999999999999',quadratic:'-999999999999999999',quartic:'999999999999999999',background:'999999999999999999',position:Array.from({length:32},(_,i)=>i%2?'999999999999999999':'-999999999999999999'),velocity:Array.from({length:32},()=> '1/999999999999999999')},r=calculateFieldModes(input);assert.equal(r.exact.expansionResidual,'0');assert(snapshotQubitWire(r));assert(JSON.stringify(r).length<2000000);assert(r.modes.every(m=>m.classification==='oscillatory'));r.selected.operatorResidual.forEach(containsZero);
});
test('bilingual mode manuscripts derive eighteen sections,worked energies and strict equations',async()=>{
 const{getLearningContent}=await import('../src/lib/learning-content');const content=await getLearningContent('field-fourier-modes-lattice-continuum');for(const language of ['ko','en']as const){assert.equal(content.outline[language].length,18);assert(content[language].includes('katex-display'));assert(!content[language].includes('katex-error'));assert(!/<h1(?:\s|>)/.test(content[language]));assert(content[language].includes('<table'));}assert(!/[가-힣]/.test(content.en));
});
