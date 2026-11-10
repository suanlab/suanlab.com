import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync,mkdirSync,writeFileSync} from 'node:fs';
import {unified} from 'unified';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import remarkRehype from 'remark-rehype';
import rehypeSlug from 'rehype-slug';
import rehypeKatex from 'rehype-katex';
import rehypeStringify from 'rehype-stringify';
import {visit} from 'unist-util-visit';
import type {Root,Element} from 'hast';
import {rayleighModel as rayleigh,rayleighKeys}from '../src/lib/learning-rayleigh';
import {ritzModel as ritz,type RitzParameters}from '../src/lib/learning-ritz';
import type {PartitionObservation as Q}from '../src/lib/learning-partition-exact-core';
const ids=['finite-state-rayleigh-energy-variance','nonorthogonal-ritz-subspace-certificates'];
const dir='docs/online-learning/physics-math-for-ai',runtime='.runtime/learning/quantum-variational-manuscripts';
const metrics:Record<string,unknown>={};
for(const[index,id]of ids.entries()){
 const sources=Object.fromEntries(['ko','en'].map(lang=>[lang,readFileSync(`${dir}/${id}-chapter.${lang}.md`,'utf8')]));
 for(const lang of ['ko','en'])test(`variational manuscript ${id}.${lang}: detailed theory,renders and complete exercises`,async()=>{
  const s=sources[lang];assert.equal((s.match(/^## /gm)||[]).length,14);assert.equal((s.match(/^\*\*\d\./gm)||[]).length,6);assert.equal((s.match(lang==='ko'?/^\*\*해설\./gm:/^\*\*Explanation\./gm)||[]).length,6);assert(!s.startsWith('---'));assert(!/^# /m.test(s));assert(!/@E\d+@/.test(s));assert(!/[\u0000-\u0008\u000b\u000c\u000e-\u001f]/.test(s));
  if(lang==='en'){assert(!/[가-힣]/.test(s));assert(s.trim().split(/\s+/).length>2500);}else assert((s.match(/[가-힣]/g)||[]).length>4500);
  const headings:string[]=[];let tables=0,equations=0;
  const result=await unified().use(remarkParse).use(remarkGfm).use(remarkMath).use(remarkRehype).use(rehypeSlug).use(rehypeKatex,{strict:'error'}).use(()=>(tree:Root)=>{
   visit(tree,'element',node=>{
    if(node.tagName==='h2'){assert.equal(typeof node.properties.id,'string');headings.push(node.properties.id as string);}
    if(node.tagName==='table'){tables++;const rows:Element[]=[];visit(node,'element',row=>{if(row.tagName==='tr')rows.push(row);});assert.equal(rows.length,4);const columns=index===0&&tables===2?4:3;for(const row of rows)assert.equal(row.children.filter(n=>n.type==='element'&&(n.tagName==='th'||n.tagName==='td')).length,columns);}
    if(Array.isArray(node.properties.className)&&node.properties.className.includes('katex-display'))equations++;
   });
  }).use(rehypeStringify).process(s);
  assert.equal(result.messages.length,0);assert(!String(result).includes('katex-error'));assert.equal(tables,2);assert.equal(equations,12);assert.equal(new Set(headings).size,14);
  metrics[`${id}.${lang}`]={sections:14,explainedExercises:6,tables,displayedEquations:equations,characters:s.length,words:s.trim().split(/\s+/).length,hangulCharacters:(s.match(/[가-힣]/g)||[]).length};mkdirSync(runtime,{recursive:true});writeFileSync(`${runtime}/${id}.${lang}.html`,String(result));
 });
 test(`variational manuscript ${id}: translations preserve all displayed derivations and exercise numbering`,()=>{
  assert.deepEqual(sources.ko.match(/\$\$[\s\S]*?\$\$/g),sources.en.match(/\$\$[\s\S]*?\$\$/g));assert.deepEqual(sources.ko.match(/^\*\*\d\./gm),sources.en.match(/^\*\*\d\./gm));
 });
}
function fraction(s:string){const[n,d='1']=s.split('/');return{n:BigInt(n),d:BigInt(d)};}
function bound(v:NonNullable<Q['bounds']>['lower']){assert.equal(v.numerator.exponent,0);assert.equal(v.denominator.exponent,0);return{n:BigInt(v.numerator.coefficient),d:BigInt(v.denominator.coefficient)};}
function rational(v:Q,expected:string,exact=true){assert(v.bounds);const a=bound(v.bounds.lower),b=bound(v.bounds.upper),x=fraction(expected);assert(a.n*x.d<=x.n*a.d&&x.n*b.d<=b.n*x.d);if(exact){assert.equal(a.n*x.d,x.n*a.d);assert.equal(b.n*x.d,x.n*b.d);}}
function near(actual:number|null,expected:number){assert.notEqual(actual,null);assert(Math.abs(actual!-expected)<1e-12*Math.max(1,Math.abs(expected)));}
const input=(extra:Record<string,number>)=>Object.assign(Object.fromEntries(rayleighKeys.map(k=>[k,0])),{dimension:2,hDenominator:1,pathDenominator:1,stateReal0:1},extra);
const ritzRefs=JSON.parse(readFileSync('scripts/fixtures/ritz-reference.json','utf8'))as{cases:{label:string;input:RitzParameters}[]};
const R=(label:string)=>{const p=ritzRefs.cases.find(x=>x.label===label);assert(p,label);return ritz(p.input);};
const analytic=JSON.parse(readFileSync('scripts/fixtures/quantum-variational-authoring-reference.json','utf8'));
test('raw complex normalization exercise has norm8,energy1 and variance1',()=>{
 const x=rayleigh(input({hReal11:2,stateReal0:2,stateImag1:2}));rational(x.rawNormSquared,'8');rational(x.energy,'1');rational(x.secondMoment,'2');rational(x.variance,'1');rational(x.ground.overlap,'1/2',false);
});
test('zero variance exercise identifies an excited state rather than a ground certificate',()=>{
 const x=rayleigh(input({hReal11:2,stateReal0:0,stateReal1:1}));rational(x.energy,'2');rational(x.variance,'0');rational(x.ground.overlap,'0',false);assert(x.classification.excitedEigenstate);assert(!x.classification.groundEigenstate);
});
test('negative-origin gap example has tight leakage1/5 and demonstrates the invalid upper-energy substitution',()=>{
 const x=rayleigh(input({hReal00:-3,hReal11:1,stateReal0:2,stateReal1:1}));rational(x.energy,'-11/5');rational(x.variance,'64/25');rational(x.ground.gap!,'4',false);rational(x.ground.leakage,'1/5',false);rational(x.ground.energyOverGap!,'1/5',false);assert(x.ground.certifiedLeakageUpperBound);assert(x.ground.certifiedLeakageUpperBound.value!>=.2);assert.equal((-11/5+2)/4<0,true);
});
test('real path derivatives include normalization and match the independent symbolic quotient',()=>{
 const x=rayleigh(input({hReal11:2,stateReal1:1,directionReal0:1,directionReal1:-1,pathNumerator:1}));rational(x.variation.firstDerivative,analytic.directionalPath.firstAtZero);rational(x.variation.secondDerivative,analytic.directionalPath.secondAtZero);assert(x.path.defined);rational(x.path.moments!.energy,analytic.directionalPath.energyAtOne);
});
test('zero vector path is absent rather than an invented zero energy sample',()=>{
 const x=rayleigh(input({hReal11:2,stateImag1:1,directionReal0:-1,directionImag1:-1,pathNumerator:1}));assert(x.classification.pathZeroVector);assert(!x.path.defined);assert.equal(x.path.moments,null);
});
test('tiny variance example retains a true positive energy statistic and exact raw norm',()=>{
 const M=BigInt(1000000),D=M*M+BigInt(1),x=rayleigh(input({hDenominator:1000000,hReal11:1,stateReal0:1000000,stateReal1:1}));rational(x.rawNormSquared,String(D));rational(x.energy,`1/${M*D}`);rational(x.variance,`${M*M}/${M*M*D*D}`);assert(!x.classification.exactEigenstate);assert(x.variance.value!>0&&x.variance.value!<1e-20);
});
test('Gaussian manuscript example has independent normalization,factorization and double-alpha answers',()=>{
 assert.equal(analytic.continuumEngineImplemented,false);assert.equal(analytic.gaussian.normalization,'1');assert.equal(analytic.gaussian.annihilationIdentity,true);assert.equal(analytic.gaussian.doubleAlphaEnergyInHbarOmega,'5/8');assert.equal(analytic.gaussian.doubleAlphaVarianceInHbarOmegaSquared,'9/32');
});
test('one-column trial example preserves the Gram denominator and energy3/2',()=>{
 const p:RitzParameters={...ritzRefs.cases.find(x=>x.label==='hand computed plane')!.input,columns:1,shearReal:0,shearImag:0};for(let i=0;i<4;i++){p[`basisReal${i}1`]=0;p[`basisImag${i}1`]=0;}const x=ritz(p);rational(x.gram[0][0].real,'2');rational(x.projectedHamiltonian[0][0].real,'3');rational(x.energy,'3/2');near(x.spectrum[0].energy.value,1.5);
});
test('plane polynomial,interlacing and nested energy example agree with the general engine',()=>{
 const x=R('hand computed plane');['1','-8/3','1'].forEach((v,i)=>rational(x.characteristicPolynomial[i],v));near(x.spectrum[0].energy.value,(4-Math.sqrt(7))/3);near(x.spectrum[1].energy.value,(4+Math.sqrt(7))/3);near(x.nestedSubspaces[0].minimumEnergy.value,1.5);assert(x.nestedSubspaces[1].minimumEnergy.value!<1.5);assert(x.spectrum[0].energy.value!>0);assert.deepEqual(x.ambientSpectrum.map(v=>v.energy.value),[0,1,3]);
});
test('degenerate Ritz stationarity exercise leaves exactly8/3 of outside variance',()=>{
 const x=R('Ritz stationary but not ambient eigenstate');rational(x.rawNormSquared,'3');rational(x.energy,'2');rational(x.variance,'8/3');rational(x.projectedVariance,'0');rational(x.outsideVariance,'8/3');assert(x.classification.selectedStateRitzStationary);assert(!x.classification.selectedStateAmbientEigenstate);assert.equal(x.spectrum.length,1);assert.equal(x.spectrum[0].multiplicity,2);x.residualNumerators.forEach((v,i)=>rational(v.real,['-2','0','2'][i]));
});
test('complex coordinate example preserves the state while bare coefficient length changes',()=>{
 const x=R('hand computed plane');assert.deepEqual(x.state,x.basisChange.state);rational(x.rawNormSquared,'2');rational(x.coefficientEuclideanNormSquared,'1');rational(x.basisChange.coefficientEuclideanNormSquared,'1/4');assert.deepEqual(x.spaceProjector,x.basisChange.spaceProjector);assert.deepEqual(x.characteristicPolynomial,x.basisChange.characteristicPolynomial);for(let i=0;i<2;i++)for(let j=0;j<2;j++){const z=x.basisChange.gram[i][j],v=analytic.ritz.gramAfterChange[i][j];assert(v);near(z.real.value,[[8,6],[6,8]][i][j]);near(z.imaginary.value,[[0,4],[-4,0]][i][j]);}
});
test('nearly dependent integer example keeps determinant1 and a positive small Gram eigenvalue',()=>{
 const x=R('nearly dependent two columns');rational(x.gramDeterminant.real,'1');assert(x.gramConditionNumber.value!>1e24);assert(x.gramSpectrum[0].eigenvalue.value!>0);assert.equal(x.columns,2);
});
test('larger dimension without inclusion need not improve the minimum energy',()=>{
 const base=ritzRefs.cases.find(x=>x.label==='exact ambient ground')!.input;const v:RitzParameters={...base,columns:1,shearReal:0,shearImag:0};for(let i=0;i<4;i++){v[`basisReal${i}1`]=0;v[`basisImag${i}1`]=0;}Object.assign(v,{hReal00:0,hReal11:1,hReal22:3});const w:RitzParameters={...base,hReal00:0,hReal11:1,hReal22:3,basisReal00:0,basisReal10:1,basisReal11:0,basisReal21:1};near(ritz(v).spectrum[0].energy.value,0);near(ritz(w).spectrum[0].energy.value,1);
});
test('moving metric example differentiates the quotient rather than ignoring overlap changes',()=>{
 assert.equal(analytic.movingMetric.atOne,'1');assert.equal(analytic.movingMetric.withoutOverlapDerivativeAtOne,'2');assert.notEqual(analytic.movingMetric.atOne,analytic.movingMetric.withoutOverlapDerivativeAtOne);
});
test('manuscript inventory records four full drafts and keeps classroom preparation distinct from reader integration',()=>{
 assert.equal(Object.keys(metrics).length,4);mkdirSync(runtime,{recursive:true});writeFileSync(`${runtime}/authoring-checks.json`,JSON.stringify({metrics,sections:56,explainedExercises:24,tables:8,displayedEquations:48,homepageIntegrated:false,wholeGoalComplete:false},null,2)+'\n');
});
