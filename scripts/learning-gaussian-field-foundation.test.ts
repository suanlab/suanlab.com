import test from 'node:test';import assert from 'node:assert/strict';import {readFileSync} from 'node:fs';
import {readGaussianFieldInput,gaussianFieldDefault,gaussianFieldExamples} from '../src/lib/learning-gaussian-field-input';
import {calculateGaussianField} from '../src/lib/learning-gaussian-field';
import {getLearningContent} from '../src/lib/learning-content';
import {learningUnits} from '../src/data/interactive-book';
import fixture from './fixtures/gaussian-field-reference.json';
const result=(id:string)=>calculateGaussianField(gaussianFieldExamples.find(e=>e.id===id)!.input);
const rational=(s:string)=>{const[n,d='1']=s.split('/');return[BigInt(n),BigInt(d)]as const;};
const times=(a:string,b:string)=>{const[n,d]=rational(a),[m,e]=rational(b);return[n*m,d*e]as const;};
for(const c of fixture.cases)test('independent bordered-matrix/SymPy reference: '+c.id,()=>{
 const {input,contract,...actual}=calculateGaussianField(c.input);assert.deepEqual(input,c.input);assert.deepEqual(actual,c.expected);assert(contract.includes('induced Euclidean measure'));assert(Object.isFrozen(input)&&Object.isFrozen(input.source)&&Object.isFrozen(input.probes));
});
test('input grammar rejects singular preparations and parameter overflow without silently adding a mass',()=>{
 for(const patch of [{massSquared:'0'},{massSquared:'0',stiffness:'0',constraint:'mean-zero'},{massSquared:'0',stiffness:'0',boundary:'dirichlet'},{temperature:'0'},{massSquared:'-1'},{stiffness:'17'},{massSquared:'1e-3'},{temperature:'0.1'},{temperature:'-0'},{sites:2},{sites:9},{source:['0']},{probes:[0,1,2,4]},{probes:[0,1,2,1.5]}])assert.equal(readGaussianFieldInput({...gaussianFieldDefault,...patch}),null,JSON.stringify(patch));
 assert(readGaussianFieldInput({...gaussianFieldDefault,massSquared:'0',boundary:'dirichlet'}));assert(readGaussianFieldInput({...gaussianFieldDefault,massSquared:'0',constraint:'mean-zero'}));
});
test('own descriptor validation never invokes getters and owns the nested arrays',()=>{
 let reads=0;const value={...gaussianFieldDefault};Object.defineProperty(value,'sites',{get(){reads++;return 4;}});assert.equal(readGaussianFieldInput(value),null);
 const source=['1','0','0','0'];Object.defineProperty(source,'1',{get(){reads++;return'0';}});assert.equal(readGaussianFieldInput({...gaussianFieldDefault,source}),null);assert.equal(reads,0);
 const plain={...gaussianFieldDefault,source:['1','0','0','0'],probes:[0,1,2,3]},owned=readGaussianFieldInput(plain)!;plain.source[0]='16';plain.probes[0]=3;assert.equal(owned.source[0],'1');assert.equal(owned.probes[0],0);
 const extra=['1','0','0','0'];Object.defineProperty(extra,Symbol('extra'),{value:1});assert.equal(readGaussianFieldInput({...gaussianFieldDefault,source:extra}),null);
 const hole=['1',,'0','0'];assert.equal(readGaussianFieldInput({...gaussianFieldDefault,source:hole}),null);
});
test('default ring has the worked Green column, connected/raw distinction and determinant',()=>{
 const r=result('localized-source');assert.deepEqual(r.connectedTwo[0],['7/15','1/5','2/15','1/5']);assert.deepEqual(r.mean,r.connectedTwo[0]);assert.equal(r.rawTwo[0][1],'22/75');assert.equal(r.normalization.restrictedPrecisionDeterminant,'45');assert.equal(r.normalization.logZRatio,'7/30');
});
test('source changes means and raw moments while preserving Gaussian connected covariance',()=>{
 const a=result('localized-source'),b=result('zero-source'),c=result('uniform-source');assert.deepEqual(a.connectedTwo,b.connectedTwo);assert.deepEqual(a.connectedTwo,c.connectedTwo);assert.deepEqual(b.mean,['0','0','0','0']);assert.deepEqual(c.mean,['1','1','1','1']);assert.notEqual(a.rawTwo[0][1],b.rawTwo[0][1]);assert.equal(b.fourPoint.raw,'22/225');
});
test('temperature changes covariance but not susceptibility; physical-source derivative factors are explicit',()=>{
 const a=result('localized-source'),b=result('hotter');assert.deepEqual(a.mean,b.mean);assert.deepEqual(a.susceptibility,b.susceptibility);assert.equal(b.connectedTwo[0][0],'14/15');assert.equal(b.sourceDerivatives.logZGradient[0],'7/30');assert.equal(b.sourceDerivatives.logZHessian[0][0],'7/30');assert.equal(b.normalization.logZRatio,'7/60');
});
test('Wick pair multiplicity survives repeated indices and raw pair subtraction is not the cumulant',()=>{
 const a=result('localized-source');assert.equal(a.fourPoint.raw,'2732/16875');assert.equal(a.fourPoint.centered,'22/225');assert.equal(a.fourPoint.connectedCumulant,'0');assert.equal(a.fourPoint.incorrectRawPairSubtraction,'-28/5625');
 const b=calculateGaussianField({...gaussianFieldDefault,source:['0','0','0','0'],probes:[0,0,0,0]});assert.equal(b.fourPoint.centered,'49/75');assert.equal(b.fourPoint.raw,'49/75');assert.equal(b.fourPoint.connectedCumulant,'0');
});
test('constrained independent sites acquire exact negative covariance without negative coupling',()=>{
 const r=calculateGaussianField({...gaussianFieldDefault,stiffness:'0',constraint:'mean-zero'});assert.deepEqual(r.connectedTwo[0],['3/4','-1/4','-1/4','-1/4']);assert.equal(r.normalization.restrictedPrecisionDeterminant,'1');assert.deepEqual(r.certificates.covarianceRowSums,['0','0','0','0']);
});
test('conditioning a pinned chain differs from subtracting its sample arithmetic mean',()=>{
 const r=result('dirichlet-constrained');assert.equal(r.connectedTwo[0][0],'3/10');assert.equal(r.normalization.restrictedPrecisionDeterminant,'20/3');assert.notEqual(r.connectedTwo[0][0],'11/36');assert.deepEqual(r.connectedTwo[0],['3/10','-1/10','-1/5']);assert.equal(r.integrationDimension,2);
});
test('massless ring requires a declared ensemble; exterior pins supply a different valid inverse',()=>{
 const a=result('massless-periodic-constrained');assert.equal(a.integrationDimension,3);assert.deepEqual(a.connectedTwo[0],['5/16','-1/16','-3/16','-1/16']);assert.equal(a.normalization.restrictedPrecisionDeterminant,'16');assert.equal(a.factorization.reducedDeterminant,'64');
 const b=result('massless-dirichlet');assert.equal(b.integrationDimension,3);assert.equal(b.normalization.restrictedPrecisionDeterminant,'4');assert.deepEqual(b.connectedTwo[0],['3/4','1/2','1/4']);assert.equal(b.certificates.constraintMeanSum,null);
});
test('constant-source shifts leave the declared zero-mean ensemble unchanged for every boundary',()=>{
 for(const boundary of ['open','periodic','dirichlet']as const){const a=calculateGaussianField({...gaussianFieldDefault,boundary,constraint:'mean-zero'}),b=calculateGaussianField({...gaussianFieldDefault,boundary,constraint:'mean-zero',source:['3','2','2','2']});assert.deepEqual(a.mean,b.mean);assert.deepEqual(a.connectedTwo,b.connectedTwo);assert.equal(a.normalization.logZRatio,b.normalization.logZRatio);assert.equal(a.certificates.constraintMeanSum,'0');}
 assert.deepEqual(result('massless-periodic-constrained').mean,result('shifted-constrained-source').mean);
});
test('joint energy/temperature rescaling preserves the whole normalized Gaussian law',()=>{
 for(const constraint of ['none','mean-zero']as const){const a=calculateGaussianField({...gaussianFieldDefault,constraint}),b=calculateGaussianField({...gaussianFieldDefault,constraint,massSquared:'2',stiffness:'2',temperature:'2',source:['2','0','0','0']});assert.deepEqual(a.mean,b.mean);assert.deepEqual(a.connectedTwo,b.connectedTwo);assert.deepEqual(a.rawTwo,b.rawTwo);assert.deepEqual(a.fourPoint,b.fourPoint);assert.equal(a.normalization.logZRatio,b.normalization.logZRatio);const[an,ad]=rational(a.susceptibility[0][0]),[bn,bd]=rational(b.susceptibility[0][0]);assert.equal(an*bd,BigInt(2)*bn*ad);}
});
test('every reference covariance obeys exact pairwise Cauchy–Schwarz and all output fractions remain finite',()=>{
 for(const c of fixture.cases){const r=calculateGaussianField(c.input),C=r.connectedTwo;for(let i=0;i<C.length;i++)for(let j=0;j<C.length;j++){const[n,d]=times(C[i][j],C[i][j]),[m,e]=times(C[i][i],C[j][j]);assert(n*e<=m*d,c.id);assert.equal(C[i][j],C[j][i]);}assert(r.factorization.diagonal.every(s=>rational(s)[0]>BigInt(0)));assert(JSON.stringify(r).length<200000,c.id);}
});
test('equivalent fraction spellings preserve their raw input but agree in computed fields',()=>{
 const a=result('localized-source'),b=result('equivalent-spelling');assert.equal(b.input.massSquared,'2/2');assert.deepEqual(a.mean,b.mean);assert.deepEqual(a.connectedTwo,b.connectedTwo);assert.deepEqual(a.normalization,b.normalization);
});
test('all seventeen named preparations are valid and stable under JSON transport',()=>{
 assert.equal(gaussianFieldExamples.length,17);assert.equal(new Set(gaussianFieldExamples.map(c=>c.id)).size,17);for(const c of gaussianFieldExamples){assert.deepEqual(readGaussianFieldInput(JSON.parse(JSON.stringify(c.input))),c.input);assert.deepEqual(calculateGaussianField(c.input).input,c.input);}
});
test('bilingual manuscripts render strict equations, twenty sections and eight explained exercises',async()=>{
 const id='finite-gaussian-field-sources-connected-correlations';assert(learningUnits.some(u=>String(u.id)===id));assert(learningUnits.length>=595);const content=await getLearningContent(id);for(const language of ['ko','en']as const){const raw=readFileSync(`content/learning/physics-math-for-ai/${id}.${language}.md`,'utf8');assert.equal(content.outline[language].length,20);assert.equal((raw.match(/^\d+\. \*\*/gm)??[]).length,8);assert(!content[language].includes('katex-error'));assert(content[language].includes('3/10')&&content[language].includes('11/36'));assert(!/<h1(?:\s|>)/.test(content[language]));}assert(!/[가-힣]/.test(content.en));
});
