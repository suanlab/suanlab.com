import test from 'node:test';import assert from 'node:assert/strict';import{readFileSync,writeFileSync}from'node:fs';
import{readNongaussianFieldInput,nongaussianFieldDefault,nongaussianFieldExamples}from'../src/lib/learning-nongaussian-field-input';import{calculateNongaussianField}from'../src/lib/learning-nongaussian-field';import{getLearningContent}from'../src/lib/learning-content';import{learningUnits}from'../src/data/interactive-book';import fixture from'./fixtures/nongaussian-field-reference.json';
const example=(id:string)=>nongaussianFieldExamples.find(c=>c.id===id)!.input,result=(id:string)=>calculateNongaussianField(example(id));
const fraction=(s:string)=>{const[n,d='1']=s.split('/');return[BigInt(n),BigInt(d)]as const;},equalProduct=(a:string,b:string,c:string,d:string)=>{const[an,ad]=fraction(a),[bn,bd]=fraction(b),[cn,cd]=fraction(c),[dn,dd]=fraction(d);return an*bn*cd*dd===cn*dn*ad*bd;};
for(const c of fixture.cases)test('independent direct-state/partition reference: '+c.id,()=>{const{input,contract,...actual}=calculateNongaussianField(c.input);assert.deepEqual(input,c.input);assert(contract.includes('four-body interaction acts on sites0,1,2,3 only'));assert.deepEqual(actual,c.expected);});
test('input rejects nonpositive factors,wrong ensembles,invalid fraction syntax and dimensions',()=>{
 const d=nongaussianFieldDefault;for(const patch of[{sites:3},{sites:9},{boundary:'dirichlet'},{pairFactor:'0'},{pairFactor:'-1'},{pairFactor:'1/17'},{pairFactor:'17'},{fourBodyFactor:'0'},{temperature:'0'},{temperature:'17'},{temperature:'1e-2'},{pairFactor:'01'},{pairFactor:'1.0'},{sourceFactors:['1']},{probes:[0,1,2,4]},{probes:[0,1,2,1.5]},{pairFactor:'1000000000000/1000000000000'}])assert.equal(readNongaussianFieldInput({...d,...patch}),null);
 assert(readNongaussianFieldInput({...d,temperature:'1/999999999999'}));assert.throws(()=>calculateNongaussianField({...d,temperature:'0'}),RangeError);
});
test('inputs own frozen vectors and never evaluate getters or inherited metadata',()=>{
 const input={...nongaussianFieldDefault,sourceFactors:['1','1','1','1'],probes:[0,1,2,3]},owned=readNongaussianFieldInput(input);assert(owned);input.sourceFactors[0]='2';input.probes[0]=3;assert.equal(owned.sourceFactors[0],'1');assert.equal(owned.probes[0],0);assert(Object.isFrozen(owned)&&Object.isFrozen(owned.sourceFactors)&&Object.isFrozen(owned.probes));let calls=0;const getter={...nongaussianFieldDefault};Object.defineProperty(getter,'pairFactor',{get(){calls++;return'2';}});assert.equal(readNongaussianFieldInput(getter),null);const source=['1','1','1','1'];Object.defineProperty(source,'0',{get(){calls++;return'1';}});assert.equal(readNongaussianFieldInput({...nongaussianFieldDefault,sourceFactors:source}),null);assert.equal(calls,0);assert.equal(readNongaussianFieldInput(Object.create(nongaussianFieldDefault)),null);assert.equal(readNongaussianFieldInput({...nongaussianFieldDefault,[Symbol('extra')]:1}),null);assert.equal(readNongaussianFieldInput({...nongaussianFieldDefault,sourceFactors:Array(4)}),null);
});
test('all seventeen examples are valid and retain nested raw input through JSON transport',()=>{assert.equal(nongaussianFieldExamples.length,17);for(const c of nongaussianFieldExamples)assert.deepEqual(readNongaussianFieldInput(JSON.parse(JSON.stringify(c.input))),c.input);});
test('equivalent fraction spelling changes neither exact probabilities nor statistics',()=>{
 const a=result('pair-ring'),b=result('equivalent-spelling');assert.notDeepEqual(a.input,b.input);const{input:ai,...av}=a,{input:bi,...bv}=b;assert(ai&&bi);assert.deepEqual(av,bv);
});
test('positive pure four-body family has identical first two moments but distinct fourth cumulants',()=>{
 const a=result('independent'),b=result('pure-four-body'),negative=result('negative-four-body');assert.deepEqual(a.mean,b.mean);assert.deepEqual(a.connectedTwo,b.connectedTwo);assert.equal(a.fourPoint.connectedCumulant,'0');assert.equal(b.fourPoint.connectedCumulant,'3/5');assert.equal(negative.fourPoint.connectedCumulant,'-3/5');assert.equal(b.normalization.partition,'20');assert.deepEqual(new Set(b.states.map(s=>s.probability)),new Set(['1/10','1/40']));
});
test('repeated independent spins retain fourth cumulant minus two and three pairing multiplicities',()=>{
 const r=result('independent-repeated');assert.equal(r.fourPoint.raw,'1');assert.equal(r.fourPoint.centered,'1');assert.deepEqual(r.fourPoint.wickPairProducts,['1','1','1']);assert.equal(r.fourPoint.connectedCumulant,'-2');
});
test('biased repeated spin exposes both third response and wrong raw subtraction',()=>{
 const r=result('biased-repeated');assert.equal(r.mean[0],'3/5');assert.equal(r.connectedTwo[0][0],'16/25');assert.equal(r.threePoint.connectedCumulant,'-96/125');assert.equal(r.fourPoint.centered,'832/625');assert.equal(r.fourPoint.connectedCumulant,'64/625');assert.equal(r.fourPoint.incorrectRawPairSubtraction,'-2');
});
test('global reversal enforces all odd subset moments and pairs state probabilities exactly',()=>{
 for(const id of['pair-ring','pure-four-body','odd-frustration','antiferromagnetic']){const r=result(id);assert.equal(r.certificates.spinFlipSymmetric,true);assert.equal(r.certificates.oddMomentMaximumAbs,'0');for(const s of r.states)assert.equal(s.probability,r.states[r.stateCount-1-s.index].probability);}
});
test('reciprocal local factors reverse means and odd cumulants while preserving even statistics',()=>{
 const a=result('mixed-interactions'),input={...a.input,sourceFactors:a.input.sourceFactors.map(s=>{const[n,d='1']=s.split('/');return d+'/'+n;})},b=calculateNongaussianField(input);const negative=(s:string)=>s==='0'?'0':s.startsWith('-')?s.slice(1):'-'+s;
 assert.deepEqual(b.mean,a.mean.map(negative));assert.deepEqual(b.connectedTwo,a.connectedTwo);assert.equal(b.threePoint.connectedCumulant,negative(a.threePoint.connectedCumulant));assert.equal(b.fourPoint.connectedCumulant,a.fourPoint.connectedCumulant);for(const s of a.states)assert.equal(s.probability,b.states[a.stateCount-1-s.index].probability);
});
test('temperature at fixed factors rescales physical response without changing the distribution',()=>{
 const a=result('pair-ring'),b=result('same-factors-hotter');assert.deepEqual(a.states,b.states);assert.deepEqual(a.connectedTwo,b.connectedTwo);for(let i=0;i<4;i++)for(let j=0;j<4;j++){assert(equalProduct(b.susceptibility[i][j],'2',a.susceptibility[i][j],'1'));assert(equalProduct(b.sourceDerivatives.logZHessian[i][j],'4',a.sourceDerivatives.logZHessian[i][j],'1'));}assert(equalProduct(b.sourceDerivatives.logZFourthSelected,'16',a.sourceDerivatives.logZFourthSelected,'1'));
});
test('fixed-energy heating changes all logarithmic factors consistently rather than holding them fixed',()=>{
 const a=example('fixed-energy-reference'),b=example('hotter-fixed-energy');assert.equal(a.temperature,'1');assert.equal(b.temperature,'2');assert(equalProduct(b.pairFactor,b.pairFactor,a.pairFactor,'1'));assert(equalProduct(b.fourBodyFactor,b.fourBodyFactor,a.fourBodyFactor,'1'));for(let i=0;i<4;i++)assert(equalProduct(b.sourceFactors[i],b.sourceFactors[i],a.sourceFactors[i],'1'));assert.notDeepEqual(calculateNongaussianField(a).states,calculateNongaussianField(b).states);
});
test('default pair ring and open chain agree with elementary hand counts',()=>{
 const a=result('pair-ring'),b=result('open-chain');assert.equal(a.normalization.partition,'353/8');assert.deepEqual(a.connectedTwo[0],['1','255/353','225/353','255/353']);assert.equal(a.fourPoint.connectedCumulant,'-101250/124609');assert.equal(b.normalization.partition,'125/4');assert.deepEqual(b.connectedTwo[0],['1','3/5','9/25','27/125']);
});
test('odd antiferromagnetic ring cannot make all edge products negative',()=>{
 const r=result('odd-frustration');for(const state of r.states){const products=r.edges.map(([i,j])=>state.spins[i]*state.spins[j]);assert.equal(products.reduce((v,p)=>v*p,1),1);assert(products.some(p=>p===1));}assert.equal(Math.min(...r.states.map(s=>r.edges.reduce((v,[i,j])=>v+s.spins[i]*s.spins[j],0))),-3);
});
test('permuting four labelled probe slots preserves raw,centered and connected fourth quantities',()=>{
 for(const c of nongaussianFieldExamples){const a=calculateNongaussianField(c.input),b=calculateNongaussianField({...c.input,probes:[c.input.probes[3],c.input.probes[1],c.input.probes[0],c.input.probes[2]]});for(const key of['raw','centered','connectedCumulant','incorrectRawPairSubtraction']as const)assert.equal(a.fourPoint[key],b.fourPoint[key],c.id);}
});
test('every reference covariance satisfies exact diagonal and pairwise Cauchy-Schwarz bounds',()=>{
 for(const c of fixture.cases){const r=calculateNongaussianField(c.input);assert.equal(r.certificates.probabilitySum,'1');assert.equal(r.certificates.zeroMaskMoment,'1');assert.equal(r.certificates.walshInversionResidual,'0');assert(r.certificates.spinDiagonalResidual.every(v=>v==='0'));for(let i=0;i<r.input.sites;i++)for(let j=0;j<r.input.sites;j++){const[a,b]=fraction(r.connectedTwo[i][j]),[u,v]=fraction(r.connectedTwo[i][i]),[x,y]=fraction(r.connectedTwo[j][j]);assert(a*a*v*y<=u*x*b*b,c.id);}for(const state of r.states)assert(fraction(state.probability)[0]>BigInt(0));}
});
test('maximum output stays bounded and has all256 probabilities and parity moments',()=>{
 const r=result('maximum-coprime'),bytes=Buffer.byteLength(JSON.stringify(r));assert.equal(r.states.length,256);assert.equal(r.parityMoments.length,256);assert(bytes<2000000);writeFileSync('.runtime/learning/nongaussian-field-foundation/maximum.json',JSON.stringify({bytes,states:r.states.length,parityMoments:r.parityMoments.length,probabilitySum:r.certificates.probabilitySum},null,2)+'\n');
});
test('bilingual manuscripts render21sections and8worked questions with strict equations',async()=>{
 const id='interacting-nongaussian-connected-correlations';assert(learningUnits.some(u=>String(u.id)===id));assert(learningUnits.length>=596);const c=await getLearningContent(id),rows=[];for(const language of['ko','en']as const){const raw=readFileSync(`content/learning/physics-math-for-ai/${id}.${language}.md`,'utf8');assert.equal(c.outline[language].length,21);assert.equal((raw.match(/^\d+\. \*\*/gm)??[]).length,8);assert(!c[language].includes('katex-error'));assert(!/<h1(?:\s|>)/.test(c[language]));assert(c[language].includes('64/625')&&c[language].includes('101250/124609'));rows.push({language,sections:c.outline[language].length,displayEquations:(raw.match(/^\$\$/gm)??[]).length/2,characters:raw.length});}assert(!/[가-힣]/.test(c.en));writeFileSync('.runtime/learning/nongaussian-field-foundation/manuscripts.json',JSON.stringify(rows,null,2)+'\n');
});
