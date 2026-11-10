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
import {tensorPureModel as pure,type TensorPureParameters} from '../src/lib/learning-tensor-pure';
import {tensorPureKeys} from '../src/lib/learning-tensor-pure-input';
import {mixedEntanglementModel as mixed,isMixedEntanglementParameters,type MixedEntanglementParameters} from '../src/lib/learning-mixed-entanglement';
import type {PartitionObservation as Q,PartitionRational} from '../src/lib/learning-partition-exact-core';

const ids=['bipartite-pure-schmidt-partial-trace','two-qubit-ppt-correlations-chsh'];
const dir='docs/online-learning/physics-math-for-ai',runtime='.runtime/learning/quantum-tensor-manuscripts';
const metrics:Record<string,{sections:number;explainedExercises:number;tables:number;displayedEquations:number;characters:number;words:number}>= {};
for(const [index,id] of ids.entries()){
 const source=Object.fromEntries(['ko','en'].map(lang=>[lang,readFileSync(`${dir}/${id}-chapter.${lang}.md`,'utf8')]));
 for(const lang of ['ko','en'])test(`tensor manuscript ${id}.${lang}: detailed prose,complete tables and strict math rendering`,async()=>{
  const s=source[lang];assert(!s.includes('{{EQ'));assert(!s.startsWith('# '));assert.equal((s.match(/^## /gm)||[]).length,14);
  assert.equal((s.match(/^\*\*\d\. /gm)||[]).length,6);assert.equal((s.match(/^\*\*(?:Explanation|해설)\./gm)||[]).length,6);
  for(const v of ['Interactive Book','null','768','512','https://www.preskill.caltech.edu/ph219/chap2_15.pdf'])assert(s.includes(v),v);
  if(lang==='en'){assert(!/[가-힣]/.test(s));assert(s.split(/\s+/).length>2500);}else assert((s.match(/[가-힣]/g)||[]).length>4500);
  assert(!/(?<![\\A-Za-z])(?:qquad|quad)\b/.test(s));let tables=0,equations=0;const headings:string[]=[];
  const result=await unified().use(remarkParse).use(remarkGfm).use(remarkMath).use(remarkRehype).use(rehypeSlug).use(rehypeKatex,{strict:'error'}).use(()=>(tree:Root)=>{
   visit(tree,'element',node=>{
    if(node.tagName==='h2'){assert.equal(typeof node.properties.id,'string');headings.push(node.properties.id as string);}
    if(node.tagName==='table'){
     tables++;const rows:Element[]=[];visit(node,'element',row=>{if(row.tagName==='tr')rows.push(row);});assert.equal(rows.length,4);
     const count=index===0?(tables===1?3:4):(tables===1?5:3);
     for(const row of rows)assert.equal(row.children.filter(n=>n.type==='element'&&(n.tagName==='th'||n.tagName==='td')).length,count);
    }
    if(Array.isArray(node.properties.className)&&node.properties.className.includes('katex-display'))equations++;
   });
  }).use(rehypeStringify).process(s);
  assert.equal(result.messages.length,0);assert(!String(result).includes('katex-error'));assert.equal(tables,2);assert.equal(equations,index===0?10:11);assert.equal(new Set(headings).size,14);
  metrics[`${id}.${lang}`]={sections:14,explainedExercises:6,tables,displayedEquations:equations,characters:s.length,words:s.trim().split(/\s+/).length};
  mkdirSync(runtime,{recursive:true});writeFileSync(`${runtime}/${id}.${lang}.html`,String(result));
 });
 test(`tensor manuscript ${id}: translations preserve displayed derivations and exercise sequence`,()=>{
  assert.deepEqual(source.ko.match(/\$\$[\s\S]*?\$\$/g),source.en.match(/\$\$[\s\S]*?\$\$/g));assert.deepEqual(source.ko.match(/^\*\*\d\./gm),source.en.match(/^\*\*\d\./gm));
 });
}
const refs=JSON.parse(readFileSync('scripts/fixtures/tensor-pure-reference.json','utf8'))as{cases:{label:string;input:TensorPureParameters}[]};
const mixedRefs=JSON.parse(readFileSync('scripts/fixtures/mixed-entanglement-reference.json','utf8'))as{cases:{label:string;input:MixedEntanglementParameters}[]};
const P=(label:string)=>pure(refs.cases.find(x=>x.label===label)!.input),M=(label:string)=>mixed(mixedRefs.cases.find(x=>x.label===label)!.input);
function near(actual:number|null,expected:number){assert.notEqual(actual,null);assert(Math.abs(actual!-expected)<1e-12,`${actual} != ${expected}`);}
function fraction(v:PartitionRational){assert.equal(v.numerator.exponent,0);assert.equal(v.denominator.exponent,0);return{n:BigInt(v.numerator.coefficient),d:BigInt(v.denominator.coefficient)};}
function rational(v:Q,n:number,d=1,exact=true){assert(v.bounds);const a=fraction(v.bounds.lower),b=fraction(v.bounds.upper),N=BigInt(n),D=BigInt(d);assert(a.n*D<=N*a.d&&N*b.d<=b.n*D);if(exact){assert.equal(a.n*D,N*a.d);assert.equal(b.n*D,N*b.d);}}
function custom(a:number,b:number,real:number[][],imag:number[][]=[]){const p=Object.fromEntries(tensorPureKeys.map(k=>[k,0]));Object.assign(p,{dimensionA:a,dimensionB:b,aProbeReal0:1,bProbeReal0:1});for(let i=0;i<a;i++)for(let j=0;j<b;j++){p[`stateReal${i}${j}`]=real[i][j];p[`stateImag${i}${j}`]=imag[i]?.[j]??0;}return p;}

test('tensor lecture coordinates distinguish A-major flattening,transpose and the complex Bob Gram sign',()=>{
 const x=P('complex B sign');rational(x.reducedB.density[0][1].imaginary,-1,2);rational(x.coefficientGramB[0][1].imaginary,1,2);rational(x.reducedA.purity,1);assert.equal(x.schmidtRank,1);
 const rectangular=pure(custom(2,3,[[0,0,0],[0,0,1]]));assert.equal(rectangular.jointDimension,6);near(rectangular.normalizedAmplitudes[5].real.value,1);for(let i=0;i<5;i++)rational(rectangular.normalizedAmplitudes[i].real,0);
});
test('product-factor exercise reconstructs the complex rectangular table and its norm30',()=>{
 const x=pure(custom(2,3,[[1,0,2],[2,0,4]],[[0,1,0],[0,2,0]]));assert.equal(x.schmidtRank,1);assert(x.productFactorization);rational(x.rawNormSquared,30);rational(x.productFactorization.normProduct,30);
 const a=x.productFactorization.rawA,b=x.productFactorization.rawB;assert.equal(a.length,2);assert.equal(b.length,3);rational(a[0].real,1);rational(a[1].real,2);rational(b[0].real,1);rational(b[1].imaginary,1);rational(b[2].real,2);rational(x.reducedA.purity,1);rational(x.reducedB.purity,1);
});
test('qutrit and ququart table entropies agree with their explicit nonbinary probability laws',()=>{
 const q=P('rank three diagonal'),v=P('rank four diagonal');assert.equal(q.schmidtRank,3);assert.equal(v.schmidtRank,4);
 for(const [x,weights,denominator,bits,nats]of [[q,[1,4,9],14,1.1981174211304033,.8304717124362917],[v,[1,4,9,16],30,1.5559130951758249,1.078476775117421]] as const){
  weights.forEach((n,i)=>rational(x.schmidtSpectrum[i].weight,n,denominator,false));near(x.entropyBits.value,bits);near(x.entropyNats.value,nats);near(x.entropyNats.value,-weights.reduce((s,n)=>s+n/denominator*Math.log(n/denominator),0));
 }
 rational(q.reducedA.purity,1,2);rational(v.reducedA.purity,59,150);rational(P('Bell plus').reducedA.purity,1,2);assert.notEqual(q.schmidtRank,P('Bell plus').schmidtRank);
});
test('qutrit coarse-complement exercise has probability2/3,mixed purity1/2 and remote no-signalling',()=>{
 const x=pure(custom(3,3,[[1,0,0],[0,1,0],[0,0,1]])),conditional=x.projectorTests.conditionBOnA.rows;
 rational(conditional[0].probability,1,3);rational(conditional[0].conditionalPurity!,1);rational(conditional[1].probability,2,3);rational(conditional[1].conditionalPurity!,1,2);
 const density=conditional[1].conditionalDensity!;rational(density[0][0].real,0);rational(density[1][1].real,1,2);rational(density[2][2].real,1,2);assert.deepEqual(x.projectorTests.conditionBOnA.nonselectiveDensity,x.reducedB.density);
});
test('qutrit partial-transpose exercise counts six positive and three negative terms with negativity1',()=>{
 const x=P('3x3 maximal Schmidt');assert.equal(x.partialTranspose.positiveWeights.length+x.partialTranspose.pairs.length,6);assert.equal(x.partialTranspose.pairs.length,3);assert.equal(x.partialTranspose.zeroMultiplicity,0);rational(x.partialTranspose.negativity,1,1,false);near(x.partialTranspose.logNegativityBits.value,Math.log2(3));
});
test('Schmidt truncation example separates discarded mass from normalized-state squared overlap',()=>{
 const x=P('rank three diagonal'),weights=x.schmidtSpectrum.map(s=>s.weight.value!);near(weights[0],1/14);near(weights[1]+weights[2],13/14);
 // Keep diagonal coefficients2,3 and normalize them by sqrt13.
 const overlap=2/Math.sqrt(14)*2/Math.sqrt(13)+3/Math.sqrt(14)*3/Math.sqrt(13);near(overlap*overlap,13/14);near(1-overlap*overlap,1/14);
});
test('local-source comparison table distinguishes equal marginals,joint coherence and entropy',()=>{
 const bell=M('Bell plus poor settings'),classical=M('classical correlated'),independent=M('independent maximally mixed');
 for(const x of [classical,independent]){assert.deepEqual(x.reducedA.density,bell.reducedA.density);assert.deepEqual(x.reducedB.density,bell.reducedB.density);}
 rational(bell.joint.purity,1);rational(classical.joint.purity,1,2);rational(independent.joint.purity,1,4);near(bell.mutualInformationBits.value,2);near(classical.mutualInformationBits.value,1);rational(independent.mutualInformationBits,0,1,false);
 rational(bell.correlationTensor[0][0],1);rational(classical.correlationTensor[0][0],0);rational(bell.correlationTensor[2][2],1);rational(classical.correlationTensor[2][2],1);near(bell.chsh.optimizedMaximum.value,2*Math.sqrt(2));near(classical.chsh.optimizedMaximum.value,2);rational(independent.chsh.optimizedMaximum,0);
});
test('leading-principal-minor counterexample is rejected rather than repaired by normalization',()=>{
 const p={...mixedRefs.cases.find(x=>x.label==='pure product zero outcomes')!.input,hReal00:1,hReal11:0,hReal22:-1,hReal33:1};assert(!isMixedEntanglementParameters(p));assert.throws(()=>mixed(p),RangeError);
});
test('Bell measurement exercise returns four exact correlations and a selected82/29 distinct from optimum',()=>{
 const x=M('Bell plus violating settings');for(const[i,j,n]of [[0,0,21],[0,1,21],[1,0,20],[1,1,-20]])rational(x.localTests.pairs[i][j].correlation,n,29);
 rational(x.chsh.selectedSigned,82,29);assert(x.chsh.selectedViolates);near(x.chsh.optimizedMaximum.value,2*Math.sqrt(2));assert(x.chsh.optimizedMaximum.value!>x.chsh.selectedAbsolute.value!);assert(!M('Bell plus poor settings').chsh.selectedViolates);
});
test('Werner exercise and rational threshold neighbours separate entanglement from CHSH violation',()=>{
 const half=M('Werner 1/2');assert(half.classification.entangledTwoQubit);assert(!half.chsh.optimizedViolates);rational(half.partialTranspose.spectrum[0].eigenvalue,-1,8,false);rational(half.partialTranspose.negativity,1,8,false);near(half.chsh.optimizedMaximum.value,Math.sqrt(2));
 assert(M('Werner 1/3').classification.separableTwoQubit);assert(!M('Werner 353553/500000').chsh.optimizedViolates);assert(M('Werner 353554/500000').chsh.optimizedViolates);
});
test('full-rank exact boundary exercise matches Bell weights,partial-transpose eigenvalues and rational K',()=>{
 const x=M('entangled exact CHSH boundary');assert.equal(x.joint.rank,4);for(const[n,i]of [[1,0],[3,1],[7,2],[29,3]])rational(x.joint.spectrum[i].eigenvalue,n,40,false);
 for(const[n,i]of [[-9,0],[13,1],[17,2],[19,3]])rational(x.partialTranspose.spectrum[i].eigenvalue,n,40,false);rational(x.partialTranspose.negativity,9,40,false);
 const cert=x.chsh.exactBoundaryCertificate;rational(cert.threshold,1,4);rational(cert.shiftedMatrix[0][0].real,11,100);rational(cert.shiftedMatrix[1][1].real,39,100);rational(cert.shiftedMatrix[2][2].real,0);rational(cert.determinant,0);assert(cert.positiveSemidefinite);assert(x.chsh.optimizedAtBoundary);rational(x.chsh.topTwoEigenvalueSum,1,1,false);near(x.chsh.optimizedMaximum.value,2);
 for(const delta of [-1,1]){const y=M(`CHSH boundary perturbation ${delta}`);assert.equal(y.joint.rank,4);rational(y.correlationTensor[0][0],15000+delta,25000);assert.equal(y.chsh.optimizedComparisonToLocalBound,delta);}
});
test('GHZ and W extension distinguishes local spectra and the discarded-party joint state',()=>{
 const ghz=Array(8).fill(0),w=Array(8).fill(0);ghz[0]=ghz[7]=1/Math.sqrt(2);w[1]=w[2]=w[4]=1/Math.sqrt(3);
 const traceLast=(z:number[])=>Array.from({length:4},(_,i)=>Array.from({length:4},(_,j)=>z[2*i]*z[2*j]+z[2*i+1]*z[2*j+1]));
 const G=traceLast(ghz),W=traceLast(w);near(G[0][0],.5);near(G[3][3],.5);near(G[1][2],0);near(W[0][0],1/3);near(W[1][2],1/3);near(W[3][3],0);
 near(G[0][0]+G[1][1],.5);near(W[0][0]+W[1][1],2/3);assert.notDeepEqual(G,W);
});
test('manuscript preparation records rendering evidence without claiming registered readers',()=>{
 assert.equal(Object.keys(metrics).length,4);mkdirSync(runtime,{recursive:true});writeFileSync(`${runtime}/rendering.json`,JSON.stringify(metrics,null,2)+'\n');
});
