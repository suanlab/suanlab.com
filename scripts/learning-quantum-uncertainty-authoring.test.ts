import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync,mkdirSync,writeFileSync} from 'node:fs';
import {unified} from 'unified';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import remarkRehype from 'remark-rehype';
import rehypeKatex from 'rehype-katex';
import rehypeStringify from 'rehype-stringify';
import {visit} from 'unist-util-visit';
import type {Root,Element} from 'hast';
import {uncertaintyCovarianceModel as covariance,type UncertaintyCovarianceParameters} from '../src/lib/learning-uncertainty-covariance';
import {povmInstrumentModel as instrument,type PovmInstrumentParameters} from '../src/lib/learning-povm-instrument';
const ids=['finite-state-covariance-uncertainty','qubit-povm-instrument-backaction'];
const metrics:Record<string,unknown>={};
for(const [index,id]of ids.entries()){
 const source=Object.fromEntries(['ko','en'].map(lang=>[lang,readFileSync(`docs/online-learning/physics-math-for-ai/${id}-chapter.${lang}.md`,'utf8')]));
 for(const lang of ['ko','en'])test(`uncertainty manuscript ${id}.${lang}: readable bilingual source and valid mathematics`,async()=>{
  const s=source[lang];assert(!s.includes('{{EQ'));assert(!s.startsWith('# '));assert.equal((s.match(/^## /gm)||[]).length,12);assert.equal((s.match(/^\*\*\d\. /gm)||[]).length,6);assert.equal((s.match(/^\*\*(?:Explanation|해설)\./gm)||[]).length,6);
  for(const required of ['Interactive Book','null','768','512','https://arxiv.org/abs/quant-ph/0510083'])assert(s.includes(required),required);
  if(lang==='en'){assert(!/[가-힣]/.test(s));assert(s.split(/\s+/).length>1800);}else assert(s.length>7000);
  assert(!/(?<![\\A-Za-z])(?:qquad|quad)\b/.test(s));
  let tables=0,equations=0;
  const result=await unified().use(remarkParse).use(remarkGfm).use(remarkMath).use(remarkRehype).use(rehypeKatex,{strict:'error'}).use(()=>(tree:Root)=>{
   visit(tree,'element',node=>{
    if(node.tagName==='table'){tables++;const rows:Element[]=[];visit(node,'element',n=>{if(n.tagName==='tr')rows.push(n);});assert(rows.length>=4);const count=index===0&&tables===2?3:4;for(const row of rows)assert.equal(row.children.filter(n=>n.type==='element'&&(n.tagName==='th'||n.tagName==='td')).length,count);}
    if(Array.isArray(node.properties.className)&&node.properties.className.includes('katex-display'))equations++;
   });
  }).use(rehypeStringify).process(s);
  assert.equal(result.messages.length,0);assert(!String(result).includes('katex-error'));assert.equal(tables,2);assert.equal(equations,index===0?8:9);
  metrics[`${id}.${lang}`]={sections:12,explainedExercises:6,tables,displayedEquations:equations,characters:s.length,words:s.trim().split(/\s+/).length};
 });
 test(`uncertainty manuscript ${id}: translations keep identical displayed mathematics and exercise sequence`,()=>{
  assert.deepEqual(source.ko.match(/\$\$[\s\S]*?\$\$/g),source.en.match(/\$\$[\s\S]*?\$\$/g));assert.deepEqual(source.ko.match(/^\*\*\d\./gm),source.en.match(/^\*\*\d\./gm));
 });
}
const refs=JSON.parse(readFileSync('scripts/fixtures/uncertainty-covariance-reference.json','utf8'))as{cases:{label:string;input:UncertaintyCovarianceParameters}[]};
const deviceRefs=JSON.parse(readFileSync('scripts/fixtures/povm-instrument-reference.json','utf8'))as{cases:{label:string;input:PovmInstrumentParameters}[]};
const C=(label:string)=>covariance(refs.cases.find(x=>x.label===label)!.input);
const P=(label:string)=>instrument(deviceRefs.cases.find(x=>x.label===label)!.input);
function near(v:number|null,expected:number){assert.notEqual(v,null);assert(Math.abs(v!-expected)<1e-13,`${v} != ${expected}`);}
test('covariance lecture: a strong saturation and vanishing weak bound coexist',()=>{
 const x=C('pure XY covariance saturation');near(x.moments.varianceA.value,16/25);near(x.moments.varianceB.value,9/25);near(x.moments.symmetricCovariance.value,-12/25);near(x.moments.commutatorHalfOverI.value,0);near(x.bounds.schrodingerSquaredBound.value,144/625);near(x.bounds.robertsonSlack.value,144/625);assert.equal(x.bounds.schrodingerSlack.sign,0);near(x.bounds.stdProduct.value,12/25);
});
test('covariance lecture: pure higher dimension and noncommuting zero expectation are distinct counterexamples',()=>{
 const q=C('qutrit pure strict');assert(q.classification.pureState);near(q.moments.varianceA.value,1);near(q.moments.varianceB.value,1);near(q.bounds.schrodingerSlack.value,1);near(q.moments.centeredCross.real.value,0);
 const m=C('mixed Pauli XY');assert(!m.classification.operatorsCommute);assert.equal(m.moments.commutatorHalfOverI.sign,0);near(m.bounds.schrodingerSlack.value,1);
});
test('projection lecture: a zero fluctuation keeps a nonzero residual and null slope',()=>{
 const x=C('zero variance A nonzero B');near(x.moments.varianceA.value,0);near(x.moments.varianceB.value,1);assert.equal(x.complexProjection.coefficientBOnA,null);assert(!x.complexProjection.coefficientUnique);near(x.complexProjection.minimumResidualVariance.value,1);assert.equal(x.bounds.schrodingerSlack.sign,0);
});
test('ensemble lecture: state covariance agrees while the split changes with its preparation labels',()=>{
 const z=C('Z ensemble maximally mixed'),x=C('X ensemble maximally mixed');assert.deepEqual(z.density,x.density);assert.deepEqual(z.quantumCovariance,x.quantumCovariance);
 // These references use X,Y; compare the X variance to expose the ensemble dependence.
 near(z.varianceDecomposition.withinA.value,1);near(z.varianceDecomposition.betweenA.value,0);near(x.varianceDecomposition.withinA.value,0);near(x.varianceDecomposition.betweenA.value,1);
});
test('Luders lecture table: the same probabilities hide different channel outputs',()=>{
 const entries=[['positive Y strength 0/1',1,0],['positive Y strength 3/5',.8,.4],['positive Y strength 1/1',0,2]]as const;
 for(const[label,y,eta]of entries){const x=P(label);assert.deepEqual(x.effectProbabilities.map(v=>v.value),[.5,.5]);assert.deepEqual(x.instruments[0].nonselectiveBloch.map(v=>v.observation.value),[0,y,0]);assert.deepEqual(x.instruments[2].nonselectiveBloch.map(v=>v.observation.value),[0,0,0]);near(x.instruments[0].disturbance.rmsSquared.observation.value,eta);}
});
test('meter lecture: mean calibration changes the variance and cannot fabricate information at zero strength',()=>{
 const x=P('positive Z strength 3/5');near(x.preparation.varianceA.value,0);near(x.rawMeter.variance.value,16/25);near(x.rawMeter.rmsErrorSquared.value,4/5);near(x.calibratedMeter!.variance.value,16/9);near(x.calibratedMeter!.rmsErrorSquared.value,16/9);near(x.calibratedMeter!.mean.value,1);assert.equal(P('positive Y strength 0/1').calibratedMeter,null);
});
test('repeatability lecture: a sharp followup and the same unsharp detector give different probabilities',()=>{
 const x=P('positive Y strength 3/5'),n=x.measurementAxis.map(v=>v.value!),r=x.instruments[2].outcomes[1].conditionalBloch!.map(v=>v.observation.value!),dot=n.reduce((s,v,i)=>s+v*r[i],0);near((1+dot)/2,1);near((1+.6*dot)/2,.8);
 const f=P('sharp flip feedback').instruments[1];near(f.selectedInputSharpARepeatLeak.observation.value,1);near(f.disturbance.rmsSquared.observation.value,4);near(f.traceDistanceToInitial.value,1);
});
test('RMS lecture: fixed state, channel rank and corrected bound are not interchangeable',()=>{
 const fixed=P('maximally mixed strength 1/1').instruments[0];near(fixed.traceDistanceToInitial.value,0);near(fixed.disturbance.rmsSquared.observation.value,2);
 const zero=P('positive Y strength 0/1');assert.equal(zero.instruments[0].channel.choiRank,1);assert.equal(zero.instruments[2].channel.choiRank,4);
 const sharp=P('positive Y strength 1/1');near(sharp.preparation.commutatorMagnitudeHalf.value,1);near(sharp.rawMeter.rmsErrorSquared.value,0);assert(sharp.instruments[0].rawOzawa.naiveProductBoundViolated);near(sharp.instruments[0].rawOzawa.correctedLeft.value,Math.SQRT2);
});
test('Gaussian extension: normalization and second moments give the stated minimal product',()=>{
 // Independent quadrature of the density and analytic derivative magnitude; no finite canonical matrices.
 const a=.7,hbar=1.3,dx=a/1000;let mass=0,x2=0,p2=0;for(let i=-12000;i<=12000;i++){const x=i*dx,w=Math.exp(-x*x/(2*a*a))/(Math.sqrt(2*Math.PI)*a),weight=i===-12000||i===12000?.5:1;mass+=weight*w*dx;x2+=weight*x*x*w*dx;p2+=weight*hbar*hbar*x*x/(4*a**4)*w*dx;}
 assert(Math.abs(mass-1)<1e-12);assert(Math.abs(x2-a*a)<1e-12);assert(Math.abs(p2-hbar*hbar/(4*a*a))<1e-12);assert(Math.abs(Math.sqrt(x2*p2)-hbar/2)<1e-12);
});
test('finite canonical obstruction: cyclic traces vanish even for noncommuting matrices',()=>{
 const a=[[1,2],[2,-1]],b=[[0,1],[1,3]],mul=(a:number[][],b:number[][])=>a.map(row=>b[0].map((_,j)=>row.reduce((s,v,k)=>s+v*b[k][j],0))),ab=mul(a,b),ba=mul(b,a),comm=ab.map((row,i)=>row.map((v,j)=>v-ba[i][j]));assert.notDeepEqual(ab,ba);assert.equal(comm[0][0]+comm[1][1],0);assert.notEqual(2,0);
});
test('manuscript rendering evidence is recorded without registering new readers',()=>{
 assert.equal(Object.keys(metrics).length,4);mkdirSync('.runtime/learning/quantum-uncertainty-manuscripts',{recursive:true});writeFileSync('.runtime/learning/quantum-uncertainty-manuscripts/rendering.json',JSON.stringify(metrics,null,2)+'\n');
});
