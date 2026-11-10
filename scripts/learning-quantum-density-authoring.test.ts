import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {unified} from 'unified';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import remarkRehype from 'remark-rehype';
import rehypeKatex from 'rehype-katex';
import rehypeStringify from 'rehype-stringify';
import {visit} from 'unist-util-visit';
import type {Root,Element} from 'hast';
import {densityEnsembleModel as ensemble,type DensityEnsembleParameters} from '../src/lib/learning-density-ensemble';
import {densitySpectrumModel as spectrum,type DensitySpectrumParameters} from '../src/lib/learning-density-spectrum';

const ids=['finite-ensemble-density-gram-tomography','qubit-density-spectrum-entropy-dephasing'];
for(const [index,id] of ids.entries()){
  const source=Object.fromEntries(['ko','en'].map(lang=>[lang,readFileSync(`docs/online-learning/physics-math-for-ai/${id}-chapter.${lang}.md`,'utf8')]));
  for(const lang of ['ko','en'])test(`density manuscript ${id}.${lang}: detailed prose, equations and intact tables`,async()=>{
    const s=source[lang];
    assert.equal((s.match(/^## /gm)||[]).length,14);
    assert.equal((s.match(/^\*\*\d\. /gm)||[]).length,8);
    assert.equal((s.match(/^\*\*(?:Explanation|해설)\./gm)||[]).length,8);
    assert(!s.startsWith('# '));assert(!s.includes('{{EQ'));
    for(const text of ['Interactive Book','null','768','512','https://www.preskill.caltech.edu/ph219/chap2_15.pdf'])assert(s.includes(text),text);
    // Math control words accidentally missing their slash are valid text to KaTeX.
    assert(!/(?<![\\A-Za-z])(?:qquad|quad)\b/.test(s));
    if(lang==='en'){assert(!/[가-힣]/.test(s));assert(s.split(/\s+/).length>1800);}else assert(s.length>8000);
    let tables=0,equations=0;
    const rendered=await unified().use(remarkParse).use(remarkGfm).use(remarkMath).use(remarkRehype).use(rehypeKatex,{strict:'error'}).use(()=>(tree:Root)=>{
      visit(tree,'element',node=>{
        if(node.tagName==='table'){
          tables++;const rows:Element[]=[];
          visit(node,'element',n=>{if(n.tagName==='tr')rows.push(n);});
          assert(rows.length>=4);const expected=index===1&&tables===2?6:4;
          for(const row of rows)assert.equal(row.children.filter(n=>n.type==='element'&&(n.tagName==='th'||n.tagName==='td')).length,expected);
        }
        if(Array.isArray(node.properties.className)&&node.properties.className.includes('katex-display'))equations++;
      });
    }).use(rehypeStringify).process(s);
    assert.equal(rendered.messages.length,0);assert(!String(rendered).includes('katex-error'));
    assert.equal(tables,2);assert.equal(equations,12);
  });
  test(`density manuscript ${id}: translations preserve displayed mathematics and exercise order`,()=>{
    assert.deepEqual(source.ko.match(/\$\$[\s\S]*?\$\$/g),source.en.match(/\$\$[\s\S]*?\$\$/g));
    assert.deepEqual(source.ko.match(/^\*\*\d\./gm),source.en.match(/^\*\*\d\./gm));
    for(const s of Object.values(source))for(const text of index===0?['19/25','29/50','9/50','21/25','6/25','Gram','Lüders']:['h₂(1/4)','0.811278','Choi','purification','ln2'])assert(s.replace(/\\frac\{(\d+)\}\{(\d+)\}/g,'$1/$2').toLowerCase().includes(text.toLowerCase()),text);
  });
}
const ensembleRefs=JSON.parse(readFileSync('scripts/fixtures/density-ensemble-reference.json','utf8')) as {cases:{label:string;input:DensityEnsembleParameters}[]};
const spectrumRefs=JSON.parse(readFileSync('scripts/fixtures/density-spectrum-reference.json','utf8')) as {cases:{label:string;input:DensitySpectrumParameters}[]};
const E=(label:string)=>ensemble(ensembleRefs.cases.find(c=>c.label===label)!.input);
const S=(label:string)=>spectrum(spectrumRefs.cases.find(c=>c.label===label)!.input);
function near(actual:number|null,expected:number){assert.notEqual(actual,null);assert(Math.abs(actual!-expected)<1e-14,`${actual} != ${expected}`);}

test('unequal lecture mixture: state purity, pinching loss and total variance are different accounts',()=>{
  const x=E('unequal nonorthogonal mixture');
  assert.deepEqual(x.density.map(row=>row.map(z=>z.real.value)),[[.7,.3],[.3,.3]]);
  near(x.purity.value,19/25);near(x.standardBasisDephasing.purity.value,29/50);near(x.standardBasisDephasing.squaredOffDiagonalCoherence.value,9/50);
  near(x.observable.mean.value,2/5);near(x.observable.variance.value,21/25);near(x.observable.withinEnsembleVariance.value,3/5);near(x.observable.betweenEnsembleMeanVariance.value,6/25);
  const eigenvalues=[(5-Math.sqrt(13))/10,(5+Math.sqrt(13))/10];near(eigenvalues[0]*eigenvalues[1],3/25);near(eigenvalues.reduce((s,v)=>s+v*v,0),x.purity.value!);
});
test('three maximally mixed preparations share rho and total variance but not its label-based split',()=>{
  const z=E('Z maximally mixed');
  for(const label of ['X maximally mixed','Y maximally mixed']){const x=E(label);assert.deepEqual(x.density,z.density);near(x.observable.variance.value,1);near(x.observable.withinEnsembleVariance.value,1);near(x.observable.betweenEnsembleMeanVariance.value,0);}
  near(z.observable.withinEnsembleVariance.value,0);near(z.observable.betweenEnsembleMeanVariance.value,1);
});
test('tomography lecture signs recover the positive Y state and include the qutrit outside outcome',()=>{
  const y=E('pure positive Y');near(y.density[0][1].imaginary.value,-.5);near(y.tomography.pairs[0].probabilities.yPlus.value,1);near(y.tomography.pairs[0].probabilities.yMinus.value,0);assert.deepEqual(y.tomography.reconstructedDensity,y.density);
  for(const row of E('dimension 3 uniform orthogonal').tomography.pairs){near(row.probabilities.outside.value,1/3);near(row.probabilities.xPlus.value!+row.probabilities.xMinus.value!+row.probabilities.outside.value!,1);}
});
test('Lüders lecture: a degenerate outcome can remain mixed and an impossible conditional state stays undefined',()=>{
  const x=E('degenerate mixed conditional'),negative=x.observable.outcomes.find(o=>o.eigenvalue.value===-1)!;
  assert.equal(negative.rank,2);near(negative.probability.value,2/3);near(negative.conditionalPurity!.value,.5);
  assert.deepEqual(negative.conditionalDensity!.map(row=>row.map(z=>z.real.value)),[[0,0,0],[0,.5,0],[0,0,.5]]);
  const impossible=E('zero outcome').observable.outcomes.find(o=>o.probability.sign===0)!;assert(impossible);assert.equal(impossible.conditionalDensity,null);assert.equal(impossible.conditionalPurity,null);
});
test('Gram and density preserve rank under near-parallel components and arbitrary ray signs',()=>{
  const x=E('near parallel rank two');assert.equal(x.rank,2);assert.equal(x.gramRank,2);near(x.density[1][1].real.value,1/1000000000001);assert.equal(x.density[1][1].real.sign,1);
  const a=E('duplicate opposite vectors');assert.equal(a.rank,1);near(a.purity.value,1);assert(a.classification.allOccupiedComponentsSameRay);
  const q=E('dimension 2 uniform orthogonal');assert.equal(q.characteristicPolynomial.density.length,3);
  const g=q.characteristicPolynomial.gram;assert.equal(g.length,5);assert.deepEqual(g.slice(0,3),q.characteristicPolynomial.density);assert.equal(g[3].sign,0);assert.equal(g[4].sign,0);
});
test('signed dephasing lecture table: entropy returns to zero while state distinguishability increases',()=>{
  const p=spectrumRefs.cases.find(c=>c.label==='phase flip endpoint')!.input,x=spectrum({...p,sampleSteps:4});
  const eta=[1,.5,0,-.5,-1],purity=[1,5/8,.5,5/8,1],fidelity=[1,.75,.5,.25,0],distance=[0,.25,.5,.75,1];
  const h=-(.25*Math.log2(.25)+.75*Math.log2(.75)),entropy=[0,h,1,h,0];assert.equal(x.records.length,5);
  x.records.forEach((r,i)=>{near(r.eta.value,eta[i]);near(r.bloch[0].value,eta[i]);near(r.purity.value,purity[i]);near(r.entropyBits.value,entropy[i]);near(r.squaredUhlmannFidelityToInitial.value,fidelity[i]);near(r.traceDistanceToInitial.value,distance[i]);});
  assert.equal(x.classification.parameterPathIsTimeEvolution,false);assert.equal(x.classification.continuousParameterPathEntropyNondecreasing,false);
});
test('spectral lecture: exact zero, tiny positive eigenvalues and basis coherence have distinct meanings',()=>{
  const x=S('X measurement basis').records[0];near(x.entropyNats.value,0);near(x.standardBasisCoherenceNats.value,Math.log(2));near(x.probeBasis.coherenceNats.value,0);assert.equal(x.spectrum[0].logEigenvalue,null);near(x.spectrum[0].entropyTermNats.value,0);
  const mixed=S('maximally mixed').records[0];assert.equal(mixed.spectrum.length,1);assert.equal(mixed.spectrum[0].rank,2);assert.equal(mixed.spectralDirection,null);near(mixed.squaredUhlmannFidelityToInitial.value,1);
  const tiny=S('minimal positive determinant').records[0];assert.equal(tiny.spectrum[0].eigenvalue.sign,1);assert.notEqual(tiny.spectrum[0].logEigenvalue,null);assert(tiny.entropyNats.value!>0);
});

// Independent small real matrices verify analytic extensions of the manuscripts.
// These helpers are tests, not a claim that either calculator accepts joint registers.
type R=number[][];
const transpose=(a:R):R=>a[0].map((_,j)=>a.map(row=>row[j]));
const times=(a:R,b:R):R=>a.map(row=>b[0].map((_,j)=>row.reduce((s,v,k)=>s+v*b[k][j],0)));
const tr=(a:R)=>a.reduce((s,row,i)=>s+row[i],0);
const outer=(v:number[]):R=>v.map(a=>v.map(b=>a*b));
const partial=(a:R):R=>[0,1].map(i=>[0,1].map(j=>[0,1].reduce((s,k)=>s+a[2*i+k][2*j+k],0)));
const compare=(a:R,b:R)=>{assert.equal(a.length,b.length);a.forEach((row,i)=>{assert.equal(row.length,b[i].length);row.forEach((v,j)=>near(v,b[i][j]));});};

test('invalid-density lecture: Hermiticity and trace one alone do not ensure positive probabilities',()=>{
  const a=[[.5,.75],[.75,.5]],minus=[1/Math.sqrt(2),-1/Math.sqrt(2)];near(tr(a),1);compare(a,transpose(a));near(times([minus],times(a,transpose([minus])))[0][0],-.25);
  // All leading principal determinants of diag(0,-1,2) vanish, while a singleton is negative.
  const leading=[0,0*(-1),0*(-1)*2];assert(leading.every(v=>v===0));assert.equal(-1<0,true);
});
test('Bell reduction and ensemble purification recover local mixing without identifying the joint state',()=>{
  const bell=outer([Math.SQRT1_2,0,0,Math.SQRT1_2]),classical=[[.5,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,.5]];
  const local=[[.5,0],[0,.5]];compare(partial(bell),local);compare(partial(classical),local);near(tr(times(bell,bell)),1);near(tr(times(classical,classical)),.5);near(tr(times(local,local)),.5);
  // sqrt(2/5)|0>|0> + sqrt(3/5)|+>|1>, order A tensor environment.
  const purified=outer([Math.sqrt(2/5),Math.sqrt(3/10),0,Math.sqrt(3/10)]);
  near(tr(purified),1);near(tr(times(purified,purified)),1);compare(partial(purified),[[.7,.3],[.3,.3]]);
  compare(partial(purified),E('unequal nonorthogonal mixture').density.map(row=>row.map(z=>z.real.value!)));
});
test('analytic reset is trace preserving but not unital and can lower entropy',()=>{
  const k=[[[1,0],[0,0]],[[0,1],[0,0]]],add=(a:R,b:R):R=>a.map((row,i)=>row.map((v,j)=>v+b[i][j]));
  compare(k.map(v=>times(transpose(v),v)).reduce(add),[[1,0],[0,1]]);
  compare(k.map(v=>times(v,transpose(v))).reduce(add),[[2,0],[0,0]]);
  for(const rho of [[[.5,0],[0,.5]],[[.7,.3],[.3,.3]],[[.5,.5],[.5,.5]]]){const result=k.map(v=>times(times(v,rho),transpose(v))).reduce(add);compare(result,[[1,0],[0,0]]);near(tr(result),1);near(tr(times(result,result)),1);}
  const entropy=(p:number[])=>-p.reduce((s,v)=>s+(v===0?0:v*Math.log(v)),0);near(entropy([.5,.5]),Math.log(2));near(entropy([1,0]),0);
});
