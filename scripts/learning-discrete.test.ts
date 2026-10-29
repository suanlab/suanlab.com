import test from 'node:test';
import assert from 'node:assert/strict';
import { choose, countSelections, discreteKinds, evaluateDiscrete as evaluate, graphAnalysis, shortestPath, directedPairs } from '../src/lib/learning-discrete';
import { discreteLabs, learningGuides, learningUnits } from '../src/data/interactive-book';

function words(n:number,k:number):number[][] { if(k===0)return [[]];return words(n,k-1).flatMap(prefix=>Array.from({length:n},(_,v)=>[...prefix,v])); }
test('all four counting rules match independent small enumerations, including empty choices',()=>{
 for(let n=0;n<=4;n++)for(let k=0;k<=5;k++) {
  const all=words(n,k),distinct=all.filter(xs=>new Set(xs).size===xs.length);
  assert.equal(countSelections(n,k,0),distinct.length);
  assert.equal(countSelections(n,k,1),new Set(distinct.map(xs=>[...xs].sort().join(','))).size);
  assert.equal(countSelections(n,k,2),all.length);
  assert.equal(countSelections(n,k,3),new Set(all.map(xs=>[...xs].sort().join(','))).size);
 }
 assert.equal(countSelections(8,8,3),6435);assert.equal(countSelections(8,8,2),16777216);
});
test('binomial identities and every pair of six-element subsets match element enumeration',()=>{
 for(let n=0;n<=12;n++){assert.equal(Array.from({length:n+1},(_,k)=>choose(n,k)).reduce((s,v)=>s+v,0),2**n);for(let k=1;k<n;k++)assert.equal(choose(n,k),choose(n-1,k-1)+choose(n-1,k));}
 for(let a=0;a<64;a++)for(let b=0;b<64;b++){
  const A=new Set([0,1,2,3,4,5].filter(i=>a&(1<<i))),B=new Set([0,1,2,3,4,5].filter(i=>b&(1<<i))),r=evaluate('binomial-inclusion-exclusion',[a,b,3]);
  assert.equal(r.stats['|A∪B|'],new Set([...A,...B]).size);assert.equal(r.stats['|A∩B|'],[...A].filter(x=>B.has(x)).length);
 }
});
test('generating functions preserve coefficient identities and distinguish rational continuation',()=>{
 const r=evaluate('recurrences-generating-functions',[1,1,8,.5]);assert.equal(r.stats.a_N,21);assert.equal(r.stats['Σ₀ᴺ a_n x^n'],1.65234375);assert.equal(r.stats['x/(1−px−qx²)'],2);
 const outside=evaluate('recurrences-generating-functions',[1,1,8,.8]);assert.equal(outside.stats['1{series converges at x}'],0);assert.ok((outside.stats['x/(1−px−qx²)'] as number)<0);assert.ok((outside.stats['Σ₀ᴺ a_n x^n'] as number)>0);
 const pole=evaluate('recurrences-generating-functions',[2,0,8,.5]);assert.equal(pole.stats['x/(1−px−qx²)'],null);assert.equal(pole.stats['1{series converges at x}'],0);
 const polynomial=evaluate('recurrences-generating-functions',[0,0,8,.8]);assert.equal(polynomial.stats['Σ₀ᴺ a_n x^n'],.8);assert.equal(polynomial.stats['radius R'],null);assert.equal(polynomial.stats['1{series converges at x}'],1);
 for(let p=0;p<=2;p++)for(let q=0;q<=2;q++){
  const terms=evaluate('recurrences-generating-functions',[p,q,12,.1]).curves[0].map(v=>v[1]);assert.deepEqual(terms.slice(0,2),[0,1]);for(let n=2;n<=12;n++)assert.equal(terms[n]-p*terms[n-1]-q*terms[n-2],0);
 }
});
function allSimplePaths(mask:number,s:number,t:number):number[][] {
 const paths:number[][]=[];function visit(path:number[]){const u=path[path.length-1];if(u===t){paths.push(path);return;}directedPairs.forEach(([a,b],i)=>{if(a===u&&(mask&(1<<i))&&!path.includes(b))visit([...path,b]);});}visit([s]);return paths;
}
test('every three-vertex graph and endpoint pair matches brute-force simple paths and two-walk counts',()=>{
 for(let mask=0;mask<64;mask++){
  const g=graphAnalysis(mask);if(g.order)for(const [u,v]of g.edges)assert.ok(g.order.indexOf(u)<g.order.indexOf(v));
  for(let s=0;s<3;s++)for(let t=0;t<3;t++){
   const paths=allSimplePaths(mask,s,t),path=shortestPath(mask,s,t),r=evaluate('paths-cycles-connectivity',[mask,s,t]);
   assert.equal(g.reach[s][t],paths.length>0);assert.equal(path===null,paths.length===0);if(path)assert.equal(path.length,Math.min(...paths.map(p=>p.length)));
   const walks=directedPairs.filter(([u,v],i)=>u===s&&(mask&(1<<i))&&directedPairs.some(([a,b],j)=>a===v&&b===t&&(mask&(1<<j)))).length;
   assert.equal(r.stats['(A²)_st'],walks);
  }
  // A directed cycle exists exactly when a vertex can return along at least one edge.
  const cycle=g.edges.some(([u,v])=>g.reach[v][u]);assert.equal(g.order===null,cycle);
 }
 assert.equal(graphAnalysis(13).underlyingTree,true);assert.equal(graphAnalysis(13).order,null);assert.equal(graphAnalysis(11).underlyingTree,false);assert.ok(graphAnalysis(11).order);assert.equal(graphAnalysis(25).strongComponents,1);assert.equal(graphAnalysis(0).strongComponents,3);
});
test('all Boolean functions reconstruct exactly from ANF; De Morgan holds on all inputs',()=>{
 for(let mask=0;mask<16;mask++)for(let x=0;x<2;x++)for(let y=0;y<2;y++){
  const r=evaluate('boolean-algebra',[mask,x,y]);const [a,b,c,d]=r.stats['ANF coefficients [1,x,y,xy]'] as number[];
  assert.equal(a^(b*x)^(c*y)^(d*x*y),r.stats['F(x,y)']);assert.equal(r.stats['1−(x∧y)'],r.stats['(¬x)∨(¬y)']);
 }
 assert.deepEqual(evaluate('boolean-algebra',[6,1,1]).stats['active minterms'],[1,2]);
});
test('both automata agree with direct language predicates for every word through length eight',()=>{
 for(let mode=0;mode<2;mode++)for(let n=0;n<=8;n++){
  let count=0;for(let w=0;w<2**n;w++){
   const word=n===0?'':w.toString(2).padStart(n,'0');const expected=mode===0?word.split('').filter(x=>x==='1').length%2===0:word.endsWith('01');const r=evaluate('automata-formal-languages',[w,n,mode]);assert.equal(r.stats['1{accepted}'],Number(expected));count+=Number(expected);
  }
  assert.equal(evaluate('automata-formal-languages',[0,n,mode]).stats['accepted words of length N'],count);
 }
 assert.equal(evaluate('automata-formal-languages',[255,0,0]).stats['word length'],0);assert.equal(evaluate('automata-formal-languages',[255,0,1]).stats['1{accepted}'],0);
});
test('every lattice pair satisfies the universal properties of meet and join',()=>{
 const subset=(a:number,b:number)=>(a&b)===a;
 for(let a=0;a<8;a++)for(let b=0;b<8;b++){
  const r=evaluate('posets-lattices',[a,b]),meet=r.stats['A∧B mask'] as number,join=r.stats['A∨B mask'] as number;
  assert.ok(subset(meet,a)&&subset(meet,b));assert.ok(subset(a,join)&&subset(b,join));
  for(let x=0;x<8;x++){if(subset(x,a)&&subset(x,b))assert.ok(subset(x,meet));if(subset(a,x)&&subset(b,x))assert.ok(subset(join,x));}
  assert.equal(r.network!.nodes.length,8);assert.equal(r.network!.edges.length,12);
 }
});
test('all 27 maps and three initial states have correct transient, least period and future state',()=>{
 for(let code=0;code<27;code++)for(let initial=0;initial<3;initial++){
  const T=[code%3,Math.floor(code/3)%3,Math.floor(code/9)%3],trace=[initial];for(let t=0;t<40;t++)trace.push(T[trace[trace.length-1]]);
  for(let n=0;n<=12;n++){
   const r=evaluate('finite-state-transitions',[code,initial,n]);assert.equal(r.stats['state after N'],trace[n]);const mu=r.stats['transient length μ'] as number,period=r.stats['period λ'] as number;
   assert.ok(mu>=0&&period>=1&&mu+period<=3);for(let t=mu;t<30;t++)assert.equal(trace[t],trace[t+period]);
   for(let k=1;k<period;k++)assert.notEqual(trace[mu],trace[mu+k]);if(mu>0)assert.notEqual(trace[mu-1],trace[mu-1+period]);
  }
 }
});
for(const kind of discreteKinds)test(`discrete content, metadata and plotted presets: ${kind}`,()=>{
 const spec=discreteLabs[kind];assert.equal(learningUnits.find(u=>u.id===kind)?.questions.length,2);assert.equal(learningGuides[kind].steps.length,4);assert.equal(learningGuides[kind].experiments.length,3);
 for(const preset of spec.presets){preset.values.forEach((v,i)=>assert.ok(v>=spec.controls[i].min&&v<=spec.controls[i].max));const r=evaluate(kind,preset.values);assert.ok(r.curves.flat(2).every(Number.isFinite));assert.ok(Object.values(r.stats).flat().filter(v=>v!==null).every(Number.isFinite));if(r.network)for(const edge of r.network.edges)assert.ok(r.network.nodes.some(n=>n.id===edge.from)&&r.network.nodes.some(n=>n.id===edge.to));}
});
test('malformed and fractional discrete inputs are rejected before drawing',()=>{
 for(const kind of discreteKinds){assert.throws(()=>evaluate(kind,[]));assert.throws(()=>evaluate(kind,[NaN]));const values=discreteLabs[kind].controls.map(c=>c.initial);values[0]=.5;assert.throws(()=>evaluate(kind,values));}
 assert.throws(()=>countSelections(9,2,0));assert.throws(()=>choose(17,2));assert.throws(()=>evaluate('recurrences-generating-functions',[1,1,3,.9]));
});
