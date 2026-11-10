import test from 'node:test';
import assert from 'node:assert/strict';
import {causalOrder,causalAncestors,causalSeparation,causalBinaryJoint,doCalculus,doGraphs,type CausalEdge} from '../src/lib/learning-causal-graphs';
import {binaryCounterfactual,observationalBounds} from '../src/lib/learning-causality';
const close=(a:number,b:number)=>assert.ok(Math.abs(a-b)<1e-10,`${a} != ${b}`);
const sum=(a:number[])=>a.reduce((s,v)=>s+v,0);
const permutations=(xs:number[]):number[][]=>xs.length?xs.flatMap((x,i)=>permutations(xs.filter((_,j)=>i!==j)).map(t=>[x,...t])):[[]];
/** Independent ancestral-moral-graph criterion, unlike the implementation's path activation. */
function moralSeparated(n:number,edges:CausalEdge[],from:number,to:number,conditioned:number[]){
 const reach=Array.from({length:n},()=>Array(n).fill(false) as boolean[]);for(const [a,b] of edges)reach[a][b]=true;
 for(let k=0;k<n;k++)for(let i=0;i<n;i++)for(let j=0;j<n;j++)reach[i][j] ||= reach[i][k]&&reach[k][j];
 const targets=[from,to,...conditioned],anc=Array.from({length:n},(_,i)=>i).filter(i=>targets.includes(i)||targets.some(t=>reach[i][t]));
 const adjacent=Array.from({length:n},()=>Array(n).fill(false) as boolean[]);
 for(const [a,b] of edges)if(anc.includes(a)&&anc.includes(b))adjacent[a][b]=adjacent[b][a]=true;
 for(const child of anc){const parents=edges.filter(([a,b])=>b===child&&anc.includes(a)).map(([a])=>a);for(const a of parents)for(const b of parents)if(a!==b)adjacent[a][b]=true;}
 const visited=new Set([from]),queue=[from];while(queue.length){const v=queue.shift()!;for(let i=0;i<n;i++)if(adjacent[v][i]&&!conditioned.includes(i)&&!visited.has(i)){visited.add(i);queue.push(i);}}
 return !visited.has(to);
}
test('path d-separation equals independent ancestral moralization on all 543 four-node DAGs',()=>{
 const graphs=new Map<string,CausalEdge[]>();for(const order of permutations([0,1,2,3])){const possible:CausalEdge[]=[];for(let i=0;i<4;i++)for(let j=i+1;j<4;j++)possible.push([order[i],order[j]]);for(let mask=0;mask<64;mask++){const edges=possible.filter((_,i)=>(mask>>i)&1);graphs.set(edges.map(e=>e.join(',')).sort().join(';'),edges);}}
 assert.equal(graphs.size,543);
 for(const edges of graphs.values())for(let from=0;from<4;from++)for(let to=from+1;to<4;to++){const rest=[0,1,2,3].filter(v=>v!==from&&v!==to);for(let bits=0;bits<4;bits++){const given=rest.filter((_,i)=>(bits>>i)&1);assert.equal(causalSeparation(4,edges,from,to,given).separated,moralSeparated(4,edges,from,to,given));}}
});
test('graph validation rejects cycles, duplicate edges, invalid endpoints and non-disjoint evidence',()=>{
 for(const edges of [[[0,1],[1,0]],[[0,1],[0,1]],[[0,0]],[[0,4]]] as CausalEdge[][])assert.throws(()=>causalOrder(4,edges));
 assert.throws(()=>causalOrder(7,[]));assert.throws(()=>causalSeparation(4,[],0,1,[1]));assert.throws(()=>causalSeparation(4,[],0,1,[2,2]));
 assert.deepEqual([...causalAncestors(4,[[3,1],[1,2]],[2])].sort(),[1,2,3]);
});
test('enumerated manipulated binary joint laws normalize and preserve assigned variables exactly',()=>{
 for(const graph of doGraphs)for(const strength of [0,.3,.6])for(const force of [{},{0:1},{0:0,1:1}] as Record<number,number>[]){
  const joint=causalBinaryJoint(graph.edges,strength,force);close(sum(joint.map(r=>r.mass)),1);assert.ok(joint.every(r=>r.mass>=0));
  for(const row of joint)for(const [k,v] of Object.entries(force))if(row.values[Number(k)]!==v)assert.equal(row.mass,0);
 }
 assert.throws(()=>causalBinaryJoint([],NaN));assert.throws(()=>causalBinaryJoint([],.7));assert.throws(()=>causalBinaryJoint([],.2,{0:.5}));
});
test('every licensed local do-calculus equality holds across graph families, queries and mechanisms',()=>{
 for(let graph=0;graph<6;graph++)for(let rule=1;rule<=3;rule++)for(const z of [0,1])for(const x of [0,1])for(const w of [0,1])for(const s of [0,.3,.6])for(const useW of [0,1]){
  const r=doCalculus(graph,rule,z,x,w,s,useW);assert.ok(r.left.mass>0&&r.right.mass>0);assert.ok(r.left.mean!==null&&r.right.mean!==null);
  if(r.separation.separated)close(r.left.mean!,r.right.mean!);if(s===0){close(r.left.mean!,.2);close(r.right.mean!,.2);}
 }
});
test('fork exchange failure and conditional repair match independent Bayes arithmetic',()=>{
 const r=doCalculus(1,2,1,0,0,.6,0);assert.equal(r.separation.separated,false);close(r.left.mean!,.8*.2+.2*.8);close(r.right.mean!,(.2*.8*.8+.8*.2*.2)/(.2*.8+.8*.2));
 const fixed=doCalculus(1,2,1,0,1,.6,1);assert.equal(fixed.separation.separated,true);close(fixed.left.mean!,.8);close(fixed.right.mean!,.8);
 const zero=doCalculus(1,2,1,0,0,0,0);assert.equal(zero.separation.separated,false);close(zero.left.mean!,zero.right.mean!);
});
test('rule-three Z(W) restriction prevents a false action deletion in Y -> Z -> W',()=>{
 const r=doCalculus(2,3,1,0,1,.6,1);assert.equal(r.zInZW,false);assert.deepEqual(r.criterion,r.base);assert.equal(r.separation.separated,false);
 const wrong=r.base.filter(([,to])=>to!==1);assert.equal(causalSeparation(4,wrong,3,1,[0,2]).separated,true);
 close(r.left.mean!,.2);close(r.right.mean!,(.2*(.8*.8+.2*.2))/(.8*.32+.2*.68));
 const noW=doCalculus(2,3,1,0,1,.6,0);assert.equal(noW.zInZW,true);assert.equal(noW.separation.separated,true);close(noW.left.mean!,.2);close(noW.right.mean!,.2);
});
test('counterfactual posteriors are exact compatible-state Bayes weights and obey factual consistency',()=>{
 for(const pi of [0,.2,.5,1])for(const ex of [0,.1,1])for(const ey of [0,.2,1])for(const mode of [0,1,2])for(const xf of [0,1])for(const yf of [0,1]){
  const r=binaryCounterfactual(pi,ex,ey,mode,xf,yf,0);let evidence=0;
  for(const atom of r.factual.atoms)if(atom.naturalX===xf&&atom.naturalY===yf)evidence+=atom.mass;
  close(r.evidence,evidence);
  if(evidence===0){assert.equal(r.posterior,null);assert.deepEqual(r.means,[null,null]);continue;}
  close(sum(r.posterior!),1);close(r.means[xf]!,yf);
  r.factual.atoms.forEach((atom,i)=>close(r.posterior![i],atom.naturalX===xf&&atom.naturalY===yf?atom.mass/evidence:0));
  for(const action of [0,1]){let expected=0;for(const atom of r.factual.atoms){if(atom.naturalX!==xf||atom.naturalY!==yf)continue;const source=mode===0?action:mode===1?atom.z:Number(action!==atom.z),y=Number(source!==atom.ey);expected+=atom.mass*y/evidence;}close(r.means[action]!,expected);}
 }
});
test('counterfactual chain, invariant fork and XOR contrast with population interventions',()=>{
 const chain=binaryCounterfactual(.5,.1,.2,0,1,1,0);close(chain.evidence,.4);close(chain.selected!,0);close(chain.populationMeans[0],.2);assert.deepEqual(chain.posterior!.filter(p=>p>0).map(p=>Math.round(p*10)),[1,9]);
 const fork=binaryCounterfactual(.5,.1,.1,1,1,1,0);close(fork.selected!,1);close(fork.populationMeans[0],.5);
 const xor=binaryCounterfactual(.5,0,0,2,1,0,0);close(xor.selected!,1);
 assert.throws(()=>binaryCounterfactual(.5,.1,.2,0,.5,1,0));assert.throws(()=>binaryCounterfactual(.5,.1,.2,0,1,1,2));
});
test('compatible potential-outcome populations retain observed cells while covering the entire sharp interval',()=>{
 for(const pi of [0,.4,1])for(const r0 of [0,.7,1])for(const r1 of [0,.2,1])for(const lambda of [0,.3,1]){
  const r=observationalBounds(pi,r0,r1,lambda,12),observed=[0,0,0,0];close(sum(r.atoms.map(a=>a.mass)),1);
  for(const atom of r.atoms)observed[2*atom.u+atom.y]+=atom.mass;observed.forEach((p,i)=>close(p,r.joint[i]));
  close(r.ate,r.lower+lambda);close(r.upper-r.lower,1);assert.ok(r.lower>=-1-1e-12&&r.upper<=1+1e-12);
  close(r.doMeans[0],(1-pi)*r0+pi*(1-lambda));close(r.doMeans[1],(1-pi)*lambda+pi*r1);
 }
});
test('all 16 missing-outcome completions independently establish sharp extrema',()=>{
 for(const pi of [0,.4,.5,1])for(const r0 of [0,.7,1])for(const r1 of [0,.2,1]){
  const r=observationalBounds(pi,r0,r1,.5,12),values=[];
  for(let mask=0;mask<16;mask++){let effect=0;for(let x=0;x<2;x++)for(let y=0;y<2;y++){const cell=2*x+y,p=r.joint[cell],missing=(mask>>cell)&1,y0=x===0?y:missing,y1=x===1?y:missing;effect+=p*(y1-y0);}values.push(effect);close(r.vertices[mask],effect);}
  close(Math.min(...values),r.lower);close(Math.max(...values),r.upper);close(r.vertices[12],r.lower);close(r.vertices[3],r.upper);
 }
});
test('perfect observational prediction permits effects zero and one; general bounds retain both effect signs',()=>{
 const zero=observationalBounds(.5,0,1,0,12),one=observationalBounds(.5,0,1,1,3);assert.deepEqual(zero.joint,one.joint);close(zero.ate,0);close(one.ate,1);
 const r=observationalBounds(.4,.7,.2,.5,12);close(r.lower,-.74);close(r.upper,.26);close(r.crude!,-.5);close(r.ate,-.24);
 assert.equal(observationalBounds(0,.7,.2,.5,12).crude,null);assert.throws(()=>observationalBounds(.5,0,1,.5,16));
});
