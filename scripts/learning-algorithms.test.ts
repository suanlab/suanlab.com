import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {algorithmLabs,learningUnits,learningGuides} from '../src/data/interactive-book';
import {getLearningContent} from '../src/lib/learning-content';
import {searchArray,searchExperiment,matrixChain,flowExperiment,randomizedApproximation,satReduction,sortingBounds,sparseSchedule,transposeCache,gossipExperiment,modeledSearchCost,evaluateAlgorithm,algorithmKinds} from '../src/lib/learning-algorithms';
const close=(a:number,b:number,t=1e-9)=>assert.ok(Math.abs(a-b)<=t*(1+Math.max(Math.abs(a),Math.abs(b))),`${a} != ${b}`);
const sum=(a:number[])=>a.reduce((s,v)=>s+v,0);
test('actual search probes return the right key or a missing witness on every small array',()=>{
 for(let N=1;N<=70;N++)for(let q=-1;q<=N;q++)for(const m of [0,1]){const a=Array.from({length:N},(_,i)=>i),r=searchArray(a,q,m);assert.equal(r.index,q>=0&&q<N?q:null);assert.equal(r.trace.length,r.comparisons);if(m===0)assert.equal(r.comparisons,q>=0&&q<N?q+1:N);else assert.ok(r.comparisons<=Math.floor(Math.log2(N))+1);for(const t of r.trace)assert.ok(t.index>=t.lo&&t.index<=t.hi);}
});
test('search best case and query family do not fabricate a universal linear-time curve',()=>{
 const first=searchExperiment(10,0,0,0),last=searchExperiment(10,1,0,0),missing=searchExperiment(10,.3,1,1);assert.ok(first.levels.every(x=>x.linear===1));assert.equal(last.selected.comparisons,1024);assert.equal(missing.selected.index,null);assert.equal(missing.selected.comparisons,11);assert.equal(searchArray([],3,1).comparisons,0);
});
test('matrix-chain dynamic program matches exhaustive independent binary-tree multiplication costs',()=>{
 const enumerate=(dims:number[],lo:number,hi:number):number[]=>{if(lo===hi)return [0];const out:number[]=[];for(let k=lo;k<hi;k++)for(const left of enumerate(dims,lo,k))for(const right of enumerate(dims,k+1,hi))out.push(left+right+dims[lo]*dims[k+1]*dims[hi+1]);return out;};
 for(const n of [2,3,4,5,6])for(const scale of [1,3])for(const shape of [0,1,2]){const r=matrixChain(n,scale,shape),costs=enumerate(r.dims,0,n-1);close(r.optimal,Math.min(...costs));close(r.recursiveCost,r.optimal);assert.equal(r.transitions,n*(n*n-1)/6);assert.equal(r.recursiveCalls,3**(n-1));assert.ok(r.prefix.every(x=>x.optimal<=x.left));}
 close(matrixChain(3,1,2).optimal,18);close(matrixChain(3,2,2).optimal,144);
});
test('DP split reconstruction certifies its claimed multiplication count',()=>{
 for(const shape of [0,1,2]){const r=matrixChain(6,2,shape),walk=(i:number,j:number):number=>{if(i===j)return 0;const k=r.split[i][j];assert.ok(i<=k&&k<j);return walk(i,k)+walk(k+1,j)+r.dims[i]*r.dims[k+1]*r.dims[j+1];};close(walk(0,5),r.optimal);assert.equal((r.parenthesization.match(/A/g)||[]).length,6);}
});
test('residual flow agrees with independent cut enumeration for every small capacity tuple',()=>{
 for(const sa of [0,1,2])for(const sb of [0,1,2])for(const ab of [0,1,2])for(const at of [0,1,2])for(const bt of [0,1,2]){const r=flowExperiment(sa,sb,ab,at,bt,8),cutValues=[sa+sb,sb+ab+at,sa+bt,at+bt];assert.equal(r.value,Math.min(...cutValues));assert.equal(r.certified,true);assert.equal(r.cutCapacity,r.value);for(let i=0;i<4;i++)for(let j=0;j<4;j++){close(r.F[i][j],-r.F[j][i]);assert.ok(r.C[i][j]-r.F[i][j]>=0);}for(const i of [1,2])close(sum(r.F[i]),0);for(const [[i,j],cap]of r.edgePairs)assert.ok(r.F[i][j]>=0&&r.F[i][j]<=cap);}
});
test('an augmentation cap is not a false maximum-flow certificate',()=>{
 const zero=flowExperiment(3,2,1,2,3,0),partial=flowExperiment(3,2,1,2,3,1),full=flowExperiment(3,2,1,2,3,8);assert.equal(zero.value,0);assert.equal(zero.cutCapacity,null);assert.equal(partial.value,2);assert.equal(partial.certified,false);assert.equal(full.value,5);assert.deepEqual(full.paths.map(p=>p.amount),[2,2,1]);for(const p of full.paths){assert.equal(p.nodes[0],0);assert.equal(p.nodes.at(-1),3);assert.ok(p.amount>0);}assert.equal(flowExperiment(0,0,0,0,0,0).certified,true);
});
test('uniform random-cut PMFs have independent expectation and variance formulas',()=>{
 for(const mode of [0,1,2,3]){const r=randomizedApproximation(mode,0,7,1);close(sum(r.probabilities),1);close(r.expectation,r.edges.length/2);close(r.variance,r.edges.length/4);close(sum(r.probabilities.map((p,k)=>p*k)),r.expectation);close(sum(r.probabilities.map((p,k)=>p*(k-r.expectation)**2)),r.variance);}const triangle=randomizedApproximation(0,0,7,0);assert.deepEqual(triangle.probabilities,[.25,0,.75,0]);
});
test('maximal matching gives a valid cover with its lower-bound and two-approximation certificate',()=>{
 for(const mode of [0,1,2,3])for(let offset=0;offset<6;offset++){const r=randomizedApproximation(mode,offset,7,0);assert.equal(r.optCover,[2,1,3,0][mode]);assert.equal(r.maximumCut,[2,5,6,0][mode]);assert.equal(r.coverSize,2*r.matching.length);assert.ok(r.matching.length<=r.optCover);assert.ok(r.coverSize<=2*r.optCover);for(const [i,j]of r.edges)assert.ok(((r.coverMask>>i)&1)||((r.coverMask>>j)&1));const vertices=r.matching.flat();assert.equal(new Set(vertices).size,vertices.length);}assert.equal(randomizedApproximation(3,0,7,0).ratio,null);
});
test('seeded cuts are reproducible but remain separate from the exact distribution',()=>{
 const a=randomizedApproximation(1,0,7,1),b=randomizedApproximation(1,0,7,1),c=randomizedApproximation(1,5,999,1);assert.equal(a.sampledMask,b.sampledMask);assert.equal(a.queryCut,5);assert.deepEqual(a.probabilities,c.probabilities);for(const seed of [1,7,19,999]){const r=randomizedApproximation(2,0,seed,0);assert.ok(r.sampledMask>=0&&r.sampledMask<64);assert.ok(r.sampledCut<=r.maximumCut);}
});
test('SAT to independent-set reduction preserves every finite truth witness in both directions',()=>{
 for(const mode of [0,1,2])for(let assignment=0;assignment<8;assignment++)for(let mask=0;mask<512;mask++){const r=satReduction(mode,assignment,mask),evaluate=(m:number)=>r.clauses.every(c=>c.some(l=>Boolean((m>>(Math.abs(l)-1))&1)===(l>0)));assert.equal(r.assignmentSatisfies,evaluate(assignment));assert.equal(r.forwardWitnessSize===3,r.assignmentSatisfies);assert.equal(r.forwardIndependent,true);assert.equal(r.maxIndependent===3,r.satisfyingAssignments>0);if(r.validSelection&&r.size===3){assert.ok(r.decoded!==null);assert.equal(evaluate(r.decoded!),true);assert.equal(r.decodedSatisfies,true);}else assert.equal(r.decoded,null);}
});
test('contradictory repeated-literal clauses are actually unsatisfiable, not a failed search timeout',()=>{
 const r=satReduction(1,0,0);assert.equal(r.satisfyingAssignments,0);assert.equal(r.maxIndependent,2);const all=satReduction(2,7,511);assert.equal(all.satisfyingAssignments,8);assert.equal(all.validSelection,false);assert.equal(all.decoded,null);
});
test('actual sorting traces meet independent decision-tree and exact insertion comparison formulas',()=>{
 for(let n=2;n<=7;n++){let factorial=1,harmonic=0;for(let k=1;k<=n;k++){factorial*=k;harmonic+=1/k;}for(const m of [0,1]){const r=sortingBounds(n,m,5039);assert.equal(r.factorial,factorial);assert.equal(r.distinctTraces,factorial);assert.ok(r.maxComparisons>=Math.ceil(Math.log2(factorial)));close(sum(r.distribution.map(x=>x.probability)),1);assert.deepEqual(r.query.output,Array.from({length:n},(_,i)=>i));assert.ok(sum(r.distribution.map(x=>x.count*2**(-x.comparisons)))<=1+1e-12);if(m===0){close(r.average,n*(n-1)/4+n-harmonic);assert.equal(r.maxComparisons,n*(n-1)/2);}else{const k=Math.ceil(Math.log2(n));assert.equal(r.maxComparisons,n*k-2**k+1);}}}
});
test('sorting query indices wrap only within the declared finite permutation set',()=>{
 const a=sortingBounds(3,0,0),b=sortingBounds(3,0,6);assert.deepEqual(a.queryInput,b.queryInput);assert.equal(a.query.comparisons,2);assert.equal(sortingBounds(3,0,5).query.comparisons,3);
});
test('CSR and genuinely scheduled reduction tasks match independently evaluated dense rows',()=>{
 for(const p of [2,3,4,5])for(const band of [0,1,3])for(const processors of [1,4,16])for(const mode of [0,1]){const r=sparseSchedule(p,band,processors,mode);assert.equal(r.values.length,r.N*(2*band+1)-band*(band+1));for(let i=0;i<r.N;i++){let exact=2*r.x[i];for(let j=Math.max(0,i-band);j<=Math.min(r.N-1,i+band);j++)if(j!==i)exact-=r.x[j];close(r.csr[i],exact);close(r.scheduled[i],exact);close(r.dense[i],exact);}assert.equal(r.work,2*r.values.length-r.N);assert.ok(r.makespan>=r.lowerBound);assert.ok(r.makespan<=Math.ceil(r.work/processors)+r.span);assert.equal(r.denseMultiplications,r.N*r.N);}
});
test('parallel schedules respect dependencies, finite processor count and task uniqueness',()=>{
 const r=sparseSchedule(4,3,3,0),finish=new Map<number,number>();r.schedule.forEach((batch,time)=>{assert.ok(batch.length<=3);for(const id of batch){assert.ok(!finish.has(id));for(const dep of r.tasks[id].deps)assert.ok(finish.has(dep)&&finish.get(dep)!<time);}batch.forEach(id=>finish.set(id,time));});assert.equal(finish.size,r.work);close(r.span,Math.max(...r.depth));const diagonal=sparseSchedule(3,0,16,0);assert.equal(diagonal.span,1);assert.equal(diagonal.makespan,1);assert.equal(sparseSchedule(3,1,1,0).makespan,sparseSchedule(3,1,1,0).work);
});
test('out-of-place transpose has the same result and access count for every traversal',()=>{
 for(const p of [2,3,4,5])for(const tile of [0,1,3,5])for(const blocked of [0,1]){const r=transposeCache(p,tile,4,8,blocked);assert.equal(r.accesses,2*r.N*r.N);assert.equal(r.hits+r.misses,r.accesses);for(let i=0;i<r.N;i++)for(let j=0;j<r.N;j++)assert.equal(r.B[j*r.N+i],r.A[i*r.N+j]);assert.ok(r.misses>=r.compulsoryLines);assert.equal(r.transfers,r.misses+r.writebacks+r.finalFlush);assert.equal(r.trafficBytes,r.transfers*4*8);}
});
test('LRU misses agree with an independent distinct-reuse-distance oracle',()=>{
 for(const blocked of [0,1])for(const capacity of [1,3,8,32]){const r=transposeCache(3,2,4,capacity,blocked),lines=r.events.map(e=>Math.floor(e.address/4));let expected=0;for(let i=0;i<lines.length;i++){let previous=i-1;while(previous>=0&&lines[previous]!==lines[i])previous--;const miss=previous<0||new Set(lines.slice(previous+1,i)).size>=capacity;assert.equal(r.events[i].miss,miss);expected+=Number(miss);}assert.equal(r.misses,expected);}
});
test('cache traffic includes dirty eviction and final flush rather than just cold loads',()=>{
 const tiny=transposeCache(2,0,1,1,0);assert.equal(tiny.misses,32);assert.equal(tiny.allWritebacks,16);assert.equal(tiny.transfers,48);const resident=transposeCache(3,2,4,32,0);assert.equal(resident.misses,32);assert.equal(resident.writebacks,0);assert.equal(resident.finalFlush,16);const naive=transposeCache(3,2,4,8,0),blocked=transposeCache(3,2,4,8,1);assert.equal(naive.misses,80);assert.equal(blocked.misses,32);assert.ok(blocked.trafficBytes<naive.trafficBytes);assert.equal(transposeCache(3,0,4,8,1).misses,naive.misses);
});
test('Metropolis matrices and pair gossip preserve component sums with explicit communication counts',()=>{
 for(const topology of [0,1,2,3,4])for(const failure of [0,1])for(const mode of [0,1]){const r=gossipExperiment(topology,30,-3,failure,mode);for(let i=0;i<6;i++){close(sum(r.W[i]),1);close(sum(r.W.map(row=>row[i])),1);for(let j=0;j<6;j++)close(r.W[i][j],r.W[j][i]);}for(const row of r.path){close(sum(row),sum(r.initial));for(const component of r.components)close(sum(component.map(i=>row[i])),sum(component.map(i=>r.initial[i])));}assert.equal(r.messages,30*(mode===0?2*r.edges.length:r.edges.length?2:0));for(const component of r.components){const mean=sum(component.map(i=>r.initial[i]))/component.length;component.forEach(i=>close(r.limit[i],mean));}}
});
test('connected complete averaging reaches consensus, while disconnected and empty graphs do not certify it',()=>{
 const full=gossipExperiment(3,1,0,0,0);full.final.forEach(v=>close(v,2.5));assert.equal(full.connected,true);const split=gossipExperiment(2,30,0,0,0);assert.equal(split.connected,false);assert.deepEqual(split.limit,[1,1,1,4,4,4]);assert.ok(split.history.at(-1)!.spread>=3-1e-10);const empty=gossipExperiment(4,30,0,1,1);assert.deepEqual(empty.final,empty.initial);assert.equal(empty.messages,0);assert.deepEqual(gossipExperiment(0,0,0,0,0).final,[0,1,2,3,4,5]);
});
test('modeled cost units can reverse finite rankings without being fabricated wall time',()=>{
 const a=modeledSearchCost(1,0,1,1,100,20),b=modeledSearchCost(10,1,1,1,0,20),c=modeledSearchCost(10,1,1,500,1000,20);assert.ok(a.linearCost<a.binaryCost);assert.ok(b.binaryCost<b.linearCost);assert.ok(c.binaryCost>c.linearCost);assert.equal(b.linear,1024);assert.equal(b.binary,11);assert.equal(b.repetitions,20);
});
const initial:Record<typeof algorithmKinds[number],number[]>={'search-asymptotic-costs':[5,.8,0,1],'matrix-chain-dynamic-programming':[4,1,1],'residual-flow-cut-certificates':[3,2,1,2,3,8],'random-cuts-matching-approximation':[0,0,7,1],'sat-independent-set-reduction':[0,7,273],'comparison-sorting-lower-bounds':[4,1,0],'sparse-work-span-scheduling':[3,1,4,0],'cache-traffic-blocked-transpose':[3,2,4,8,1],'distributed-gossip-communication':[0,10,0,0,0],'operation-counts-wall-time':[8,.9,1,5,50,100]};
for(const kind of algorithmKinds)test(`algorithm result contract and finite network: ${kind}`,()=>{const p=initial[kind],r=evaluateAlgorithm(kind,p);assert.ok(r.curves.flat(2).every(Number.isFinite));for(const v of Object.values(r.stats).flat())assert.ok(v===null||Number.isFinite(v));assert.ok(!/[가-힣]/.test(r.statement??''));if(r.network){assert.equal(new Set(r.network.nodes.map(n=>n.id)).size,r.network.nodes.length);assert.ok(r.network.edges.every(e=>r.network!.nodes.some(n=>n.id===e.from)&&r.network!.nodes.some(n=>n.id===e.to)));assert.ok(r.network.nodes.every(n=>Number.isFinite(n.x)&&Number.isFinite(n.y)&&n.x>20&&n.x<330&&n.y>20&&n.y<300));}assert.throws(()=>evaluateAlgorithm(kind,[]));const invalid=p.slice();invalid[0]=NaN;assert.throws(()=>evaluateAlgorithm(kind,invalid));});

for(const id of algorithmKinds)test(`algorithm presets, bounds and bilingual guide: ${id}`,()=>{
 const spec=algorithmLabs[id],guide=learningGuides[id],unit=learningUnits.find(u=>u.id===id)!;assert.equal(guide.steps.length,4);assert.equal(guide.experiments.length,3);assert.equal(unit.questions.length,2);const vectors=[spec.controls.map(c=>c.initial),...spec.presets.map(p=>p.values)];for(let i=0;i<spec.controls.length;i++)for(const v of [spec.controls[i].min,spec.controls[i].max]){const p=spec.controls.map(c=>c.initial);p[i]=v;vectors.push(p);}for(const p of vectors){const r=evaluateAlgorithm(id,p);assert.ok(r.curves.flat(2).every(Number.isFinite));for(const v of Object.values(r.stats).flat())assert.ok(v===null||Number.isFinite(v));assert.ok(!/[가-힣]/.test(r.statement??''));if(r.network)assert.ok(r.network.nodes.every(n=>Number.isFinite(n.x)&&Number.isFinite(n.y)));}assert.throws(()=>evaluateAlgorithm(id,[]));const invalid=spec.controls.map(c=>c.initial);invalid[0]=NaN;assert.throws(()=>evaluateAlgorithm(id,invalid));const int=spec.controls.findIndex(c=>c.step===1);if(int>=0){const p=spec.controls.map(c=>c.initial);p[int]=spec.controls[int].min+.5;assert.throws(()=>evaluateAlgorithm(id,p));}
});
for(const id of algorithmKinds)for(const lang of ['ko','en'] as const)test(`substantive algorithm manuscript and rendered equations: ${id}.${lang}`,async()=>{
 const raw=await fs.readFile(`content/learning/physics-math-for-ai/${id}.${lang}.md`,'utf8'),content=await getLearningContent(id);assert.ok((raw.match(/^## /gm)||[]).length>=5);assert.ok(raw.includes('$$'));assert.ok(raw.includes('https://'));assert.ok(raw.includes(lang==='en'?'Exercise:':'직접 풀기:'));if(lang==='en'){assert.ok(raw.split(/\s+/).length>350);assert.ok(!/[가-힣]/.test(raw));}else assert.ok((raw.match(/[가-힣]/g)||[]).length>700);assert.ok(content[lang].includes('katex-display'));assert.ok(!content[lang].includes('katex-error'));assert.ok(content[lang].length>4000);
});
