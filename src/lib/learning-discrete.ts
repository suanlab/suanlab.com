import type { FoundationResult } from './learning-foundations';
import type { Vec2 } from './learning-simulations';

export const discreteKinds = ['counting-permutations','binomial-inclusion-exclusion','recurrences-generating-functions','graphs-trees-dags','paths-cycles-connectivity','boolean-algebra','automata-formal-languages','posets-lattices','finite-state-transitions'] as const;
export type DiscreteKind = typeof discreteKinds[number];
export function isDiscreteKind(kind: string): kind is DiscreteKind { return (discreteKinds as readonly string[]).includes(kind); }
export interface DiscreteNetwork {
  nodes: { id: number; label: string; x: number; y: number; active?: boolean; accepting?: boolean }[];
  edges: { from: number; to: number; label?: string; active?: boolean }[];
  directed: boolean;
}
export interface DiscreteResult extends FoundationResult {
  network?: DiscreteNetwork;
  table?: { headers: string[]; rows: (number | string)[][] };
}
function integer(value: number, min: number, max: number) { if (!Number.isInteger(value) || value < min || value > max) throw new Error(`Expected integer in [${min}, ${max}].`); }
/** Small counts are exact integers, well within Number.MAX_SAFE_INTEGER. */
export function choose(n: number, k: number): number {
  integer(n, 0, 16); integer(k, 0, 16);
  if (k > n) return 0;
  let value = 1;
  for (let i = 1; i <= Math.min(k, n - k); i++) value = value * (n - i + 1) / i;
  return Math.round(value);
}
export function countSelections(n: number, k: number, mode: number): number {
  integer(n, 0, 8); integer(k, 0, 8); integer(mode, 0, 3);
  if (k === 0) return 1;
  if (n === 0) return 0;
  if (mode === 1) return choose(n, k);
  if (mode === 2) return n ** k;
  if (mode === 3) return choose(n + k - 1, k);
  if (k > n) return 0;
  let value = 1; for (let i = 0; i < k; i++) value *= n - i;
  return value;
}
export const directedPairs = [[0,1],[0,2],[1,0],[1,2],[2,0],[2,1]] as const;
export function graphFromMask(mask: number) { integer(mask, 0, 63); return directedPairs.filter((_, i) => (mask & (1 << i)) !== 0).map(([a,b]) => [a,b] as [number,number]); }
export function graphAnalysis(mask: number) {
  const edges = graphFromMask(mask), adjacency = Array.from({length:3}, () => [] as number[]);
  edges.forEach(([u,v]) => adjacency[u].push(v));
  const reach = Array.from({length:3}, (_,u) => Array.from({length:3}, (_,v) => u === v || adjacency[u].includes(v)));
  for (let k=0;k<3;k++) for (let u=0;u<3;u++) for(let v=0;v<3;v++) reach[u][v] ||= reach[u][k] && reach[k][v];
  const indegree = [0,0,0]; edges.forEach(([,v]) => indegree[v]++);
  const queue = [0,1,2].filter(v => indegree[v] === 0), order:number[] = [];
  while(queue.length) { const u=queue.shift()!; order.push(u); adjacency[u].forEach(v => { if (--indegree[v] === 0) queue.push(v); }); }
  const undirected = new Set(edges.map(([u,v]) => `${Math.min(u,v)}:${Math.max(u,v)}`));
  const seen = new Set<number>(); let weakComponents=0;
  for(let u=0;u<3;u++) if(!seen.has(u)) { weakComponents++; const q=[u]; seen.add(u); while(q.length) { const v=q.shift()!; for(let w=0;w<3;w++) if(!seen.has(w) && (adjacency[v].includes(w)||adjacency[w].includes(v))) { seen.add(w); q.push(w); } } }
  const assigned = new Set<number>(); let strongComponents=0;
  for(let u=0;u<3;u++) if(!assigned.has(u)) { strongComponents++; for(let v=0;v<3;v++) if(reach[u][v] && reach[v][u]) assigned.add(v); }
  return { edges, adjacency, reach, order:order.length === 3 ? order : null, weakComponents, strongComponents, undirectedEdges:undirected.size, underlyingTree:weakComponents === 1 && undirected.size === 2 };
}
function graphNetwork(mask: number, path: number[] = []): DiscreteNetwork {
  return { directed:true, nodes:[{id:0,label:'0',x:175,y:65},{id:1,label:'1',x:75,y:225},{id:2,label:'2',x:275,y:225}].map(node=>({...node,active:path.includes(node.id)})), edges:graphFromMask(mask).map(([from,to])=>({from,to,active:path.some((v,i)=>i+1<path.length&&v===from&&path[i+1]===to)})) };
}
export function shortestPath(mask: number, source: number, target: number): number[] | null {
  integer(source,0,2); integer(target,0,2); const {adjacency}=graphAnalysis(mask);
  const queue:number[][]=[[source]],seen=new Set([source]);
  while(queue.length) { const path=queue.shift()!,u=path[path.length-1]; if(u===target)return path; for(const v of adjacency[u]) if(!seen.has(v)){seen.add(v);queue.push([...path,v]);} }
  return null;
}
export function automatonStep(mode: number, state: number, symbol: number): number {
  integer(mode,0,1); integer(symbol,0,1); integer(state,0,mode===0?1:2);
  return mode===0 ? (state+symbol)%2 : symbol===0 ? 1 : state===1 ? 2 : 0;
}
export function evaluateDiscrete(kind: DiscreteKind, p: number[]): DiscreteResult {
  const counts:Record<DiscreteKind,number>={'counting-permutations':3,'binomial-inclusion-exclusion':3,'recurrences-generating-functions':4,'graphs-trees-dags':1,'paths-cycles-connectivity':3,'boolean-algebra':3,'automata-formal-languages':3,'posets-lattices':2,'finite-state-transitions':3};
  if(p.length!==counts[kind]||!p.every(Number.isFinite))throw new Error('Use finite inputs of the expected shape.');
  const [a,b,c,d]=p;
  if(kind==='counting-permutations') {
    const count=countSelections(a,b,c); const labels=['P(n,k)','C(n,k)','n^k','C(n+k−1,k)'];
    return {statement:`mode ${c}: ${labels[c]}; n=${a}, k=${b}`,stats:{'count':count,'n':a,'k':b,'mode':c},curves:[Array.from({length:9},(_,k)=>[k,countSelections(a,k,c)] as Vec2)],table:{headers:['k','count'],rows:Array.from({length:9},(_,k)=>[k,countSelections(a,k,c)])}};
  }
  if(kind==='binomial-inclusion-exclusion') {
    integer(a,0,63);integer(b,0,63);integer(c,0,6);
    const A=Array.from({length:6},(_,i)=>(a>>i)&1),B=Array.from({length:6},(_,i)=>(b>>i)&1),size=(xs:number[])=>xs.reduce((s,x)=>s+x,0);
    return {statement:`U={0,1,2,3,4,5}; A mask=${a}; B mask=${b}; k=${c}`,stats:{'|A|':size(A),'|B|':size(B),'|A∩B|':size(A.map((x,i)=>x*B[i])),'|A∪B|':size(A.map((x,i)=>Math.max(x,B[i]))),'C(6,k)':choose(6,c),'Σ C(6,j)':64},curves:[Array.from({length:7},(_,k)=>[k,choose(6,k)] as Vec2)],table:{headers:['u','1_A','1_B','1_(A∩B)','1_(A∪B)'],rows:A.map((x,i)=>[i,x,B[i],x*B[i],Math.max(x,B[i])])}};
  }
  if(kind==='recurrences-generating-functions') {
    integer(a,0,2);integer(b,0,2);integer(c,0,12);if(Math.abs(d)>.8)throw new Error('Use |x|≤0.8.');
    const terms=[0,1];for(let i=2;i<=Math.max(1,c);i++)terms.push(a*terms[i-1]+b*terms[i-2]);
    const rho=(a+Math.sqrt(a*a+4*b))/2,denom=1-a*d-b*d*d,converges=rho===0||Math.abs(d)<1/rho;
    const partial=terms.slice(0,c+1).reduce((s,v,i)=>s+v*d**i,0);
    return {statement:`a₀=0; a₁=1; aₙ=${a}aₙ₋₁+${b}aₙ₋₂; x=${d}`,stats:{'a_N':terms[c],'Σ₀ᴺ a_n x^n':partial,'x/(1−px−qx²)':denom===0?null:d/denom,'1{series converges at x}':converges?1:0,'radius R':rho===0?null:1/rho},curves:[terms.slice(0,c+1).map((v,i)=>[i,v])],table:{headers:['n','a_n','a_n x^n'],rows:terms.slice(0,c+1).map((v,i)=>[i,v,Number((v*d**i).toPrecision(8))])}};
  }
  if(kind==='graphs-trees-dags') {
    const g=graphAnalysis(a);return {statement:`V={0,1,2}; edge mask=${a}`,stats:{'|E|':g.edges.length,'weak components':g.weakComponents,'underlying undirected edges':g.undirectedEdges,'1{underlying tree}':g.underlyingTree?1:0,'1{DAG}':g.order?1:0,'topological order':g.order},curves:[],network:graphNetwork(a),table:{headers:['bit','u','v','A_uv'],rows:directedPairs.map(([u,v],i)=>[i,u,v,(a>>i)&1])}};
  }
  if(kind==='paths-cycles-connectivity') {
    const g=graphAnalysis(a),path=shortestPath(a,b,c);const walks=g.adjacency[b].filter(v=>g.adjacency[v].includes(c)).length;
    return {statement:`edge mask=${a}; source=${b}; target=${c}`,stats:{'shortest path':path,'distance':path?path.length-1:null,'1{reachable incl. length 0}':g.reach[b][c]?1:0,'(A²)_st':walks,'strong components':g.strongComponents,'1{directed cycle}':g.order?0:1},curves:[],network:graphNetwork(a,path??[]),table:{headers:['u','reach(u,0)','reach(u,1)','reach(u,2)'],rows:g.reach.map((row,i)=>[i,...row.map(Number)])}};
  }
  if(kind==='boolean-algebra') {
    integer(a,0,15);integer(b,0,1);integer(c,0,1);const output=(x:number,y:number)=>(a>>(2*x+y))&1;
    return {statement:`truth mask=${a}; row index=2x+y; x=${b}, y=${c}`,stats:{'F(x,y)':output(b,c),'1−(x∧y)':1-b*c,'(¬x)∨(¬y)':Math.max(1-b,1-c),'active minterms':Array.from({length:4},(_,i)=>i).filter(i=>(a>>i)&1),'ANF coefficients [1,x,y,xy]':[output(0,0),output(1,0)^output(0,0),output(0,1)^output(0,0),output(0,0)^output(1,0)^output(0,1)^output(1,1)]},curves:[Array.from({length:4},(_,i)=>[i,output(i>>1,i&1)] as Vec2)],table:{headers:['x','y','F','minterm'],rows:Array.from({length:4},(_,i)=>[i>>1,i&1,output(i>>1,i&1),`m${i}`])}};
  }
  if(kind==='automata-formal-languages') {
    integer(a,0,255);integer(b,0,8);integer(c,0,1);const word=Array.from({length:b},(_,i)=>(a>>(b-1-i))&1);let state=0;const rows:(number|string)[][]=[[0,'ε',state]];
    word.forEach((symbol,i)=>{state=automatonStep(c,state,symbol);rows.push([i+1,symbol,state]);});
    const accepting=c===0?0:2,states=c===0?[0,1]:[0,1,2];
    return {statement:`mode=${c}; input=${word.length?word.join(''):'ε'}; read left→right; keep lowest ${b} bits`,stats:{'final state':state,'1{accepted}':state===accepting?1:0,'word length':b,'accepted words of length N':c===0?(b===0?1:2**(b-1)):(b<2?0:2**(b-2))},curves:[],network:{directed:true,nodes:states.map((id)=>({id,label:`q${id}`,x:c===0?100+150*id:55+120*id,y:145,active:id===state,accepting:id===accepting})),edges:states.flatMap(from=>[0,1].map(symbol=>({from,to:automatonStep(c,from,symbol),label:String(symbol)})))},table:{headers:['consumed','symbol','state'],rows}};
  }
  if(kind==='posets-lattices') {
    integer(a,0,7);integer(b,0,7);const nodes=Array.from({length:8},(_,id)=>{const rank=((id>>0)&1)+((id>>1)&1)+((id>>2)&1);const level=Array.from({length:8},(_,i)=>i).filter(i=>((i&1)+((i>>1)&1)+((i>>2)&1))===rank);return {id,label:id===0?'∅':`{${[0,1,2].filter(i=>id&(1<<i)).join(',')}}`,x:175+(level.indexOf(id)-(level.length-1)/2)*100,y:255-rank*70,active:id===a||id===b};});
    return {statement:`B₃=(P({0,1,2}),⊆); A mask=${a}, B mask=${b}`,stats:{'A∧B mask':a&b,'A∨B mask':a|b,'1{A⊆B}':(a&b)===a?1:0,'1{B⊆A}':(a&b)===b?1:0,'1{comparable}':(a&b)===a||(a&b)===b?1:0},curves:[],network:{nodes,edges:nodes.flatMap(node=>[0,1,2].filter(i=>!(node.id&(1<<i))).map(i=>({from:node.id,to:node.id|(1<<i)}))),directed:false},table:{headers:['mask','subset'],rows:nodes.map(n=>[n.id,n.label])}};
  }
  integer(a,0,26);integer(b,0,2);integer(c,0,12);const transition=[a%3,Math.floor(a/3)%3,Math.floor(a/9)%3];
  const orbit:number[]=[],seen=new Map<number,number>();let state=b;
  while(!seen.has(state)){seen.set(state,orbit.length);orbit.push(state);state=transition[state];}
  const mu=seen.get(state)!,period=orbit.length-mu;const stateAt=(t:number)=>t<orbit.length?orbit[t]:orbit[mu+(t-mu)%period];
  return {statement:`T(0,1,2)=(${transition.join(',')}); s₀=${b}; N=${c}`,stats:{'T(0,1,2)':transition,'state after N':stateAt(c),'transient length μ':mu,'period λ':period,'distinct visited states':orbit.length},curves:[],network:{directed:true,nodes:[0,1,2].map(id=>({id,label:String(id),x:65+110*id,y:150,active:id===stateAt(c)})),edges:transition.map((to,from)=>({from,to}))},table:{headers:['t','T^t(s₀)'],rows:Array.from({length:c+1},(_,t)=>[t,stateAt(t)])}};
}
