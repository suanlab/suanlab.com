/** Small DAG utilities. No probabilities or data imports enter the separation test. */
export type CausalEdge = [number, number];
export interface CausalNetwork {
  nodes: { name: string; x: number; y: number; conditioned?: boolean; intervened?: boolean; hidden?: boolean }[];
  edges: { from: number; to: number; removed?: boolean }[];
}
export function causalOrder(n: number, edges: CausalEdge[]) {
  if (!Number.isInteger(n) || n < 2 || n > 6 || edges.some(([a,b])=>!Number.isInteger(a)||!Number.isInteger(b)||a<0||b<0||a>=n||b>=n||a===b) || new Set(edges.map(e=>e.join(','))).size!==edges.length) throw new Error('Use a simple small directed graph.');
  const remaining=new Set(Array.from({length:n},(_,i)=>i)),order:number[]=[];
  while(remaining.size){const next=[...remaining].find(v=>!edges.some(([a,b])=>b===v&&remaining.has(a)));if(next===undefined)throw new Error('Use an acyclic causal graph.');order.push(next);remaining.delete(next);}
  return order;
}
export function causalAncestors(n:number,edges:CausalEdge[],targets:number[]){
  causalOrder(n,edges);if(targets.some(v=>!Number.isInteger(v)||v<0||v>=n))throw new Error('Use graph nodes.');
  const seen=new Set(targets);let changed=true;
  while(changed){changed=false;for(const [a,b] of edges)if(seen.has(b)&&!seen.has(a)){seen.add(a);changed=true;}}
  return seen;
}
export function causalSeparation(n:number,edges:CausalEdge[],from:number,to:number,conditioned:number[]){
  causalOrder(n,edges);const nodes=[from,to,...conditioned];if(new Set(nodes).size!==nodes.length||nodes.some(v=>!Number.isInteger(v)||v<0||v>=n))throw new Error('Use disjoint endpoint and conditioning nodes.');
  const ancestors=causalAncestors(n,edges,conditioned),paths:number[][]=[];
  const visit=(path:number[])=>{const last=path.at(-1)!;if(last===to){paths.push(path);return;}for(const [a,b] of edges){const next=a===last?b:b===last?a:-1;if(next>=0&&!path.includes(next))visit([...path,next]);}};visit([from]);
  const into=(a:number,b:number)=>edges.some(([x,y])=>x===a&&y===b);
  const active=paths.filter(path=>path.slice(1,-1).every((v,i)=>{const collider=into(path[i],v)&&into(path[i+2],v);return collider?ancestors.has(v):!conditioned.includes(v);}));
  return {separated:active.length===0,paths,active};
}
export const doGraphs: { name: string; edges: CausalEdge[] }[] = [
  {name:'Z -> X -> Y; W independent',edges:[[1,0],[0,3]]},
  {name:'W -> Z; W -> Y; X independent',edges:[[2,1],[2,3]]},
  {name:'Y -> Z -> W; X independent',edges:[[3,1],[1,2]]},
  {name:'Z -> W <- Y; X independent',edges:[[1,2],[3,2]]},
  {name:'Z -> Y; X,W independent',edges:[[1,3]]},
  {name:'X -> Z <- W -> Y',edges:[[0,1],[2,1],[2,3]]},
];
const bitProbability=(p:number,v:number)=>v?p:1-p;
/** Uniform-noise structural assignments with q=0.2+strength*mean(parents). */
export function causalBinaryJoint(edges:CausalEdge[],strength:number,forced:Record<number,number>={}){
  causalOrder(4,edges);if(!Number.isFinite(strength)||strength<0||strength>.6||Object.entries(forced).some(([k,v])=>!Number.isInteger(Number(k))||Number(k)<0||Number(k)>3||(v!==0&&v!==1)))throw new Error('Use a bounded causal mechanism and binary interventions.');
  return Array.from({length:16},(_,mask)=>{const values=Array.from({length:4},(_,i)=>(mask>>i)&1);let mass=1;for(let v=0;v<4;v++){
    if(v in forced){mass*=Number(values[v]===forced[v]);continue;}
    const parents=edges.filter(([,b])=>b===v).map(([a])=>a),average=parents.length?parents.reduce((s,i)=>s+values[i],0)/parents.length:0;
    mass*=bitProbability(.2+strength*average,values[v]);
  }return {values,mass};});
}
const conditionalMean=(joint:ReturnType<typeof causalBinaryJoint>,conditions:Record<number,number>)=>{
  const selected=joint.filter(row=>Object.entries(conditions).every(([k,v])=>row.values[Number(k)]===v)),mass=selected.reduce((s,r)=>s+r.mass,0);
  return {mass,mean:mass>0?selected.reduce((s,r)=>s+r.mass*r.values[3],0)/mass:null};
};
/** Three do-calculus graph criteria, including the Z(W) ancestor restriction. */
export function doCalculus(graph:number,rule:number,z:number,x:number,w:number,strength:number,conditionW:number){
  for(const [v,lo,hi] of [[graph,0,5],[rule,1,3],[z,0,1],[x,0,1],[w,0,1],[conditionW,0,1]])if(!Number.isInteger(v)||v<lo||v>hi)throw new Error('Use the stated integer do-calculus choices.');
  const original=doGraphs[graph].edges,base=original.filter(([,b])=>b!==0),conditioned=conditionW?[0,2]:[0];
  const zInZW=!conditionW||!causalAncestors(4,base,[2]).has(1);
  const criterion=base.filter(([a,b])=>rule===2?a!==1:rule===3&&zInZW?b!==1:true);
  const separation=causalSeparation(4,criterion,3,1,conditioned),W:Record<number,number>=conditionW?{2:w}:{};
  const baseline=causalBinaryJoint(original,strength,{0:x}),acted=causalBinaryJoint(original,strength,{0:x,1:z});
  const left=rule===1?conditionalMean(baseline,{...W,1:z}):conditionalMean(acted,W);
  const right=conditionalMean(baseline,rule===2?{...W,1:z}:W);
  return {original,base,criterion,zInZW,separation,left,right,baseline,acted};
}
