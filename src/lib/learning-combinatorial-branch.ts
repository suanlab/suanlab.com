export const branchWeights=[2,3,4] as const;
export type BranchStatus='open'|'split'|'bound'|'none'|'int';
export interface BranchRelaxation {upper12:number|null;point12:number[]|null;fractional:number|null;}
export interface BranchNode extends BranchRelaxation {id:number;parent:number|null;depth:number;slot:number;fixed:number[];status:BranchStatus;closedAt:number|null;incumbentAtClose:number|null;}
export interface BranchState {step:number;nodes:BranchNode[];frontier:number[];incumbent:number;witness:number[];upper12:number;processed:number|null;}
const integer=(v:number,min:number,max:number)=>{if(!Number.isInteger(v)||v<min||v>max)throw new Error('Use integer values, capacity and trace controls in the stated domain.');};
const objective=(values:number[],point:number[])=>point.reduce((s,x,i)=>s+values[i]*x,0);
export const twelfthsLabel=(value:number|null)=>{if(value===null)return '—';let a=value,b=12;while(b){const t=a%b;a=b;b=t;}return 12/a===1?String(value/a):`${value/a}/${12/a}`;};
export function branchRelaxation(values:number[],capacity:number,fixed:number[]):BranchRelaxation{
 if(values.length!==3||fixed.length!==3)throw new Error('Use three items.');values.forEach(v=>integer(v,0,9));integer(capacity,0,9);fixed.forEach(v=>integer(v,-1,1));
 let remaining=capacity-fixed.reduce((s,x,i)=>s+(x===1?branchWeights[i]:0),0);
 if(remaining<0)return {upper12:null,point12:null,fractional:null};
 const point12:number[]=fixed.map(x=>x===1?12:0),order=[0,1,2].filter(i=>fixed[i]===-1&&values[i]>0).sort((i,j)=>values[j]*branchWeights[i]-values[i]*branchWeights[j]||i-j);
 for(const i of order){const used=Math.min(branchWeights[i],remaining);point12[i]=used*(12/branchWeights[i]);remaining-=used;}
 return {upper12:objective(values,point12),point12,fractional:point12.findIndex(x=>x>0&&x<12)<0?null:point12.findIndex(x=>x>0&&x<12)};
}
export function branchBoundKnapsack(v0:number,v1:number,v2:number,capacity:number,requested:number,query:number){
 const values=[v0,v1,v2];values.forEach(v=>integer(v,0,9));integer(capacity,0,9);integer(requested,0,15);integer(query,0,14);
 const nodes:BranchNode[]=[],frontier:number[]=[],history:BranchState[]=[];
 const create=(fixed:number[],parent:number|null,depth:number,slot:number)=>{const node:BranchNode={id:nodes.length,parent,depth,slot,fixed:fixed.slice(),status:'open',closedAt:null,incumbentAtClose:null,...branchRelaxation(values,capacity,fixed)};nodes.push(node);return node.id;};
 create([-1,-1,-1],null,0,0);frontier.push(0);
 let witness=nodes[0].point12!.map(x=>x===12?1:0),incumbent=objective(values,witness);
 const save=(processed:number|null)=>history.push({step:history.length,nodes:nodes.map(n=>({...n,fixed:n.fixed.slice(),point12:n.point12?.slice()??null})),frontier:frontier.slice(),incumbent,witness:witness.slice(),upper12:Math.max(12*incumbent,...frontier.map(id=>nodes[id].upper12??-Infinity)),processed});
 save(null);
 while(frontier.length){
  const id=frontier.pop()!,node=nodes[id];
  if(node.upper12===null)node.status='none';
  else{
   const candidate=node.point12!.map(x=>x===12?1:0),value=objective(values,candidate);
   if(value>incumbent){incumbent=value;witness=candidate;}
   if(node.fractional===null)node.status='int';
   else if(node.upper12<=12*incumbent)node.status='bound';
   else{
    node.status='split';const i=node.fractional,zero=node.fixed.slice(),one=node.fixed.slice();zero[i]=0;one[i]=1;
    const left=create(zero,id,node.depth+1,2*node.slot),right=create(one,id,node.depth+1,2*node.slot+1);
    frontier.push(right,left);
   }
  }
  node.closedAt=history.length;node.incumbentAtClose=incumbent;save(id);
 }
 const selectedStep=Math.min(requested,history.length-1),current=history[selectedStep],selectedId=Math.min(query,current.nodes.length-1),selected=current.nodes[selectedId];
 return {values,capacity,requested,query,history,current,selectedStep,selectedId,selected,complete:current.frontier.length===0,certified:current.upper12===12*current.incumbent};
}
