export type ProofType='P'|'Q'|'R'|{from:ProofType;to:ProofType};
export type ProofTerm={tag:'var';index:number}|{tag:'lambda';annotation:ProofType;body:ProofTerm}|{tag:'apply';fn:ProofTerm;arg:ProofTerm};
export const arrow=(from:ProofType,to:ProofType):ProofType=>({from,to});
const P:ProofType='P',Q:ProofType='Q',R:ProofType='R',pq=arrow(P,Q),qr=arrow(Q,R),pr=arrow(P,R);
export const proofGoals:ProofType[]=[P,Q,R,arrow(P,P),pq,pr,arrow(pq,pq),arrow(P,arrow(Q,P)),arrow(pq,arrow(qr,pr)),arrow(arrow(pq,P),P)];
export const proofContexts=[{types:[] as ProofType[],names:[] as string[]},{types:[pq,qr],names:['f','g']},{types:[P,pq],names:['p','f']},{types:[P],names:['p']},{types:[pq],names:['f']}];
const V=(index:number):ProofTerm=>({tag:'var',index}),L=(annotation:ProofType,body:ProofTerm):ProofTerm=>({tag:'lambda',annotation,body}),A=(fn:ProofTerm,arg:ProofTerm):ProofTerm=>({tag:'apply',fn,arg});
export const proofCandidates:ProofTerm[]=[L(P,V(0)),L(P,A(V(2),A(V(1),V(0)))),A(V(0),V(0)),V(0),A(V(1),V(0)),L(P,L(Q,V(1))),L(pq,L(qr,L(P,A(V(1),A(V(2),V(0)))))),L(pq,V(0))];
export const proofTypeKey=(t:ProofType):string=>typeof t==='string'?t:`(${proofTypeKey(t.from)}>${proofTypeKey(t.to)})`;
export const sameProofType=(a:ProofType,b:ProofType)=>proofTypeKey(a)===proofTypeKey(b);
export function proofTypeLabel(t:ProofType):string{return typeof t==='string'?t:`${typeof t.from==='string'?t.from:`(${proofTypeLabel(t.from)})`}→${proofTypeLabel(t.to)}`;}
export const proofTermSize=(t:ProofTerm):number=>t.tag==='var'?1:t.tag==='lambda'?1+proofTermSize(t.body):1+proofTermSize(t.fn)+proofTermSize(t.arg);
export function proofTermLabel(t:ProofTerm,names:string[]=[],depth=0):string{return t.tag==='var'?(names[t.index]??`#${t.index}`):t.tag==='lambda'?`(λx${depth}:${proofTypeLabel(t.annotation)}. ${proofTermLabel(t.body,[`x${depth}`,...names],depth+1)})`:`(${proofTermLabel(t.fn,names,depth)} ${proofTermLabel(t.arg,names,depth)})`;}
export interface ProofNode {id:number;parent:number|null;depth:number;term:ProofTerm;context:ProofType[];names:string[];children:number[];}
export interface ProofStep {nodeId:number;rule:'var'|'lambda'|'apply';context:ProofType[];type:ProofType|null;error:'unbound-variable'|'function-required'|'argument-mismatch'|'premise-failed'|null;premises:number[];}
export function checkProofTerm(term:ProofTerm,context:ProofType[],names=context.map((_,i)=>`h${i}`)){
 const nodes:ProofNode[]=[],steps:ProofStep[]=[];
 const index=(t:ProofTerm,ctx:ProofType[],ns:string[],depth:number,lambdaDepth:number,parent:number|null):number=>{const node:ProofNode={id:nodes.length,parent,depth,term:t,context:ctx.slice(),names:ns.slice(),children:[]};nodes.push(node);if(t.tag==='lambda')node.children.push(index(t.body,[t.annotation,...ctx],[`x${lambdaDepth}`,...ns],depth+1,lambdaDepth+1,node.id));if(t.tag==='apply')node.children.push(index(t.fn,ctx,ns,depth+1,lambdaDepth,node.id),index(t.arg,ctx,ns,depth+1,lambdaDepth,node.id));return node.id;};
 index(term,context,names,0,0,null);
 const visit=(id:number):ProofStep=>{const n=nodes[id],t=n.term,premises:number[]=[];let type:ProofType|null=null,error:ProofStep['error']=null;
  if(t.tag==='var'){if(!Number.isInteger(t.index)||t.index<0||t.index>=n.context.length)error='unbound-variable';else type=n.context[t.index];}
  else if(t.tag==='lambda'){const b=visit(n.children[0]);premises.push(b.nodeId);if(b.type===null)error='premise-failed';else type=arrow(t.annotation,b.type);}
  else{const f=visit(n.children[0]);premises.push(f.nodeId);if(f.type===null)error='premise-failed';else if(typeof f.type==='string')error='function-required';else{const a=visit(n.children[1]);premises.push(a.nodeId);if(a.type===null)error='premise-failed';else if(!sameProofType(f.type.from,a.type))error='argument-mismatch';else type=f.type.to;}}
  const step:ProofStep={nodeId:id,rule:t.tag,context:n.context.slice(),type,error,premises};steps.push(step);return step;
 };
 const root=visit(0);return {nodes,steps,type:root.type,valid:root.type!==null};
}
function* proofBudgetPartitions(total:number,count:number):Generator<number[]>{if(count===0){if(total===0)yield [];return;}if(total<count)return;for(let first=1;first<=total-count+1;first++)for(const tail of proofBudgetPartitions(total-first,count-1))yield [first,...tail];}
export function boundedProofSearch(goal:ProofType,context:ProofType[],budget:number){
 if(!Number.isInteger(budget)||budget<1||budget>9)throw new Error('Use a normal-term node budget from1 to9.');
 const cache=new Map<string,ProofTerm|null>();let subgoals=0;
 const exact=(target:ProofType,ctx:ProofType[],size:number):ProofTerm|null=>{if(size<1)return null;const key=`${proofTypeKey(target)}|${ctx.map(proofTypeKey).join(',')}|${size}`;if(cache.has(key))return cache.get(key)!;subgoals++;let result:ProofTerm|null=null;
  if(typeof target!=='string'&&size>=2){const body=exact(target.to,[target.from,...ctx],size-1);if(body)result=L(target.from,body);}
  if(!result)for(let j=0;j<ctx.length&&!result;j++){let tail=ctx[j];const args:ProofType[]=[];while(true){if(sameProofType(tail,target))for(const allocation of proofBudgetPartitions(size-1-args.length,args.length)){const terms=args.map((arg,i)=>exact(arg,ctx,allocation[i]));if(terms.every((x):x is ProofTerm=>x!==null)){result=terms.reduce((fn,arg)=>A(fn,arg),V(j));break;}}if(result||typeof tail==='string')break;args.push(tail.from);tail=tail.to;}}
  cache.set(key,result);return result;
 };
 let term:ProofTerm|null=null,size:number|null=null;for(let n=1;n<=budget;n++){term=exact(goal,context,n);if(term){size=n;break;}}
 const checked=term?checkProofTerm(term,context):null;if(checked&&(!checked.type||!sameProofType(checked.type,goal)))throw new Error('Search result failed the separate type checker.');
 return {term,size,subgoals,checked};
}
export const proofBooleanValue=(t:ProofType,assignment:number):boolean=>typeof t==='string'?Boolean(assignment&(1<<['P','Q','R'].indexOf(t))):!proofBooleanValue(t.from,assignment)||proofBooleanValue(t.to,assignment);
const integer=(v:number,min:number,max:number)=>{if(!Number.isInteger(v)||v<min||v>max)throw new Error('Use the stated goal, context, budget, candidate and rule-query indices.');};
export function typedProofLab(goalIndex:number,contextIndex:number,budget:number,candidateIndex:number,requestedStep:number){
 integer(goalIndex,0,9);integer(contextIndex,0,4);integer(budget,1,9);integer(candidateIndex,0,7);integer(requestedStep,0,14);
 const goal=proofGoals[goalIndex],context=proofContexts[contextIndex],candidate=proofCandidates[candidateIndex],checked=checkProofTerm(candidate,context.types,context.names),candidateProves=checked.type!==null&&sameProofType(checked.type,goal),sweep=Array.from({length:9},(_,i)=>boundedProofSearch(goal,context.types,i+1)),search=sweep[budget-1],selectedStep=Math.min(requestedStep,checked.steps.length-1),selected=checked.steps[selectedStep],countermodels=Array.from({length:8},(_,i)=>i).filter(m=>context.types.every(t=>proofBooleanValue(t,m))&&!proofBooleanValue(goal,m));
 return {goalIndex,contextIndex,budget,candidateIndex,requestedStep,goal,context,candidate,checked,candidateProves,sweep,search,selectedStep,selected,countermodels,candidateSize:proofTermSize(candidate)};
}
