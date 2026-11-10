export interface DifferenceEdge {from:number;to:number;bound:number;atom:number;truth:boolean;}
const integer=(v:number,min:number,max:number)=>{if(!Number.isInteger(v)||v<min||v>max)throw new Error('Use the stated integer-difference and Boolean-mask domains.');};
export function differenceTheory(a:number,b:number,c:number,assignment:number){
 [a,b,c].forEach(v=>integer(v,-2,2));integer(assignment,0,7);
 const edges:DifferenceEdge[]=[[1,0,a],[2,1,b],[0,2,c]].map(([from,to,bound],atom)=>{const truth=Boolean((assignment>>atom)&1);return truth?{from,to,bound,atom,truth}:{from:to,to:from,bound:-bound-1,atom,truth};}),distances=[0,0,0],predecessor=[-1,-1,-1];let updated=-1;
 for(let pass=0;pass<3;pass++){updated=-1;for(let i=0;i<edges.length;i++){const edge=edges[i];if(distances[edge.to]>distances[edge.from]+edge.bound){distances[edge.to]=distances[edge.from]+edge.bound;predecessor[edge.to]=i;updated=edge.to;}}}
 if(updated<0)return {edges,feasible:true,witness:distances.map(v=>v-distances[0]),cycle:[] as DifferenceEdge[],cycleSum:null as number|null,learnedClause:null as number|null};
 let node=updated;for(let i=0;i<3;i++)node=edges[predecessor[node]].from;const start=node,cycle:DifferenceEdge[]=[];
 do{const edge=edges[predecessor[node]];cycle.push(edge);node=edge.from;}while(node!==start);cycle.reverse();
 return {edges,feasible:false,witness:null,cycle,cycleSum:cycle.reduce((sum,e)=>sum+e.bound,0),learnedClause:cycle.reduce((mask,e)=>mask|(1<<(e.atom+(e.truth?3:0))),0)};
}
export const clauseTruth=(clause:number,assignment:number)=>[0,1,2].some(i=>Boolean((clause>>i)&1)&&Boolean((assignment>>i)&1)||Boolean((clause>>(i+3))&1)&&!Boolean((assignment>>i)&1));
const clauseMasks=Array.from({length:64},(_,clause)=>Array.from({length:8},(_,A)=>A).reduce((mask,A)=>mask|(Number(clauseTruth(clause,A))<<A),0));
export function classifyCnf(c1:number,c2:number,c3:number,theoryMask:number){[c1,c2,c3].forEach(c=>integer(c,0,63));integer(theoryMask,0,255);const booleanMask=clauseMasks[c1]&clauseMasks[c2]&clauseMasks[c3];return {booleanMask,smtMask:booleanMask&theoryMask};}
export const satLiteralName=(slot:number)=>`${slot>=3?'¬':''}${['P','Q','R'][slot%3]}`;
export const satClauseName=(mask:number)=>mask===0?'FALSE':[0,1,2,3,4,5].filter(i=>(mask>>i)&1).map(satLiteralName).join(' ∨ ');
const pop=(mask:number)=>[0,1,2].filter(i=>(mask>>i)&1).length;
export function satSmtDifference(a:number,b:number,c:number,c1:number,c2:number,c3:number,selected:number){
 [a,b,c].forEach(v=>integer(v,-2,2));[c1,c2,c3].forEach(v=>integer(v,0,63));integer(selected,0,7);const clauses=[c1,c2,c3],theories=Array.from({length:8},(_,A)=>differenceTheory(a,b,c,A)),theoryMask=theories.reduce((mask,T,A)=>mask|(Number(T.feasible)<<A),0),classification=classifyCnf(c1,c2,c3,theoryMask),booleanModels=Array.from({length:8},(_,A)=>A).filter(A=>(classification.booleanMask>>A)&1),smtModels=booleanModels.filter(A=>theories[A].feasible),current=theories[selected],selectedClauses=clauses.map(C=>clauseTruth(C,selected)),selectedBoolean=selectedClauses.every(Boolean);
 const core=(allowable:number)=>{if(allowable===0)throw new Error('The empty theory has a model.');return Array.from({length:8},(_,S)=>S).sort((A,B)=>pop(A)-pop(B)||A-B).find(S=>clauses.reduce((mask,C,i)=>(S>>i)&1?mask&clauseMasks[C]:mask,allowable)===0)??null;};
 return {thresholds:[a,b,c],clauses,selected,current,theories,theoryMask,booleanModels,smtModels,selectedClauses,selectedBoolean,selectedSmt:selectedBoolean&&current.feasible,globalAssignment:smtModels[0]??null,globalWitness:smtModels.length?theories[smtModels[0]].witness:null,booleanCore:core(255),smtCore:core(theoryMask)};
}
