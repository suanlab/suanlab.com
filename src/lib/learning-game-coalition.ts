// Exact three-player coalition geometry; all exposed payoff numerators use denominator12.
export const coalitionOrders=[[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]] as const;
export const coalitionDenominator=12;
const sumFor=(x:readonly number[],mask:number)=>x.reduce((s,v,i)=>s+((mask&(1<<i))?v:0),0);
export function coalitionValues(family:number,scale:number){
 if(!Number.isInteger(family)||family<0||family>4||!Number.isInteger(scale)||scale<1||scale>4)throw new Error('Use a stated coalition family and integer scale.');
 return Array.from({length:8},(_,S)=>scale*(family===0?sumFor([1,2,3],S):family===1?(S===7?6:0):family===2?([1,2,4].filter(b=>S&b).length>=2?6:0):family===3?((S&1)&&(S&6)?6:0):((S&3)===3?6:0)));
}
export function coalitionCoreVertices(values:readonly number[]){
 if(values.length!==8||values[0]!==0||values.some(v=>!Number.isInteger(v)||Math.abs(v)>96))throw new Error('Use eight bounded integer coalition values, with empty coalition value0.');
 const total=values[7],constraints=Array.from({length:6},(_,i)=>{const mask=i+1,z=Number(Boolean(mask&4));return [Number(Boolean(mask&1))-z,Number(Boolean(mask&2))-z,values[mask]-z*total];}),found=new Map<string,number[]>();
 for(let i=0;i<6;i++)for(let j=i+1;j<6;j++){
  const [a,b,c]=constraints[i],[d,e,f]=constraints[j],det=a*e-b*d;if(!det)continue;
  const x=12*(c*e-b*f)/det,y=12*(a*f-c*d)/det;
  if(!Number.isInteger(x)||!Number.isInteger(y))throw new Error('Three-player integer-value intersection lost its stated denominator.');
  if(constraints.every(([A,B,C])=>A*x+B*y>=12*C)){const v=[x===0?0:x,y===0?0:y,12*total-x-y];found.set(v.join(','),v);}
 }
 const vertices=[...found.values()],cx=vertices.reduce((s,v)=>s+v[0],0)/Math.max(1,vertices.length),cy=vertices.reduce((s,v)=>s+v[1],0)/Math.max(1,vertices.length);
 vertices.sort((a,b)=>Math.atan2(a[1]-cy,a[0]-cx)-Math.atan2(b[1]-cy,b[0]-cx));
 const dimension=vertices.length===0?null:vertices.length===1?0:vertices.every(v=>(vertices[1][0]-vertices[0][0])*(v[1]-vertices[0][1])===(vertices[1][1]-vertices[0][1])*(v[0]-vertices[0][0]))?1:2;
 return {vertices,dimension,constraints};
}
export function coalitionLab(family:number,scale:number,queryP0:number,queryP1:number,coalition:number,permutation:number,player:number){
 for(const [v,lo,hi] of [[family,0,4],[scale,1,4],[queryP0,0,12],[queryP1,0,12],[coalition,0,7],[permutation,0,5],[player,0,2]])if(!Number.isInteger(v)||v<lo||v>hi)throw new Error('Use the stated integer game, scale, allocation, coalition, order and player controls.');
 const values=coalitionValues(family,scale),total=values[7],orders=coalitionOrders.map(order=>{let S=0;const marginals=[0,0,0],steps=[{coalition:0,value:0,player:null as number|null,marginal:0}];for(const i of order){const after=S|(1<<i),delta=values[after]-values[S];marginals[i]=delta;steps.push({coalition:after,value:values[after],player:i,marginal:delta});S=after;}return {order,marginals,steps};}),shapleyNumerators=[0,1,2].map(i=>2*orders.reduce((s,row)=>s+row.marginals[i],0)),allocationNumerators=[total*queryP0,total*queryP1,total*(12-queryP0-queryP1)],slacks=(x:readonly number[])=>values.map((v,S)=>sumFor(x,S)-12*v),allocationSlacks=slacks(allocationNumerators),shapleySlacks=slacks(shapleyNumerators),core=coalitionCoreVertices(values),sum=allocationNumerators.reduce((s,v)=>s+v,0),efficient=sum===12*total,individuallyRational=[1,2,4].every(S=>allocationSlacks[S]>=0),inCore=efficient&&allocationSlacks.every(v=>v>=0),selected=orders[permutation],selectedMarginal=selected.marginals[player];
 let running=0;const marginalAverages=orders.map((row,i)=>{running+=row.marginals[player];return running/(i+1);});
 return {family,scale,queryP0,queryP1,coalition,permutation,player,D:12,values,total,orders,shapleyNumerators,allocationNumerators,allocationSlacks,shapleySlacks,core,efficient,individuallyRational,inCore,shapleyInCore:shapleySlacks.every(v=>v>=0),selected,selectedMarginal,marginalAverages,coalitionAllocationNumerator:sumFor(allocationNumerators,coalition),coalitionSlackNumerator:allocationSlacks[coalition]};
}
