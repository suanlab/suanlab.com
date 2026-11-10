export const fptasWeights=[2,3,4,5] as const;
export interface FptasCell {weight:number;mask:number;}
const integer=(v:number,min:number,max:number)=>{if(!Number.isInteger(v)||v<min||v>max)throw new Error('Use the stated integer value, capacity, epsilon-tenths and DP-query domains.');};
export const fptasMaskValue=(values:number[],mask:number)=>values.reduce((s,v,i)=>s+v*((mask>>i)&1),0);
export function knapsackFptas(v0:number,v1:number,v2:number,v3:number,capacity:number,epsilonTenths:number){
 const values=[v0,v1,v2,v3];values.forEach(v=>integer(v,0,20));integer(capacity,0,14);integer(epsilonTenths,1,9);
 const eligible=fptasWeights.map(w=>w<=capacity),vmax=Math.max(0,...values.filter((_,i)=>eligible[i])),kNumerator=vmax>0?epsilonTenths*vmax:null,scaled=values.map((v,i)=>eligible[i]&&kNumerator!==null?Math.floor(40*v/kNumerator):0),scoreLimit=scaled.reduce((s,v)=>s+v,0),rows:(FptasCell|null)[][]=[Array.from({length:scoreLimit+1},(_,score)=>score===0?{weight:0,mask:0}:null)];
 for(let i=0;i<4;i++){
  const previous=rows[i],row=previous.slice();
  if(eligible[i])for(let score=scaled[i];score<=scoreLimit;score++){
   const source=previous[score-scaled[i]];if(!source)continue;
   const candidate={weight:source.weight+fptasWeights[i],mask:source.mask|(1<<i)},keep=row[score];
   if(keep===null||candidate.weight<keep.weight||(candidate.weight===keep.weight&&candidate.mask<keep.mask))row[score]=candidate;
  }
  rows.push(row);
 }
 let bestScore=0;for(let score=1;score<=scoreLimit;score++)if(rows[4][score]!==null&&rows[4][score]!.weight<=capacity)bestScore=score;
 const cell=rows[4][bestScore]!;
 return {values,capacity,epsilonTenths,eligible,vmax,kNumerator,kDenominator:40,scaled,scoreLimit,rows,bestScore,mask:cell.mask,weight:cell.weight,value:fptasMaskValue(values,cell.mask),denseCells:5*(scoreLimit+1),finiteCells:rows.reduce((s,row)=>s+row.filter(Boolean).length,0)};
}
// A separate 16-subset reference for teaching comparisons; the FPTAS never calls it.
export function knapsackFptasReference(values:number[],capacity:number){
 if(values.length!==4)throw new Error('Use four values.');values.forEach(v=>integer(v,0,20));integer(capacity,0,14);
 const feasible=Array.from({length:16},(_,mask)=>({mask,weight:fptasMaskValue([...fptasWeights],mask),value:fptasMaskValue(values,mask)})).filter(x=>x.weight<=capacity),optimum=Math.max(...feasible.map(x=>x.value)),optimizers=feasible.filter(x=>x.value===optimum);
 return {feasible,optimum,optimizers};
}
export function fptasComparison(v0:number,v1:number,v2:number,v3:number,capacity:number,epsilonTenths:number,row:number,score:number){
 integer(row,0,4);integer(score,0,160);const approximate=knapsackFptas(v0,v1,v2,v3,capacity,epsilonTenths),reference=knapsackFptasReference(approximate.values,capacity),inTable=score<=approximate.scoreLimit,cell=inTable?approximate.rows[row][score]:null,sweep=Array.from({length:9},(_,i)=>{const a=knapsackFptas(v0,v1,v2,v3,capacity,i+1);return {epsilon:(i+1)/10,value:a.value,denseCells:a.denseCells,guaranteed:(9-i)*reference.optimum/10};});
 return {approximate,reference,row,score,inTable,cell,sweep,ratio:reference.optimum>0?approximate.value/reference.optimum:null,guaranteed:(10-epsilonTenths)*reference.optimum/10,additiveAllowance:epsilonTenths*approximate.vmax/10};
}
