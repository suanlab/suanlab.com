/** Discounted, known finite MDP. Costs are minimized; synchronous Bellman updates. */
export type BellmanVector=[number,number,number,number];
export const bellmanCheapCosts=[1,2,4] as const;
const sup=(a:number[],b:number[])=>Math.max(...a.map((v,i)=>Math.abs(v-b[i])));
const checked=(x:number,lo:number,hi:number)=>{if(!Number.isFinite(x)||x<lo||x>hi)throw new Error('Use the stated discounted Bellman domain.');};
export function bellmanBackup(V:BellmanVector,gamma:number,stay:number,fastCost:number){
 const Q=bellmanCheapCosts.map((c,s)=>[c+gamma*(stay*V[s]+(1-stay)*V[s+1]),fastCost+gamma*V[s+1]]),policy=Q.map(q=>q[0]<=q[1]?0:1),values=[...Q.map(q=>Math.min(...q)),0] as BellmanVector;
 return {Q,policy,values};
}
/** Upper triangular policy evaluation by back substitution, separate from Bellman iteration. */
export function evaluateBellmanPolicy(policy:number[],gamma:number,stay:number,fastCost:number):BellmanVector{
 const V:BellmanVector=[0,0,0,0];for(let s=2;s>=0;s--){const slow=policy[s]===0,Vnext=V[s+1];V[s]=slow?(bellmanCheapCosts[s]+gamma*(1-stay)*Vnext)/(1-gamma*stay):fastCost+gamma*Vnext;}return V;
}
export function discountedBellman(gamma:number,stay:number,fastCost:number,iterations:number,selected:number,initial:number){
 checked(gamma,0,.95);checked(stay,0,1);checked(fastCost,1,8);checked(iterations,0,40);checked(selected,0,2);checked(initial,-4,8);if(!Number.isInteger(iterations)||!Number.isInteger(selected))throw new Error('Use integer Bellman iteration and state choices.');
 const policyValues=Array.from({length:8},(_,code)=>{const policy=[code&1,(code>>1)&1,(code>>2)&1];return {policy,values:evaluateBellmanPolicy(policy,gamma,stay,fastCost)};}),optimal=Array.from({length:4},(_,s)=>Math.min(...policyValues.map(p=>p.values[s]))) as BellmanVector;
 let V:BellmanVector=[initial,initial,initial,0];const history=[{iteration:0,values:V.slice(),error:sup(V,optimal)}];for(let k=1;k<=iterations;k++){V=bellmanBackup(V,gamma,stay,fastCost).values;history.push({iteration:k,values:V.slice(),error:sup(V,optimal)});}
 const backup=bellmanBackup(V,gamma,stay,fastCost),residual=sup(backup.values,V),bound=residual/(1-gamma),greedyValue=evaluateBellmanPolicy(backup.policy,gamma,stay,fastCost),optimalBackup=bellmanBackup(optimal,gamma,stay,fastCost);
 return {V,optimal,history,policyValues,backup,residual,bound,error:sup(V,optimal),greedyValue,policyGap:sup(greedyValue,optimal),optimalPolicy:optimalBackup.policy,selected,selectedCosts:[bellmanCheapCosts[selected],fastCost],ties:backup.Q.map(q=>Number(q[0]===q[1]))};
}
