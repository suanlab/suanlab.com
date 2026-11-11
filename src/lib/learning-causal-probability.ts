/** Log probabilities preserve structural support before products round to zero. */
export function causalLogBit(p:number,bit:number){
 if(!Number.isFinite(p)||p<0||p>1||(bit!==0&&bit!==1))throw new Error('Use a binary probability and bit.');
 return bit?Math.log(p):Math.log1p(-p);
}
export function normalizeCausalLogs(logWeights:number[]){
 if(logWeights.some(v=>Number.isNaN(v)||v===Infinity))throw new Error('Use finite log weights or structural negative infinity.');
 const maximum=Math.max(-Infinity,...logWeights);
 if(maximum===-Infinity)return {probabilities:null,logMass:null,mass:0};
 const scaled=logWeights.map(v=>Math.exp(v-maximum)),total=scaled.reduce((s,v)=>s+v,0),logMass=maximum+Math.log(total);
 return {probabilities:scaled.map(v=>v/total),logMass,mass:Math.exp(logMass)};
}
