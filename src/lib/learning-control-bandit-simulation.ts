/** Reproducible finite Bernoulli experiments; policy selection never receives true means. */
export const banditPolicies=['epsilon-greedy','UCB1','Thompson'] as const;
export function banditUniform(seed:number){let state=seed>>>0;const next=()=>{state=(Math.imul(1664525,state)+1013904223)>>>0;return (state+.5)/4294967296;};for(let i=0;i<8;i++)next();return next;}
/** a-th uniform order statistic has Beta(a,b) law for positive integer a,b. */
export function sampleIntegerBeta(a:number,b:number,uniform:()=>number){const draws=Array.from({length:a+b-1},()=>uniform()).sort((x,y)=>x-y);return draws[a-1];}
export interface BanditTrial {t:number;action:number;reward:number;counts:number[];successes:number[];scores:(number|null)[];reason:string;}
export function banditTrials(policy:number,rewards:number[][],N:number,epsilon:number,uniform:()=>number):BanditTrial[]{
 const counts=[0,0],successes=[0,0],rows:BanditTrial[]=[];for(let t=1;t<=N;t++){let action:number,scores:(number|null)[]=[null,null],reason='initial';if(t<=2)action=t-1;else{const means=counts.map((n,i)=>successes[i]/n);if(policy===0){scores=means;const explore=uniform()<epsilon,randomArm=Math.floor(2*uniform());action=explore?randomArm:means[0]>=means[1]?0:1;reason=explore?'explore':'greedy';}else if(policy===1){scores=means.map((m,i)=>m+Math.sqrt(2*Math.log(t-1)/counts[i]));action=scores[0]!>=scores[1]!?0:1;reason='ucb';}else{scores=counts.map((n,i)=>sampleIntegerBeta(successes[i]+1,n-successes[i]+1,uniform));action=scores[0]!>=scores[1]!?0:1;reason='posterior';}}
  const reward=rewards[action][counts[action]];counts[action]++;successes[action]+=reward;rows.push({t,action,reward,counts:counts.slice(),successes:successes.slice(),scores,reason});
 }return rows;
}
export function banditExperiment(mu0:number,mu1:number,N:number,epsilon:number,seed:number,selected:number){
 const range=(v:number,lo:number,hi:number)=>{if(!Number.isFinite(v)||v<lo||v>hi)throw new Error('Use the stated Bernoulli experiment domain.');};[mu0,mu1].forEach(v=>range(v,0,1));range(N,2,128);range(epsilon,0,.5);range(seed,1,999);range(selected,0,2);if(!Number.isInteger(N)||!Number.isInteger(seed)||!Number.isInteger(selected))throw new Error('Use integer horizon, seed and policy choices.');
 const means=[mu0,mu1],roles=[0x9e3779b9,0x85ebca6b],algorithmRoles=[0xc2b2ae35,0x27d4eb2f,0x165667b1],streams=means.map((mu,i)=>{const uniform=banditUniform(seed^roles[i]);return Array.from({length:N},()=>Number(uniform()<mu));}),best=Math.max(...means),results=banditPolicies.map((name,policy)=>{let rewardSum=0,pseudoRegret=0;const rows=banditTrials(policy,streams,N,epsilon,banditUniform(seed^algorithmRoles[policy])).map(row=>{rewardSum+=row.reward;pseudoRegret+=best-means[row.action];return {...row,rewardSum,pseudoRegret,shortfall:row.t*best-rewardSum};});return {name,rows,last:rows.at(-1)!};});
 return {streams,results,selected:results[selected],baseline:results[1]};
}
