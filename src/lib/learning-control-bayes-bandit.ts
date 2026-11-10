/** Exact finite Bayesian decision calculations for independent Beta/Bernoulli arms. */
export type BanditBelief=[number,number,number,number];
const integer=(v:number,lo:number,hi:number)=>{if(!Number.isInteger(v)||v<lo||v>hi)throw new Error('Use integer Beta counts, budget and outcome choices.');};
const factorial=(n:number)=>{let v=1;for(let k=2;k<=n;k++)v*=k;return v;};
export function integerBetaDensity(a:number,b:number,x:number){return factorial(a+b-1)/(factorial(a-1)*factorial(b-1))*x**(a-1)*(1-x)**(b-1);}
export function betaBanditUpdate(state:BanditBelief,arm:number,reward:number):BanditBelief{const next=state.slice() as BanditBelief;next[2*arm+(reward===1?0:1)]++;return next;}
export function exactBayesBandit(a0:number,b0:number,a1:number,b1:number,H:number,arm:number,reward:number){
 [a0,b0,a1,b1].forEach(v=>integer(v,1,8));integer(H,1,6);integer(arm,0,1);integer(reward,0,1);
 const initial:BanditBelief=[a0,b0,a1,b1],cache=new Map<string,{optimal:number;greedy:number;Q:[number,number];action:number}>();
 const solve=(state:BanditBelief,h:number):{optimal:number;greedy:number;Q:[number,number];action:number}=>{const key=state.join(',')+':'+h,old=cache.get(key);if(old)return old;if(h===0)return {optimal:0,greedy:0,Q:[0,0],action:0};const means=[state[0]/(state[0]+state[1]),state[2]/(state[2]+state[3])],Q:[number,number]=[0,0],G=[0,0];for(const i of [0,1]){const win=solve(betaBanditUpdate(state,i,1),h-1),lose=solve(betaBanditUpdate(state,i,0),h-1),p=means[i];Q[i]=p+p*win.optimal+(1-p)*lose.optimal;G[i]=p+p*win.greedy+(1-p)*lose.greedy;}const action=Q[0]>=Q[1]?0:1,greedyArm=means[0]>=means[1]?0:1,result={optimal:Q[action],greedy:G[greedyArm],Q,action};cache.set(key,result);return result;};
 const selected=solve(initial,H),means=[a0/(a0+b0),a1/(a1+b1)],variances=[a0*b0/((a0+b0)**2*(a0+b0+1)),a1*b1/((a1+b1)**2*(a1+b1+1))],updated=betaBanditUpdate(initial,arm,reward),prior=Array.from({length:201},(_,i)=>[i/200,integerBetaDensity(initial[2*arm],initial[2*arm+1],i/200)] as [number,number]),posterior=prior.map(([x])=>[x,integerBetaDensity(updated[2*arm],updated[2*arm+1],x)] as [number,number]),history=Array.from({length:H+1},(_,h)=>({h,...solve(initial,h)}));
 return {selected,means,variances,updated,prior,posterior,history,greedyArm:means[0]>=means[1]?0:1,advantage:selected.optimal-selected.greedy};
}
