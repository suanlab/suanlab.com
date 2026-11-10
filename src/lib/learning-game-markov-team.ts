// Exact rational policy evaluation and comparisons for two finite common-payoff Markov games.
type Q=readonly[bigint,bigint];
const Z=BigInt(0),O=BigInt(1),abs=(v:bigint)=>v<Z?-v:v;
const gcd=(a:bigint,b:bigint):bigint=>{a=abs(a);b=abs(b);while(b!==Z){const r=a%b;a=b;b=r;}return a;};
const q=(n:number|bigint,d:number|bigint=1):Q=>{let a=BigInt(n),b=BigInt(d);if(b===Z)throw new Error('Zero rational denominator.');if(b<Z){a=-a;b=-b;}const k=gcd(a,b);return[a/k,b/k];};
const add=(a:Q,b:Q)=>q(a[0]*b[1]+b[0]*a[1],a[1]*b[1]),neg=(a:Q)=>q(-a[0],a[1]),sub=(a:Q,b:Q)=>add(a,neg(b)),mul=(a:Q,b:Q)=>q(a[0]*b[0],a[1]*b[1]),div=(a:Q,b:Q)=>q(a[0]*b[1],a[1]*b[0]),cmp=(a:Q,b:Q)=>a[0]*b[1]-b[0]*a[1],zero=q(0),one=q(1),sum=(a:readonly Q[])=>a.reduce(add,zero);
const num=(a:Q)=>Number(a[0])/Number(a[1]),label=(a:Q)=>a[1]===O?String(a[0]):`${a[0]}/${a[1]}`;
const int=(v:number,lo:number,hi:number)=>{if(!Number.isInteger(v)||v<lo||v>hi)throw new Error('Use the stated integer family, discount, state policies and query controls.');};
export const markovTeamFamilies=[
 {rewards:[[4,0,0,3],[2,0,0,1]],nextOneEighths:[[0,0,0,0],[8,8,8,8]]},
 {rewards:[[4,2,2,1],[0,1,1,-1]],nextOneEighths:[[4,2,2,1],[8,6,6,0]]},
] as const;
function evaluate(family:number,gamma:number,p:readonly number[],r:readonly number[]){
 const g=q(gamma,100),f=markovTeamFamilies[family],masses=[0,1].map(s=>{const a=q(p[s],100),b=q(r[s],100);return [mul(a,b),mul(a,sub(one,b)),mul(sub(one,a),b),mul(sub(one,a),sub(one,b))];}),reward=masses.map((row,s)=>sum(row.map((w,i)=>mul(w,q(f.rewards[s][i]))))),nextOne=masses.map((row,s)=>sum(row.map((w,i)=>mul(w,q(f.nextOneEighths[s][i],8))))),a=sub(one,mul(g,sub(one,nextOne[0]))),b=neg(mul(g,nextOne[0])),c=neg(mul(g,sub(one,nextOne[1]))),d=sub(one,mul(g,nextOne[1])),det=sub(mul(a,d),mul(b,c)),V=[div(sub(mul(d,reward[0]),mul(b,reward[1])),det),div(sub(mul(a,reward[1]),mul(c,reward[0])),det)],Qvalues=f.rewards.map((row,s)=>row.map((v,i)=>{const t=q(f.nextOneEighths[s][i],8);return add(q(v),mul(g,add(mul(sub(one,t),V[0]),mul(t,V[1]))));})),residual=V.map((v,s)=>sub(v,add(reward[s],mul(g,add(mul(sub(one,nextOne[s]),V[0]),mul(nextOne[s],V[1]))))));
 return {V,reward,nextOne,masses,Qvalues,residual,det};
}
const policies=[[0,0],[0,100],[100,0],[100,100]] as const;
const expose=(e:ReturnType<typeof evaluate>)=>({V:e.V.map(num),exactV:e.V.map(label),reward:e.reward.map(num),exactReward:e.reward.map(label),nextOne:e.nextOne.map(num),exactNextOne:e.nextOne.map(label),masses:e.masses.map(row=>row.map(num)),Qvalues:e.Qvalues.map(row=>row.map(num)),exactQvalues:e.Qvalues.map(row=>row.map(label)),residual:e.residual.map(num),exactResidual:e.residual.map(label),determinant:num(e.det)});
function optimal<T extends {value:ReturnType<typeof evaluate>}>(rows:T[]){
 const best=rows.reduce((a,b)=>cmp(sum(a.value.V),sum(b.value.V))>=Z?a:b);
 if(!rows.every(row=>row.value.V.every((v,i)=>cmp(best.value.V[i],v)>=Z)))throw new Error('The enumerated finite MDP candidates lack a componentwise optimum.');
 return {best,all:rows.filter(row=>row.value.V.every((v,i)=>cmp(v,best.value.V[i])===Z))};
}
export function markovTeamPolicyValue(family:number,gamma:number,p0:number,p1:number,q0:number,q1:number){int(family,0,1);int(gamma,0,95);for(const v of [p0,p1,q0,q1])int(v,0,100);return expose(evaluate(family,gamma,[p0,p1],[q0,q1]));}
export function markovTeamLab(family:number,gammaPercent:number,p0:number,p1:number,q0:number,q1:number,state:number,joint:number){
 int(family,0,1);int(gammaPercent,0,95);for(const v of [p0,p1,q0,q1])int(v,0,100);int(state,0,1);int(joint,0,3);
 const p=[p0,p1],r=[q0,q1],current=evaluate(family,gammaPercent,p,r),team=policies.flatMap((a,i)=>policies.map((b,j)=>({id:4*i+j,p:a,q:b,value:evaluate(family,gammaPercent,a,b)}))),teamOpt=optimal(team),responses=[0,1].map(player=>{const candidates=policies.map((a,id)=>({id,policy:a,value:player===0?evaluate(family,gammaPercent,a,r):evaluate(family,gammaPercent,p,a)})),o=optimal(candidates);return {candidates,...o};}),gains=responses.map(o=>o.best.value.V.map((v,s)=>sub(v,current.V[s]))),teamGap=teamOpt.best.value.V.map((v,s)=>sub(v,current.V[s])),immediateBest=[Math.max(...markovTeamFamilies[family].rewards[0]),Math.max(...markovTeamFamilies[family].rewards[1])];
 return {family,gammaPercent,gamma:gammaPercent/100,p,q:r,state,joint,current:expose(current),team:team.map(o=>({id:o.id,p:[...o.p],q:[...o.q],...expose(o.value),optimal:teamOpt.all.includes(o)})),teamOptimalIds:teamOpt.all.map(o=>o.id),teamOptimalValue:teamOpt.best.value.V.map(num),exactTeamOptimalValue:teamOpt.best.value.V.map(label),responses:responses.map(o=>({optimalIds:o.all.map(c=>c.id),value:o.best.value.V.map(num),exactValue:o.best.value.V.map(label),candidates:o.candidates.map(c=>({id:c.id,policy:[...c.policy],...expose(c.value),optimal:o.all.includes(c)}))})),gains:gains.map(a=>a.map(num)),exactGains:gains.map(a=>a.map(label)),isStationaryNash:gains.every(a=>a.every(v=>v[0]===Z)),teamGap:teamGap.map(num),exactTeamGap:teamGap.map(label),isTeamOptimal:teamGap.every(v=>v[0]===Z),query:{state,joint,actions:[Math.floor(joint/2),joint%2],mass:num(current.masses[state][joint]),reward:markovTeamFamilies[family].rewards[state][joint],nextOne:markovTeamFamilies[family].nextOneEighths[state][joint]/8,Q:num(current.Qvalues[state][joint]),exactQ:label(current.Qvalues[state][joint]),immediateBest:immediateBest[state]}};
}
