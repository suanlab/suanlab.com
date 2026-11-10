export interface HedgeRound {
 round:number;p:number[];q:number[]|null;g:number[];h:number[]|null;
 reward:number;columnReward:number|null;total:number;columnTotal:number|null;
 fixed:number[];columnFixed:number[]|null;regret:number;columnRegret:number|null;
 next:number[];columnNext:number[]|null;bound:number|null;
}
const integer=(v:number,lo:number,hi:number)=>{if(!Number.isInteger(v)||v<lo||v>hi)throw new Error('Use the stated integer algorithm, environment, rate, prior, horizon, prefix and round controls.');};
export function hedgeMixture(cumulative:readonly number[],eta:number,prior:number,algorithm:number){
 if(algorithm===1)return cumulative[0]>=cumulative[1]?[1,0]:[0,1];
 const a=Math.log(prior)+eta*cumulative[0],b=Math.log1p(-prior)+eta*cumulative[1],m=Math.max(a,b),x=Math.exp(a-m),y=Math.exp(b-m);return [x/(x+y),y/(x+y)];
}
export function hedgeLab(algorithm:number,mode:number,etaPercent:number,priorPercent:number,horizon:number,requestedPrefix:number,requestedRound:number){
 for(const [v,lo,hi] of [[algorithm,0,1],[mode,0,4],[etaPercent,0,400],[priorPercent,1,99],[horizon,1,64],[requestedPrefix,0,64],[requestedRound,0,63]])integer(v,lo,hi);
 const eta=etaPercent/100,prior=priorPercent/100,prefix=Math.min(horizon,requestedPrefix),C=Math.log(1/Math.min(prior,1-prior)),fixed=[0,0],columnFixed=[0,0],rows:HedgeRound[]=[];let total=0,columnTotal=0,pSum=0,qSum=0;
 const boundAt=(n:number)=>algorithm===0&&eta>0?C/eta+eta*n/8:null;
 for(let t=0;t<prefix;t++){
  const p=hedgeMixture(fixed,eta,prior,algorithm),q=mode===4?hedgeMixture(columnFixed,eta,1-prior,algorithm):null;
  const b=mode===0?0:mode===1?t%2:mode===2?Number(t>=Math.floor(horizon/2)):p[0]>=p[1]?1:0;
  const g=q?[q[0],q[1]]:b===0?[1,0]:[0,1],h=q?[p[1],p[0]]:null,reward=p[0]*g[0]+p[1]*g[1],columnReward=q&&h?q[0]*h[0]+q[1]*h[1]:null;
  total+=reward;fixed[0]+=g[0];fixed[1]+=g[1];pSum+=p[0];
  if(q&&h&&columnReward!==null){columnTotal+=columnReward;columnFixed[0]+=h[0];columnFixed[1]+=h[1];qSum+=q[0];}
  rows.push({round:t,p,q,g,h,reward,columnReward,total,columnTotal:q?columnTotal:null,fixed:fixed.slice(),columnFixed:q?columnFixed.slice():null,regret:Math.max(...fixed)-total,columnRegret:q?Math.max(...columnFixed)-columnTotal:null,next:hedgeMixture(fixed,eta,prior,algorithm),columnNext:q?hedgeMixture(columnFixed,eta,1-prior,algorithm):null,bound:boundAt(t+1)});
 }
 const last=rows.at(-1),regret=Math.max(...fixed)-total,columnRegret=mode===4?Math.max(...columnFixed)-columnTotal:null,averageP=prefix?pSum/prefix:null,averageQ=mode===4&&prefix?qSum/prefix:null,lower=averageP===null||mode!==4?null:2*Math.min(averageP,1-averageP)-1,upper=averageQ===null?null:2*Math.max(averageQ,1-averageQ)-1,gap=lower===null||upper===null?null:upper-lower,bound=boundAt(prefix),selectedRound=prefix?Math.min(requestedRound,prefix-1):null;
 return {algorithm,mode,eta,prior,horizon,requestedPrefix,requestedRound,prefix,C,rows,total,columnTotal:mode===4?columnTotal:null,fixed:fixed.slice(),columnFixed:mode===4?columnFixed.slice():null,regret,positiveRegret:Math.max(0,regret),columnRegret,bound,boundSlack:bound===null?null:bound-regret,averageRegret:prefix?regret/prefix:null,averageBound:prefix&&bound!==null?bound/prefix:null,tunedEta:Math.sqrt(8*C/horizon),next:last?.next??hedgeMixture([0,0],eta,prior,algorithm),columnNext:mode===4?(last?.columnNext??hedgeMixture([0,0],eta,1-prior,algorithm)):null,selectedRound,selected:selectedRound===null?null:rows[selectedRound],averageP,averageQ,lower,upper,gap,gapFromRegrets:prefix&&columnRegret!==null?2*(regret+columnRegret)/prefix:null,gapBound:prefix&&mode===4&&bound!==null?4*bound/prefix:null};
}
