export const repeatedGames=[
 {name:'coordination',row:[2,0,0,1],column:[2,0,0,1]},
 {name:'opposed payoffs',row:[1,-1,-1,1],column:[-1,1,1,-1]},
 {name:'social dilemma',row:[3,0,5,1],column:[3,5,0,1]},
] as const;
export interface RepeatedRound {round:number;a:number;b:number;rowPayoff:number;columnPayoff:number;cumulativeRow:number;cumulativeColumn:number;fixedRewards:[number,number];dynamicReward:number;externalRegret:number;}
const integer=(x:number,lo:number,hi:number)=>{if(!Number.isInteger(x)||x<lo||x>hi)throw new Error('Use the stated integer game, rule, horizon, prefix, discount and round controls.');};
export function repeatedGameTrace(game:number,opponent:number,policy:number,horizon:number,length=horizon){
 integer(game,0,2);integer(opponent,0,3);integer(policy,0,3);integer(horizon,1,24);integer(length,0,horizon);
 const g=repeatedGames[game],rows:RepeatedRound[]=[];let previousA=0,previousB=0,rowTotal=0,columnTotal=0,fixed0=0,fixed1=0,dynamic=0;
 for(let t=0;t<length;t++){
  const a=policy===0?0:policy===1?1:policy===2?t%2:previousB,b=opponent===0?0:opponent===1?t%2:opponent===2?Number(t>=Math.floor(horizon/2)):previousA,rowPayoff=g.row[2*a+b],columnPayoff=g.column[2*a+b];
  rowTotal+=rowPayoff;columnTotal+=columnPayoff;fixed0+=g.row[b];fixed1+=g.row[2+b];dynamic+=Math.max(g.row[b],g.row[2+b]);
  rows.push({round:t,a,b,rowPayoff,columnPayoff,cumulativeRow:rowTotal,cumulativeColumn:columnTotal,fixedRewards:[fixed0,fixed1],dynamicReward:dynamic,externalRegret:Math.max(fixed0,fixed1)-rowTotal});previousA=a;previousB=b;
 }
 return rows;
}
export function repeatedRegretLab(game:number,opponent:number,policy:number,horizon:number,requestedPrefix:number,discountPercent:number,requestedRound:number){
 integer(requestedPrefix,0,24);integer(discountPercent,0,100);integer(requestedRound,0,23);const prefix=Math.min(requestedPrefix,horizon),rows=repeatedGameTrace(game,opponent,policy,horizon,prefix),last=rows.at(-1),fixedRewards=last?.fixedRewards.slice()??[0,0],actual=last?.cumulativeRow??0,columnActual=last?.cumulativeColumn??0,dynamicReward=last?.dynamicReward??0,counterfactualTraces=[0,1].map(a=>repeatedGameTrace(game,opponent,a,horizon,prefix)),counterfactualRewards=counterfactualTraces.map(rs=>rs.at(-1)?.cumulativeRow??0),bestFixed=Math.max(...fixedRewards),bestPolicy=Math.max(...counterfactualRewards),externalRegret=bestFixed-actual,policyRegret=bestPolicy-actual,gamma=discountPercent/100,weights=rows.map((r,i)=>i===0?1:gamma**i),discountWeight=weights.reduce((s,v)=>s+v,0),discountedRow=rows.reduce((s,r,i)=>s+weights[i]*r.rowPayoff,0),discountedColumn=rows.reduce((s,r,i)=>s+weights[i]*r.columnPayoff,0),selectedRound=prefix===0?null:Math.min(requestedRound,prefix-1),selected=selectedRound===null?null:rows[selectedRound];
 return {game,opponent,policy,horizon,requestedPrefix,discountPercent,requestedRound,prefix,g:repeatedGames[game],rows,actual,columnActual,fixedRewards,bestFixed,counterfactualTraces,counterfactualRewards,bestPolicy,externalRegret,positiveRegret:Math.max(0,externalRegret),policyRegret,dynamicReward,dynamicRegret:dynamicReward-actual,averageRegret:prefix?externalRegret/prefix:null,averagePositiveRegret:prefix?Math.max(0,externalRegret)/prefix:null,gamma,weights,discountWeight,discountedRow,discountedColumn,discountAverage:discountWeight?discountedRow/discountWeight:null,selectedRound,selected};
}
