/** Three private-value agents. Rules see reports only; values enter only the evaluator. */
export type MechanismRule=0|1|2;
const sum=(xs:readonly number[])=>xs.reduce((a,b)=>a+b,0);
function integer(x:number,lo:number,hi:number){if(!Number.isInteger(x)||x<lo||x>hi)throw new Error('Use the complete stated integer mechanism, values, reports, reserve/cost and agent controls.');}
export function directOutcome(rule:number,reports:readonly number[],parameter:number){
 integer(rule,0,2);integer(parameter,0,30);if(reports.length!==3)throw new Error('Exactly three submitted reports are required.');reports.forEach(b=>integer(b,0,10));
 const allocation=[0,0,0],payments=[0,0,0];let winner:number|null=null,build=false;
 if(rule===2){build=sum(reports)>=parameter;if(build)for(let i=0;i<3;i++){allocation[i]=1;payments[i]=Math.max(0,parameter-sum(reports.filter((_,j)=>j!==i)));}}
 else {const highest=Math.max(...reports);if(highest>=parameter){winner=reports.indexOf(highest);allocation[winner]=1;payments[winner]=rule===0?highest:Math.max(parameter,...reports.filter((_,j)=>j!==winner));}}
 return{allocation,payments,winner,build,revenue:sum(payments),resourceCost:rule===2&&build?parameter:0};
}
export function mechanismLab(rule:number,v0:number,v1:number,v2:number,b0:number,b1:number,b2:number,parameter:number,agent:number){
 integer(rule,0,2);integer(parameter,0,30);integer(agent,0,2);const values=[v0,v1,v2],reports=[b0,b1,b2];values.forEach(v=>integer(v,0,10));reports.forEach(b=>integer(b,0,10));
 const evaluate=(b:readonly number[])=>{const o=directOutcome(rule,b,parameter),benefits=o.allocation.map((x,i)=>x*values[i]),utilities=benefits.map((v,i)=>v-o.payments[i]),operatorNet=o.revenue-o.resourceCost,welfare=sum(benefits)-o.resourceCost;return{...o,benefits,utilities,operatorNet,welfare};};
 const current=evaluate(reports),truthful=evaluate(values),curves=values.map((_,i)=>Array.from({length:11},(_,b)=>{const bs=reports.slice();bs[i]=b;return evaluate(bs).utilities[i];})),bestUtilities=curves.map(v=>Math.max(...v)),bestReports=curves.map((v,i)=>v.flatMap((u,b)=>u===bestUtilities[i]?[b]:[])),conditionalGains=bestUtilities.map((u,i)=>u-current.utilities[i]),truthUtilities=curves.map((v,i)=>v[values[i]]),truthGains=truthUtilities.map((u,i)=>u-current.utilities[i]);
 // Exhaustive selected-type check: both opponents' reports vary, own true type stays fixed.
 const opponents=[0,1,2].filter(i=>i!==agent),cells:{opponents:number[];truthUtility:number;bestUtility:number;advantage:number;bestReports:number[]}[]=[];let witness:{opponents:number[];report:number;truthUtility:number;deviationUtility:number;gain:number}|null=null,maxTruthViolation=0;
 for(let x=0;x<=10;x++)for(let y=0;y<=10;y++){const bs=reports.slice();bs[opponents[0]]=x;bs[opponents[1]]=y;bs[agent]=values[agent];const tu=evaluate(bs).utilities[agent],us=Array.from({length:11},(_,b)=>{bs[agent]=b;return evaluate(bs).utilities[agent];}),best=Math.max(...us),advantage=best-tu,br=us.flatMap((u,b)=>u===best?[b]:[]);cells.push({opponents:[x,y],truthUtility:tu,bestUtility:best,advantage,bestReports:br});if(advantage>maxTruthViolation){maxTruthViolation=advantage;const report=br[0];witness={opponents:[x,y],report,truthUtility:tu,deviationUtility:us[report],gain:advantage};}}
 const efficientWelfare=rule===2?Math.max(0,sum(values)-parameter):Math.max(...values),otherReports=reports.filter((_,i)=>i!==agent),criticalLevel=rule===2?parameter-sum(otherReports):Math.max(parameter,...otherReports),pivotWithout=rule===2?Math.max(0,sum(otherReports)-parameter):null,pivotChosenOthers=rule===2?(sum(otherReports)-parameter)*Number(current.build):null;
 return{rule,values,reports,parameter,agent,current,truthful,curves,bestUtilities,bestReports,conditionalGains,truthUtilities,truthGains,cells,opponents,maxTruthViolation,witness,selectedTypeTruthDominant:maxTruthViolation===0,opponentProfiles:121,reportComparisons:1331,efficientWelfare,welfareGap:efficientWelfare-current.welfare,truthfulWelfareGap:efficientWelfare-truthful.welfare,allReportsTruthful:values.every((v,i)=>v===reports[i]),criticalLevel,pivotWithout,pivotChosenOthers};
}
