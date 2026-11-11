import {twoStepPaths,twoStepValue} from './learning-control-policy-gradient';
const finite=(x:number)=>Number.isFinite(x)?x:null;
const logProduct=(f:number[])=>f.some(x=>x===0)?-Infinity:f.reduce((sum,x)=>sum+Math.log(x),0);
const term=(logMass:number,amount:number)=>amount===0||logMass===-Infinity?0:Math.exp(logMass+Math.log(amount));
/** Exact known-model moments. Support uses positive factors, never a rounded path product. */
export function sequentialOffPolicy(p:number,q:number,b0:number,b1:number,alpha:number,gamma:number,cap:number){
 const range=(v:number,lo:number,hi:number)=>{if(!Number.isFinite(v)||v<lo||v>hi)throw new Error('Use the stated sequential importance-weight domain.');};[p,q,b0,b1,gamma].forEach(v=>range(v,0,1));range(alpha,.5,1);range(cap,.25,4);
 const target=twoStepPaths(p,q,alpha),behavior=twoStepPaths(b0,b1,alpha),oracle=twoStepValue(p,q,alpha,gamma).J;
 const rows=target.map((v,i)=>{const b=behavior[i],targetPositive=v.factors.every(x=>x>0),behaviorPositive=b.factors.every(x=>x>0),logTarget=logProduct(v.factors),logBehavior=logProduct(b.factors),logWeight=targetPositive&&behaviorPositive?logTarget-logBehavior:-Infinity,logPrefix=v.factors[0]===0?-Infinity:Math.log(v.factors[0])-Math.log(b.factors[0]),G=v.r0+gamma*v.r1,raw=behaviorPositive?term(logWeight,1):null,clipped=behaviorPositive?term(Math.min(Math.log(cap),logWeight),1):null,fullValue=behaviorPositive?finite(term(logWeight,G)):null,pdisValue=behaviorPositive?finite(term(logPrefix,v.r0)+term(logWeight,gamma*v.r1)):null;return {...v,behaviorProbability:b.probability,targetPositive,behaviorPositive,logTarget,logBehavior,logWeight,logPrefix,G,rawWeight:raw===null?null:finite(raw),rawOverflow:raw!==null&&!Number.isFinite(raw),clippedWeight:clipped,fullValue,pdisValue};});
 const fullSupport=rows.every(v=>!v.targetPositive||v.behaviorPositive),firstSupport=[0,1].every(a=>(a?p:1-p)===0||(a?b0:1-b0)>0),pdisSupport=firstSupport&&(gamma===0||fullSupport),observedFull=rows.reduce((sum,v)=>sum+(v.behaviorPositive?term(v.logTarget,v.G):0),0),firstMean=b0<1?1-p:0,observedPdis=firstMean+gamma*rows.reduce((sum,v)=>sum+(v.behaviorPositive?term(v.logTarget,v.r1):0),0),coveredMass=rows.reduce((sum,v)=>sum+(v.behaviorPositive?term(v.logTarget,1):0),0);
 const clipMean=(c:number)=>rows.reduce((sum,v)=>sum+(v.behaviorPositive?term(v.logBehavior+Math.min(Math.log(c),v.logWeight),v.G):0),0);
 const expandedVariance=(kind:'full'|'pdis',mean:number)=>{let result=0;for(const v of rows){if(!v.behaviorPositive)continue;const weighted=kind==='full'?term(v.logTarget-.5*v.logBehavior,v.G):term(.5*v.logBehavior+v.logPrefix,v.r0)+term(v.logTarget-.5*v.logBehavior,gamma*v.r1),center=term(.5*v.logBehavior,mean);result+=(weighted-center)**2;}return finite(result);};
 // Var(X) = sum_{i<j} B_i B_j (X_i-X_j)^2. Keep reward components
 // separate until subtraction: rounding 1+gamma first erases small effects.
 const difference=(left:number,right:number,scale:number)=>{
  if(left===right)return 0;
  const hi=Math.max(left,right),lo=Math.min(left,right);
  const magnitude=Math.exp(scale+hi+Math.log(-Math.expm1(lo-hi)));
  return left>right?magnitude:-magnitude;
 };
 const variance=(kind:'full'|'pdis')=>{
  const supported=rows.filter(v=>v.behaviorPositive),differences:number[]=[];
  for(let i=0;i<supported.length;i++)for(let j=i+1;j<supported.length;j++){
   const x=supported[i],y=supported[j],scale=.5*(x.logBehavior+y.logBehavior);
   const first=(v:typeof x)=>v.r0===0?-Infinity:kind==='full'?v.logWeight:v.logPrefix;
   const last=(v:typeof x)=>v.r1===0?-Infinity:v.logWeight;
   differences.push(difference(first(x),first(y),scale)+(gamma===0?0:difference(last(x),last(y),scale+Math.log(gamma))));
  }
  const scale=Math.max(0,...differences.map(Math.abs));
  if(!Number.isFinite(scale))return null;
  return scale===0?0:finite(scale*(scale*differences.reduce((sum,v)=>sum+(v/scale)**2,0)));
 };
 const clippedMean=clipMean(cap),clipCurve=Array.from({length:101},(_,i)=>{const c=.25+3.75*i/100;return [c,clipMean(c)] as [number,number];});
 return {rows,oracle,fullSupport,firstSupport,pdisSupport,coveredMass,observedFull,observedPdis,fullMean:fullSupport?observedFull:null,pdisMean:pdisSupport?observedPdis:null,fullVariance:fullSupport?variance('full'):null,pdisVariance:pdisSupport?variance('pdis'):null,expandedFullVariance:fullSupport?expandedVariance('full',observedFull):null,expandedPdisVariance:pdisSupport?expandedVariance('pdis',observedPdis):null,clippedMean,clippedBias:clippedMean-oracle,clipCurve,overflowCount:rows.filter(v=>v.rawOverflow).length};
}
