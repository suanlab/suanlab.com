// Analytic scalar mechanisms; plots are floating evaluations, never exact rational certificates.
const integer=(v:number,l:number,h:number)=>{if(!Number.isInteger(v)||v<l||v>h)throw new Error('Use the stated integer sensitivity/noise controls.');};
const logSqrt2Pi=.5*Math.log(2*Math.PI);
// Laplace continued fraction for the Gaussian Mills ratio, used only at z>=1.
function mills(z:number){let t=0;for(let j=256;j>=1;j--)t=j/(z+t);return 1/(z+t);}
function logPhi(x:number):number{
 if(x<=-1)return -.5*x*x-logSqrt2Pi+Math.log(mills(-x));
 if(x>=1)return Math.log1p(-Math.exp(logPhi(-x)));
 let term=x,sum=x;for(let k=1;k<30;k++){term*=(-x*x/2)/k;sum+=term/(2*k+1);}return Math.log(.5+sum/Math.sqrt(2*Math.PI));
}
export function gaussianLogDelta(gap:number,sigma:number,epsilon:number):number|null{
 if(!Number.isFinite(gap)||gap<0||gap>10||(gap>0&&gap<1e-6)||!Number.isFinite(sigma)||sigma<.1||sigma>10||!Number.isFinite(epsilon)||epsilon<0||epsilon>10)throw new Error('Use finite nonnegative gap/epsilon and positive sigma.');
 if(gap===0)return null;
 const h=gap/sigma,a=h/2-epsilon/h,b=-h/2-epsilon/h;
 if(a<=-1){
  // phi(a)=exp(epsilon)phi(b). Evaluate R(-a)-R(-b) without cancellation.
  const u=-a,v=u+h;let tu=0,tv=0,diff=0;
  for(let j=256;j>=1;j--){const du=u+tu,dv=v+tv;diff=j*(h-diff)/(du*dv);tu=j/du;tv=j/dv;}
  return -.5*a*a-logSqrt2Pi+Math.log(h)+Math.log1p(-diff/h)-Math.log(u+tu)-Math.log(v+tv);
 }
 const p=logPhi(a),q=epsilon+logPhi(b),ratio=q-p;
 return p+Math.log(-Math.expm1(Math.min(-Number.EPSILON,ratio)));
}
export function scalarNoiseProfile(family:'laplace'|'gaussian',gap:number,scale:number,epsilon:number){
 if(!Number.isFinite(gap)||gap<0||gap>10||(gap>0&&gap<1e-6)||!Number.isFinite(scale)||scale<.1||scale>10||!Number.isFinite(epsilon)||epsilon<0||epsilon>10)throw new Error('Use finite nonnegative gap/epsilon and positive noise scale.');
 if(family!=='laplace'&&family!=='gaussian')throw new Error('Use a supported noise family.');
 const eta=gap/scale,zero=gap===0||(family==='laplace'&&epsilon>=eta),logDelta=zero?null:family==='gaussian'?gaussianLogDelta(gap,scale,epsilon):Math.log(-Math.expm1((epsilon-eta)/2)),delta=logDelta===null?0:Math.exp(logDelta),threshold=zero?null:family==='laplace'?(gap-scale*epsilon)/2:gap/2-epsilon*scale*scale/gap;
 return{delta,logDelta,log10Delta:logDelta===null?null:logDelta/Math.LN10,analyticPositive:!zero,underflow:!zero&&delta===0,threshold,pureEpsilon:family==='laplace'?eta:gap===0?0:null,pureInfinite:family==='gaussian'&&gap>0};
}
export function boundedNoiseLab(family:'laplace'|'gaussian',n:number,B:number,query:number,noiseTimes10:number,epsilonTimes10:number,deltaPermille:number,requestedRecord:number){
 for(const [v,l,h]of [[n,1,32],[B,1,10],[query,0,3],[noiseTimes10,1,100],[epsilonTimes10,0,100],[deltaPermille,0,1000],[requestedRecord,0,10]])integer(v,l,h);
 const scale=noiseTimes10/10,epsilon=epsilonTimes10/10,claimedDelta=deltaPermille/1000,record=Math.min(requestedRecord,B),sensitivity=query===0?B:query===1?B/n:query===2?1:0,gap=query===0?record:query===1?record/n:query===2?Number(record>B/2):0,base=query===3?7:0,global=scalarNoiseProfile(family,sensitivity,scale,epsilon),pair=scalarNoiseProfile(family,gap,scale,epsilon);
 // Pure Laplace boundary uses integer cross-products, avoiding a rounded equality.
 const purePass=family==='laplace'?(query===3||epsilonTimes10*noiseTimes10*(query===1?n:1)>=(query===2?1:B)*100):sensitivity===0;
 const zero=(r:ReturnType<typeof scalarNoiseProfile>)=>{r.delta=0;r.logDelta=null;r.log10Delta=null;r.analyticPositive=false;r.underflow=false;r.threshold=null;};
 if(family==='laplace'&&purePass)zero(global);
 if(family==='laplace'&&(query===3||epsilonTimes10*noiseTimes10*(query===1?n:1)>=(query===2?gap:record)*100))zero(pair);
 const passes=purePass||claimedDelta>0&&(global.logDelta===null||global.logDelta<=Math.log(claimedDelta)),MSE=(family==='laplace'?2:1)*scale*scale,MAE=family==='laplace'?scale:scale*Math.sqrt(2/Math.PI),left=-6*scale,right=sensitivity+6*scale,density=(x:number,c:number)=>family==='laplace'?Math.exp(-Math.abs(x-c)/scale)/(2*scale):Math.exp(-.5*((x-c)/scale)**2)/(scale*Math.sqrt(2*Math.PI)),densityCurves=[0,sensitivity].map(c=>Array.from({length:241},(_,i)=>{const x=left+(right-left)*i/240;return[x,density(x,c)] as [number,number];})),curve=Array.from({length:101},(_,i)=>[i/10,family==='laplace'&&(query===3||i*noiseTimes10*(query===1?n:1)>=(query===2?1:B)*100)?0:scalarNoiseProfile(family,sensitivity,scale,i/10).delta] as [number,number]);
 return{family,n,B,query,scale,epsilon,claimedDelta,requestedRecord,record,gap,sensitivity,base,neighborValue:base+gap,global,pair,purePass,passes,MSE,MAE,densityCurves,densityBounds:[left,right,0,(family==='laplace'?1/(2*scale):1/(scale*Math.sqrt(2*Math.PI)))*1.12],curve,unitVarianceRiskForMean:query===1?MSE:null};
}
