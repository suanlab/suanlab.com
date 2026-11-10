// Exact stability decisions with floating trajectories for a two-player quadratic saddle game.
type V=readonly[number,number];type M=readonly[V,V];
const cleanZero=(v:number)=>v===0?0:v;
const apply=(m:M,[x,y]:V):V=>[cleanZero(m[0][0]*x+m[0][1]*y),cleanZero(m[1][0]*x+m[1][1]*y)];
const validate=(x:number,l:number,h:number)=>{if(!Number.isInteger(x)||x<l||x>h)throw new Error('Use the complete stated integer game/dynamics controls.');};
export function adversarialLab(rule:number,a:number,dampingIndex:number,etaIndex:number,xIndex:number,yIndex:number,steps:number,requestedStep:number){
 for(const [v,l,h] of [[rule,0,3],[a,-2,2],[dampingIndex,0,100],[etaIndex,0,200],[xIndex,-4,4],[yIndex,-4,4],[steps,0,64],[requestedStep,0,63]])validate(v,l,h);
 const mu=dampingIndex/100,h=etaIndex/100,d=1-h*mu,q=h*a,c=1-h*mu+h*h*(mu*mu-a*a),b=-q+2*h*h*mu*a,den=(1+h*mu)**2+q*q;
 const matrices:M[]=[[[d,-q],[q,d]],[[d,-q],[q*d,d-q*q]],[[c,b],[-b,c]],[[(1+h*mu)/den,-q/den],[q/den,(1+h*mu)/den]]];
 const zero=BigInt(0),D=BigInt(10000),D2=D*D,v=BigInt(etaIndex*dampingIndex),qn=BigInt(100*etaIndex*a),dn=D-v;
 const rationalLabel=(N:bigint,E:bigint)=>{let a=N,b=E;while(b!==zero){const t=a%b;a=b;b=t;}return E/a===BigInt(1)?String(N/a):`${N/a}/${E/a}`;};
 const conformal=(N:bigint,E:bigint)=>({exactNormSquaredFactor:rationalLabel(N,E),strict:N<E,closed:N<=E,critical:N===E,normSquaredFactor:Number(N)/Number(E)});
 const cn=D2-v*D+v*v-qn*qn,bn=-qn*D+BigInt(2)*v*qn,traceNumerator=BigInt(2)*dn*D-qn*qn,detNumerator=dn*dn,jury=[D2-traceNumerator+detNumerator,D2+traceNumerator+detNumerator,D2-detNumerator];
 const certs=[conformal(dn*dn+qn*qn,D2),{strict:jury.every(n=>n>zero),closed:jury.every(n=>n>=zero),critical:jury.every(n=>n>=zero)&&jury.some(n=>n===zero),normSquaredFactor:null,exactNormSquaredFactor:null},conformal(cn*cn+bn*bn,D2*D2),conformal(D2,(D+v)*(D+v)+qn*qn)];
 const cert=certs[rule],zeroField=a===0&&mu===0,selectedMatrix=matrices[rule],trace=selectedMatrix[0][0]+selectedMatrix[1][1],det=selectedMatrix[0][0]*selectedMatrix[1][1]-selectedMatrix[0][1]*selectedMatrix[1][0],disc=trace*trace-4*det;
 const rho=rule===1?(disc<0?Math.sqrt(Math.max(0,det)):Math.max(Math.abs((trace+Math.sqrt(disc))/2),Math.abs((trace-Math.sqrt(disc))/2))):Math.sqrt(certs[rule].normSquaredFactor!);
 const jordan=rule===1&&cert.critical&&qn!==zero&&traceNumerator*traceNumerator===BigInt(4)*detNumerator*D2;
 const initial:V=[xIndex/2,yIndex/2],F=([x,y]:V):V=>[mu*x+a*y,mu*y-a*x],norm=(v:V)=>Math.hypot(...v),residual=(v:V)=>norm(F(v)),distance=(v:V)=>zeroField?0:norm(v);
 const histories=matrices.map(m=>{const xs:V[]=[initial];for(let t=0;t<steps;t++)xs.push(apply(m,xs[t]));return xs;}),history=histories[rule],continuous=history.map((_,t):V=>{const time=t*h,e=Math.exp(-mu*time),C=Math.cos(a*time),S=Math.sin(a*time);return[e*(C*initial[0]-S*initial[1]),e*(S*initial[0]+C*initial[1])];});
 const averages:(V|null)[]=[null];let sx=0,sy=0;for(let t=1;t<=steps;t++){sx+=history[t-1][0];sy+=history[t-1][1];averages.push([sx/t,sy/t]);}
 const query=steps===0?null:Math.min(requestedStep,steps-1),z=query===null?null:history[query],gradient=z===null?null:F(z),predictor=z===null?null:[z[0]-h*gradient![0],z[1]-h*gradient![1]] as V,next=query===null?null:history[query+1],average=averages[steps];
 return{rule,a,mu,h,steps,requestedStep,query,zeroField,initial,matrices,selectedMatrix,history,histories,continuous,averages,final:history[steps],average,finalResidual:residual(history[steps]),averageResidual:average===null?null:residual(average),finalDistanceToSaddleSet:distance(history[steps]),spectralRadius:rho,certificate:{strictlyStable:cert.strict,radiusAtMostOne:cert.closed,radiusExactlyOne:cert.critical,allInitialConvergeToSaddleSet:cert.strict||zeroField,powersBounded:cert.closed&&!jordan,unitJordanBlock:jordan,normSquaredFactor:cert.normSquaredFactor,exactNormSquaredFactor:cert.exactNormSquaredFactor,exactAlternatingJuryNumerators:jury.map(String),exactJuryDenominator:String(D2)},selected:{z,next,gradient,predictor,updateSize:z===null?null:Math.hypot(next![0]-z[0],next![1]-z[1]),residual:z===null?null:residual(z)},alternatingInvariant:mu===0&&Math.abs(q)<2?{positiveDefinite:true,initial:initial[0]**2+initial[1]**2-q*initial[0]*initial[1],values:histories[1].map(([x,y])=>x*x+y*y-q*x*y)}:null,logResiduals:history.map((v,t)=>[t,Math.log1p(residual(v))/Math.LN10] as V),logContinuousResiduals:continuous.map((v,t)=>[t,Math.log1p(residual(v))/Math.LN10] as V),logAverageResiduals:averages.flatMap((v,t)=>v===null?[]:[[t,Math.log1p(residual(v))/Math.LN10] as V])};
}
