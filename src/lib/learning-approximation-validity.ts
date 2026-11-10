import type {Vec2} from './learning-simulations';
function controls(values:number[],ranges:readonly (readonly [number,number])[]){
 if(values.length!==ranges.length)throw new Error('Use the stated approximation control vector.');
 values.forEach((v,i)=>{if(!Number.isInteger(v)||v<ranges[i][0]||v>ranges[i][1])throw new Error('Use bounded integer approximation controls.');});
}
function pendulumRatio(degrees:number){
 let a=1,b=Math.cos(degrees*Math.PI/360),iterations=0;
 for(;iterations<16;iterations++){const nextA=(a+b)/2,nextB=Math.sqrt(a*b);a=nextA;b=nextB;if(Math.abs(a-b)<=Number.EPSILON*a)break;}
 return{ratio:2/(a+b),agmGap:Math.abs(a-b),iterations:iterations+1};
}
export function pendulumApproximation(values:number[]){
 controls(values,[[0,150],[1,20],[1,20],[0,1],[1,100]]);
 const[angleDegrees,lengthCode,gravityCode,approximation,toleranceCode]=values;
 const angleRadians=angleDegrees*Math.PI/180,length=lengthCode/4,gravity=gravityCode/2,tolerance=toleranceCode/1000,limitPeriod=2*Math.PI*Math.sqrt(length/gravity),r=pendulumRatio(angleDegrees),approximationRatio=approximation===0?1:1+angleRadians**2/16;
 const hasOscillation=angleDegrees>0,period=hasOscillation?limitPeriod*r.ratio:null,approximatePeriod=hasOscillation?limitPeriod*approximationRatio:null,relativeError=hasOscillation?Math.abs(r.ratio-approximationRatio)/r.ratio:null;
 const points=Array.from({length:151},(_,degrees)=>{const ratio=pendulumRatio(degrees).ratio,alpha=degrees*Math.PI/180,corrected=1+alpha*alpha/16;return{degrees,ratio,linearError:Math.abs(ratio-1)/ratio,correctedError:Math.abs(ratio-corrected)/ratio,approximationRatio:approximation===0?1:corrected};});
 const curves=[points.map(p=>[p.degrees,p.ratio] as Vec2),points.map(p=>[p.degrees,p.approximationRatio] as Vec2)];
 // Error at zero amplitude is unavailable; curve points start at one degree. The ratio plot includes a labeled limit at zero.
 const errorCurves=[points.slice(1).map(p=>[p.degrees,p.linearError*100] as Vec2),points.slice(1).map(p=>[p.degrees,p.correctedError*100] as Vec2),points.slice(1).map(p=>[p.degrees,tolerance*100] as Vec2)];
 return{angleDegrees,angleRadians,length,gravity,approximation,tolerance,limitPeriod,hasOscillation,period,approximatePeriod,ratio:r.ratio,approximationRatio,relativeError,absoluteError:period===null||approximatePeriod===null?null:Math.abs(period-approximatePeriod),meetsTolerance:relativeError===null?null:relativeError<=tolerance,agmGap:r.agmGap,agmIterations:r.iterations,curves,errorCurves};
}

// Stable entire functions. The position formula needs more than expm1 alone near z=0.
function response(z:number){
 const phi1=z===0?1:-Math.expm1(-z)/z;
 let phi2:number;
 if(z<.05){let term=.5;phi2=term;for(let j=1;j<=12;j++){term*=-z/(j+2);phi2+=term;}}
 else phi2=(z+Math.expm1(-z))/(z*z);
 return{phi1,phi2};
}
export function linearDragRegimes(values:number[]){
 controls(values,[[1,5],[0,30],[1,20],[0,100],[0,1],[0,1],[1,100]]);
 const[mass,dragCode,gravityCode,timeCode,quantity,approximation,toleranceCode]=values;
 const drag=dragCode/10,gravity=gravityCode/2,time=timeCode/10,tolerance=toleranceCode/1000,tau=drag===0?null:mass/drag,terminalVelocity=tau===null?null:gravity*tau;
 const sample=(t:number)=>{const z=drag*t/mass,{phi1,phi2}=response(z),velocity=gravity*t*phi1,position=gravity*t*t*phi2,reference=quantity===0?velocity:position,early=quantity===0?gravity*t:gravity*t*t/2,late=tau===null?null:quantity===0?terminalVelocity!:terminalVelocity!*(t-tau),selected=approximation===0?early:late,absoluteError=selected===null?null:Math.abs(selected-reference),relativeError=absoluteError===null||reference===0?null:absoluteError/Math.abs(reference);return{time:t,z,velocity,position,reference,early,late,selected,absoluteError,relativeError};};
 const query=sample(time),history=Array.from({length:101},(_,n)=>sample(n/10));
 const curves=[history.map(q=>[q.time,q.reference] as Vec2),history.filter(q=>q.selected!==null).map(q=>[q.time,q.selected!] as Vec2)];
 // A separate normalized comparison makes the early/late regimes visible independently of dimensional scale.
 const dimensionlessCurves=tau===null?[]:[Array.from({length:121},(_,n)=>{const z=n/20,{phi1,phi2}=response(z);return[z,quantity===0?z*phi1:z*z*phi2] as Vec2;}),Array.from({length:121},(_,n)=>{const z=n/20;return[z,approximation===0?(quantity===0?z:z*z/2):(quantity===0?1:z-1)] as Vec2;})];
 return{mass,drag,gravity,time,quantity,approximation,tolerance,tau,terminalVelocity,query,hasFiniteTerminalScale:tau!==null,relativeError:query.relativeError,meetsTolerance:query.relativeError===null?null:query.relativeError<=tolerance,history,curves,dimensionlessCurves};
}
