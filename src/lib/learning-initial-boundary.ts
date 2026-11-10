import type {Vec2} from './learning-simulations';

function controls(values:number[],ranges:readonly (readonly [number,number])[]){
 if(values.length!==ranges.length)throw new Error('Use the stated initial/boundary control vector.');
 values.forEach((v,i)=>{if(!Number.isInteger(v)||v<ranges[i][0]||v>ranges[i][1])throw new Error('Use bounded integer initial/boundary controls.');});
}
const signed=[-3,3] as const;
// Analytic one-mode families on a rod/string of length L=1 m; these are not numerical time integrators.
export function initialDataOrder(values:number[]){
 controls(values,[[0,1],signed,signed,signed,signed,[1,4],[0,40]]);
 const[mode,initialAmplitude,initialVelocity,candidateAmplitude,candidateVelocity,rate,timeCode]=values;
 const time=timeCode/10,diffusivity=rate/100,speed=rate/2,decay=diffusivity*Math.PI**2,omega=speed*Math.PI;
 const state=(a:number,b:number,t:number)=>mode===0?{q:a*Math.exp(-decay*t),velocity:-decay*a*Math.exp(-decay*t)}:{q:a*Math.cos(omega*t)+b/omega*Math.sin(omega*t),velocity:-a*omega*Math.sin(omega*t)+b*Math.cos(omega*t)};
 const query=state(candidateAmplitude,candidateVelocity,time),target=state(initialAmplitude,initialVelocity,time);
 const displacementMatches=candidateAmplitude===initialAmplitude,velocityMatches=mode===1?candidateVelocity===initialVelocity:null;
 const temporalCurves=[Array.from({length:161},(_,i)=>[i/40,state(candidateAmplitude,candidateVelocity,i/40).q] as Vec2),Array.from({length:161},(_,i)=>[i/40,state(initialAmplitude,initialVelocity,i/40).q] as Vec2)];
 const spatialCurves=[query.q,target.q].map(a=>Array.from({length:81},(_,i)=>{const x=i/80;return[x,i===0||i===80?0:a*Math.sin(Math.PI*x)] as Vec2;}));
 return{mode,time,rate,diffusivity,speed,decay,omega,initialAmplitude,initialVelocity,candidateAmplitude,candidateVelocity,displacementMatches,velocityMatches,allRequiredDataMatch:displacementMatches&&(velocityMatches??true),velocityControlsInactive:mode===0,candidateInitialTimeDerivative:mode===0?-decay*candidateAmplitude:candidateVelocity,targetInitialTimeDerivative:mode===0?-decay*initialAmplitude:initialVelocity,query,target,temporalCurves,spatialCurves,curves:temporalCurves};
}

export function heatBoundaryTraces(values:number[]){
 controls(values,[[0,1],signed,signed,signed,signed,signed,signed,[1,4],[0,40]]);
 const[boundary,c0,c1,c2,h0,h1,h2,rate,timeCode]=values,time=timeCode/10,diffusivity=rate/100,decay=diffusivity*Math.PI**2;
 const coefficients=[c0,c1,c2],initialCoefficients=[h0,h1,h2];
 const profile=(c:number[],x:number,t:number)=>c[0]+Math.exp(-decay*t)*(c[1]*Math.cos(Math.PI*x)+c[2]*Math.sin(Math.PI*x));
 const traces=(t:number)=>{const e=Math.exp(-decay*t);return{leftValue:c0+c1*e,rightValue:c0-c1*e,leftDerivative:Math.PI*c2*e,rightDerivative:-Math.PI*c2*e};};
 const candidateBoundaryMatches=boundary===0?c0===0&&c1===0:c2===0;
 const targetCornerCompatible=boundary===0?h0===0&&h1===0:h2===0;
 const initialProfileMatches=coefficients.every((c,i)=>c===initialCoefficients[i]);
 const spatialCurves=[Array.from({length:81},(_,i)=>[i/80,profile(coefficients,i/80,time)] as Vec2),Array.from({length:81},(_,i)=>[i/80,profile(coefficients,i/80,0)] as Vec2),Array.from({length:81},(_,i)=>[i/80,profile(initialCoefficients,i/80,0)] as Vec2)];
 const traceCurves=[0,1].map(side=>Array.from({length:161},(_,i)=>{const t=i/40,q=traces(t);return[t,boundary===0?(side===0?q.leftValue:q.rightValue):(side===0?q.leftDerivative:q.rightDerivative)] as Vec2;}));
 return{boundary,time,rate,diffusivity,decay,coefficients,initialCoefficients,queryTraces:traces(time),candidateBoundaryMatches,targetCornerCompatible,initialProfileMatches,allDeclaredConditionsMatch:candidateBoundaryMatches&&initialProfileMatches,spatialCurves,traceCurves,curves:spatialCurves};
}

export function poissonBoundaryFamily(values:number[]){
 controls(values,[[0,2],signed,signed,signed,[0,1],signed,signed]);
 const[boundary,source,left,right,gauge,meanValue,offset]=values;
 // All coefficients have denominator six. This represents the entire C2 solution family of -u''=source.
 const a6=boundary===0?6*(right-left)+3*source:boundary===1?6*(right+source):6*left;
 const b6=boundary===2?(gauge===1?6*meanValue+source-3*left:6*offset):6*left;
 const compatibilityResidual=boundary===2?right-left+source:0,hasSolution=compatibilityResidual===0;
 const solutionClass=!hasSolution?'none':boundary===2&&gauge===0?'affine-family':'unique';
 const a=a6/6,b=b6/6,meanNumerator=-2*source+a6+2*b6;
 const candidateAt=(x:number)=>-source*x*x/2+a*x+b;
 const candidate=Array.from({length:81},(_,i)=>[i/80,candidateAt(i/80)] as Vec2);
 const selectedMean=meanNumerator/12;
 return{boundary,source,left,right,gauge,meanValue,offset,a,b,aNumerator:a6,bNumerator:b6,coefficientDenominator:6,compatibilityResidual,hasSolution,solutionClass,ungaugedConstantFree:boundary===2&&hasSolution,gaugeActive:boundary===2&&gauge===1,offsetActive:boundary===2&&gauge===0,candidateLabel:hasSolution?'selected solution':'interior candidate; right datum fails',leftValue:b,rightValue:(-3*source+a6+b6)/6,leftDerivative:a,rightDerivative:(a6-6*source)/6,leftOutwardDerivative:-a,rightOutwardDerivative:(a6-6*source)/6,selectedMean,meanNumerator,meanDenominator:12,curves:[candidate],familyFormula:boundary===2&&hasSolution?`u(x)=-(${source})x²/2+(${left})x+C, C∈R`:null};
}
