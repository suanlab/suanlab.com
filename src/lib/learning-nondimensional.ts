import type {Vec2} from './learning-simulations';
function controls(values:number[],ranges:readonly (readonly [number,number])[]){
 if(values.length!==ranges.length)throw new Error('Use the stated nondimensional control vector.');
 values.forEach((v,i)=>{if(!Number.isInteger(v)||v<ranges[i][0]||v>ranges[i][1])throw new Error('Use bounded integer nondimensional controls.');});
}
const sinc=(z:number)=>Math.abs(z)<1e-3?1-z*z/6+z**4/120-z**6/5040:Math.sin(z)/z;
function response(zeta:number,deltaSquared:number,regime:number,A:number,B:number,s:number){
 if(regime>0){
  const d=Math.sqrt(deltaSquared),rMinus=-zeta-d,eMinus=Math.exp(rMinus*s),ePlus=Math.exp((-zeta+d)*s),S=ePlus*(-Math.expm1(-2*d*s))/(2*d),coefficient=B+(zeta+d)*A;
  return{u:A*eMinus+coefficient*S,derivative:rMinus*A*eMinus+coefficient*(ePlus+rMinus*S)};
 }
 const d=regime===0?0:Math.sqrt(deltaSquared),C=Math.cos(d*s),S=s*sinc(d*s),e=Math.exp(-zeta*s),q=B+zeta*A,u=e*(A*C+q*S),derivative=-zeta*u+e*(-A*d*d*S+q*C);
 return{u,derivative};
}
export function oscillatorSimilarity(values:number[]){
 controls(values,[[1,5],[1,5],[0,40],[-20,20],[-20,20],[1,4],[1,4],[1,4],[0,20],[0,80]]);
 const[mass,stiffness,dampingCode,xCode,vCode,massScale,rateScale,lengthScale,extraDampingCode,queryCode]=values;
 const x0=xCode/10,v0=vCode/10,omega=Math.sqrt(stiffness/mass),zeta=dampingCode/(20*Math.sqrt(mass*stiffness)),A=x0,B=v0/omega,queryTime=queryCode/10;
 const make=(m:number,k:number,bCode:number,X:number,w:number,z:number)=>{const discriminantNumerator=bCode*bCode-400*m*k,regime=Math.sign(discriminantNumerator),deltaSquared=Math.abs(discriminantNumerator)/(400*m*k);return{mass:m,stiffness:k,damping:bCode/10,referenceLength:X,omega:w,zeta:z,x0:X*A,v0:X*w*B,normalizedInitial:[A,B],discriminantNumerator,regime,deltaSquared};};
 const worlds=[make(mass,stiffness,dampingCode,1,omega,zeta),make(massScale*mass,massScale*rateScale**2*stiffness,massScale*rateScale*dampingCode+extraDampingCode,lengthScale,rateScale*omega,zeta+extraDampingCode/(20*massScale*rateScale*Math.sqrt(mass*stiffness)))];
 // The scaling construction proves identical normalized initial data. Use their canonical coefficients directly.
 if(extraDampingCode===0)worlds[1].deltaSquared=worlds[0].deltaSquared;
 const evaluate=(i:number,s:number)=>response(worlds[i].zeta,worlds[i].deltaSquared,worlds[i].regime,A,B,s);
 const normalizedCurves=worlds.map((_,i)=>Array.from({length:161},(_,n)=>{const s=n/20;return[s,evaluate(i,s).u] as Vec2;}));
 const physicalCurves=worlds.map((w,i)=>Array.from({length:161},(_,n)=>{const t=n/8;return[t,w.referenceLength*evaluate(i,w.omega*t).u] as Vec2;}));
 const query=worlds.map((w,i)=>{const q=evaluate(i,queryTime);return{normalizedTime:queryTime,physicalTime:queryTime/w.omega,normalizedDisplacement:q.u,normalizedVelocity:q.derivative,physicalDisplacement:w.referenceLength*q.u,physicalVelocity:w.referenceLength*w.omega*q.derivative,normalizedEnergy:(q.u*q.u+q.derivative*q.derivative)/2,normalizedEnergyDerivative:-2*w.zeta*q.derivative*q.derivative};});
 return{massScale,rateScale,lengthScale,extraDampingCode,queryTime,worlds,query,normalizedInitial:[A,B],sameNormalizedLaw:extraDampingCode===0,sameNormalizedInitialData:true,sameNormalizedProblem:extraDampingCode===0,zeroExcitation:xCode===0&&vCode===0,maxSampledNormalizedGap:Math.max(...normalizedCurves[0].map((p,j)=>Math.abs(p[1]-normalizedCurves[1][j][1]))),normalizedCurves,physicalCurves,curves:normalizedCurves};
}

export const buckinghamDimensionMatrix:readonly (readonly number[])[]=[[1,1,0,0,1],[1,-3,1,1,-1],[-2,0,-1,0,-1]];
export function buckinghamGroups(values:number[]){
 controls(values,[[-3,3],[-3,3],[-3,3],[-3,3],[-3,3],[1,100],[1,120]]);
 const exponents=values.slice(0,5),reCode=values[5],forceCode=values[6],reynolds=reCode/10,forceGroup=forceCode/10,defect=buckinghamDimensionMatrix.map(row=>row.reduce((v,c,j)=>v+c*exponents[j],0)),dimensionless=defect.every(x=>x===0);
 const residualNumerators=[forceCode*reCode-100,forceCode*reCode-10*(reCode+10)],residualDenominator=10*reCode;
 const hypotheses=[10/reCode,1+10/reCode],curves=[Array.from({length:100},(_,j)=>{const r=(j+1)/10;return[r,1/r] as Vec2;}),Array.from({length:100},(_,j)=>{const r=(j+1)/10;return[r,1+1/r] as Vec2;})];
 return{exponents,defect,dimensionless,rank:3,nullity:2,basisCoefficients:dimensionless?[exponents[0],exponents[4]]:null,reynolds,forceGroup,hypotheses,residualNumerators,residualDenominator,hypothesisMatches:residualNumerators.map(x=>x===0),residuals:residualNumerators.map(x=>x/residualDenominator),curves};
}
