import {timeGaugeHamiltonianEnergy} from './learning-hamiltonian-canonical';
import type {Vec2,Vec3,Mat3} from './learning-simulations';

/** Canonical bracket of gradients, ordered (q,p). Components need compatible physical units. */
export function canonicalPoissonBracket(fq:number[],fp:number[],gq:number[],gp:number[]):number {
 const n=fq.length;
 if(n<1||n>3||[fq,fp,gq,gp].some(a=>a.length!==n))throw new Error('Use matching gradients for one to three canonical pairs.');
 let value=0;
 for(let j=0;j<n;j++){
  if(![fq[j],fp[j],gq[j],gp[j]].every(Number.isFinite))throw new Error('Canonical gradients must be finite and dense.');
  value+=fq[j]*gp[j]-fp[j]*gq[j];
 }
 if(!Number.isFinite(value))throw new Error('Canonical bracket overflow.');
 return value;
}
function controls(p:number[],ranges:number[][]){
 if(p.length!==ranges.length)throw new Error('Use the complete Poisson-bracket control vector.');
 for(let j=0;j<p.length;j++)if(!Number.isInteger(p[j])||p[j]<ranges[j][0]||p[j]>ranges[j][1])throw new Error('Use bounded integer control codes.');
}
const bracket1=(fq:number,fp:number,gq:number,gp:number)=>canonicalPoissonBracket([fq],[fp],[gq],[gp]);

/** [m,f,2q0,2v0,a,observableCode,beta,20t]. a=1..4 dimensionless.
 * A=A_base+beta*t*H [Js], base code0:qp,1:m*nu*q²/2,2:p²/(2m*nu),3:H/nu.
 * nu=Omega for f>0,1/s for f=0. Analytic flow is reused, not integrated anew.
 */
export function poissonObservableCanonicalFlow(p:number[]){
 controls(p,[[1,4],[0,4],[-4,4],[-4,4],[1,4],[0,3],[-2,2],[0,40]]);
 const [mass,f,qc,vc,scale,observableCode,beta,tc]=p,flow=timeGaugeHamiltonianEnergy([mass,f,qc,vc,0,tc]),k=flow.stiffness,nu=f>0?flow.omegaNatural:1;
 const state=(j:number)=>{
  const time=j/100,q=flow.curves[0][j][1],momentum=mass*flow.curves[1][j][1],velocity=momentum/mass,force=-k*q,H=flow.curves[6][j][1];
  const baseValues=[q*momentum,mass*nu*q*q/2,momentum*momentum/(2*mass*nu),H/nu],baseQ=[momentum,mass*nu*q,0,k*q/nu],baseP=[q,0,momentum/(mass*nu),momentum/(mass*nu)];
  const base=baseValues[observableCode],aq=baseQ[observableCode]+beta*time*k*q,ap=baseP[observableCode]+beta*time*momentum/mass,observable=base+beta*time*H;
  const poisson=bracket1(aq,ap,k*q,momentum/mass),explicitTimeDerivative=beta*H;
  const directBaseRate=[momentum*momentum/mass-k*q*q,nu*q*momentum,-k*q*momentum/(mass*nu),0][observableCode];
  const totalTimeDerivative=directBaseRate+explicitTimeDerivative,chainTimeDerivative=aq*velocity+ap*force+explicitTimeDerivative;
  const Q=scale*q,P=momentum/scale,badP=scale*momentum,transformedBracket=bracket1(aq/scale,ap*scale,k*Q/(scale*scale),scale*scale*P/mass),badNaiveBracket=bracket1(aq/scale,ap/scale,k*Q/(scale*scale),badP/(mass*scale*scale));
  const F=baseValues[1],G=baseValues[2],C=baseValues[0],FG=bracket1(mass*nu*q,0,0,momentum/(mass*nu));
  const jacobiTerms:Vec3=[0,0,0];
  // {F,{G,C}}=-2{F,G}, {G,{C,F}}=-2{G,F}, {C,{F,G}}={C,C}.
  jacobiTerms[0]=bracket1(mass*nu*q,0,0,-2*momentum/(mass*nu));
  jacobiTerms[1]=bracket1(0,momentum/(mass*nu),-2*mass*nu*q,0);
  jacobiTerms[2]=bracket1(momentum,q,momentum,q);
  return {time,position:q,momentum,velocity,force,energy:H,baseObservable:base,explicitTimeContribution:beta*time*H,observable,gradientQ:aq,gradientP:ap,poisson,reversePoisson:bracket1(k*q,momentum/mass,aq,ap),explicitTimeDerivative,totalTimeDerivative,chainTimeDerivative,flowIdentityResidual:chainTimeDerivative-totalTimeDerivative,bracketIdentityResidual:poisson-directBaseRate,canonicalPosition:Q,canonicalMomentum:P,badMomentum:badP,canonicalFundamentalBracket:scale/scale,badFundamentalBracket:scale*scale,transformedBracket,canonicalInvarianceResidual:transformedBracket-poisson,badNaiveBracket,badTensorCorrectedBracket:scale*scale*badNaiveBracket,badTensorResidual:scale*scale*badNaiveBracket-poisson,physicalQRate:scale*velocity,badNaiveQRate:badP/(mass*scale*scale),physicalBadPRate:scale*force,badNaivePRate:-k*Q/(scale*scale),quadraticBasis:[F,G,C]as Vec3,quadraticFGBracket:FG,quadraticFGResidual:FG-C,jacobiTerms,jacobiResidual:jacobiTerms.reduce((s,x)=>s+x,0)};
 };
 const states=Array.from({length:201},(_,j)=>state(j)),curves:Vec2[][]=Array.from({length:18},()=>[]),phaseCurves:Vec2[][]=[[],[],[]];
 for(const s of states){[s.position,s.momentum,s.canonicalPosition,s.canonicalMomentum,s.observable,s.baseObservable,s.explicitTimeContribution,s.poisson,s.explicitTimeDerivative,s.totalTimeDerivative,s.chainTimeDerivative,s.flowIdentityResidual,s.energy,s.energy-flow.initialEnergy,s.badNaiveBracket,s.transformedBracket,s.badTensorCorrectedBracket,s.canonicalInvarianceResidual].forEach((v,j)=>curves[j].push([s.time,v]));phaseCurves[0].push([s.position,s.momentum]);phaseCurves[1].push([s.canonicalPosition,s.canonicalMomentum]);phaseCurves[2].push([s.canonicalPosition,s.badMomentum]);}
 const globallyConservedObservable=(observableCode===3&&beta===0)||(f===0&&((observableCode===0&&beta===-2)||(observableCode===2&&beta===0))),zeroEnergyOrbit=f===0?vc===0:qc===0&&vc===0;
 return {mass,frequencyCode:f,stiffness:k,omegaNatural:flow.omegaNatural,referenceFrequency:nu,scale,observableCode,beta,horizon:2,initial:states[0],query:states[tc*5],final:states[200],hasExplicitTimeDependence:beta!==0,globallyConservedObservable,selectedOrbitHasConstantObservable:globallyConservedObservable||zeroEnergyOrbit,canonicalPoissonMatrix:[[0,1],[-1,0]],badPoissonMatrix:[[0,scale*scale],[-scale*scale,0]],curves,phaseCurves,maximumSampledFlowIdentityResidual:Math.max(...states.map(s=>Math.abs(s.flowIdentityResidual))),maximumSampledCanonicalInvarianceResidual:Math.max(...states.map(s=>Math.abs(s.canonicalInvarianceResidual))),maximumSampledEnergyResidual:Math.max(...states.map(s=>Math.abs(s.energy-flow.initialEnergy)))};
}

const positions:Vec3[]=[[1,0,0],[1,1,1],[1,-1,.5],[0,0,0],[-1,.5,1]];
const velocities:Vec3[]=[[0,1,0],[1,0,1],[-1,1,.5],[0,0,0],[.5,-1,1]];
const cross=(a:Vec3,b:Vec3):Vec3=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const axes:Vec3[]=[[1,0,0],[0,1,0],[0,0,1]];
/** [m,fx,fy,fz,positionPreset,velocityPreset,selectedAxis,20t].
 * Independent Cartesian harmonic/free flows; no new3D ODE integrator.
 * Bracket algebra is independent of H; conservation depends on its stiffnesses.
 */
export function angularMomentumPoissonAlgebra(p:number[]){
 controls(p,[[1,4],[0,4],[0,4],[0,4],[0,4],[0,4],[0,2],[0,40]]);
 const [mass,fx,fy,fz,pc,vc,selectedAxis,tc]=p,frequencies=[fx,fy,fz],q0:Vec3=[...positions[pc]],v0:Vec3=[...velocities[vc]],flows=frequencies.map((f,j)=>timeGaugeHamiltonianEnergy([mass,f,q0[j]*2,v0[j]*2,0,tc])),stiffness=flows.map(f=>f.stiffness)as Vec3,initialEnergy=flows.reduce((s,f)=>s+f.initialEnergy,0);
 const state=(j:number)=>{
  const time=j/100,position=flows.map(f=>f.curves[0][j][1])as Vec3,velocity=flows.map(f=>f.curves[1][j][1])as Vec3,momentum=velocity.map(v=>mass*v)as Vec3,force=position.map((x,i)=>-stiffness[i]*x)as Vec3,angularMomentum=cross(position,momentum),torque=cross(position,force),velocityCrossMomentum=cross(velocity,momentum),directAngularMomentumRate=torque.map((t,i)=>t+velocityCrossMomentum[i])as Vec3;
  const gradientsQ=axes.map(e=>cross(momentum,e)),gradientsP=axes.map(e=>cross(e,position)),gradientHq=force.map(x=>-x),gradientHp=velocity;
  const bracketMatrix=axes.map((_,a)=>axes.map((__,b)=>canonicalPoissonBracket(gradientsQ[a],gradientsP[a],gradientsQ[b],gradientsP[b])))as Mat3;
  const [Lx,Ly,Lz]=angularMomentum,expectedBracketMatrix:Mat3=[[0,Lz,-Ly],[-Lz,0,Lx],[Ly,-Lx,0]],algebraResidualMatrix=bracketMatrix.map((row,i)=>row.map((x,j)=>x-expectedBracketMatrix[i][j]))as Mat3;
  const hamiltonBrackets=axes.map((_,i)=>canonicalPoissonBracket(gradientsQ[i],gradientsP[i],gradientHq,gradientHp))as Vec3,torqueResidual=hamiltonBrackets.map((x,i)=>x-torque[i])as Vec3;
  const energy=flows.reduce((s,f)=>s+f.curves[6][j][1],0),rotationPosition=cross(axes[selectedAxis],position),rotationMomentum=cross(axes[selectedAxis],momentum);
  // Jacobi for (x,Lx,Ly): {x,Lz} + {Lx,-z} + {Ly,0}.
  const jacobiRotationTerms:Vec3=[gradientsP[2][0],gradientsP[0][2],0];
  return {time,position,velocity,momentum,force,angularMomentum,torque,directAngularMomentumRate,gradientsQ,gradientsP,bracketMatrix,expectedBracketMatrix,algebraResidualMatrix,hamiltonBrackets,torqueResidual,energy,energyResidual:energy-initialEnergy,rotationPosition,rotationMomentum,jacobiRotationTerms,jacobiRotationResidual:jacobiRotationTerms.reduce((s,x)=>s+x,0)};
 };
 const states=Array.from({length:201},(_,j)=>state(j)),curves:Vec2[][]=Array.from({length:23},()=>[]),spatialProjections:Vec2[][]=[[],[],[]];
 for(const s of states){[...s.angularMomentum,...s.torque,...s.bracketMatrix[selectedAxis],s.energy,s.energyResidual,...s.algebraResidualMatrix[selectedAxis],...s.torqueResidual,...s.position,...s.momentum].forEach((v,i)=>curves[i].push([s.time,v]));spatialProjections[0].push([s.position[0],s.position[1]]);spatialProjections[1].push([s.position[1],s.position[2]]);spatialProjections[2].push([s.position[2],s.position[0]]);}
 const globallyConservedComponents=[fy===fz,fz===fx,fx===fy],zeroAxes=q0.map((x,i)=>x===0&&v0[i]===0),selectedOrbitHasConstantComponents=globallyConservedComponents.map((v,i)=>v||zeroAxes[(i+1)%3]||zeroAxes[(i+2)%3]);
 return {mass,frequencyCodes:frequencies,stiffness,initialPosition:q0,initialVelocity:v0,selectedAxis,horizon:2,isotropic:fx===fy&&fy===fz,globallyConservedComponents,selectedOrbitHasConstantComponents,initial:states[0],query:states[tc*5],final:states[200],curves,spatialProjections,maximumSampledAlgebraResidual:Math.max(...states.flatMap(s=>s.algebraResidualMatrix.flat().map(Math.abs))),maximumSampledTorqueResidual:Math.max(...states.flatMap(s=>s.torqueResidual.map(Math.abs))),maximumSampledEnergyResidual:Math.max(...states.map(s=>Math.abs(s.energyResidual)))};
}
