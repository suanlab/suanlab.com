import type {Vec2} from './learning-simulations';
import {polarLagrangeNewtonEquivalence} from './learning-lagrange-equations';

/** Same seven controls as the prior Cartesian/polar flow. No new ODE solver.
 * Computes the velocity Legendre inverse and canonical derivatives, with
 * separate units for p_r [kg m/s] and p_theta [kg m²/s]. No mixed-unit norm.
 */
export function polarHamiltonianLegendreMap(p:number[]){
  const flow=polarLagrangeNewtonEquivalence(p),m=flow.mass,k=flow.stiffness;
  const canonical=(radius:number,radialRate:number,angularRate:number,energy:number,angularMomentum:number,radialAcceleration:number,angularAcceleration:number)=>{
    const pr=m*radialRate,ptheta=m*radius*radius*angularRate,radialKinetic=pr*pr/(2*m),angularKinetic=ptheta*ptheta/(2*m*radius*radius),potential=k*radius*radius/2;
    const lagrangian=m*(radialRate*radialRate+radius*radius*angularRate*angularRate)/2-potential,hamiltonian=radialKinetic+angularKinetic+potential,legendreValue=pr*radialRate+ptheta*angularRate-lagrangian;
    const radialVelocity=pr/m,angularVelocity=ptheta/(m*radius*radius),partialRadius=-ptheta*ptheta/(m*radius**3)+k*radius,prDot=-partialRadius,pthetaDot=0;
    const radialAccelerationFromHamilton=prDot/m,angularAccelerationFromHamilton=pthetaDot/(m*radius*radius)-2*ptheta*radialVelocity/(m*radius**3);
    return {radius,radialMomentum:pr,angularMomentum:ptheta,radialKinetic,angularKinetic,potential,lagrangian,hamiltonian,legendreValue,legendreDiscrepancy:hamiltonian-legendreValue,physicalEnergyDiscrepancy:hamiltonian-energy,angularMomentumDiscrepancy:ptheta-angularMomentum,velocityHessianDiagonal:[m,m*radius*radius]as Vec2,inverseVelocityHessianDiagonal:[1/m,1/(m*radius*radius)]as Vec2,velocityHessianDeterminant:m*m*radius*radius,partialRadius,partialAngle:0,radialVelocity,angularVelocity,radialMomentumRate:prDot,angularMomentumRate:pthetaDot,radialVelocityDiscrepancy:radialVelocity-radialRate,angularVelocityDiscrepancy:angularVelocity-angularRate,radialMomentumRateDiscrepancy:prDot-m*radialAcceleration,angularMomentumRateDiscrepancy:pthetaDot-m*(2*radius*radialRate*angularRate+radius*radius*angularAcceleration),radialAccelerationFromHamilton,angularAccelerationFromHamilton,radialAccelerationDiscrepancy:radialAccelerationFromHamilton-radialAcceleration,angularAccelerationDiscrepancy:angularAccelerationFromHamilton-angularAcceleration};
  };
  const transform=(q:typeof flow.query)=>({time:q.time,position:q.position,velocity:q.velocity,radius:q.radius,principalAngle:q.polar?.angle??null,chartRegular:q.chartRegular,analyticOrigin:q.analyticOrigin,physicalEnergy:q.energy,physicalAngularMomentum:q.angularMomentum,canonical:q.polar?canonical(q.radius,q.polar.radialRate,q.polar.angularRate,q.energy,q.angularMomentum,q.polar.radialAcceleration,q.polar.angularAcceleration):null});
  const query=transform(flow.query),initial=transform(flow.initial),final=transform(flow.final);
  const byTime=flow.curves.map(c=>new Map(c)),angularByTime=new Map(flow.polarSegments[1].flat()),rddotByTime=new Map(flow.polarSegments[2].flat()),alphaByTime=new Map(flow.polarSegments[3].flat());
  const canonicalSegments:Vec2[][][]=Array.from({length:12},()=>[]),radialPhaseSegments:Vec2[][]=[];
  let maximumSampledEnergyDiscrepancy=0,maximumSampledLegendreDiscrepancy=0,maximumSampledAngularMomentumDiscrepancy=0;
  for(const segment of flow.polarSegments[0]){
    const groups:Vec2[][]=Array.from({length:12},()=>[]),phase:Vec2[]=[];
    for(const[t,rd]of segment){const r=byTime[12].get(t)!,q=canonical(r,rd,angularByTime.get(t)!,byTime[8].get(t)!,byTime[9].get(t)!,rddotByTime.get(t)!,alphaByTime.get(t)!);
      // p_r,p_theta,H,K_r,K_theta,U,r_dot,theta_dot,p_r_dot,
      // velocity-Hessian determinant,H-E,Legendre discrepancy. Separate units.
      [q.radialMomentum,q.angularMomentum,q.hamiltonian,q.radialKinetic,q.angularKinetic,q.potential,q.radialVelocity,q.angularVelocity,q.radialMomentumRate,q.velocityHessianDeterminant,q.physicalEnergyDiscrepancy,q.legendreDiscrepancy].forEach((v,j)=>groups[j].push([t,v]));phase.push([r,q.radialMomentum]);
      maximumSampledEnergyDiscrepancy=Math.max(maximumSampledEnergyDiscrepancy,Math.abs(q.physicalEnergyDiscrepancy));maximumSampledLegendreDiscrepancy=Math.max(maximumSampledLegendreDiscrepancy,Math.abs(q.legendreDiscrepancy));maximumSampledAngularMomentumDiscrepancy=Math.max(maximumSampledAngularMomentumDiscrepancy,Math.abs(q.angularMomentumDiscrepancy));
    }
    groups.forEach((g,j)=>canonicalSegments[j].push(g));radialPhaseSegments.push(phase);
  }
  return {mass:m,stiffness:k,omegaNatural:flow.omegaNatural,referenceFrequency:flow.referenceFrequency,horizon:2,query,initial,final,circularByInitialData:flow.circularByInitialData,analyticOriginTimes:flow.analyticOriginTimes,physicalCurves:flow.curves,cartesianPath:flow.phase,canonicalSegments,radialPhaseSegments,maximumSampledEnergyDiscrepancy,maximumSampledLegendreDiscrepancy,maximumSampledAngularMomentumDiscrepancy};
}

function controls(p:number[]){
  const ranges=[[1,4],[0,4],[-4,4],[-4,4],[-8,8],[0,40]];
  if(p.length!==ranges.length)throw new Error('Use the six time-gauge Hamiltonian controls.');
  for(let j=0;j<ranges.length;j++)if(!Number.isInteger(p[j])||p[j]<ranges[j][0]||p[j]>ranges[j][1])throw new Error('Use bounded integer control codes.');
}
function sinCosPi(n:number,d:number):Vec2{
  const j=((n%(2*d))+2*d)%(2*d);
  if(j===0)return[0,1];if(2*j===d)return[1,0];if(j===d)return[0,-1];if(2*j===3*d)return[-1,0];
  return[Math.sin(n*Math.PI/d),Math.cos(n*Math.PI/d)];
}
/** [m,f,2x0,2v0,2gamma,20queryTime]. Omega=pi*f/4,k=m*Omega².
 * F=gamma*t*x²/2, gamma [kg/s²]. P=m*v+gamma*t*x.
 * H_gamma=(P-gamma*t*x)²/(2m)+(k-gamma)*x²/2=E-gamma*x²/2.
 * Actual analytic Cartesian flow, numerical evaluation, not an ODE integrator.
 */
export function timeGaugeHamiltonianEnergy(p:number[]){
  controls(p);
  const [mass,f,xc,vc,gc,tc]=p,omegaNatural=Math.PI*f/4,stiffness=mass*omegaNatural**2,initialPosition=xc/2,initialVelocity=vc/2,gamma=gc/2;
  const initialEnergy=mass*initialVelocity**2/2+stiffness*initialPosition**2/2;
  const state=(j:number)=>{
    const time=j/100,[sn,cs]=sinCosPi(f*j,400),position=f===0?initialPosition+initialVelocity*time:initialPosition*cs+initialVelocity*sn/omegaNatural,velocity=f===0?initialVelocity:-omegaNatural*initialPosition*sn+initialVelocity*cs,acceleration=-(omegaNatural**2)*position;
    const kinetic=mass*velocity*velocity/2,potential=stiffness*position*position/2,physicalEnergy=kinetic+potential,mechanicalMomentum=mass*velocity,mechanicalForce=-stiffness*position;
    const gaugeFunction=gamma*time*position*position/2,gaugeDx=gamma*time*position,gaugePartialTime=gamma*position*position/2,gaugeTotalTime=gaugePartialTime+gaugeDx*velocity;
    const canonicalMomentum=mechanicalMomentum+gaugeDx,lagrangianBase=kinetic-potential,lagrangianGauge=lagrangianBase+gaugeTotalTime,recoveredVelocity=(canonicalMomentum-gamma*time*position)/mass;
    const hamiltonian=(canonicalMomentum-gamma*time*position)**2/(2*mass)+(stiffness-gamma)*position*position/2,legendreValue=canonicalMomentum*velocity-lagrangianGauge,hamiltonianFromEnergy=physicalEnergy-gaugePartialTime;
    const partialPosition=-gamma*time*recoveredVelocity+(stiffness-gamma)*position,partialMomentum=recoveredVelocity,partialTime=-gamma*position*recoveredVelocity;
    const canonicalMomentumRate=-partialPosition,canonicalMomentumRateFromPhysical=mass*acceleration+gamma*position+gamma*time*velocity,lagrangianPositionDerivative=-stiffness*position+gamma*time*velocity+gamma*position;
    const physicalEnergyRate=velocity*(mass*acceleration+stiffness*position),hamiltonianTotalTime=physicalEnergyRate-gamma*position*velocity,hamiltonianTotalFromCanonical=partialPosition*partialMomentum+partialMomentum*canonicalMomentumRate+partialTime;
    // d(m*x*v/2)/dt=K-U on this free/harmonic flow. F(0)=0.
    const baseCumulativeAction=mass*(position*velocity-initialPosition*initialVelocity)/2,gaugeCumulativeAction=baseCumulativeAction+gaugeFunction;
    return {time,position,velocity,acceleration,mechanicalForce,kinetic,potential,physicalEnergy,physicalEnergyRate,physicalEnergyResidual:physicalEnergy-initialEnergy,mechanicalMomentum,canonicalMomentum,canonicalMomentumRate,canonicalMomentumRateFromPhysical,gaugeFunction,gaugeDx,gaugePartialTime,gaugeTotalTime,lagrangianBase,lagrangianGauge,hamiltonian,hamiltonianFromEnergy,legendreValue,partialPosition,partialMomentum,partialTime,hamiltonianTotalTime,hamiltonianTotalFromCanonical,hamiltonXResidual:recoveredVelocity-velocity,hamiltonPResidual:canonicalMomentumRate-canonicalMomentumRateFromPhysical,eulerLagrangeResidual:canonicalMomentumRateFromPhysical-lagrangianPositionDerivative,newtonResidual:mass*acceleration+stiffness*position,energyRelationDiscrepancy:hamiltonian-hamiltonianFromEnergy,legendreDiscrepancy:hamiltonian-legendreValue,totalMinusExplicitTime:hamiltonianTotalTime-partialTime,oneFormPositionDiscrepancy:canonicalMomentum-mechanicalMomentum-gaugeDx,oneFormTimeDiscrepancy:hamiltonian-physicalEnergy+gaugePartialTime,oneFormAlongPathDiscrepancy:(canonicalMomentum*velocity-hamiltonian)-(mechanicalMomentum*velocity-physicalEnergy)-gaugeTotalTime,baseCumulativeAction,gaugeCumulativeAction};
  };
  const states=Array.from({length:201},(_,j)=>state(j)),query=states[tc*5],curves:Vec2[][]=Array.from({length:20},()=>[]),phaseCurves:Vec2[][]=[[],[]];
  for(const q of states){
    // x,v,p,P,K,U,E,H,dH/dt,partial_tH,Pdot,Fphysical,Fgauge[Js],
    // dFgauge/dt[J],E-E0,xdot discrepancy,Pdot discrepancy,S0,Sg,H-(E-Ft).
    [q.position,q.velocity,q.mechanicalMomentum,q.canonicalMomentum,q.kinetic,q.potential,q.physicalEnergy,q.hamiltonian,q.hamiltonianTotalTime,q.partialTime,q.canonicalMomentumRate,q.mechanicalForce,q.gaugeFunction,q.gaugeTotalTime,q.physicalEnergyResidual,q.hamiltonXResidual,q.hamiltonPResidual,q.baseCumulativeAction,q.gaugeCumulativeAction,q.energyRelationDiscrepancy].forEach((v,j)=>curves[j].push([q.time,v]));phaseCurves[0].push([q.position,q.mechanicalMomentum]);phaseCurves[1].push([q.position,q.canonicalMomentum]);
  }
  return {mass,frequencyCode:f,omegaNatural,stiffness,initialPosition,initialVelocity,gamma,horizon:2,initialEnergy,initial:states[0],query,final:states[200],explicitlyTimeDependent:gc!==0,selectedOrbitHasConstantHamiltonian:gc===0||(f===0?vc===0:xc===0&&vc===0),hamiltonianRange:[Math.min(...states.map(q=>q.hamiltonian)),Math.max(...states.map(q=>q.hamiltonian))]as Vec2,maximumSampledEnergyResidual:Math.max(...states.map(q=>Math.abs(q.physicalEnergyResidual))),maximumSampledPositionEquationResidual:Math.max(...states.map(q=>Math.abs(q.hamiltonXResidual))),maximumSampledMomentumEquationResidual:Math.max(...states.map(q=>Math.abs(q.hamiltonPResidual))),maximumSampledEnergyRelationDiscrepancy:Math.max(...states.map(q=>Math.abs(q.energyRelationDiscrepancy))),maximumSampledLegendreDiscrepancy:Math.max(...states.map(q=>Math.abs(q.legendreDiscrepancy))),curves,phaseCurves};
}
