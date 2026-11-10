import type {Vec2} from './learning-simulations';
type State4=[number,number,number,number];
type Matrix2=[[number,number],[number,number]];
const dot=(a:Vec2,b:Vec2)=>a[0]*b[0]+a[1]*b[1];
const cross=(a:Vec2,b:Vec2)=>a[0]*b[1]-a[1]*b[0];
const scale=(a:Vec2,k:number):Vec2=>[k*a[0],k*a[1]];
const plus=(a:Vec2,b:Vec2):Vec2=>[a[0]+b[0],a[1]+b[1]];
const minus=(a:Vec2,b:Vec2):Vec2=>[a[0]-b[0],a[1]-b[1]];
function controls(p:number[],ranges:readonly(readonly[number,number])[]){
  if(p.length!==ranges.length)throw new Error('Use the stated Lagrange controls.');
  for(let j=0;j<ranges.length;j++)if(!Number.isInteger(p[j])||p[j]<ranges[j][0]||p[j]>ranges[j][1])throw new Error('Use bounded integer control codes.');
}
/** Exact known cardinal identities, never a tolerance-based angle snap. */
function sinCosPi(n:number,d:number):Vec2 {
  const k=((n%(2*d))+2*d)%(2*d);
  if(k===0)return[0,1];if(2*k===d)return[1,0];if(k===d)return[0,-1];if(2*k===3*d)return[-1,0];
  return[Math.sin(n*Math.PI/d),Math.cos(n*Math.PI/d)];
}
function sinCosAngle(a:number):Vec2 {
  const k=Math.round(a/(Math.PI/2));
  return a===k*Math.PI/2?sinCosPi(k,2):[Math.sin(a),Math.cos(a)];
}

/** [m,f,2r0,2theta0/pi,2rdot0,W,20queryTime]. Natural Omega=pi*f/4,
 * k=m*Omega²; initial angular rate=W*Omega_ref/2, Omega_ref=Omega or1/s.
 * Exact Cartesian harmonic/free flow, numerical evaluation; horizon2s.
 * Polar inverse is null at analytically represented origins, with no radius clamp.
 */
export function polarLagrangeNewtonEquivalence(p:number[]){
  controls(p,[[1,4],[0,4],[1,6],[-4,4],[-4,4],[-4,4],[0,40]]);
  const [mass,f,rc,angleCode,vc,wc,tc]=p,omegaNatural=Math.PI*f/4,stiffness=mass*omegaNatural**2,referenceFrequency=f===0?1:omegaNatural;
  const radius0=rc/2,radialRate0=vc/2,angularRate0=wc*referenceFrequency/2,[s0,c0]=sinCosPi(angleCode,2),er0:Vec2=[c0,s0],et0:Vec2=[-s0,c0];
  const position0=scale(er0,radius0),velocity0=plus(scale(er0,radialRate0),scale(et0,radius0*angularRate0));
  const initialEnergy=mass*dot(velocity0,velocity0)/2+stiffness*radius0**2/2,initialAngularMomentum=mass*cross(position0,velocity0);
  const analyticOriginTimes:number[]=[];
  if(wc===0){
    if(f===0){if(vc<0){const root=-rc/vc;if(root<=2)analyticOriginTimes.push(root);}}
    else for(let n=0;n<4;n++){
      // With zero radial rate, the rational event formula also preserves
      // endpoint identities. Other radial events use the analytic atan2 root.
      const root=vc===0?(2+4*n)/f:(Math.atan2(omegaNatural*radius0,-radialRate0)+n*Math.PI)/omegaNatural;
      if(root>2)break;if(root>=0)analyticOriginTimes.push(root);
    }
  }
  const state=(j:number,originEvent=false)=>{
    const time=j/100,[sn,cs]=sinCosPi(f*j,400),position=f===0?plus(position0,scale(velocity0,time)):plus(scale(position0,cs),scale(velocity0,sn/omegaNatural));
    const velocity=f===0?[...velocity0]as Vec2:plus(scale(position0,-omegaNatural*sn),scale(velocity0,cs));
    // These are exact control/phase identities, not a small-radius decision.
    const analyticOrigin=originEvent||wc===0&&(f===0?rc*100+vc*j===0:vc===0&&cs===0);
    if(analyticOrigin){position[0]=0;position[1]=0;}
    const radius=Math.hypot(...position),acceleration=scale(position,-(omegaNatural**2)),force=scale(position,-stiffness),kinetic=mass*dot(velocity,velocity)/2,potential=stiffness*dot(position,position)/2,energy=kinetic+potential,angularMomentum=mass*cross(position,velocity);
    let polar:null|{angle:number;radialRate:number;angularRate:number;radialAcceleration:number;angularAcceleration:number;physicalRadialAcceleration:number;physicalTangentialAcceleration:number;radialELResidual:number;angularELResidual:number;reconstructedAcceleration:Vec2;accelerationDiscrepancy:number;angularMomentumRate:number}=null;
    if(radius>0){
      const er=scale(position,1/radius),et:Vec2=[-er[1],er[0]],radialRate=dot(er,velocity),angularRate=cross(position,velocity)/(radius*radius),radialAcceleration=radius*angularRate**2-omegaNatural**2*radius,angularAcceleration=-2*radialRate*angularRate/radius;
      const ar=radialAcceleration-radius*angularRate**2,at=radius*angularAcceleration+2*radialRate*angularRate,reconstructedAcceleration=plus(scale(er,ar),scale(et,at));
      polar={angle:Math.atan2(position[1],position[0]),radialRate,angularRate,radialAcceleration,angularAcceleration,physicalRadialAcceleration:ar,physicalTangentialAcceleration:at,radialELResidual:mass*ar+stiffness*radius,angularELResidual:mass*(radius*radius*angularAcceleration+2*radius*radialRate*angularRate),reconstructedAcceleration,accelerationDiscrepancy:Math.hypot(...minus(reconstructedAcceleration,acceleration)),angularMomentumRate:mass*(2*radius*radialRate*angularRate+radius*radius*angularAcceleration)};
    }
    return {time,position,velocity,acceleration,force,radius,analyticOrigin,chartRegular:radius>0,polar,kinetic,potential,energy,angularMomentum,energyResidual:energy-initialEnergy,angularMomentumResidual:angularMomentum-initialAngularMomentum,newtonResidual:Math.hypot(...minus(scale(acceleration,mass),force))};
  };
  // Include every analytic crossing, including those between the 0.01s grid
  // points. A null chart event breaks each polar curve instead of joining it
  // through an undefined state. The physical Cartesian path remains continuous.
  const samples=new Map<number,boolean>(Array.from({length:201},(_,j)=>[j,false]as[number,boolean]));
  analyticOriginTimes.forEach(t=>samples.set(t*100,true));
  const states=[...samples].sort((a,b)=>a[0]-b[0]).map(([j,event])=>state(j,event)),query=state(tc*5),curves:Vec2[][]=Array.from({length:13},()=>[]),phase:Vec2[]=[],polarSegments:Vec2[][][]=Array.from({length:4},()=>[]);
  const active:Vec2[][]=Array.from({length:4},()=>[]);
  for(const st of states){[...st.position,...st.velocity,...st.acceleration,st.kinetic,st.potential,st.energy,st.angularMomentum,st.energyResidual,st.angularMomentumResidual,st.radius].forEach((v,i)=>curves[i].push([st.time,v]));phase.push(st.position);
    const values=st.polar?[st.polar.radialRate,st.polar.angularRate,st.polar.radialAcceleration,st.polar.angularAcceleration]:null;
    for(let j=0;j<4;j++){if(values)active[j].push([st.time,values[j]]);else if(active[j].length){polarSegments[j].push(active[j].splice(0));}}
  }
  active.forEach((v,j)=>{if(v.length)polarSegments[j].push(v);});
  return {mass,frequencyCode:f,omegaNatural,stiffness,referenceFrequency,radius0,angle0:angleCode*Math.PI/2,radialRate0,angularRate0,position0,velocity0,initialEnergy,initialAngularMomentum,horizon:2,query,initial:states[0],final:states[states.length-1],circularByInitialData:f>0&&vc===0&&Math.abs(wc)===2,freeLimit:f===0,analyticOriginTimes,minimumSampledRadius:Math.min(...states.map(s=>s.radius)),maximumEnergyResidual:Math.max(...states.map(s=>Math.abs(s.energyResidual))),maximumAngularMomentumResidual:Math.max(...states.map(s=>Math.abs(s.angularMomentumResidual))),curves,phase,polarSegments};
}

type DoubleParameters={m1:number;m2:number;l1:number;l2:number};
function doubleTerms({m1,m2,l1,l2}:DoubleParameters,z:State4){
  const [a,b,wa,wb]=z,[sa,ca]=sinCosAngle(a),[sb,cb]=sinCosAngle(b),[sd,cd]=sinCosAngle(a-b),beta=m2*l1*l2,g=10;
  const matrix:Matrix2=[[(m1+m2)*l1*l1,beta*cd],[beta*cd,m2*l2*l2]],coriolis:Vec2=[beta*sd*wb*wb,-beta*sd*wa*wa],gravityGradient:Vec2=[(m1+m2)*g*l1*sa,m2*g*l2*sb];
  const determinant=m2*l1*l1*l2*l2*(m1+m2*sd*sd),numericalDeterminant=matrix[0][0]*matrix[1][1]-matrix[0][1]*matrix[1][0];
  const solve=(rhs:Vec2):Vec2=>[(matrix[1][1]*rhs[0]-matrix[0][1]*rhs[1])/determinant,(matrix[0][0]*rhs[1]-matrix[0][1]*rhs[0])/determinant];
  const accelerations=solve(scale(plus(coriolis,gravityGradient),-1)),naiveAccelerations=solve(scale(gravityGradient,-1));
  return {a,b,wa,wb,sa,ca,sb,cb,sd,cd,beta,g,matrix,coriolis,gravityGradient,determinant,numericalDeterminant,accelerations,naiveAccelerations};
}
function doubleState(parameters:DoubleParameters,z:State4){
  const {m1,m2,l1,l2}=parameters;
  const {a,b,wa,wb,sa,ca,sb,cb,sd,beta,g,matrix,coriolis,gravityGradient,determinant,numericalDeterminant,accelerations,naiveAccelerations}=doubleTerms(parameters,z);
  const e1:Vec2=[sa,-ca],e2:Vec2=[sb,-cb],t1:Vec2=[ca,sa],t2:Vec2=[cb,sb],p1=scale(e1,l1),p2=plus(p1,scale(e2,l2)),v1=scale(t1,l1*wa),v2=plus(v1,scale(t2,l2*wb));
  const a1=plus(scale(t1,l1*accelerations[0]),scale(e1,-l1*wa*wa)),a2=plus(a1,plus(scale(t2,l2*accelerations[1]),scale(e2,-l2*wb*wb)));
  const rod2Force=scale(plus(a2,[0,g]),m2),rod1Force=plus(scale(plus(a1,[0,g]),m1),rod2Force),constraintPower=dot(minus(rod1Force,rod2Force),v1)+dot(rod2Force,v2);
  const el:Vec2=plus([matrix[0][0]*accelerations[0]+matrix[0][1]*accelerations[1],matrix[1][0]*accelerations[0]+matrix[1][1]*accelerations[1]],plus(coriolis,gravityGradient));
  const kinetic=matrix[0][0]*wa*wa/2+matrix[1][1]*wb*wb/2+matrix[0][1]*wa*wb,cartesianKinetic=m1*dot(v1,v1)/2+m2*dot(v2,v2)/2;
  const [sha]=sinCosAngle(a/2),[shb]=sinCosAngle(b/2),potential=2*g*((m1+m2)*l1*sha*sha+m2*l2*shb*shb);
  const kineticRate=matrix[0][0]*wa*accelerations[0]+matrix[1][1]*wb*accelerations[1]+matrix[0][1]*(accelerations[0]*wb+wa*accelerations[1])-beta*sd*(wa-wb)*wa*wb,potentialRate=dot(gravityGradient,[wa,wb]);
  return {angles:[a,b]as Vec2,rates:[wa,wb]as Vec2,accelerations,naiveAccelerations,naiveAccelerationError:Math.hypot(...minus(naiveAccelerations,accelerations)),matrix,coriolis,gravityGradient,determinant,numericalDeterminant,determinantDiscrepancy:numericalDeterminant-determinant,eulerLagrangeResidual:el,kinetic,cartesianKinetic,kineticDiscrepancy:kinetic-cartesianKinetic,potential,energy:kinetic+potential,kineticRate,potentialRate,energyRate:kineticRate+potentialRate,pose:{p1,p2},velocities:[v1,v2]as[Vec2,Vec2],cartesianAccelerations:[a1,a2]as[Vec2,Vec2],rod1Force,rod2Force,rod1SignedTension:-dot(rod1Force,e1),rod2SignedTension:-dot(rod2Force,e2),rodTangentialResiduals:[dot(rod1Force,t1),dot(rod2Force,t2)]as Vec2,constraintPower,rodLengthSquaredResiduals:[dot(p1,p1)-l1*l1,dot(minus(p2,p1),minus(p2,p1))-l2*l2]as Vec2};
}
/** [m1,m2,2l1,2l2,12a0/pi,12b0/pi,10wa0,10wb0,20queryTime,resolution1..4].
 * Absolute angles from downward vertical; unforced rigid rods;g=10.
 * Explicit RK4 at dt=.0025/2^(resolution-1), horizon2s, plus dt/2 comparison.
 * Numerical ODE trajectories; energy errors and refinement gaps are diagnostics.
 */
export function doublePendulumLagrangeDynamics(p:number[]){
  controls(p,[[1,4],[1,4],[1,6],[1,6],[-12,12],[-12,12],[-10,10],[-10,10],[0,40],[1,4]]);
  const [m1,m2,l1c,l2c,ac,bc,wac,wbc,tc,resolution]=p,parameters={m1,m2,l1:l1c/2,l2:l2c/2},initialState:State4=[ac*Math.PI/12,bc*Math.PI/12,wac/10,wbc/10],substeps=4*2**(resolution-1),stepSize=.01/substeps;
  const derivative=(z:State4):State4=>{const st=doubleTerms(parameters,z);return[z[2],z[3],...st.accelerations];};
  const shifted=(z:State4,k:State4,h:number)=>z.map((x,j)=>x+h*k[j])as State4;
  const rk4=(z:State4,h:number):State4=>{const k1=derivative(z),k2=derivative(shifted(z,k1,h/2)),k3=derivative(shifted(z,k2,h/2)),k4=derivative(shifted(z,k3,h));return z.map((x,j)=>x+h*(k1[j]+2*k2[j]+2*k3[j]+k4[j])/6)as State4;};
  const integrate=(n:number)=>{
    let z=[...initialState]as State4;const states=[doubleState(parameters,z)],raw:State4[]=[[...z]];
    for(let j=1;j<=200;j++){for(let k=0;k<n;k++)z=rk4(z,.01/n);if(!z.every(Number.isFinite))throw new Error('Non-finite pendulum integration.');states.push(doubleState(parameters,z));raw.push([...z]);}
    return {states,raw};
  };
  const coarse=integrate(substeps),fine=integrate(2*substeps),initialEnergy=coarse.states[0].energy,curves:Vec2[][]=Array.from({length:15},()=>[]),phaseCurves:Vec2[][]=[[],[]];
  let maxEnergyError=0,maxRefinedEnergyError=0,maxAngleRefinementGap=0,maxRateRefinementGap=0;
  for(let j=0;j<=200;j++){
    const t=j/100,st=coarse.states[j],ft=fine.states[j],energyError=st.energy-initialEnergy,fineError=ft.energy-initialEnergy,angleGap=Math.hypot(st.angles[0]-ft.angles[0],st.angles[1]-ft.angles[1]),rateGap=Math.hypot(st.rates[0]-ft.rates[0],st.rates[1]-ft.rates[1]);
    maxEnergyError=Math.max(maxEnergyError,Math.abs(energyError));maxRefinedEnergyError=Math.max(maxRefinedEnergyError,Math.abs(fineError));maxAngleRefinementGap=Math.max(maxAngleRefinementGap,angleGap);maxRateRefinementGap=Math.max(maxRateRefinementGap,rateGap);
    [...st.angles,...st.rates,st.kinetic,st.potential,st.energy,energyError,fineError,angleGap,rateGap,...st.accelerations,st.rod1SignedTension,st.rod2SignedTension].forEach((v,i)=>curves[i].push([t,v]));phaseCurves[0].push(st.pose.p1);phaseCurves[1].push(st.pose.p2);
  }
  const query=coarse.states[tc*5],refinedQuery=fine.states[tc*5];
  return {...parameters,gravity:10,unforced:true as const,initialState,initial:coarse.states[0],query,refinedQuery,final:coarse.states[200],time:tc/20,horizon:2,resolution,stepSize,refinedStepSize:stepSize/2,method:'RK4' as const,integrationSteps:200*substeps,refinedIntegrationSteps:400*substeps,initialEnergy,queryEnergyError:query.energy-initialEnergy,queryRefinedEnergyError:refinedQuery.energy-initialEnergy,maxEnergyError,maxRefinedEnergyError,maxAngleRefinementGap,maxRateRefinementGap,curves,phaseCurves};
}
