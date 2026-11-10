import type {Vec2} from './learning-simulations';

type Matrix=number[][];
const dot=(a:number[],b:number[])=>a.reduce((sum,x,j)=>sum+x*b[j],0);
const norm=(a:number[])=>Math.hypot(...a);
const add=(a:number[],b:number[])=>a.map((x,j)=>x+b[j]);
const sub=(a:number[],b:number[])=>a.map((x,j)=>x-b[j]);
const scale=(a:number[],k:number)=>a.map(x=>k*x);
const mv=(a:Matrix,b:number[])=>a.map(row=>dot(row,b));
const transpose=(a:Matrix)=>a[0].map((_,j)=>a.map(row=>row[j]));
const mm=(a:Matrix,b:Matrix)=>a.map(row=>transpose(b).map(col=>dot(row,col)));
const matrixError=(a:Matrix,b:Matrix)=>norm(a.flatMap((row,i)=>sub(row,b[i])));
const det2=(a:Matrix)=>a[0][0]*a[1][1]-a[0][1]*a[1][0];
const eigen2=(a:Matrix)=>{const big=(a[0][0]+a[1][1]+Math.hypot(a[0][0]-a[1][1],2*a[0][1]))/2;return [det2(a)/big,big];};
function controls(p:number[],ranges:readonly(readonly[number,number])[]){
  if(p.length!==ranges.length)throw new Error('Use the stated generalized-coordinate controls.');
  for(let j=0;j<ranges.length;j++)if(!Number.isInteger(p[j])||p[j]<ranges[j][0]||p[j]>ranges[j][1])throw new Error('Use bounded integer control codes.');
}

/** Instantaneous polar kinematics/Newton acceleration, not an integrated path.
 * [m,2r,12theta/pi,10rdot,10omega,10Fx,10Fy,10deltaR,10deltaTheta,100epsilon].
 * Angle representations are not a global one-to-one chart. At r=0 inverse
 * polar acceleration is undefined, while Cartesian acceleration stays finite.
 */
export function polarGeneralizedCoordinates(p:number[]){
  controls(p,[[1,4],[0,6],[-24,24],[-20,20],[-20,20],[-40,40],[-40,40],[-10,10],[-10,10],[0,100]]);
  const [mass,rc,ac,vc,wc,fxc,fyc,drc,dtc,ec]=p,radius=rc/2,angle=ac*Math.PI/12,radialRate=vc/10,angularRate=wc/10,force:Vec2=[fxc/10,fyc/10],variation:Vec2=[drc/10,dtc/10],epsilon=ec/100,c=Math.cos(angle),s=Math.sin(angle),er:Vec2=[c,s],et:Vec2=[-s,c];
  const position:Vec2=[radius*c,radius*s],jacobian:Matrix=[[c,-radius*s],[s,radius*c]],rates:Vec2=[radialRate,angularRate],velocity=mv(jacobian,rates)as Vec2,cartesianAcceleration=scale(force,1/mass)as Vec2;
  const metric:Matrix=[[mass,0],[0,mass*radius*radius]],metricFromCartesian=mm(transpose(jacobian),jacobian).map(row=>scale(row,mass));
  const radialForce=dot(er,force),tangentialForce=dot(et,force),generalizedForce=mv(transpose(jacobian),force)as Vec2,regular=rc>0;
  const curvatureAcceleration=add(scale(er,-radius*angularRate*angularRate),scale(et,2*radialRate*angularRate))as Vec2;
  const generalizedAcceleration:Vec2|null=regular?[radialForce/mass+radius*angularRate*angularRate,(tangentialForce/mass-2*radialRate*angularRate)/radius]:null;
  const naiveAcceleration=generalizedAcceleration?mv(jacobian,generalizedAcceleration)as Vec2:null,reconstructedAcceleration=naiveAcceleration?add(naiveAcceleration,curvatureAcceleration)as Vec2:null;
  const kineticCartesian=mass*dot(velocity,velocity)/2,kineticMetric=dot(rates,mv(metric,rates))/2,kineticFormula=mass*(radialRate*radialRate+radius*radius*angularRate*angularRate)/2;
  const cartesianPower=dot(force,velocity),generalizedPower=dot(generalizedForce,rates),virtualDisplacement=mv(jacobian,variation)as Vec2,virtualWorkCartesian=dot(force,virtualDisplacement),virtualWorkGeneralized=dot(generalizedForce,variation);
  const boundary=variation[0]<0&&radius/(-variation[0])<=1?radius/(-variation[0]):null;
  const probe=(e:number)=>{
    // This explicit boundary value is the domain event, not a tolerance clamp.
    const nextRadius=boundary!==null&&e===boundary?0:radius+e*variation[0],linearizedPoint=add(position,scale(virtualDisplacement,e))as Vec2,valid=nextRadius>=0;
    const exactPoint:Vec2|null=valid?[nextRadius*Math.cos(angle+e*variation[1]),nextRadius*Math.sin(angle+e*variation[1])]:null;
    const finiteWork=exactPoint?dot(force,sub(exactPoint,position)):null,linearWork=e*virtualWorkCartesian;
    return {epsilon:e,nextRadius,inParameterDomain:valid,regularTarget:nextRadius>0,exactPoint,linearizedPoint,linearizationError:exactPoint?norm(sub(exactPoint,linearizedPoint)):null,finiteWork,linearWork,workLinearizationError:finiteWork===null?null:finiteWork-linearWork};
  };
  const samples=Array.from({length:101},(_,j)=>j/100);if(boundary!==null&&!samples.includes(boundary))samples.push(boundary);samples.sort((a,b)=>a-b);
  const actualProbeCurve:Vec2[]=[],linearProbeCurve:Vec2[]=[],linearizationCurve:Vec2[]=[],finiteWorkCurve:Vec2[]=[],linearWorkCurve:Vec2[]=[];
  for(const e of samples){const v=probe(e);if(v.inParameterDomain){actualProbeCurve.push(v.exactPoint!);linearProbeCurve.push(v.linearizedPoint);linearizationCurve.push([e,v.linearizationError!]);finiteWorkCurve.push([e,v.finiteWork!]);linearWorkCurve.push([e,v.linearWork]);}}
  return {mass,radius,angle,angleCode:ac,radialRate,angularRate,force,variation,epsilon,position,basis:{radial:er,tangential:et},jacobian,numericalJacobianDeterminant:det2(jacobian),analyticJacobianDeterminant:radius,metric,metricFromCartesian,metricResidual:matrixError(metric,metricFromCartesian),analyticMetricDeterminant:mass*mass*radius*radius,numericalMetricDeterminant:det2(metricFromCartesian),localChartRegular:regular,localJacobianRank:regular?2:1,physicalDOF:2,globallyOneToOne:false as const,velocity,cartesianAcceleration,radialForce,tangentialForce,generalizedForce,curvatureAcceleration,generalizedAcceleration,naiveAcceleration,reconstructedAcceleration,accelerationResidual:reconstructedAcceleration?norm(sub(reconstructedAcceleration,cartesianAcceleration)):null,naiveAccelerationError:naiveAcceleration?norm(sub(naiveAcceleration,cartesianAcceleration)):null,kineticCartesian,kineticMetric,kineticFormula,kineticResidual:kineticCartesian-kineticMetric,cartesianPower,generalizedPower,powerResidual:cartesianPower-generalizedPower,virtualDisplacement,virtualWorkCartesian,virtualWorkGeneralized,virtualWorkResidual:virtualWorkCartesian-virtualWorkGeneralized,domainBoundaryEpsilon:boundary,queryProbe:probe(epsilon),actualProbeCurve,linearProbeCurve,linearizationCurve,finiteWorkCurve,linearWorkCurve,instantaneousOnly:true as const,dynamicsIntegrated:false as const};
}

/** Two point masses on massless rigid links, fixed pivot, g=10.
 * Absolute angles from downward vertical. Relative chart u=(a,b-a).
 * [m1,m2,2l1,2l2,12a/pi,12b/pi,10omega1,10omega2,
 *  10tipFx,10tipFy,10deltaA,10deltaB,100epsilon].
 * This is a compatible instantaneous configuration/rate/force assignment,
 * not double-pendulum time integration; constraint reactions are not solved.
 */
export function doublePendulumGeneralizedMetric(p:number[]){
  controls(p,[[1,4],[1,4],[1,6],[1,6],[-12,12],[-12,12],[-20,20],[-20,20],[-30,30],[-30,30],[-10,10],[-10,10],[0,100]]);
  const [m1,m2,lc1,lc2,ac,bc,wc1,wc2,fxc,fyc,dac,dbc,ec]=p,l1=lc1/2,l2=lc2/2,a=ac*Math.PI/12,b=bc*Math.PI/12,absoluteRates:Vec2=[wc1/10,wc2/10],absoluteVariation:Vec2=[dac/10,dbc/10],epsilon=ec/100,gravity=10,tipForce:Vec2=[fxc/10,fyc/10];
  const geometry=(aa:number,bb:number)=>{const p1:Vec2=[l1*Math.sin(aa),-l1*Math.cos(aa)],p2=add(p1,[l2*Math.sin(bb),-l2*Math.cos(bb)])as Vec2;return {p1,p2,stack:[...p1,...p2]};};
  const h=(z:number[])=>[z[0]*z[0]+z[1]*z[1]-l1*l1,(z[2]-z[0])**2+(z[3]-z[1])**2-l2*l2];
  const pose=geometry(a,b),d=sub(pose.p2,pose.p1),jacobian:Matrix=[[l1*Math.cos(a),0],[l1*Math.sin(a),0],[l1*Math.cos(a),l2*Math.cos(b)],[l1*Math.sin(a),l2*Math.sin(b)]],weights=[m1,m1,m2,m2];
  const constraintJacobian:Matrix=[[2*pose.p1[0],2*pose.p1[1],0,0],[-2*d[0],-2*d[1],2*d[0],2*d[1]]],constraintValues=h(pose.stack),constraintTangentResidual=mm(constraintJacobian,jacobian),velocityStack=mv(jacobian,absoluteRates),velocities:[Vec2,Vec2]=[velocityStack.slice(0,2)as Vec2,velocityStack.slice(2)as Vec2],constraintVelocity=mv(constraintJacobian,velocityStack);
  const off=m2*l1*l2*Math.cos(a-b),metricAbsolute:Matrix=[[(m1+m2)*l1*l1,off],[off,m2*l2*l2]],weightedJ=jacobian.map((row,j)=>scale(row,weights[j])),metricFromPoints=mm(transpose(jacobian),weightedJ),determinantFormula=m2*l1*l1*l2*l2*(m1+m2*Math.sin(a-b)**2);
  const chartMatrix:Matrix=[[1,0],[1,1]],relativeAngles:Vec2=[a,b-a],relativeRates:Vec2=[absoluteRates[0],absoluteRates[1]-absoluteRates[0]],relativeVariation:Vec2=[absoluteVariation[0],absoluteVariation[1]-absoluteVariation[0]],relativeJacobian=mm(jacobian,chartMatrix),metricRelative=mm(transpose(chartMatrix),mm(metricAbsolute,chartMatrix));
  const kineticByPoint=[m1*dot(velocities[0],velocities[0])/2,m2*dot(velocities[1],velocities[1])/2],kineticCartesian=kineticByPoint[0]+kineticByPoint[1],kineticAbsolute=dot(absoluteRates,mv(metricAbsolute,absoluteRates))/2,kineticRelative=dot(relativeRates,mv(metricRelative,relativeRates))/2,diagonalKinetic=(metricAbsolute[0][0]*absoluteRates[0]**2+metricAbsolute[1][1]*absoluteRates[1]**2)/2,couplingKinetic=off*absoluteRates[0]*absoluteRates[1];
  const forceStack=[0,-m1*gravity,tipForce[0],tipForce[1]-m2*gravity],generalizedForceAbsolute=mv(transpose(jacobian),forceStack)as Vec2,generalizedForceRelative=mv(transpose(chartMatrix),generalizedForceAbsolute)as Vec2,gravityGeneralized:Vec2=[-(m1+m2)*gravity*l1*Math.sin(a),-m2*gravity*l2*Math.sin(b)],tipGeneralized=mv(transpose(jacobian.slice(2)),tipForce)as Vec2;
  const cartesianPower=dot(forceStack,velocityStack),absolutePower=dot(generalizedForceAbsolute,absoluteRates),relativePower=dot(generalizedForceRelative,relativeRates),potential=m1*gravity*pose.p1[1]+m2*gravity*pose.p2[1];
  const virtualDisplacement=mv(jacobian,absoluteVariation),virtualWorkCartesian=dot(forceStack,virtualDisplacement),virtualWorkAbsolute=dot(generalizedForceAbsolute,absoluteVariation),virtualWorkRelative=dot(generalizedForceRelative,relativeVariation);
  const probe=(e:number)=>{const exactPose=geometry(a+e*absoluteVariation[0],b+e*absoluteVariation[1]),linearPose=add(pose.stack,scale(virtualDisplacement,e)),exactConstraints=h(exactPose.stack),linearConstraints=h(linearPose),finiteWork=dot(forceStack,sub(exactPose.stack,pose.stack)),linearWork=e*virtualWorkCartesian;return {epsilon:e,exactPose,linearPose,exactConstraints,linearConstraints,linearizationError:norm(sub(exactPose.stack,linearPose)),finiteWork,linearWork,workLinearizationError:finiteWork-linearWork};};
  const probeCurves:Vec2[][]=Array.from({length:7},()=>[]);
  for(let j=0;j<=100;j++){const e=j/100,s=probe(e);[s.linearizationError,...s.linearConstraints,...s.exactConstraints,s.finiteWork,s.linearWork].forEach((v,k)=>probeCurves[k].push([e,v]));}
  return {masses:[m1,m2]as Vec2,lengths:[l1,l2]as Vec2,gravity,absoluteAngles:[a,b]as Vec2,relativeAngles,absoluteRates,relativeRates,absoluteVariation,relativeVariation,epsilon,tipForce,pose,jacobian,relativeJacobian,chartMatrix,constraintValues,constraintJacobian,constraintTangentResidual,constraintVelocity,constraintRankByAssumptions:2,ambientDOF:4,physicalDOF:2,velocities,velocityStack,metricAbsolute,metricRelative,metricFromPoints,metricResidual:matrixError(metricAbsolute,metricFromPoints),determinantFormula,numericalDeterminant:det2(metricAbsolute),positiveDefiniteByAssumptions:true as const,eigenvaluesAbsolute:eigen2(metricAbsolute),eigenvaluesRelative:eigen2(metricRelative),kineticByPoint,kineticCartesian,kineticAbsolute,kineticRelative,diagonalKinetic,couplingKinetic,kineticAbsoluteResidual:kineticCartesian-kineticAbsolute,kineticRelativeResidual:kineticCartesian-kineticRelative,potential,forceStack,generalizedForceAbsolute,generalizedForceRelative,gravityGeneralized,tipGeneralized,generalizedForceResidual:norm(sub(generalizedForceAbsolute,add(gravityGeneralized,tipGeneralized))),cartesianPower,absolutePower,relativePower,powerAbsoluteResidual:cartesianPower-absolutePower,powerRelativeResidual:cartesianPower-relativePower,virtualDisplacement,virtualWorkCartesian,virtualWorkAbsolute,virtualWorkRelative,virtualWorkAbsoluteResidual:virtualWorkCartesian-virtualWorkAbsolute,virtualWorkRelativeResidual:virtualWorkCartesian-virtualWorkRelative,queryProbe:probe(epsilon),probeCurves,instantaneousOnly:true as const,dynamicsIntegrated:false as const,constraintReactionsSolved:false as const};
}
