/** Prescribed static SI potentials on a bounded teaching domain. Neither a
 * material solver nor a time-dependent Maxwell evolution. Numerical residuals
 * from shared expressions are diagnostics, not independent certificates. */
import {teachingCoulombConstant} from './learning-charge-current-fields';
import {teachingVacuumPermeability,type StaticVector} from './learning-electro-magnetostatics';
export type PotentialMatrix=[StaticVector,StaticVector,StaticVector];
export interface QuadraticPotential {hessian:PotentialMatrix;linear:StaticVector;reference:number;coulombConstant?:number;}
export interface CylinderPotential {radius:number;current:number;permeability?:number;gaugeQuadratic?:number;gaugeCubic?:number;gaugeAxialConstant?:number;}
type Value=number|null;
function bound(x:number,lo:number,hi:number,name:string){if(!Number.isFinite(x)||x<lo||x>hi)throw new RangeError(`${name} outside [${lo},${hi}]`);}
function vector(x:StaticVector,limit:number,name:string){if(!Array.isArray(x)||x.length!==3)throw new RangeError(`${name} requires three components`);for(let i=0;i<3;i++)bound(x[i],-limit,limit,name);}
// A lost nonzero intermediate is unavailable, even if a later sum could hide it.
// This is a conservative arithmetic contract, not a physical singularity.
function product(...xs:Value[]):Value{if(xs.includes(0))return 0;if(xs.some(x=>x===null))return null;let v=1;for(const x of xs){v*=x!;if(!Number.isFinite(v)||v===0)return null;}return v;}
function sum(xs:Value[]):Value{if(xs.some(x=>x===null))return null;let s=0,c=0;for(const x of xs){const y=x!-c,t=s+y;c=(t-s)-y;s=t;}return Number.isFinite(s)?(s===0?0:s):null;}
const state=(v:Value)=>v===null?'arithmetic-range' as const:'finite' as const;
const numericVector=(xs:Value[]):StaticVector|null=>xs.some(x=>x===null)?null:xs as StaticVector;
const dot=(a:StaticVector,b:StaticVector)=>sum(a.map((x,i)=>product(x,b[i])));
function validate(s:QuadraticPotential){
 if(!Array.isArray(s.hessian)||s.hessian.length!==3)throw new RangeError('Hessian requires three rows');
 for(const row of s.hessian)vector(row,4,'Hessian V/m²');vector(s.linear,4,'Linear V/m');bound(s.reference,-100,100,'Reference V');
 for(let i=0;i<3;i++)for(let j=0;j<3;j++)if(s.hessian[i][j]!==s.hessian[j][i])throw new RangeError('Hessian must be symmetric; no silent symmetrization');
 const k=s.coulombConstant===undefined?teachingCoulombConstant:s.coulombConstant;bound(k,1e-12,1e12,'Coulomb coefficient');return k;
}
export function quadraticPotentialObservation(s:QuadraticPotential,position:StaticVector,testCharge=1){
 const k=validate(s);vector(position,4,'Position m');bound(testCharge,-4,4,'Test charge C');
 const quadratic:Value[]=[];for(let i=0;i<3;i++)for(let j=i;j<3;j++)quadratic.push(product(i===j?.5:1,s.hessian[i][j],position[i],position[j]));
 const shapePotential=sum([...quadratic,dot(s.linear,position)]),potential=sum([shapePotential,s.reference]);
 const field=numericVector(s.hessian.map((row,i)=>product(-1,sum([...row.map((x,j)=>product(x,position[j])),s.linear[i]]))));
 const divergence=-sum([s.hessian[0][0],s.hessian[1][1],s.hessian[2][2]])!||0;
 const density=product(divergence,1/(4*Math.PI*k));
 return{position:[...position]as StaticVector,shapePotential,potential,potentialStatus:state(potential),reference:s.reference,
  field,fieldStatus:field===null?'arithmetic-range' as const:'finite' as const,
  force:numericVector([0,1,2].map(i=>product(testCharge,field?.[i]??null))),potentialEnergy:product(testCharge,potential),
  testCharge,divergence,curl:[0,0,0]as StaticVector,chargeDensity:density,chargeDensityStatus:state(density),coulombConstant:k,
  model:'bounded-prescribed-symmetric-quadratic-electrostatic-potential' as const};
}
const nodes4=[-.8611363115940526,-.3399810435848563,.3399810435848563,.8611363115940526],weights4=[.3478548451374538,.6521451548625461,.6521451548625461,.3478548451374538];
function quadrature(f:(t:number)=>Value,nodes=nodes4,weights=weights4):Value{return sum(nodes.map((x,i)=>product(.5,weights[i],f((x+1)/2))));}
export function quadraticPotentialPathWork(s:QuadraticPotential,start:StaticVector,end:StaticVector,control:StaticVector,testCharge=1){
 validate(s);for(const p of [start,end,control])vector(p,4,'Path position m');bound(testCharge,-4,4,'Test charge C');
 const initial=quadraticPotentialObservation(s,start,testCharge),final=quadraticPotentialObservation(s,end,testCharge);
 const delta=end.map((x,i)=>x-start[i])as StaticVector,combined=end.map((x,i)=>x+start[i])as StaticVector;
 // C cancels before evaluation; avoid subtraction of two nearly equal C+V.
 const difference=sum([...s.hessian.flatMap((row,i)=>row.map((x,j)=>product(.5,delta[i],x,combined[j]))),dot(s.linear,delta)]);
 const lerp=(a:number,b:number,t:number)=>a+(b-a)*t;
 const at=(t:number,curved:boolean):{position:StaticVector;velocity:StaticVector}=>({position:start.map((x,i)=>curved?lerp(lerp(x,control[i],t),lerp(control[i],end[i],t),t):lerp(x,end[i],t))as StaticVector,velocity:start.map((x,i)=>curved?2*((1-t)*(control[i]-x)+t*(end[i]-control[i])):delta[i])as StaticVector});
 const integrand=(curved:boolean)=>(t:number)=>{const p=at(t,curved),field=quadraticPotentialObservation(s,p.position,0).field;return field===null?null:dot(field,p.velocity);};
 const straightIntegral=quadrature(integrand(false)),bezierIntegral=quadrature(integrand(true));
 const lowerOrder=quadrature(integrand(true),[-1/Math.sqrt(3),1/Math.sqrt(3)],[1,1]);
 return{initial,final,start:[...start]as StaticVector,end:[...end]as StaticVector,control:[...control]as StaticVector,
  potentialDifference:difference,potentialDifferenceStatus:state(difference),naiveEndpointDifference:initial.potential===null||final.potential===null?null:final.potential-initial.potential,
  straightFieldIntegral:straightIntegral,bezierFieldIntegral:bezierIntegral,quadratureOrderDifference:sum([bezierIntegral,product(-1,lowerOrder)]),
  fieldWorkStraight:product(testCharge,straightIntegral),fieldWorkBezier:product(testCharge,bezierIntegral),potentialEnergyChange:product(testCharge,difference),quasistaticExternalWork:product(testCharge,difference),
  straightResidual:sum([straightIntegral,difference]),bezierResidual:sum([bezierIntegral,difference]),
  quadratureMeaning:'polynomial-path-quadrature-difference-not-certified-error-bound' as const,
  externalWorkMeaning:'negative-field-work-only-for-quasistatic-or-zero-net-kinetic-change' as const};
}
export function cylinderVectorPotential(s:CylinderPotential,position:StaticVector){
 bound(s.radius,.25,4,'Radius m');bound(s.current,-4,4,'Current A');vector(position,4,'Position m');
 const mu=s.permeability===undefined?teachingVacuumPermeability:s.permeability,alpha=s.gaugeQuadratic===undefined?0:s.gaugeQuadratic,g=s.gaugeCubic===undefined?0:s.gaugeCubic,axial=s.gaugeAxialConstant===undefined?0:s.gaugeAxialConstant;
 bound(mu,1e-12,2,'Permeability T m/A');bound(alpha,-4,4,'Quadratic gauge T');bound(g,-4,4,'Cubic gauge T/m');bound(axial,-4,4,'Axial gauge T m');
 const [x,y,z]=position,a=s.radius,I=s.current,r=Math.hypot(x,y),inside=r<=a,boundary=r===a&&I!==0;
 const az=product(mu,I,1/(4*Math.PI),inside?((a-r)/a)*(1+r/a):-2*Math.log1p((r-a)/a));
 const coefficient=product(-mu,I,1/(2*Math.PI),inside?1/(a*a):1/(r*r));
 const gradient=numericVector([product(coefficient,x),product(coefficient,y),0]);
 const sourceField=gradient===null?null:[gradient[1],-gradient[0]||0,0]as StaticVector;
 const gaugeGradient=numericVector([sum([product(alpha,x),product(g,y,z)]),sum([product(alpha,y),product(g,x,z)]),sum([product(alpha,z),product(g,x,y),axial])]);
 const gaugeFunction=sum([product(.5,alpha,sum([product(x,x),product(y,y),product(z,z)])),product(g,x,y,z),product(axial,z)]);
 const sourceA: [Value,Value,Value]=[0,0,az];
 const transformed=numericVector([0,1,2].map(i=>sum([sourceA[i],gaugeGradient?.[i]??null])));
 const secondInside=product(-mu,I,1/(2*Math.PI*a*a)),secondOutsideAtInterface=product(mu,I,1/(2*Math.PI*a*a));
 const second=boundary?null:inside?secondInside:product(mu,I,1/(2*Math.PI*r*r));
 return{position:[...position]as StaticVector,radius:a,current:I,permeability:mu,radialDistance:r,region:r<a?'interior' as const:r>a?'exterior' as const:'boundary' as const,
  sourceAz:az,sourceAzStatus:state(az),sourceVectorPotential:numericVector(sourceA),sourceAzGradient:gradient,sourceAzGradientStatus:gradient===null?'arithmetic-range' as const:'finite' as const,
  gaugeFunction,gaugeGradient,gaugeQuadratic:alpha,gaugeCubic:g,gaugeAxialConstant:axial,vectorPotential:transformed,vectorPotentialStatus:transformed===null?'arithmetic-range' as const:'finite' as const,
  magneticField:sourceField,magneticFieldStatus:sourceField===null?'arithmetic-range' as const:'finite' as const,
  sourceAzRadialDerivative:r===0?0:product(coefficient,r),sourceAzSecondRadialDerivative:second,
  sourceAzLaplacian:boundary?null:inside?product(2,secondInside):0,
  secondDerivativeStatus:boundary?'boundary-one-sided-only' as const:state(second),sourceDivergenceA:0,gaugeDivergenceA:3*alpha||0,divergenceA:3*alpha||0,
  interfaceLimits:{sourceAz:0,radialFirstDerivative:product(-mu,I,1/(2*Math.PI*a)),secondRadialInside:secondInside,secondRadialOutside:secondOutsideAtInterface},
  reference:'source-Az-zero-at-source-radius-not-at-infinity' as const,gaugeMeaning:'static-smooth-single-valued-gradient-not-a-physical-background' as const};
}
export function cylinderPotentialRectangle(s:CylinderPotential,left:number,right:number,bottom:number,top:number,y=0){
 for(const value of [left,right,bottom,top,y])bound(value,-4,4,'Rectangle coordinate m');
 const vertices:StaticVector[]=[[left,y,bottom],[right,y,bottom],[right,y,top],[left,y,top]];
 const observations=vertices.map(p=>cylinderVectorPotential(s,p));
 // Iterate the actual four oriented edges,including the gauge contribution.
 const edges=vertices.map((p,i)=>{const end=vertices[(i+1)%4],delta=end.map((x,j)=>x-p[j])as StaticVector;return quadrature(t=>{const a=cylinderVectorPotential(s,p.map((x,j)=>x+t*delta[j])as StaticVector).vectorPotential;return a===null?null:dot(a,delta);});});
 const circulation=sum(edges),mu=s.permeability===undefined?teachingVacuumPermeability:s.permeability;
 const rLeft=observations[0].radialDistance,rRight=observations[1].radialDistance,deltaSquared=product(right-left,right+left);
 // At fixed y, rRight²-rLeft²=(right-left)(right+left). Cancel
 // nearly equal source-potential values before subtraction on a thin loop.
 const logarithmArgument=product(deltaSquared,1/(left*left+y*y));
 const deltaAz=rLeft<=s.radius&&rRight<=s.radius?product(-mu,s.current,1/(4*Math.PI*s.radius*s.radius),deltaSquared)
  :rLeft>=s.radius&&rRight>=s.radius?product(-mu,s.current,1/(4*Math.PI),logarithmArgument===null?null:Math.log1p(logarithmArgument))
  :sum([observations[1].sourceAz,product(-1,observations[0].sourceAz)]);
 const sourceCirculation=product(top-bottom,deltaAz);
 return{vertices,edgeIntegrals:edges,circulation,sourceCirculation,orientedAreaY:-(right-left)*(top-bottom)||0,gaugeClosedIntegral:sum([circulation,product(-1,sourceCirculation)]),
  diagnosticMeaning:'shared-potential-endpoint-identity-not-independent-flux-verification' as const};
}
