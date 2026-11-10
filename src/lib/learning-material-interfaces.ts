/** Stationary isotropic linear half-spaces with constant positive coefficients.
 * One-sided limits are explicit; analytic balances are not independent certificates.
 */
import {maxwellMedium,teachingMaxwellPermittivity as eps0} from './learning-maxwell';
import type {StaticVector} from './learning-electro-magnetostatics';
type Value=number|null;
function bound(x:number,lo:number,hi:number,name:string){if(!Number.isFinite(x)||x<lo||x>hi)throw new RangeError(`${name} outside [${lo},${hi}]`);}
function product(...xs:Value[]):Value{if(xs.includes(0))return 0;if(xs.includes(null))return null;let v=1;for(const x of xs){v*=x!;if(!Number.isFinite(v)||v===0)return null;}return v;}
function sum(...xs:Value[]):Value{if(xs.includes(null))return null;let s=0,c=0;for(const x of xs){const y=x!-c,t=s+y;c=(t-s)-y;s=t;}return Number.isFinite(s)?s||0:null;}
function divide(x:Value,y:Value):Value{if(x===null||y===null||y===0)return null;const q=x/y;return !Number.isFinite(q)||x!==0&&q===0?null:q||0;}
const vector=(v:Value[]):StaticVector|null=>v.includes(null)?null:v as StaticVector;
const status=(x:Value)=>x===null?'arithmetic-range' as const:'finite' as const;
function position(p:StaticVector){if(!Array.isArray(p)||p.length!==3)throw new RangeError('Three coordinates required');for(let i=0;i<3;i++)bound(p[i],-4,4,'Coordinate m');}

export interface DielectricInterface{
 permittivity1?:number;permittivity2?:number;normalDisplacement1:number;freeSurfaceCharge:number;tangentialElectric:number;
 tangentialJump?:number;normalMode?:'specified-charge'|'equal-normal-field-negative-control';
}
function dielectricParameters(s:DielectricInterface){
 const e1=maxwellMedium({permittivity:s.permittivity1}).permittivity,e2=maxwellMedium({permittivity:s.permittivity2}).permittivity;
 bound(s.normalDisplacement1,-4e-8,4e-8,'Normal D C/m²');bound(s.freeSurfaceCharge,-4e-8,4e-8,'Free surface charge C/m²');bound(s.tangentialElectric,-4,4,'Tangential E V/m');
 const tangent=s.tangentialJump??0,mode=s.normalMode??'specified-charge';bound(tangent,-4,4,'Tangential jump V/m');
 if(mode!=='specified-charge'&&mode!=='equal-normal-field-negative-control')throw new RangeError('Unknown dielectric normal mode');return{e1,e2,tangent,mode};
}
export function dielectricInterfaceState(s:DielectricInterface,p:StaticVector){
 const {e1,e2,tangent,mode}=dielectricParameters(s);position(p);
 const d1=s.normalDisplacement1,en1=divide(d1,e1),targetD2=sum(d1,s.freeSurfaceCharge);
 const en2=mode==='specified-charge'?divide(targetD2,e2):en1,d2=mode==='specified-charge'?targetD2:product(e2,en1),et2=sum(s.tangentialElectric,tangent);
 const E1=vector([s.tangentialElectric,0,en1]),E2=vector([et2,0,en2]);
 const D1=vector([product(e1,s.tangentialElectric),0,d1]),D2=vector([product(e2,et2),0,d2]);
 const P1=vector([product(e1-eps0,s.tangentialElectric),0,product(e1-eps0,en1)]),P2=vector([product(e2-eps0,et2),0,product(e2-eps0,en2)]);
 // Cancel the common normal background analytically, before rounded face values.
 const stableNormalEJump=mode==='specified-charge'?sum(divide(s.freeSurfaceCharge,e2),product(d1,divide(e1-e2,product(e1,e2)))):0;
 const requiredFreeCharge=mode==='specified-charge'?s.freeSurfaceCharge:product(e2-e1,en1);
 const totalSurfaceCharge=product(eps0,stableNormalEJump);
 const boundSurfaceCharge=mode==='specified-charge'?sum(product(eps0,d1,divide(e1-e2,product(e1,e2))),product(-1,divide(e2-eps0,e2),s.freeSurfaceCharge)):product(-1,requiredFreeCharge);
 const V1=sum(product(-s.tangentialElectric,p[0]),product(-1,en1,p[2])),V2=sum(product(-1,et2,p[0]),product(-1,en2,p[2]));
 const region=p[2]<0?'medium-1' as const:p[2]>0?'medium-2' as const:'interface-two-sided' as const;
 const constraint=mode==='specified-charge'?0:exactProductSum([[d1,e2],[-d1,e1],[-s.freeSurfaceCharge,e1]]);
 const normalResidual=divide(constraint,e1);
 return{position:[...p] as StaticVector,permittivity1:e1,permittivity2:e2,vacuumReferencePermittivity:eps0,normalMode:mode,tangentialJump:tangent,
  medium1:{electricField:E1,displacement:D1,polarization:P1,potential:V1},medium2:{electricField:E2,displacement:D2,polarization:P2,potential:V2},
  region,electricField:region==='interface-two-sided'?null:region==='medium-1'?E1:E2,displacement:region==='interface-two-sided'?null:region==='medium-1'?D1:D2,
  potential:region==='interface-two-sided'?null:region==='medium-1'?V1:V2,
  interfacePotential1:product(-s.tangentialElectric,p[0]),interfacePotential2:product(-1,et2,p[0]),stablePotentialJump:product(-tangent,p[0]),
  specifiedFreeSurfaceCharge:s.freeSurfaceCharge,requiredFreeSurfaceCharge:requiredFreeCharge,boundSurfaceCharge,totalSurfaceCharge,
  stableNormalEJump,naiveNormalEJump:sum(en2,product(-1,en1)),naiveNormalDJump:sum(d2,-d1),naivePolarizationSurfaceCharge:sum(P1?.[2]??null,product(-1,P2?.[2]??null)),
  normalSourceResidual:normalResidual,naiveNormalSourceResidual:sum(requiredFreeCharge,-s.freeSurfaceCharge),
  electrostaticFamilyCompatible:tangent!==0?false:constraint===null?null:constraint===0,
  boundaryNormal:[0,0,1] as StaticVector,bulkFreeChargeDensity:0,bulkBoundChargeDensity:0,
  fieldStatus:E1===null||E2===null?'arithmetic-range' as const:'finite' as const,
  jumpMeaning:'analytic-prescribed-family-jumps-not-rounded-face-subtraction' as const,
  diagnosticMeaning:'planar-prescribed-sources-not-an-isolated-device-or-independent-certificate' as const};
}
export function dielectricInterfaceMeasures(s:DielectricInterface,span:number,depth:number,halfHeight:number){
 for(const[x,name]of[[span,'Span'],[depth,'Depth'],[halfHeight,'Half height']]as const)bound(x,0,4,`${name} m`);
 const q=dielectricInterfaceState(s,[0,0,0]),area=product(span,depth),halfSpan=product(.5,span);
 const limits=q.medium1.electricField===null||q.medium2.electricField===null?null:[q.medium1.electricField,q.medium2.electricField];
 const sides=limits===null?null:[product(limits[0][0],span),product(limits[0][2],halfHeight),product(limits[1][2],halfHeight),product(-limits[1][0],span)];
 const normalFlux=product(q.requiredFreeSurfaceCharge,area),totalFlux=product(q.totalSurfaceCharge,area),loopLimit=product(-q.tangentialJump,span);
 const hasGeometry=halfHeight>0&&(span===0||halfSpan!==null);
 return{span,depth,halfHeight,area,vertices:hasGeometry?[[product(-1,halfSpan),0,-halfHeight],[halfSpan,0,-halfHeight],[halfSpan,0,halfHeight],[product(-1,halfSpan),0,halfHeight]]:null,
  topDFlux:halfHeight===0?null:product(q.medium2.displacement?.[2]??null,area),bottomDFlux:halfHeight===0?null:product(-1,q.medium1.displacement?.[2]??null,area),
  stableDFlux:halfHeight===0?null:normalFlux,stableVacuumEFlux:halfHeight===0?null:totalFlux,
  naiveDFlux:halfHeight===0?null:product(q.naiveNormalDJump,area),specifiedFreeCharge:product(s.freeSurfaceCharge,area),requiredFreeCharge:product(q.requiredFreeSurfaceCharge,area),boundCharge:product(q.boundSurfaceCharge,area),
  electricCirculation:hasGeometry?loopLimit:null,limitingElectricCirculation:loopLimit,
  loopEdgeIntegrals:hasGeometry&&sides?[sides[0],sum(sides[1],sides[2]),sides[3],product(-1,sum(sides[1],sides[2]))]:null,
  geometryStatus:halfHeight===0?'collapsed-height-two-sided-limit' as const:!hasGeometry?'arithmetic-range' as const:'finite' as const,
  normalizationStatus:area===0?'zero-area-undefined' as const:status(area),
  diagnosticMeaning:'fixed-pillbox-outward-faces-and-loop-normal-minus-y-analytic-limits-not-independent-quadrature' as const};
}

export interface NormalIncidenceInterface{
 permittivity1?:number;permittivity2?:number;permeability1?:number;permeability2?:number;
 amplitude:number;wavenumber1:number;reflectionMode?:'matched-boundary'|'forced-zero-reflection';
}
// Dekker error-free product splitting: inputs are bounded normal coefficients.
function twoProduct(a:number,b:number){const p=a*b,c=134217729*a,ah=c-(c-a),al=a-ah,d=134217729*b,bh=d-(d-b),bl=b-bh;return[p,((ah*bh-p)+ah*bl+al*bh)+al*bl];}
function exactProductSum(pairs:[number,number][]):Value{
 let expansion:number[]=[];
 for(const[a,b]of pairs){
  const p=a*b;if(a!==0&&b!==0&&Math.abs(p)<Number.MIN_VALUE/Number.EPSILON)return null;
  for(const term of twoProduct(a,b)){
   let Q=term;const next:number[]=[];
   for(const e of expansion){const s=Q+e,bv=s-Q,error=(Q-(s-bv))+(e-bv);if(error!==0)next.push(error);Q=s;}
   if(Q!==0||!next.length)next.push(Q);expansion=next;
  }
 }
 return sum(...expansion);
}
function determinant(a:number,b:number,c:number,d:number){return exactProductSum([[a,b],[-c,d]])!;}
export function normalInterfaceCoefficients(s:NormalIncidenceInterface){
 const m1=maxwellMedium({permittivity:s.permittivity1,permeability:s.permeability1}),m2=maxwellMedium({permittivity:s.permittivity2,permeability:s.permeability2});
 bound(s.amplitude,-4,4,'Amplitude V/m');bound(s.wavenumber1,0,16,'Incident wavenumber rad/m');const mode=s.reflectionMode??'matched-boundary';
 if(mode!=='matched-boundary'&&mode!=='forced-zero-reflection')throw new RangeError('Unknown reflection mode');
 const Z1=Math.sqrt(m1.permeability/m1.permittivity),Z2=Math.sqrt(m2.permeability/m2.permittivity),zSum=Z1+Z2;
 const det=determinant(m2.permeability,m1.permittivity,m1.permeability,m2.permittivity),den=(Math.sqrt(m2.permeability*m1.permittivity)+Math.sqrt(m1.permeability*m2.permittivity))**2;
 const referenceReflection=det/den,referenceTransmission=2*Z2/zSum,referenceHFactor=2*Z1/zSum;
 const reflection=mode==='matched-boundary'?referenceReflection:0,transmission=mode==='matched-boundary'?referenceTransmission:1,hFactor=mode==='matched-boundary'?referenceHFactor:1;
 const omega=product(s.wavenumber1,m1.speed),k2=divide(omega,m2.speed);
 const requiredSheetCurrentAmplitude=mode==='matched-boundary'?0:product(s.amplitude,referenceReflection,1/Z1+1/Z2);
 const incidentCycleMean=omega===null||omega===0?null:product(.5,s.amplitude,s.amplitude,1/Z1);
 const reflectedCycleMean=omega===null||omega===0?null:product(-.5,s.amplitude,s.amplitude,reflection,reflection,1/Z1);
 const transmittedCycleMean=omega===null||omega===0?null:product(.5,s.amplitude,s.amplitude,transmission,transmission,1/Z2);
 return{medium1:m1,medium2:m2,impedance1:Z1,impedance2:Z2,reflectionMode:mode,referenceReflection,referenceTransmission,referenceHFactor,naiveReflection:(Z2-Z1)/zSum,impedanceDeterminant:det,impedanceMatched:det===0,
  reflection,transmission,hFactor,reflectance:product(reflection,reflection),transmittance:product(Z1/Z2,transmission,transmission),referenceReflectance:product(referenceReflection,referenceReflection),referenceTransmittance:product(Z1/Z2,referenceTransmission,referenceTransmission),
  omega,wavenumber2:k2,requiredSheetCurrentAmplitude,incidentCycleMean,reflectedSignedCycleMean:reflectedCycleMean,transmittedCycleMean,
  observedCycleReflectance:divide(product(-1,reflectedCycleMean),incidentCycleMean),observedCycleTransmittance:divide(transmittedCycleMean,incidentCycleMean),
  cycleStatus:omega===0?'no-cycle-static-family' as const:omega===null?'arithmetic-range' as const:'finite-frequency' as const,
  noSheetFamilyCompatible:mode==='matched-boundary'||det===0||s.amplitude===0,
  diagnosticMeaning:'analytic-scattering-family-not-independent-boundary-or-power-verification' as const};
}
export function normalInterfaceState(s:NormalIncidenceInterface,z:number,time:number){
 const q=normalInterfaceCoefficients(s);bound(z,-4,4,'Axial position m');bound(time,-4,4,'Time s');
 const temporal=product(q.omega,time),spatial1=product(s.wavenumber1,z),spatial2=product(q.wavenumber2,z);
 for(const phase of[temporal,spatial1,spatial2,sum(spatial1,temporal),sum(spatial1,product(-1,temporal)),sum(spatial2,product(-1,temporal))])if(phase!==null&&Math.abs(phase)>1e6)throw new RangeError('Phase exceeds supported 1e6 rad budget');
 const trig=(x:Value,f:(x:number)=>number)=>x===null?null:f(x),C=trig(spatial1,Math.cos),S=trig(spatial1,Math.sin),U=trig(temporal,Math.cos),V=trig(temporal,Math.sin);
 const a=q.transmission,b=q.hFactor,A=s.amplitude,Z1=q.impedance1,Z2=q.impedance2;
 // Stable standing-wave basis preserves a small 1+r through direct transmission.
 const e1=product(A,sum(product(a,C,U),product(b,S,V))),h1=product(A,1/Z1,sum(product(b,C,U),product(a,S,V)));
 const e1z=product(A,s.wavenumber1,sum(product(-a,S,U),product(b,C,V))),e1t=product(A,q.omega,sum(product(-a,C,V),product(b,S,U)));
 const h1z=product(A,s.wavenumber1,1/Z1,sum(product(-b,S,U),product(a,C,V))),h1t=product(A,q.omega,1/Z1,sum(product(-b,C,V),product(a,S,U)));
 const phase2=sum(spatial2,product(-1,temporal)),cos2=trig(phase2,Math.cos),sin2=trig(phase2,Math.sin);
 const e2=product(A,q.transmission,cos2),h2=divide(e2,Z2),e2z=product(-A,q.transmission,q.wavenumber2,sin2),e2t=product(A,q.transmission,q.omega,sin2),h2z=divide(e2z,Z2),h2t=divide(e2t,Z2);
 function observation(E:Value,H:Value,Ez:Value,Et:Value,Hz:Value,Ht:Value,medium:typeof q.medium1){
  const B=product(medium.permeability,H),uE=product(.5,medium.permittivity,E,E),uB=product(.5,medium.permeability,H,H);
  return{...medium,electricValue:E,magneticIntensity:H,magneticValue:B,electricField:vector([E,0,0]),magneticField:vector([0,B,0]),energyDensity:sum(uE,uB),electricEnergyDensity:uE,magneticEnergyDensity:uB,poyntingZ:product(E,H),
   electricSpatialDerivative:Ez,electricTimeDerivative:Et,magneticIntensitySpatialDerivative:Hz,magneticIntensityTimeDerivative:Ht,
   faradayResidualY:sum(Ez,product(medium.permeability,Ht)),ampereResidualX:sum(product(-1,Hz),product(-medium.permittivity,Et)),fieldStatus:status(E)};
 }
 const medium1=observation(e1,h1,e1z,e1t,h1z,h1t,q.medium1),medium2=observation(e2,h2,e2z,e2t,h2z,h2t,q.medium2);
 const boundaryE=product(A,q.transmission,U),leftH=product(A,q.hFactor,U,1/Z1),rightH=product(A,q.transmission,U,1/Z2);
 const sheetCurrent=product(q.requiredSheetCurrentAmplitude,U),sheetWork=product(sheetCurrent,boundaryE);
 const region=z<0?'medium-1' as const:z>0?'medium-2' as const:'interface-two-sided' as const;
 return{...q,z,time,medium1,medium2,region,electricValue:region==='interface-two-sided'?null:region==='medium-1'?e1:e2,magneticIntensity:region==='interface-two-sided'?null:region==='medium-1'?h1:h2,
  boundary:{electricLeft:boundaryE,electricRight:boundaryE,magneticIntensityLeft:leftH,magneticIntensityRight:rightH,naiveHJump:sum(rightH,product(-1,leftH)),requiredSheetCurrentX:sheetCurrent,
   requiredSheetWork:sheetWork,analyticPowerMismatchWithoutSheet:sheetWork,naivePowerDifference:sum(product(boundaryE,leftH),product(-1,boundaryE,rightH)),
   diagnosticMeaning:'source-aware-boundary-work-with-continuous-E-not-an-independent-certificate' as const},
  referenceMeaning:'medium-1-and-medium-2-are-one-sided-extensions-not-an-interface-average' as const};
}
