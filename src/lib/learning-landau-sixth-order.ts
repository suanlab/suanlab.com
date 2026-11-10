/** Phenomenological scalar sixth-order Landau potential on the whole real line.
 * Exact polynomial root counts/multiplicities, rational free-energy enclosures.
 * This is minimization of a specified effective theory, not finite-T sampling.
 */
import { binaryProduct } from './learning-binary-products';
import { partitionObservationFromRange, type PartitionObservation } from './learning-partition-exact-core';
import { exactQ as q,exactAdd as add,exactSub as sub,exactNeg as neg,exactMul as mul,exactDiv as div,exactCompare as cmp,exactAbs as absolute,
 exactPolynomialEvaluate as evaluate,exactPolynomialDerivative as derivative,isolateExactPolynomialRoots, type ExactQ as Q,type ExactPolynomial as Polynomial,type ExactRootInterval } from './learning-exact-polynomial-roots';
export interface LandauSixthOrderParameters { quadraticA:number;quarticB:number;sexticC:number;fieldH:number }
export interface LandauSixthOrderOptions {minimumPrecision?:number;maximumPrecision?:number}
interface I {lower:Q;upper:Q}
const Z=BigInt(0),O=BigInt(1),integer=(n:number)=>q(BigInt(n));
const binary=(n:number):Q=>{const x=binaryProduct(n);return x.exponent>=0?q(x.coefficient<<BigInt(x.exponent)):q(x.coefficient,O<<BigInt(-x.exponent));};
const point=(v:Q):I=>({lower:v,upper:v});
const sum=(a:I,b:I):I=>({lower:add(a.lower,b.lower),upper:add(a.upper,b.upper)});
const min=(a:Q,b:Q)=>cmp(a,b)<Z?a:b,max=(a:Q,b:Q)=>cmp(a,b)>Z?a:b;
const negative=(a:I):I=>({lower:neg(a.upper),upper:neg(a.lower)});
const difference=(a:I,b:I)=>sum(a,negative(b));
const squared=(x:I):I=>({lower:x.lower.n<=Z&&x.upper.n>=Z?integer(0):min(mul(x.lower,x.lower),mul(x.upper,x.upper)),upper:max(mul(x.lower,x.lower),mul(x.upper,x.upper))});
function times(a:I,b:I):I{const values=[mul(a.lower,b.lower),mul(a.lower,b.upper),mul(a.upper,b.lower),mul(a.upper,b.upper)];return {lower:values.reduce(min),upper:values.reduce(max)};}
function range(p:Polynomial,x:I):I {let result=point(integer(0));for(let j=p.length-1;j>=0;j--)result=sum(times(result,x),point(p[j]));return result;}
const bv=(coefficient:bigint)=>({coefficient,exponent:0});
const observe=(x:I)=>partitionObservationFromRange({lower:{n:bv(x.lower.n),d:bv(x.lower.d)},upper:{n:bv(x.upper.n),d:bv(x.upper.d)}});
const serialize=(x:Q)=>({numerator:{coefficient:x.n.toString(),exponent:0},denominator:{coefficient:x.d.toString(),exponent:0}});
const bounds=(x:I)=>({lower:serialize(x.lower),upper:serialize(x.upper)});
const infinity=(sign:number):PartitionObservation=>({value:null,sign,logMagnitude:null,logResolution:'not-applicable',resolution:'mathematical-infinity',bounds:null});
const ready=(x:PartitionObservation)=>x.resolution!=='unresolved-at-precision-ceiling';
const precisions=[128,256,512,1024,2048,4096,8192];
export function landauSixthOrderModel(input:LandauSixthOrderParameters,options:LandauSixthOrderOptions={}) {
 if(!input||![input.quadraticA,input.quarticB,input.sexticC,input.fieldH].every(Number.isFinite)||Math.abs(input.quadraticA)>4||Math.abs(input.quarticB)>4||input.sexticC<.25||input.sexticC>4||Math.abs(input.fieldH)>4)throw new RangeError('Expected |a|,|b|,|H| <=4 and c=0.25..4');
 const minimum=options.minimumPrecision??128,maximum=options.maximumPrecision??8192;
 if(!precisions.includes(minimum)||!precisions.includes(maximum)||minimum>maximum)throw new RangeError('Invalid precision range');
 const parameters=Object.freeze({quadraticA:input.quadraticA,quarticB:input.quarticB,sexticC:input.sexticC,fieldH:input.fieldH}),a=binary(input.quadraticA),b=binary(input.quarticB),c=binary(input.sexticC),H=binary(input.fieldH);
 const g=[neg(H),a,integer(0),b,integer(0),c],phi=[integer(0),neg(H),div(a,integer(2)),integer(0),div(b,integer(4)),integer(0),div(c,integer(6))];
 const curve=derivative(g),coexistenceCoefficients=b.n<Z&&cmp(mul(integer(16),mul(a,c)),mul(integer(3),mul(b,b)))===Z;
 const coexistence=!H.n&&coexistenceCoefficients,yStar=div(mul(integer(-3),b),mul(integer(4),c));
 const attempts:{precisionBits:number;reason:string|null}[]=[];
 function calculate(P:number){
  const isolation=isolateExactPolynomialRoots(g,{precisionBits:P});
  let numericReady=true;
  const rows=isolation.roots.map((root:ExactRootInterval)=>{
   const {lower,upper,multiplicity}=root,x={lower,upper},mid=div(add(lower,upper),integer(2));
   let leadingPolynomial=g;for(let j=0;j<multiplicity;j++)leadingPolynomial=derivative(leadingPolynomial);
   const leading=range(leadingPolynomial,x),leadingSign=leading.lower.n>Z?1:leading.upper.n<Z?-1:0;
   const localKind=multiplicity%2===0?'stationary-inflection' as const:leadingSign>0?'minimum' as const:leadingSign<0?'maximum' as const:'unresolved' as const;
   const gradient=range(g,x),maxGradient=max(absolute(gradient.lower),absolute(gradient.upper));
   const error=mul(div(sub(upper,lower),integer(2)),maxGradient),center=evaluate(phi,mid);
   // Exact coexistence factorization proves every minimum has Phi=0.
   let energy=coexistence&&localKind==='minimum'?point(integer(0)):{lower:sub(center,error),upper:add(center,error)};
   if(coexistenceCoefficients&&H.n){
    // At a stationary point H=(c/3)m(m^2-y*)(3m^2-y*).
    // Thus Phi=3H^2/[2c(3m^2-y*)^2]-Hm: no O(1) cancellation.
    const d=difference(times(point(integer(3)),squared(x)),point(yStar)),d2=squared(d);
    if(d2.lower.n>Z){const factored=difference(times(point(div(mul(integer(3),mul(H,H)),mul(integer(2),c))),{lower:div(integer(1),d2.upper),upper:div(integer(1),d2.lower)}),times(point(H),x));energy={lower:max(energy.lower,factored.lower),upper:min(energy.upper,factored.upper)};}
   }
   const curvature=multiplicity>1?point(integer(0)):range(curve,x);
   let branchFieldDerivative:PartitionObservation|null=null;
   if(multiplicity%2) {
    if(multiplicity>1&&leadingSign)branchFieldDerivative=infinity(leadingSign);
    else if(curvature.lower.n>Z||curvature.upper.n<Z)branchFieldDerivative=observe({lower:div(integer(1),curvature.upper),upper:div(integer(1),curvature.lower)});
   }
   const publicRoot={magnetization:observe(x),curvature:observe(curvature),constrainedFreeEnergy:observe(energy),leadingStationaryDerivative:observe(leading),
    stationaryMultiplicity:multiplicity,localKind,branchFieldDerivative,
    derivativeStatus:multiplicity%2===0?'fold-no-single-valued-derivative':multiplicity>1?'singular-limiting-derivative':'regular-branch',
    residualEnclosure:bounds(gradient)};
   if(!leadingSign||![publicRoot.magnetization,publicRoot.curvature,publicRoot.constrainedFreeEnergy,publicRoot.leadingStationaryDerivative].every(ready)||multiplicity%2&&(!branchFieldDerivative||!ready(branchFieldDerivative)))numericReady=false;
   return {root,publicRoot,energy};
  });
  const minima=rows.map((x,i)=>x.publicRoot.localKind==='minimum'?i:-1).filter(i=>i>=0);
  let global:number[]=[];let candidates=minima;
  if(!H.n){
   if(b.n<Z){const comparison=cmp(mul(integer(16),mul(a,c)),mul(integer(3),mul(b,b)));
    global=comparison===Z?minima:minima.filter(i=>comparison<Z?rows[i].publicRoot.magnetization.sign!==0:rows[i].publicRoot.magnetization.sign===0);
   }else global=minima.filter(i=>a.n<Z?rows[i].publicRoot.magnetization.sign!==0:rows[i].publicRoot.magnetization.sign===0);
   candidates=global;
  }else{
   // A minimum of the wrong sign is dominated by its reflected state.
   candidates=minima.filter(i=>rows[i].publicRoot.magnetization.sign===(H.n>Z?1:-1));
   candidates=candidates.filter(i=>!candidates.some(j=>j!==i&&cmp(rows[j].energy.upper,rows[i].energy.lower)<Z));
   if(candidates.length===1||candidates.length>1&&candidates.every(i=>cmp(rows[i].energy.lower,rows[i].energy.upper)===Z&&cmp(rows[i].energy.lower,rows[candidates[0]].energy.lower)===Z))global=candidates;
  }
  if(rows.some(x=>x.publicRoot.localKind==='unresolved')){global=[];candidates=rows.map((x,i)=>x.publicRoot.localKind==='minimum'||x.publicRoot.localKind==='unresolved'?i:-1).filter(i=>i>=0);}
  if(!minima.length||!candidates.length)numericReady=false;
  const selectionResolved=global.length>0;
  const lowest=selectionResolved?rows[global[0]].energy:null;
  const roots=rows.map((row,i)=>{
   const isGlobal=selectionResolved?global.includes(i):null;
   const freeEnergyExcess=lowest?(isGlobal?observe(point(integer(0))):observe(difference(row.energy,lowest))):null;
   if(freeEnergyExcess&&!ready(freeEnergyExcess))numericReady=false;
   return {...row.publicRoot,globalMinimum:isGlobal,
    role:row.publicRoot.localKind==='minimum'?(isGlobal===null?'minimum-global-unresolved':isGlobal?'global-minimum':'metastable-minimum'):row.publicRoot.localKind==='maximum'?'unstable-maximum':row.publicRoot.localKind,
    freeEnergyExcess};
  });
  const barriers:{minimumIndex:number;maximumIndex:number;barrier:PartitionObservation}[]=[];
  for(let j=0;j<rows.length;j++)if(rows[j].publicRoot.localKind==='maximum'){
   for(const direction of [-1,1]){let i=j+direction;while(i>=0&&i<rows.length&&rows[i].publicRoot.localKind==='stationary-inflection')i+=direction;
    if(i>=0&&i<rows.length&&rows[i].publicRoot.localKind==='minimum'){const barrier=observe(difference(rows[j].energy,rows[i].energy));barriers.push({minimumIndex:i,maximumIndex:j,barrier});if(!ready(barrier))numericReady=false;}
   }
  }
  const fullyResolved=numericReady&&selectionResolved;
  return {fullyResolved,precisionBits:P,rootCount:isolation.distinctRootCount,realRootMultiplicity:rows.reduce((n,x)=>n+x.root.multiplicity,0),roots,
   rootsComplete:true,globalMinimumIndices:global,globalMinimumCandidates:candidates,globalSelectionResolved:selectionResolved,
   minimumFreeEnergy:lowest?observe(lowest):null,barriers,bisections:isolation.bisections,
   exactCoexistence:coexistence,unresolvedReason:fullyResolved?null:!selectionResolved?'global free-energy comparison unresolved':'stationary observable enclosure unresolved'};
 }
 let latest:ReturnType<typeof calculate>|undefined;
 for(const P of precisions.filter(p=>p>=minimum&&p<=maximum)){latest=calculate(P);attempts.push({precisionBits:P,reason:latest.unresolvedReason});if(latest.fullyResolved)break;}
 return {scope:'phenomenological-sixth-order-landau-minimization' as const,parameters,...latest!,attempts,
  approximationContract:'Phi=a*m^2/2+b*m^4/4+c*m^6/6-H*m on m in R; exact actual binary64 coefficients with |a|,|b|,|H|<=4,c=0.25..4. Sturm counts and gcd chains certify all real stationary roots and multiplicities. Directed rational free-energy bounds use a midpoint plus a mean-value bound; observations require relative width <=2^-48. Zero-field symmetry and exact coexistence are algebraic identities. Nearby roots are not merged by a distance tolerance. Branch derivatives are not equilibrium fluctuation susceptibilities; folds have no single-valued local derivative. No finite-temperature integration, transition time, measured hysteresis or nearest-neighbor universality is claimed.'};
}
