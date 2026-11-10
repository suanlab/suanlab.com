/** Exact rational algebra with directed dyadic exponentials for a finite BVP.
 * This is an analytic family, not a finite-difference solver or a general PDE.
 */
import {f,zero,one,add,sub,mul,div,neg,eq,le,type F} from './learning-quantum-hilbert-core';
import {createTranscendentalContext} from './learning-interval-transcendentals';
import {readBoundaryLayerInput,type BoundaryLayerInput} from './learning-boundary-layer-input';
export interface BoundaryLayerInterval {lower:string;upper:string;}
type I={lower:F;upper:F};
const B=BigInt,precisionBits=192,S=B(1)<<B(precisionBits),context=createTranscendentalContext(precisionBits);
const parse=(s:string)=>{const[n,d='1']=s.split('/');return f(B(n),B(d));};
const text=(x:F)=>x.d===B(1)?String(x.n):`${x.n}/${x.d}`;
const point=(x:F):I=>({lower:x,upper:x});
const wire=(x:I):BoundaryLayerInterval=>({lower:text(x.lower),upper:text(x.upper)});
const plus=(a:I,b:I):I=>({lower:add(a.lower,b.lower),upper:add(a.upper,b.upper)});
const minus=(a:I,b:I):I=>({lower:sub(a.lower,b.upper),upper:sub(a.upper,b.lower)});
const scaled=(a:I,s:F):I=>s.n>=B(0)?{lower:mul(a.lower,s),upper:mul(a.upper,s)}:{lower:mul(a.upper,s),upper:mul(a.lower,s)};
const positiveDivision=(a:I,b:I):I=>{if(a.lower.n<B(0)||b.lower.n<=B(0))throw Error('Positive interval division required');return{lower:div(a.lower,b.upper),upper:div(a.upper,b.lower)};};
const absolute=(x:F)=>x.n<B(0)?neg(x):x;
function exponential(x:F):I{
 if(eq(x,zero))return point(one);
 if(x.n<B(0))throw Error('Nonnegative exponent magnitude required');
 const n={coefficient:x.n,exponent:0},d={coefficient:x.d,exponent:0},initial=context.expNegativeRatio(n,d);
 if(initial.upper-initial.lower===B(1))return{lower:f(initial.lower,S),upper:f(initial.upper,S)};
 for(let bits=256;bits<=2048;bits+=64){const next=createTranscendentalContext(bits).expNegativeRatio(n,d),shift=B(bits-precisionBits),lo=next.lower>>shift,hi=next.upper>>shift;
  if(lo===hi)return{lower:f(lo,S),upper:f(lo+B(1),S)};
 }
 throw Error('Exponential target cell unresolved');
}
export interface BoundaryLayerPoint {x:string;canonicalDistance:string;stretchedCoordinate:string|null;exact:BoundaryLayerInterval;outer:string|null;inner:BoundaryLayerInterval|null;composite:BoundaryLayerInterval|null;}
export function calculateBoundaryLayer(value:BoundaryLayerInput){
 const input=readBoundaryLayerInput(value);if(!input)throw new RangeError('Invalid boundary-layer input');
 const epsilon=parse(input.epsilon),a=parse(input.drift),forcing=parse(input.forcing),A=parse(input.left),C=parse(input.right),probeX=parse(input.probe);
 const gridSpacing=div(one,f(B(input.panels))),physicalGrid=Array.from({length:65},(_,i)=>f(B(i),B(64)));
 if(eq(a,zero)){
  const slope=sub(C,A),curvature=div(forcing,epsilon);
  const evaluate=(x:F):BoundaryLayerPoint=>({x:text(x),canonicalDistance:text(x),stretchedCoordinate:null,exact:wire(point(add(add(A,mul(slope,x)),div(mul(curvature,mul(x,sub(x,one))),f(B(2)))))),outer:null,inner:null,composite:null});
  return{input,branch:'zero-drift' as const,precisionBits,layer:null,probe:{...evaluate(probeX),derivative:wire(point(add(slope,mul(curvature,sub(probeX,f(B(1),B(2))))))),curvature:wire(point(curvature)),compositeDifference:null},physicalSamples:physicalGrid.map(evaluate),layerSamples:[] as BoundaryLayerPoint[],grid:{spacing:text(gridSpacing),cellOverLayer:null,pointsPerEfold:null,fourPointsPerEfold:null,firstCellLayerFraction:null},analyticResidual:'0' as const,contract:'Finite constant-coefficient BVP; zero drift uses the exact quadratic, not a boundary-layer formula.'};
 }
 const reflected=a.n<B(0),b=absolute(a),s=reflected?neg(one):one,near=reflected?C:A,far=reflected?A:C,t=div(b,epsilon),delta=div(epsilon,b),k=div(forcing,b),D=sub(sub(far,near),k),q=exponential(t),denominator=minus(point(one),q);
 const cache=new Map<string,I>();const decay=(y:F)=>{const z=mul(t,y),key=text(z),old=cache.get(key);if(old)return old;const next=exponential(z);cache.set(key,next);return next;};
 const ratio=(y:F,e:I):I=>eq(y,zero)?point(zero):eq(y,one)?point(one):positiveDivision(minus(point(one),e),denominator);
 const evaluateCanonical=(y:F):BoundaryLayerPoint=>{const x=reflected?sub(one,y):y,e=decay(y),outer=add(far,mul(k,sub(y,one))),exact=eq(y,zero)?point(near):eq(y,one)?point(far):plus(point(add(near,mul(k,y))),scaled(ratio(y,e),D));
  return{x:text(x),canonicalDistance:text(y),stretchedCoordinate:text(mul(t,y)),exact:wire(exact),outer:text(outer),inner:wire(minus(point(sub(far,k)),scaled(e,D))),composite:wire(minus(point(outer),scaled(e,D)))};};
 const py=reflected?sub(one,probeX):probeX,pe=decay(py),gain=positiveDivision(pe,denominator),qOverDen=positiveDivision(q,denominator),window=le(t,f(B(8)))?t:f(B(8));
 const derivative=scaled(plus(point(k),scaled(gain,mul(D,t))),s),curvature=scaled(gain,neg(mul(D,mul(t,t))));
 const difference=scaled(positiveDivision({lower:mul(q.lower,ratio(py,pe).lower),upper:mul(q.upper,ratio(py,pe).upper)},point(one)),D);
 const cellRatio=mul(t,gridSpacing),firstCell=minus(point(one),decay(gridSpacing));
 return{input,branch:'nonzero-drift' as const,precisionBits,layer:{side:reflected?'right' as const:'left' as const,delta:text(delta),inverseDelta:text(t),canonicalNear:text(near),canonicalFar:text(far),outerSlope:text(k),mismatch:text(D),hasLayerAmplitude:!eq(D,zero),farDecay:wire(q),normalizer:wire(denominator),outerUniformError:text(absolute(D)),compositeUniformError:wire(scaled(q,absolute(D))),compositeDerivativeUniformError:wire(scaled(qOverDen,mul(absolute(D),t))),compositeNearBoundaryResidual:'0' as const,compositeFarBoundaryResidual:wire(scaled(q,neg(D))),asymptoticallyThinAtCurrentInput:le(delta,f(B(1),B(10))),thinLabelThreshold:'1/10'},probe:{...evaluateCanonical(py),derivative:wire(derivative),curvature:wire(curvature),compositeDifference:wire(difference)},physicalSamples:physicalGrid.map(x=>evaluateCanonical(reflected?sub(one,x):x)),layerSamples:Array.from({length:65},(_,i)=>evaluateCanonical(div(mul(window,f(B(i),B(64))),t))),grid:{spacing:text(gridSpacing),cellOverLayer:text(cellRatio),pointsPerEfold:text(div(f(B(input.panels)),t)),fourPointsPerEfold:le(mul(f(B(4)),cellRatio),one),firstCellLayerFraction:wire(firstCell)},analyticResidual:'0' as const,contract:'Finite constant-coefficient BVP; exact analytic solution, matched composite and whole-domain certificates. Four points per e-fold and delta<=1/10 are display heuristics, not discretization error proofs.'};
}
export type BoundaryLayerResult=ReturnType<typeof calculateBoundaryLayer>;
