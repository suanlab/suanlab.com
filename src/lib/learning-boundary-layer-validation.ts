/** Rebuild every result leaf from owned input, using independent fractions and
 * negative-Taylor exponentials. Untrusted output strings are compared, never
 * parsed as arithmetic. Only trusted reconstructions enter the bounded cache.
 */
import {Rational as Q} from './learning-graph-gauge-validation-math';
import {boundaryLayerIndependentDecay} from './learning-boundary-layer-validation-exp';
import {readBoundaryLayerInput,type BoundaryLayerInput} from './learning-boundary-layer-input';
import {snapshotQubitWire} from './learning-qubit-wire';
import type {BoundaryLayerResult} from './learning-boundary-layer-protocol';
const B=BigInt,Z=new Q(B(0)),O=new Q(B(1)),integer=(n:number)=>new Q(B(n));
type I=readonly[Q,Q];
const lt=(a:Q,b:Q)=>a.n*b.d<b.n*a.d,abs=(a:Q)=>a.n<B(0)?a.neg():a,pt=(q:Q):I=>[q,q];
const sum=(a:I,b:I):I=>[a[0].add(b[0]),a[1].add(b[1])],difference=(a:I,b:I):I=>[a[0].sub(b[1]),a[1].sub(b[0])];
const scale=(a:I,c:Q):I=>c.n<B(0)?[a[1].mul(c),a[0].mul(c)]:[a[0].mul(c),a[1].mul(c)];
const divide=(a:I,b:I):I=>{if(lt(a[0],Z)||!lt(Z,b[0]))throw Error('Positive division required');return[a[0].div(b[1]),a[1].div(b[0])];};
const wire=(r:I)=>({lower:r[0].text(),upper:r[1].text()});
function reconstruct(input:Readonly<BoundaryLayerInput>){
 const epsilon=Q.read(input.epsilon),a=Q.read(input.drift),force=Q.read(input.forcing),left=Q.read(input.left),right=Q.read(input.right),probe=Q.read(input.probe),h=O.div(integer(input.panels));
 const physical=Array.from({length:65},(_,i)=>new Q(B(i),B(64)));
 if(a.n===B(0)){
  const slope=right.sub(left),curvature=force.div(epsilon);
  const at=(x:Q)=>({x:x.text(),canonicalDistance:x.text(),stretchedCoordinate:null,exact:wire(pt(left.add(slope.mul(x)).add(curvature.mul(x).mul(x.sub(O)).div(integer(2))))),outer:null,inner:null,composite:null});
  return{input,branch:'zero-drift',precisionBits:192,layer:null,probe:{...at(probe),derivative:wire(pt(slope.add(curvature.mul(probe.sub(new Q(B(1),B(2))))))),curvature:wire(pt(curvature)),compositeDifference:null},physicalSamples:physical.map(at),layerSamples:[],grid:{spacing:h.text(),cellOverLayer:null,pointsPerEfold:null,fourPointsPerEfold:null,firstCellLayerFraction:null},analyticResidual:'0',contract:'Finite constant-coefficient BVP; zero drift uses the exact quadratic, not a boundary-layer formula.'};
 }
 const reverse=a.n<B(0),b=abs(a),near=reverse?right:left,far=reverse?left:right,k=force.div(b),delta=epsilon.div(b),t=b.div(epsilon),D=far.sub(near).sub(k),exponentials=new Map<string,I>();
 const decay=(y:Q):I=>{const z=t.mul(y),key=z.text(),old=exponentials.get(key);if(old)return old;const result=boundaryLayerIndependentDecay(z.n,z.d);exponentials.set(key,result);return result;};
 const q=decay(O),den=difference(pt(O),q);
 const ratio=(y:Q,e:I):I=>y.n===B(0)?pt(Z):y.text()==='1'?pt(O):divide(difference(pt(O),e),den);
 const at=(y:Q)=>{const e=decay(y),outer=far.add(k.mul(y.sub(O))),exact=y.n===B(0)?pt(near):y.text()==='1'?pt(far):sum(pt(near.add(k.mul(y))),scale(ratio(y,e),D));return{x:(reverse?O.sub(y):y).text(),canonicalDistance:y.text(),stretchedCoordinate:t.mul(y).text(),exact:wire(exact),outer:outer.text(),inner:wire(difference(pt(far.sub(k)),scale(e,D))),composite:wire(difference(pt(outer),scale(e,D)))};};
 const py=reverse?O.sub(probe):probe,pe=decay(py),gain=divide(pe,den),r=ratio(py,pe),amplitude=abs(D),window=lt(t,integer(8))?t:integer(8);
 return{input,branch:'nonzero-drift',precisionBits:192,layer:{side:reverse?'right':'left',delta:delta.text(),inverseDelta:t.text(),canonicalNear:near.text(),canonicalFar:far.text(),outerSlope:k.text(),mismatch:D.text(),hasLayerAmplitude:D.n!==B(0),farDecay:wire(q),normalizer:wire(den),outerUniformError:amplitude.text(),compositeUniformError:wire(scale(q,amplitude)),compositeDerivativeUniformError:wire(scale(divide(q,den),amplitude.mul(t))),compositeNearBoundaryResidual:'0',compositeFarBoundaryResidual:wire(scale(q,D.neg())),asymptoticallyThinAtCurrentInput:!lt(new Q(B(1),B(10)),delta),thinLabelThreshold:'1/10'},probe:{...at(py),derivative:wire(scale(sum(pt(k),scale(gain,D.mul(t))),reverse?O.neg():O)),curvature:wire(scale(gain,D.mul(t).mul(t).neg())),compositeDifference:wire(scale([q[0].mul(r[0]),q[1].mul(r[1])],D))},physicalSamples:physical.map(x=>at(reverse?O.sub(x):x)),layerSamples:Array.from({length:65},(_,i)=>at(window.mul(new Q(B(i),B(64))).div(t))),grid:{spacing:h.text(),cellOverLayer:t.mul(h).text(),pointsPerEfold:integer(input.panels).div(t).text(),fourPointsPerEfold:!lt(O,t.mul(h).mul(integer(4))),firstCellLayerFraction:wire(difference(pt(O),decay(h)))},analyticResidual:'0',contract:'Finite constant-coefficient BVP; exact analytic solution, matched composite and whole-domain certificates. Four points per e-fold and delta<=1/10 are display heuristics, not discretization error proofs.'};
}
function same(value:unknown,expected:unknown):boolean{
 if(expected===null||typeof expected!=='object')return value===expected;
 if(!value||typeof value!=='object'||Array.isArray(value)!==Array.isArray(expected))return false;
 if(Array.isArray(expected))return Array.isArray(value)&&value.length===expected.length&&expected.every((x,i)=>same(value[i],x));
 const a=value as Record<string,unknown>,b=expected as Record<string,unknown>,keys=Object.keys(b);return Object.keys(a).length===keys.length&&keys.every(k=>Object.hasOwn(a,k)&&same(a[k],b[k]));
}
const cache=new Map<string,unknown>();
export function validateBoundaryLayerResult(source:BoundaryLayerInput,value:unknown):BoundaryLayerResult|null{
 try{const input=readBoundaryLayerInput(source);if(!input)return null;const owned=snapshotQubitWire(value),key=JSON.stringify(input);let expected=cache.get(key);
  if(!expected){expected=snapshotQubitWire(reconstruct(input));if(cache.size===16)cache.delete(cache.keys().next().value!);cache.set(key,expected);}
  return same(owned,expected)?owned as BoundaryLayerResult:null;
 }catch{return null;}
}
