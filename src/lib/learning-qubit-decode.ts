/** Heavy scalar decoding shared by Workers; no product model value imports. */
import type { QubitInput,QubitWireScalar,QubitWireMatrix,QubitWireTerm,QubitWireMap } from './learning-qubit-input';
import type { QubitMap,QubitInstrument } from './learning-qubit-channel';
import { f,complex,type F,type C,type M } from './learning-qubit-channel-algebra';
export function decodeQubitFraction(s:string):F {const [n,d='1']=s.split('/');return f(BigInt(n),BigInt(d));}
const scalar=(s:QubitWireScalar):C=>complex([decodeQubitFraction(s.real[0]),decodeQubitFraction(s.real[1])],[decodeQubitFraction(s.imaginary[0]),decodeQubitFraction(s.imaginary[1])]);
const matrix=(m:QubitWireMatrix):M=>m.map(row=>row.map(scalar));
const terms=(xs:readonly QubitWireTerm[])=>xs.map(t=>({weight:decodeQubitFraction(t.weight),operator:matrix(t.operator)}));
function map(m:QubitWireMap):QubitMap {
  if(m.kind==='transpose')return {kind:'transpose'};
  if(m.kind==='operator-sum')return {kind:'operator-sum',terms:terms(m.terms)};
  return {kind:'matrix-units',images:m.images.map(matrix) as [M,M,M,M]};
}
export function decodeQubitInput(input:QubitInput){
  switch(input.kind){
    case 'circuit':return {kind:input.kind,sites:input.sites,gates:input.gates,input:input.input.map(scalar)};
    case 'channel':return {kind:input.kind,map:map(input.map),state:matrix(input.state)};
    case 'environment':return {kind:input.kind,parameter:decodeQubitFraction(input.parameter),state:matrix(input.state)};
    case 'instrument':return {kind:input.kind,instrument:{outcomes:input.outcomes.map(terms) as QubitInstrument['outcomes']},state:matrix(input.state),correctZ:input.correctZ};
    case 'deferred':return {kind:input.kind,state:matrix(input.state)};
  }
}
