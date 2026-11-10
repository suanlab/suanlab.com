/** Ideal BB84 intercept/resend enumeration, not a key-security proof. */
import { f, zero, one, add, sub, mul, div, le, q, sum, type F } from './learning-quantum-hilbert-core';
type Basis='z'|'x';type Matrix=F[][];
const half=f(BigInt(1),BigInt(2));
const projector=(basis:Basis,bit:number):Matrix=>basis==='z'?[[bit?zero:one,zero],[zero,bit?one:zero]]
  :[[half,bit?sub(zero,half):half],[bit?sub(zero,half):half,half]];
const traceProduct=(a:Matrix,b:Matrix)=>sum(a.flatMap((row,i)=>row.map((x,j)=>mul(x,b[j][i]))));
function probability(x:F){
  if(!x||typeof x.n!=='bigint'||typeof x.d!=='bigint'||x.d<=BigInt(0)||x.n.toString().replace('-','').length>32||x.d.toString().length>32||!le(zero,x)||!le(x,one))throw new RangeError('Probability in[0,1] with at most32 rational digits required');
}
export function idealBb84InterceptResend(attackFraction:F,eveZProbability:F=half,sampleSize=20){
  probability(attackFraction);probability(eveZProbability);
  if(!Number.isSafeInteger(sampleSize)||sampleSize<0||sampleSize>256)throw new RangeError('Use a test sample between0 and256 sifted positions');
  const rows:{aliceBasis:Basis;aliceBit:number;bobBasis:Basis;bobBit:number;attacked:boolean;eveBasis:Basis|null;eveBit:number|null;weight:F}[]=[];
  for(const aliceBasis of ['z','x'] as const)for(const aliceBit of [0,1])for(const bobBasis of ['z','x'] as const){
    const input=projector(aliceBasis,aliceBit),initial=f(BigInt(1),BigInt(8));
    for(const bobBit of [0,1])rows.push({aliceBasis,aliceBit,bobBasis,bobBit,attacked:false,eveBasis:null,eveBit:null,
      weight:mul(mul(initial,sub(one,attackFraction)),traceProduct(projector(bobBasis,bobBit),input))});
    for(const eveBasis of ['z','x'] as const)for(const eveBit of [0,1])for(const bobBit of [0,1]){
      const resent=projector(eveBasis,eveBit),evePrior=eveBasis==='z'?eveZProbability:sub(one,eveZProbability);
      const weight=mul(mul(mul(mul(initial,attackFraction),evePrior),traceProduct(resent,input)),traceProduct(projector(bobBasis,bobBit),resent));
      rows.push({aliceBasis,aliceBit,bobBasis,bobBit,attacked:true,eveBasis,eveBit,weight});
    }
  }
  const sifted=rows.filter(r=>r.aliceBasis===r.bobBasis),mass=sum(sifted.map(r=>r.weight)),error=sum(sifted.filter(r=>r.aliceBit!==r.bobBit).map(r=>r.weight));
  const known=sum(sifted.filter(r=>r.attacked&&r.eveBasis===r.aliceBasis).map(r=>r.weight)),knownFraction=div(known,mass),qber=div(error,mass);
  const basisQber=(basis:Basis)=>{const selected=sifted.filter(r=>r.aliceBasis===basis);return div(sum(selected.filter(r=>r.aliceBit!==r.bobBit).map(r=>r.weight)),sum(selected.map(r=>r.weight)));};
  const noError=f(sub(one,qber).n**BigInt(sampleSize),sub(one,qber).d**BigInt(sampleSize));
  return {attackFraction:q(attackFraction),eveZProbability:q(eveZProbability),totalProbability:q(sum(rows.map(r=>r.weight))),siftProbability:q(mass),siftedQber:q(qber),basisQber:{z:q(basisQber('z')),x:q(basisQber('x'))},
    eveKnownSiftedFraction:q(knownFraction),eveIdealInformationBits:q(knownFraction),eveGuessSuccess:q(add(half,mul(half,knownFraction))),
    sampleSize,probabilityAtLeastOneSampleError:q(sub(one,noError)),rows:rows.map(r=>({...r,weight:q(r.weight)})),
    assumptions:['Uniform independent Alice bits and Z/X bases; uniform independent Bob bases','Independent attack decisions and Eve basis choices; projective intercept/resend; ideal noiseless single-photon qubits','After sifting Eve knows public bases and her attack flags/outcomes; wrong-basis observations reveal no Alice-bit information','Authenticated public classical discussion is required in real QKD; error correction, privacy amplification and finite-key security are not modeled'],
    contract:'Exact finite enumeration and IID sample-error probability for this attack only. Eve information is the known-bit/unknown-bit flagged model after basis announcement. A zero-error sample does not certify secrecy. No modern QKD security threshold, key rate, implementation loophole analysis, Worker, public input reader or UI.'};
}
