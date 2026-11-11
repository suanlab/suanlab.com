import type { MeasureResult } from './learning-measure';
import type { Vec2 } from './learning-simulations';
import { binaryEntropy, bernoulliChannelInformation } from './learning-information';

export const codingKinds = ['binary-channel-capacity', 'repetition-error-correction', 'hamming-linear-codes', 'lossless-compression-reconstruction', 'finite-description-length', 'kolmogorov-uncomputability', 'two-part-mdl-selection', 'budgeted-description-search', 'communication-fingerprints'] as const;
export type CodingKind = typeof codingKinds[number];
export const isCodingKind = (kind: string): kind is CodingKind =>
  (codingKinds as readonly string[]).includes(kind);

export function binaryEntropyBits(p: number): number {
  return binaryEntropy(p);
}

/** Joint rows are P(X=x,Y=y), with erasure output at column 2 for BEC. */
export function binaryChannel(inputOne: number, noise: number, erasure: boolean) {
  binaryEntropyBits(inputOne);
  binaryEntropyBits(noise);
  const joint = erasure
    ? [[(1-inputOne)*(1-noise), 0, (1-inputOne)*noise], [0, inputOne*(1-noise), inputOne*noise]]
    : [[(1-inputOne)*(1-noise), (1-inputOne)*noise], [inputOne*noise, inputOne*(1-noise)]];
  const output = joint[0].map((_, y) => joint[0][y]+joint[1][y]);
  const mutualInformation = erasure ? (1-noise)*binaryEntropyBits(inputOne)
    : bernoulliChannelInformation(inputOne,noise);
  const capacity = erasure ? 1-noise : bernoulliChannelInformation(.5,noise);
  return { joint, output, mutualInformation, capacity };
}

export function evaluateCoding(kind: CodingKind, values: number[]): MeasureResult {
  if (kind !== 'binary-channel-capacity') return evaluateCodingExtension(kind, values);
  if (values.length !== 3 ||
      !values.every(Number.isFinite) || ![0,1].includes(values[0])) {
    throw new Error('Use channel mode 0 BSC or 1 BEC and two probabilities.');
  }
  const [mode, p, noise] = values;
  const r = binaryChannel(p, noise, mode === 1);
  const underflow=r.mutualInformation===0&&p>0&&p<1&&(mode===1?noise<1:noise!==.5);
  const curve: Vec2[] = Array.from({length:101}, (_,i) => [i/100, binaryChannel(i/100,noise,mode===1).mutualInformation]);
  const rows = r.joint.flatMap((row,x) => row.map((v,y) => [x,y===2?'erasure':y,v,r.output[y]]));
  return {
    plot: 'continuous',
    warning: underflow?'numeric-range':undefined,
    statement: `${mode === 1 ? 'BEC' : 'BSC'}; independent stationary channel uses; P(X=1)=${p}; noise=${noise}. Capacity is asymptotic bits per channel use.`,
    stats: {'I(X;Y) (bits/use)':underflow?null:r.mutualInformation,'capacity (bits/use)':r.capacity,
      'input entropy (bits)':binaryEntropyBits(p),'gap to capacity':Math.max(0,r.capacity-r.mutualInformation),
      'output probabilities':r.output},
    curves:[curve,[[0,r.capacity],[1,r.capacity]]],curveColors:['blue','orange'],
    dots:underflow?[]:[[p,r.mutualInformation]],dotColors:['orange'],bounds:[0,1,0,1.05],
    table:{headers:['X','Y','P(X,Y)','P(Y)'],rows},
  };
}

const bitCount = (x: number) => {let n=0;for(;x;x>>>=1)n+=x&1;return n;};
const binary = (x:number,n:number) => x.toString(2).padStart(n,'0').slice(-n);
function integer(x:number,lo:number,hi:number) {if(!Number.isInteger(x)||x<lo||x>hi)throw new Error('Integer outside the stated finite domain.');}
function choose(n:number,k:number) {let c=1;for(let i=1;i<=k;i++)c=c*(n-i+1)/i;return c;}
export function repetitionError(n:number,noise:number) {
  integer(n,1,9);if(n%2===0)throw new Error('Use an odd block length.');binaryEntropyBits(noise);
  return Array.from({length:n+1},(_,k)=>choose(n,k)*noise**k*(1-noise)**(n-k)).slice((n+1)/2).reduce((a,b)=>a+b,0);
}
export function hammingEncode(data:number) {
  integer(data,0,15);let word=0;[3,5,6,7].forEach((pos,i)=>{word|=((data>>i)&1)<<(pos-1);});
  for(const check of [1,2,4]){let parity=0;for(let pos=1;pos<=7;pos++)if(pos&check)parity^=(word>>(pos-1))&1;word|=parity<<(check-1);}
  return word;
}
export function hammingDecode(received:number) {
  integer(received,0,127);let syndrome=0;for(let pos=1;pos<=7;pos++)if((received>>(pos-1))&1)syndrome^=pos;
  const corrected=received^(syndrome?1<<(syndrome-1):0);
  const data=[3,5,6,7].reduce((v,pos,i)=>v+((corrected>>(pos-1))&1)*2**i,0);
  return {syndrome,corrected,data};
}
export function encodeRuns(word:string) {
  const n=word.length;if(![8,16].includes(n)||!/^[01]+$/.test(word))throw new Error('Use an eight- or sixteen-bit word.');
  const runs=word.match(/0+|1+/g)!;const width=Math.log2(n);
  const rle='1'+word[0]+runs.map(run=>binary(run.length-1,width)).join('');const raw='0'+word;
  return {raw,rle,encoded:rle.length<raw.length?rle:raw,runs:runs.map(run=>({bit:Number(run[0]),length:run.length,code:binary(run.length-1,width)}))};
}
export function decodeRuns(encoded:string,n:number) {
  if(![8,16].includes(n)||!/^[01]+$/.test(encoded))throw new Error('Invalid code or external word length.');
  if(encoded[0]==='0'){if(encoded.length!==n+1)throw new Error('Invalid raw length.');return encoded.slice(1);}
  const width=Math.log2(n);if(encoded.length<width+2||(encoded.length-2)%width)throw new Error('Incomplete run.');
  let bit=encoded[1],word='';for(let i=2;i<encoded.length;i+=width){word+=bit.repeat(parseInt(encoded.slice(i,i+width),2)+1);bit=bit==='0'?'1':'0';if(word.length>n)throw new Error('Run exceeds external word length.');}
  if(word.length!==n)throw new Error('Runs do not cover the word.');return word;
}
export interface FiniteProgram {code:string;output:number;family:number;}
/** A finite prefix decoder, deliberately not a universal machine. */
export function finitePrograms(level=3):FiniteProgram[] {
  integer(level,0,3);const out:FiniteProgram[]=Array.from({length:256},(_,x)=>({code:'0'+binary(x,8),output:x,family:0}));
  if(level>=1)for(let x=0;x<2;x++)out.push({code:'10'+x,output:x?255:0,family:1});
  if(level>=2)for(let x=0;x<4;x++)out.push({code:'110'+binary(x,2),output:parseInt(binary(x,2).repeat(4),2),family:2});
  if(level>=3)for(let x=0;x<16;x++)out.push({code:'1110'+binary(x,4),output:parseInt(binary(x,4).repeat(2),2),family:3});
  return out;
}
export function finiteDecode(code:string):number {
  const p=finitePrograms().find(p=>p.code===code);if(!p)throw new Error('Invalid finite program.');return p.output;
}
export function finiteSearch(target:number,budget:number,literalFirst:boolean) {
  integer(target,0,255);integer(budget,1,278);const programs=finitePrograms();if(!literalFirst)programs.sort((a,b)=>a.code.length-b.code.length||a.code.localeCompare(b.code));
  let best:FiniteProgram|null=null;const history:{checked:number;best:number|null}[]=[];
  programs.slice(0,budget).forEach((p,i)=>{if(p.output===target&&(!best||p.code.length<best.code.length))best=p;history.push({checked:i+1,best:best?best.code.length:null});});
  const oracle=Math.min(...programs.filter(p=>p.output===target).map(p=>p.code.length));
  return {best:best as FiniteProgram|null,history,oracle};
}
export function twoPartMDL(successes:number,repeats:number) {
  integer(successes,0,16);integer(repeats,1,16);const N=16*repeats,K=successes*repeats;
  const models=[{header:'0',p:.5},{header:'10',p:.25},{header:'110',p:.5},{header:'111',p:.75}].map(m=>{const data=-K*Math.log2(m.p)-(N-K)*Math.log2(1-m.p);return {...m,data,total:m.header.length+data};});
  return {N,K,models,winner:models.reduce((best,m,i)=>m.total<models[best].total?i:best,0)};
}
export function fingerprintRows(k:number,seed:number) {
  integer(k,0,8);integer(seed,1,16);let state=seed;return Array.from({length:k},()=>{state=(73*state+41)%256;return state;});
}
export function fingerprint(value:number,rows:number[]) {integer(value,0,255);return rows.reduce((word,row,i)=>{integer(row,0,255);return word+(bitCount(row&value)%2)*2**i;},0);}

function evaluateCodingExtension(kind:Exclude<CodingKind,'binary-channel-capacity'>,v:number[]):MeasureResult {
  const domains:Record<Exclude<CodingKind,'binary-channel-capacity'>,[number,number,boolean][]>={
    'repetition-error-correction':[[0,4,true],[0,1,false],[0,1,true],[0,511,true]],
    'hamming-linear-codes':[[0,15,true],[0,127,true]],
    'lossless-compression-reconstruction':[[0,1,true],[0,65535,true]],
    'finite-description-length':[[0,255,true],[0,3,true]],
    'kolmogorov-uncomputability':[[0,9,true],[0,255,true]],
    'two-part-mdl-selection':[[0,16,true],[1,16,true]],
    'budgeted-description-search':[[0,255,true],[1,278,true],[0,1,true]],
    'communication-fingerprints':[[0,255,true],[0,255,true],[0,8,true],[0,1,true],[1,16,true]],
  };
  const domain=domains[kind];if(v.length!==domain.length||v.some((x,i)=>!Number.isFinite(x)||x<domain[i][0]||x>domain[i][1]||(domain[i][2]&&!Number.isInteger(x))))throw new Error('Use the stated finite coding example domain.');
  const indexed=(values:number[]):Vec2[]=>values.map((y,x)=>[x,y]);const [a,b,c,d,e]=v;
  if(kind==='repetition-error-correction') {
    const n=2*a+1,received=d&((1<<n)-1),errors=bitCount(received^(c?((1<<n)-1):0)),decoded=Number(bitCount(received)>n/2),mass=Array.from({length:n+1},(_,k)=>choose(n,k)*b**k*(1-b)**(n-k));
    const blockError=repetitionError(n,b),wordProbability=b**errors*(1-b)**(n-errors),blockUnderflow=blockError===0&&b>0,wordUnderflow=wordProbability===0&&b>0&&b<1;
    return {plot:'discrete',warning:blockUnderflow||wordUnderflow?'numeric-range':undefined,statement:`Odd repetition n=${n}; independent BSC flips; majority decoder, including when noise > 1/2. Received mask uses only the lowest n bits.`,stats:{'block length n':n,'rate (data bits/use)':1/n,'minimum distance':n,'guaranteed corrected flips':(n-1)/2,'introduced errors':errors,'decoded bit':decoded,'1{recovered payload}':Number(decoded===c),'majority block error probability':blockUnderflow?null:blockError,'selected received-word probability':wordUnderflow?null:wordProbability},curves:[indexed(mass)],dots:mass.flatMap((y,k)=>k>n/2?[[k,y] as Vec2]:[]),dotColors:mass.slice((n+1)/2).map(()=>'orange'),bounds:[0,n,-.02,1.05],table:{headers:['position','sent','received','error'],rows:Array.from({length:n},(_,i)=>[i+1,c,(received>>i)&1,((received>>i)&1)^c])}};
  }
  if(kind==='hamming-linear-codes') {
    const sent=hammingEncode(a),received=sent^b,r=hammingDecode(received);
    return {plot:'discrete',statement:'GF(2) Hamming(7,4); parity positions 1,2,4; data 3,5,6,7; H column j is binary(j). Nonzero syndrome triggers one flip.',stats:{'sent payload':a,'introduced errors':bitCount(b),'syndrome as position':r.syndrome,'decoded payload':r.data,'1{payload recovered}':Number(r.data===a),'post-correction syndrome':hammingDecode(r.corrected).syndrome,'rate':4/7,'minimum distance':3},curves:[Array.from({length:7},(_,i)=>[i+1,(sent>>i)&1] as Vec2),Array.from({length:7},(_,i)=>[i+1,(received>>i)&1] as Vec2)],curveColors:['blue','orange'],bounds:[0,7,-.1,1.1],table:{headers:['position','H column (top to bottom)','role','sent','received','corrected'],rows:Array.from({length:7},(_,i)=>[i+1,binary(i+1,3).split('').reverse().join(''),[1,2,4].includes(i+1)?'parity':'data',(sent>>i)&1,(received>>i)&1,(r.corrected>>i)&1])}};
  }
  if(kind==='lossless-compression-reconstruction') {
    const n=a?16:8,word=binary(b,n),r=encodeRuns(word),restored=decodeRuns(r.encoded,n);
    return {plot:'discrete',statement:`External length N=${n}; target=${word}; stream=${r.encoded}; restored=${restored}. Header 0 raw, 1 RLE. RLE first bit + fixed-width (length−1) tokens.`,stats:{'raw protocol bits':r.raw.length,'run protocol bits':r.rle.length,'chosen protocol bits':r.encoded.length,'number of runs':r.runs.length,'1{exact reconstruction}':Number(restored===word),'1{RLE selected}':Number(r.encoded[0]==='1')},curves:[indexed([r.raw.length,r.rle.length,r.encoded.length])],table:{headers:['run index','bit','length','encoded length minus one'],rows:r.runs.map((run,i)=>[i,run.bit,run.length,run.code])}};
  }
  if(kind==='finite-description-length') {
    const candidates=finitePrograms(b).filter(p=>p.output===a),best=Math.min(...candidates.map(p=>p.code.length));
    return {plot:'discrete',statement:`Eight-bit target ${binary(a,8)}. Fixed finite prefix decoder, not universal Kolmogorov complexity; family level=${b}. Decoder is shared and is not charged to these finite code lengths.`,stats:{'shortest finite description bits':best,'literal description bits':9,'saving vs literal':9-best,'available target programs':candidates.length,'Kraft sum of enabled programs':finitePrograms(b).reduce((s,p)=>s+2**(-p.code.length),0)},curves:[indexed(candidates.map(p=>p.code.length))],table:{headers:['family (0 literal,1 uniform,2 period2,3 period4)','program','bits','decoded byte'],rows:candidates.map(p=>[p.family,p.code,p.code.length,p.output])}};
  }
  if(kind==='kolmogorov-uncomputability') {
    const programs=finitePrograms(),allowed=programs.filter(p=>p.code.length<=a),covered=new Set(allowed.map(p=>p.output)),shortest=Math.min(...programs.filter(p=>p.output===b).map(p=>p.code.length));
    return {plot:'discrete',statement:'Finite decoder coverage only. No universal K oracle is implemented. Universal noncomputability is proved in the text, not inferred from this bounded enumeration.',stats:{'program length budget L':a,'programs within budget':allowed.length,'distinct eight-bit outputs':covered.size,'uncovered eight-bit outputs':256-covered.size,'prefix counting upper bound':Math.min(256,2**a),'selected finite minimum bits':shortest,'1{selected byte covered}':Number(covered.has(b))},curves:[Array.from({length:10},(_,L)=>[L,new Set(programs.filter(p=>p.code.length<=L).map(p=>p.output)).size] as Vec2),Array.from({length:10},(_,L)=>[L,Math.min(256,2**L)] as Vec2)],curveColors:['blue','orange'],table:{headers:['program length','program count','distinct outputs up to length'],rows:Array.from({length:10},(_,L)=>[L,programs.filter(p=>p.code.length===L).length,new Set(programs.filter(p=>p.code.length<=L).map(p=>p.output)).size])}};
  }
  if(kind==='two-part-mdl-selection') {
    const r=twoPartMDL(a,b),p=a/16,mleData=p===0||p===1?0:-r.K*Math.log2(p)-(r.N-r.K)*Math.log2(1-p);
    return {plot:'discrete',statement:`Ordered Bernoulli data N=${r.N}, K=${r.K}. Four predeclared models; prefix headers 0,10,110,111. Data lengths −log2 P are ideal fractional lengths, not an implemented binary codec.`,stats:{'observations N':r.N,'ones K':r.K,'selected model index':r.winner,'selected total ideal bits':r.models[r.winner].total,'MLE data-only ideal bits':mleData,'1{MLE includes parameter cost}':0},curves:[indexed(r.models.map(m=>m.total)),indexed(r.models.map(m=>m.data))],curveColors:['blue','orange'],table:{headers:['model','p','header','header bits','data ideal bits','total ideal bits'],rows:r.models.map((m,i)=>[i,m.p,m.header,m.header.length,m.data,m.total])}};
  }
  if(kind==='budgeted-description-search') {
    const r=finiteSearch(a,b,c===1),points=r.history.filter(x=>x.best!==null).map(x=>[x.checked,x.best!] as Vec2),milestones=r.history.filter((x,i,h)=>i===0||i===h.length-1||x.best!==h[i-1].best);
    return {plot:'continuous',statement:`Fixed finite decoder; target=${binary(a,8)}; ${c?'literal-first':'shortest-first'} enumeration; checked=${b}/278. Not found means unknown within this budget. Full-family oracle is finite and computable.`,stats:{'programs evaluated':b,'best found upper bound bits':r.best?r.best.code.length:null,'finite-family oracle bits':r.oracle,'1{target found}':Number(r.best!==null),'gap to finite oracle':r.best?r.best.code.length-r.oracle:null},curves:[points,[[1,r.oracle],[278,r.oracle]]],curveColors:['blue','orange'],dots:r.best?[[b,r.best.code.length]]:[],dotColors:['blue'],bounds:[0,278,0,10],table:{headers:['programs checked','best discovered length'],rows:milestones.map(x=>[x.checked,x.best??'unknown'])}};
  }
  // Equality of distributed eight-bit representations. Fixed seeds illustrate one
  // deterministic transcript; the probability statement is for ideal public coins.
  const rows=d===0?Array.from({length:c},(_,i)=>1<<i):fingerprintRows(c,e),ha=fingerprint(a,rows),hb=fingerprint(b,rows),same=a===b;
  return {plot:'discrete',statement:`One-way equality of eight-bit words; k=${c}; ${d?'fixed seeded GF(2) fingerprint':'lowest-bit truncation'}. Collision bound 2^−k assumes independent uniform public rows; it is not a guarantee for this fixed seed.`,stats:{'bits communicated k':c,'Alice fingerprint':ha,'Bob fingerprint':hb,'1{protocol declares equal}':Number(ha===hb),'1{actual equality}':Number(same),'1{false positive in displayed transcript}':Number(!same&&ha===hb),'ideal random false-positive probability':same?0:2**(-c),'deterministic exact one-way lower bound bits':8},curves:[Array.from({length:9},(_,k)=>[k,same?0:2**(-k)] as Vec2),Array.from({length:9},(_,k)=>[k,k<8?1:0] as Vec2)],curveColors:['blue','orange'],bounds:[0,8,-.02,1.05],table:{headers:['row','public mask','Alice parity','Bob parity'],rows:rows.map((row,i)=>[i,binary(row,8),bitCount(row&a)%2,bitCount(row&b)%2])}};
}
