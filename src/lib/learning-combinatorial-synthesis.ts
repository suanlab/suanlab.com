export interface SynthesisProgram {id:number;coefficients:[number,number,number];cost:number;}
export interface SynthesisRound {round:number;examples:number[];viableIds:number[];selectedId:number|null;counterexample:number|null;status:'counterexample'|'verified-finite'|'grammar-empty';}
const integer=(v:number,min:number,max:number)=>{if(!Number.isInteger(v)||v<min||v>max)throw new Error('Use the stated integer specification, grammar, verifier-domain, seed, round and probe controls.');};
export const synthesisSpecLabels=['2x + 1','x²','abs(x)','3x + 1'] as const;
export function synthesisSpecification(index:number,x:number){integer(index,0,3);if(!Number.isInteger(x)||Math.abs(x)>3)throw new Error('Use an integer input from-3 to3.');return index===0?2*x+1:index===1?x*x:index===2?Math.abs(x):3*x+1;}
export const synthesisProgramValue=(p:SynthesisProgram,x:number)=>p.coefficients[0]*x*x+p.coefficients[1]*x+p.coefficients[2];
export function synthesisProgramLabel(p:SynthesisProgram){const [a,b,c]=p.coefficients,parts:[number,string][]=[[a,'x²'],[b,'x'],[c,'']];let text='';for(const [v,suffix] of parts){if(v===0)continue;const magnitude=suffix&&Math.abs(v)===1?'':String(Math.abs(v));text+=(text?(v<0?' − ':' + '):(v<0?'−':''))+magnitude+suffix;}return text||'0';}
export function synthesisGrammar(bound:number){integer(bound,1,3);const raw:Omit<SynthesisProgram,'id'>[]=[];for(let a=-bound;a<=bound;a++)for(let b=-bound;b<=bound;b++)for(let c=-bound;c<=bound;c++)raw.push({coefficients:[a,b,c],cost:Math.abs(a)+Math.abs(b)+Math.abs(c)});raw.sort((a,b)=>a.cost-b.cost||a.coefficients[0]-b.coefficients[0]||a.coefficients[1]-b.coefficients[1]||a.coefficients[2]-b.coefficients[2]);return raw.map((p,id)=>({id,...p}));}
export function counterexampleSynthesis(specIndex:number,bound:number,radius:number,seedMask:number,requestedRound:number,probe:number){
 integer(specIndex,0,3);integer(bound,1,3);integer(radius,0,2);integer(seedMask,0,31);integer(requestedRound,0,5);integer(probe,-3,3);
 const grammar=synthesisGrammar(bound),domain=Array.from({length:2*radius+1},(_,i)=>i-radius),requestedSeeds=[-2,-1,0,1,2].filter(x=>(seedMask>>(x+2))&1),ignoredSeeds=requestedSeeds.filter(x=>!domain.includes(x)),initialExamples=requestedSeeds.filter(x=>domain.includes(x)),rounds:SynthesisRound[]=[];let examples=initialExamples.slice();
 while(true){
  const viable=grammar.filter(p=>examples.every(x=>synthesisProgramValue(p,x)===synthesisSpecification(specIndex,x))),selected=viable[0]??null,counterexample=selected?(domain.find(x=>synthesisProgramValue(selected,x)!==synthesisSpecification(specIndex,x))??null):null,status:SynthesisRound['status']=selected===null?'grammar-empty':counterexample===null?'verified-finite':'counterexample';
  rounds.push({round:rounds.length,examples:examples.slice(),viableIds:viable.map(p=>p.id),selectedId:selected?.id??null,counterexample,status});
  if(status!=='counterexample')break;
  examples=[...examples,counterexample!].sort((a,b)=>a-b);
 }
 const actualRound=Math.min(requestedRound,rounds.length-1),current=rounds[actualRound],program=current.selectedId===null?null:grammar[current.selectedId],probeExpected=synthesisSpecification(specIndex,probe),probeValue=program?synthesisProgramValue(program,probe):null,sliceC=program?.coefficients[2]??0,slice=grammar.filter(p=>p.coefficients[2]===sliceC);
 return {specIndex,bound,radius,seedMask,requestedRound,probe,grammar,domain,requestedSeeds,ignoredSeeds,initialExamples,rounds,actualRound,current,program,sliceC,slice,probeInside:domain.includes(probe),probeExpected,probeValue,probeMatch:probeValue===null?null:probeValue===probeExpected};
}
