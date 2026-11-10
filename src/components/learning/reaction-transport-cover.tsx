import{reversibleReactionAt,fisherFrontAt}from '@/lib/learning-continuum-reaction-transport';
import type{ContinuumReactionKind}from '@/lib/learning-continuum-reaction-transport-lab';
export default function ReactionTransportCover({kind}:{kind:ContinuumReactionKind}){
 const reaction=kind==='reversible-reaction-advection-diffusion-budget';
 const data=Array.from({length:161},(_,i)=>{const x=-4+i/20;if(reaction){const q=reversibleReactionAt({mass:2,area:1,initialSigma:.5,initialMean:0,velocity:.4,diffusivity:.1,forwardRate:1,reverseRate:.5,initialFractionA:.8},x,.7);return[x,q.a.concentration??0,q.b.concentration??0,q.total.concentration??0];}const q=fisherFrontAt({diffusivity:1/6,growthRate:1,carryingConcentration:1,bulkVelocity:0,initialFront:0,area:1},x,0);return[x,q.concentration??0,q.source??0];});
 const max=Math.max(...data.flatMap(v=>v.slice(1))),path=(index:number)=>data.map((p,i)=>(i===0?'M':'L')+(40+(p[0]+4)*35).toFixed(3)+' '+(195-145*p[index]/max).toFixed(3)).join(' ');
 return <svg viewBox="0 0 360 240" aria-hidden="true" className="ai-book-art"><path d="M40 35V195H320" fill="none" stroke="var(--learning-neutral)"/>{[1,2,...(reaction?[3]:[])].map((index,i)=><path key={index} d={path(index)} fill="none" stroke={['var(--learning-blue)','var(--learning-orange)','var(--learning-green)'][i]} strokeWidth="3"/>)}<g fill="currentColor" fontSize="14"><text x="43" y="25">{reaction?'A ⇄ B · A+B':'Fisher–KPP · c, S'}</text><text x="180" y="222" textAnchor="middle">x</text></g></svg>;
}
