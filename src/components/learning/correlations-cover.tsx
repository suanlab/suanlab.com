import React from 'react';
import type {CorrelationKind} from '@/lib/learning-correlations';

/** Decorative fixed examples; numerical experiments run only in their worker. */
export default function CorrelationsCover({kind}:{kind:CorrelationKind}) {
 const ring=kind==='periodic-ising-ring-correlation-length';
 const average=kind==='finite-correlated-average-ar1-uncertainty';
 const blue='var(--learning-blue)',orange='var(--learning-orange)';
 const sites=Array.from({length:8},(_,i)=>({x:180+78*Math.cos(i*Math.PI/4),y:125+78*Math.sin(i*Math.PI/4)}));
 return <svg viewBox="0 0 360 240" aria-hidden="true" className="ai-book-art" data-correlation-cover={kind}>
  <text x="180" y="27" textAnchor="middle" fontSize="14" fill="currentColor">{ring?'C(r) = C(N−r)':average?'Var(mean) = Σᵢⱼ Cov(Xᵢ,Xⱼ)/n²':'Cᵢⱼ = E[sᵢsⱼ] − E[sᵢ]E[sⱼ]'}</text>
  {ring?<>{sites.map((p,i)=><line key={`edge-${i}`} x1={p.x} y1={p.y} x2={sites[(i+1)%8].x} y2={sites[(i+1)%8].y} stroke={i<2?orange:blue} strokeWidth="4"/>)}{sites.map((p,i)=><circle key={i} cx={p.x} cy={p.y} r={i===0||i===2?8:5} fill={i===0||i===2?orange:blue}/>)}<text x="180" y="130" textAnchor="middle" fontSize="14" fill="currentColor">r = 2 · N−r = 6</text></>:average?<>{Array.from({length:4},(_,i)=>Array.from({length:4},(_,j)=><rect key={`${i}-${j}`} x={100+j*40} y={46+i*40} width="36" height="36" fill={blue} opacity={.7*2**(-Math.abs(i-j))}/>))}<text x="180" y="232" textAnchor="middle" fontSize="12" fill="currentColor">n = 4 · ρ = 1/2 · V = 33/64</text></>:<>{[[1,-.5],[-.5,1]].map((row,i)=>row.map((v,j)=><g key={`${i}-${j}`}><rect x={100+80*j} y={49+80*i} width="76" height="76" fill={v<0?orange:blue} opacity="0.25"/><text x={138+80*j} y={92+80*i} textAnchor="middle" fontSize="21" fill="currentColor">{v}</text></g>))}<text x="180" y="232" textAnchor="middle" fontSize="12" fill="currentColor">Var(s₁+s₂) = 1 · Var(s₁−s₂) = 3</text></>}
 </svg>;
}
