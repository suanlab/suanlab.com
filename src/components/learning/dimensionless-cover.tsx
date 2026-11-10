import { pecletAt, damkohlerAt } from '@/lib/learning-continuum-dimensionless';
import type { ContinuumDimensionlessKind } from '@/lib/learning-continuum-dimensionless-lab';
export default function DimensionlessCover({kind}:{kind:ContinuumDimensionlessKind}) {
  const pe=kind==='advection-diffusion-peclet-boundary-layer';
  const paths=[0,1,2].map(index=>Array.from({length:101},(_,i)=>{
    const xi=i/100;
    const y=pe?pecletAt({velocity:[0,.2,1][index],diffusivity:.1,length:1,area:1,leftConcentration:0,rightConcentration:1},xi).basisShape.value:
      damkohlerAt({diffusivity:.25,length:1,area:1,reactionRate:[0,1,16][index],surfaceConcentration:1},xi).basisShape.value;
    return `${i?'L':'M'}${45+270*xi} ${190-145*(y??0)}`;
  }).join(' '));
  return <svg viewBox="0 0 360 240" aria-hidden="true" className="ai-book-art">
    <path d="M45 35V190H320" fill="none" stroke="var(--learning-neutral)"/>
    {paths.map((d,i)=><path key={i} d={d} fill="none" stroke={`var(--learning-${['blue','orange','green'][i]})`} strokeWidth="3" strokeDasharray={i?'5 3':undefined}/>)}
    <g fill="currentColor" fontSize="14"><text x="45" y="24">{pe?'Pe · advection / diffusion':'Da = φ² · reaction / diffusion'}</text><text x="45" y="218">{pe?'θ(ξ) · boundary layer':'C(ξ) · sealed at ξ=1'}</text><text x="315" y="208">ξ</text></g>
  </svg>;
}
