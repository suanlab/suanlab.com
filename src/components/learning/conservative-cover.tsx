import { nonuniformVolumeSimulation } from '@/lib/learning-continuum-volume-discretization';
import { burgersRiemannSimulation } from '@/lib/learning-continuum-godunov';
import type { ContinuumConservativeKind } from '@/lib/learning-continuum-conservative-lab';
export default function ConservativeCover({kind}:{kind:ContinuumConservativeKind}) {
  const volume=kind==='nonuniform-volume-shared-flux-mass-norm';
  const rows=volume?nonuniformVolumeSimulation({cellCountExponent:3,length:2,area:1,meshWarp:.6,velocity:.5,diffusivity:.1,meanConcentration:1,relativeContrast:.8,mode:1,phaseCycles:.125,timeStep:.01,steps:8}).cells.map(c=>({left:c.left,right:c.right,q:c.concentration.value!,other:c.wrongConcentration.value!}))
    :burgersRiemannSimulation({cellCountExponent:3,length:2,leftState:-1,rightState:1,interfaceFraction:.5,timeStep:.01,steps:12}).godunov.cells.map(c=>({left:c.left,right:c.right,q:c.value.value!,other:c.initial}));
  const peak=Math.max(...rows.flatMap(c=>[Math.abs(c.q),Math.abs(c.other)])),x=(v:number)=>45+135*v,y=(v:number)=>volume?210-150*v/peak:130-90*v/peak;
  return <svg viewBox="0 0 360 240" aria-hidden="true" className="ai-book-art"><path d={`M45 30V225M45 ${y(0)}H320`} fill="none" stroke="var(--learning-neutral)"/>{rows.map((c,i)=><g key={i}><rect x={x(c.left)} y={Math.min(y(0),y(c.q))} width={x(c.right)-x(c.left)} height={Math.abs(y(0)-y(c.q))} fill="var(--learning-blue)" fillOpacity=".2" stroke="var(--learning-blue)"/><path d={`M${x(c.left)} ${y(c.other)}H${x(c.right)}`} fill="none" stroke="var(--learning-orange)" strokeWidth="2.5" strokeDasharray="4 2"/></g>)}<text x="45" y="24" fill="currentColor" fontSize="14">{volume?'Vᵢ q̇ᵢ = Fleft − Fright':'Godunov · sonic rarefaction'}</text></svg>;
}
