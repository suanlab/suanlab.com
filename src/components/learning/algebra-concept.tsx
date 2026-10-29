import { useId } from 'react';
import { leastSquaresState, projectionState, symmetricEigen2, applySymmetric2 } from '@/lib/learning-linear-algebra';
import type { Vec2 } from '@/lib/learning-simulations';
import type { AlgebraKind } from './algebra-model';
export default function AlgebraConcept({ kind, step, label }: { kind: AlgebraKind; step: number; label: string }) {
  const id = useId();
  const origin: Vec2 = kind === 'least-squares' ? [45, 225] : [170, 150]; const scale = kind === 'least-squares' ? 65 : 42;
  const xy = (p: Vec2): Vec2 => [origin[0] + p[0] * scale, origin[1] - p[1] * scale];
  const arrow = (from: Vec2, to: Vec2, role: string) => <path d={`M${xy(from)}L${xy(to)}`} stroke={`var(--learning-${role})`} strokeWidth="3" fill="none" markerEnd={`url(#${id}-${role})`} />;
  const labelAt = (p: Vec2, text: string, role = 'neutral') => <text x={xy(p)[0]} y={xy(p)[1]} fontSize="12" fill={`var(--learning-${role})`}>{text}</text>;
  const proj = projectionState([3, 2, 0], [2, 0, 0]); const ls = leastSquaresState([1, 2, 2], step === 2 ? 11 / 14 : .5); const eigen = symmetricEigen2(2, 1, 2);
  const first: Vec2 = step === 0 ? [1, 0] : eigen.first; const mapped = applySymmetric2(2, 1, 2, first);
  return <svg className="ai-concept-diagram" viewBox="0 0 400 280" role="img" aria-label={label}>
    <defs>{['blue', 'orange', 'green'].map(role => <marker key={role} id={`${id}-${role}`} markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0L7 3.5L0 7Z" fill={`var(--learning-${role})`} /></marker>)}</defs>
    {Array.from({ length: 8 }, (_, i) => i - 3).map(i => <path key={i} d={`M${xy([-4, i])}L${xy([5, i])}M${xy([i, -4])}L${xy([i, 5])}`} stroke="var(--learning-grid)" strokeOpacity=".2" fill="none" />)}
    <path d={`M15 ${origin[1]}H385M${origin[0]} 15V265`} fill="none" stroke="var(--learning-neutral)" />
    {labelAt(kind === 'least-squares' ? [3.5, -.3] : [4.5, -.4], 'x')}{labelAt(kind === 'least-squares' ? [.15, 3] : [.15, 2.9], 'y')}
    {kind === 'inner-products' && <>
      {arrow([0, 0], [3, 2], 'blue')}{arrow([0, 0], [2, 0], 'orange')}{labelAt([2.5, 2.25], 'u=(3,2)', 'blue')}{labelAt([1, -.5], 'v=(2,0)', 'orange')}
      {step >= 1 && <>{arrow([0, 0], [proj.projected![0], proj.projected![1]], 'green')}{labelAt([3, -.6], 'p=(3,0)', 'green')}</>}
      {step >= 2 && <><path d={`M${xy([3, 0])}L${xy([3, 2])}`} stroke="var(--learning-neutral)" strokeWidth="2" strokeDasharray="4 3" /><path d={`M${xy([2.7, 0])}L${xy([2.7, .3])}L${xy([3, .3])}`} fill="none" stroke="var(--learning-neutral)" />{labelAt([3.2, 1], 'r')}</>}
    </>}
    {kind === 'least-squares' && <>
      {[1, 2, 2].map((value, i) => <g key={i}>{step >= 1 && <path d={`M${xy([i + 1, value])}L${xy([i + 1, ls.predicted[i]])}`} stroke="var(--learning-orange)" strokeWidth="2" strokeDasharray="4 3" />}<circle cx={xy([i + 1, value])[0]} cy={xy([i + 1, value])[1]} r="5" fill="var(--learning-blue)" />{labelAt([i + .9, -.35], String(i + 1))}</g>)}
      {step >= 1 && <path d={`M${xy([0, 0])}L${xy([3.5, (step === 2 ? 11 / 14 : .5) * 3.5])}`} stroke="var(--learning-green)" strokeWidth="3" fill="none" />}
      {labelAt([.3, 3], step === 0 ? 'y=(1,2,2)' : step === 1 ? 'a=0.5; L=0.75' : 'a*=11/14; L=5/28')}
    </>}
    {kind === 'eigenvalues' && <>
      {step === 2 ? <><polyline points={Array.from({ length: 65 }, (_, i) => { const theta = i / 64 * 2 * Math.PI; const u = Math.sqrt(3) * Math.cos(theta); const v = Math.sin(theta); return xy([eigen.first[0] * u + eigen.second[0] * v, eigen.first[1] * u + eigen.second[1] * v]).join(','); }).join(' ')} stroke="var(--learning-green)" strokeWidth="2" fill="none" />{arrow([0, 0], [Math.sqrt(3) * eigen.first[0], Math.sqrt(3) * eigen.first[1]], 'green')}{arrow([0, 0], eigen.second, 'orange')}{labelAt([1.3, 1.5], '√3', 'green')}{labelAt([-1.4, 1], '1', 'orange')}{labelAt([1.4, -.6], '75%', 'green')}</>
        : <>{arrow([0, 0], mapped, 'orange')}{arrow([0, 0], first, 'blue')}{labelAt([mapped[0] + .15, mapped[1] + .1], step === 0 ? 'Av=(2,1)' : 'Av=3v', 'orange')}{labelAt([first[0] + .15, first[1] - .4], 'v', 'blue')}{step === 1 && arrow([0, 0], eigen.second, 'green')}</>}
    </>}
  </svg>;
}
