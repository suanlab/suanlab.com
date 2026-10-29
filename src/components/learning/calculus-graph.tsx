'use client';
import { useId } from 'react';
import { useLanguage } from '@/components/language-provider';
import { neuronState, neuronStep } from '@/lib/learning-calculus';
import { formatNumber as n } from '@/lib/learning-simulations';
import { calculusValues, defaultCalculus, type CalculusKind, type CalculusSettings } from './calculus-model';

export default function CalculusGraph({ kind, stage, label, settings = defaultCalculus, reference = false }: {kind: CalculusKind; stage: number; label: string; settings?: CalculusSettings; reference?: boolean}) {
  const { language } = useLanguage(); const ko = language === 'ko';
  const id = useId(); const showReverse = reference ? stage >= (kind === 'backpropagation' ? 1 : kind === 'jacobians' ? 3 : 2) : stage >= 2; const v = calculusValues(settings); const c = v.chain; const j = v.jacobian; const b = v.neuron;
  const nodes = kind === 'chain-rule' ? [['x', n(settings.x)], ['u=ax+b', n(c.u)], ['y=u²', n(c.output)]]
    : kind === 'jacobians' ? [['(x,y)', settings.point.map(x => n(x)).join(', ')], ['xy', n(j.output[0])], ['x²+y', n(j.output[1])]]
    : [['z=w·x+b', n(b.z)], ['h=φ(z)', n(b.hidden)], ['ŷ=vh+c', n(b.prediction)], ['L=½(ŷ−t)²', n(b.loss)]];
  const positions = kind === 'jacobians' ? [[90,145],[290,112],[290,212]] : nodes.map((_,i) => [65+i*(270/(nodes.length-1)),135]);
  const edges = kind === 'jacobians' ? [[0,1],[0,2]] : nodes.slice(1).map((_,i) => [i,i+1]);
  const reverse = kind === 'chain-rule' ? [`dy/dx=${n(c.derivative)}`,`dy/du=${n(c.localOuter)}`]
    : kind === 'jacobians' ? [`Jᵀq=(${j.vjp.map(x => n(x)).join(', ')})`] : [`∂L/∂z=${n(b.adjointZ)}`,`∂L/∂h=${n(b.adjointHidden)}`,`∂L/∂ŷ=${n(b.error)}`];
  return <div className="ai-calculus-diagram" tabIndex={0} role="region" aria-label={ko ? '계산 그래프 · 작은 화면에서 가로로 이동하여 확인하세요' : 'Computation graph · Scroll horizontally on narrow screens'}><svg className="ai-concept-diagram ai-calculus-graph" viewBox="0 0 400 280" role="img" aria-label={label}>
    <defs>{['blue','orange'].map(color => <marker key={color} id={`${id}-${color}`} markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0L7 3.5L0 7Z" fill={`var(--learning-${color})`} /></marker>)}</defs>
    {edges.map(([a,z],i) => <g key={i}><path d={`M${positions[a][0]+34} ${positions[a][1]}L${positions[z][0]-34} ${positions[z][1]}`} fill="none" stroke="var(--learning-blue)" strokeWidth="2.5" markerEnd={`url(#${id}-blue)`} opacity={reference||stage>=1?1:.3}/>{showReverse&&<path d={`M${positions[z][0]-34} ${positions[z][1]+38}L${positions[a][0]+34} ${positions[a][1]+38}`} fill="none" stroke="var(--learning-orange)" strokeWidth="2.5" markerEnd={`url(#${id}-orange)`}/>}</g>)}
    {nodes.map(([name,value],i) => <g key={name}><rect x={positions[i][0]-32} y={positions[i][1]-28} width="64" height="56" rx="9" fill="hsl(var(--card))" stroke={`var(--learning-${i===nodes.length-1?'green':'blue'})`} strokeWidth="2"/><text x={positions[i][0]} y={positions[i][1]-42} textAnchor="middle" fill="var(--learning-neutral)" fontSize="13">{name}</text><text x={positions[i][0]} y={positions[i][1]+5} textAnchor="middle" fill="hsl(var(--foreground))" fontSize="15" textLength={value.length>7?56:undefined} lengthAdjust="spacingAndGlyphs">{reference||stage>=1||i===0?value:'?'}</text></g>)}
    {showReverse&&reverse.map((text,i) => <text key={i} x="200" y={kind==='jacobians'?265:208+i*22} textAnchor="middle" fill="var(--learning-orange)" fontSize="13">{text}</text>)}
    {kind==='jacobians'&&(reference||stage>=1)&&<text x="200" y="24" textAnchor="middle" fill="var(--learning-blue)" fontSize="13">J = [{j.matrix.map(r=>`[${r.map(x=>n(x)).join(',')}]`).join(',')}]</text>}
    {kind==='jacobians'&&stage>=2&&<text x="200" y="46" textAnchor="middle" fill="var(--learning-green)" fontSize="13">Jd=({j.jvp.map(x=>n(x)).join(', ')})</text>}
    {kind==='backpropagation'&&<text x="200" y="25" textAnchor="middle" fill="var(--learning-blue)" fontSize="13">x=({settings.input.map(x=>n(x)).join(', ')}) · t={n(settings.target)}</text>}
    {kind==='backpropagation'&&reference&&stage===3&&<text x="200" y="49" textAnchor="middle" fill="var(--learning-green)" fontSize="13">Lnew={n(neuronState(settings.input,neuronStep(settings.input,settings.parameters,settings.target,settings.activation,settings.rate),settings.target,settings.activation).loss)}</text>}
    {kind==='chain-rule'&&(reference||stage>=1)&&<text x="200" y="36" textAnchor="middle" fill="var(--learning-blue)" fontSize="14">du/dx = {n(c.localInner)} · dy/du = {n(c.localOuter)}</text>}
  </svg></div>;
}
