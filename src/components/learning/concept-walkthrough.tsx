'use client';
import { isProbabilityKind } from '@/lib/learning-probability';

import { isAbstractAlgebraKind } from '@/lib/learning-abstract-algebra';
import { isTensorKind } from '@/lib/learning-tensors';
import { isComplexKind } from '@/lib/learning-complex';
import { isAnalysisKind } from '@/lib/learning-analysis';
import { isMatrixStructureKind } from '@/lib/learning-matrix-structures';
import { isDiscreteKind } from '@/lib/learning-discrete';
import { useId, useState } from 'react';
import { Button } from '@/components/ui/button';
import { useLanguage } from '@/components/language-provider';
import { learningGuides, type LearningUnitId } from '@/data/interactive-book';
import AlgebraConcept from './algebra-concept';
import { LogicConcept } from './logic-lab';
import { isLogicKind } from '@/lib/learning-logic';
import { FoundationConcept } from './foundations-lab';
import { isElementaryKind } from '@/lib/learning-elementary';
import { isFoundationKind } from '@/lib/learning-foundations';
import CalculusGraph from './calculus-graph';
import { isCalculusKind } from './calculus-model';
import { isAlgebraKind } from './algebra-model';
import { gradientStep, transform3, type Vec2 } from '@/lib/learning-simulations';

function Diagram({ kind, step, label }: { kind: LearningUnitId; step: number; label: string }) {
  const id = useId();
  if (isLogicKind(kind)) return <LogicConcept kind={kind} step={step} description={label}/>;
  if (isFoundationKind(kind)||isElementaryKind(kind)||isDiscreteKind(kind)||isMatrixStructureKind(kind)||isAnalysisKind(kind)||isComplexKind(kind)||isProbabilityKind(kind)||isAbstractAlgebraKind(kind)||isTensorKind(kind)) return <FoundationConcept kind={kind} label={label} step={step} />;
  if (isCalculusKind(kind)) return <CalculusGraph kind={kind} stage={step} label={label} reference />;
  if (isAlgebraKind(kind)) return <AlgebraConcept kind={kind} step={step} label={label} />;
  const origin: Vec2 = kind === 'vectors-and-forces' ? [60, 225] : kind === 'linear-transforms' ? [55, 225] : [200, 140];
  const scale = kind === 'vectors-and-forces' ? 45 : kind === 'linear-transforms' ? 72 : 32;
  const xy = (p: Vec2): Vec2 => [origin[0] + p[0] * scale, origin[1] - p[1] * scale];
  const arrow = (from: Vec2, to: Vec2, color: string, dashed = false) => <path d={`M${xy(from)}L${xy(to)}`} stroke={`var(--learning-${color})`} strokeWidth="3" strokeDasharray={dashed ? '5 4' : undefined} fill="none" markerEnd={`url(#${id}-${color})`} />;
  const dot = (p: Vec2, color: string) => <circle cx={xy(p)[0]} cy={xy(p)[1]} r="5" fill={`var(--learning-${color})`} />;
  const text = (p: Vec2, value: string, color = 'neutral') => <text x={xy(p)[0]} y={xy(p)[1]} fill={`var(--learning-${color})`} fontSize="13">{value}</text>;
  const shear = (p: Vec2): Vec2 => transform3([[1, 1, 0], [0, 1, 0], [0, 0, 1]], [...p, 0]).slice(0, 2) as Vec2;
  const start: Vec2 = [2, 2]; const next = gradientStep(2, 1, start, .25); const next2 = gradientStep(2, 1, next, .25); const large = gradientStep(2, 1, start, 1.25);
  return <svg viewBox="0 0 400 280" role="img" aria-label={label} className="ai-concept-diagram">
    <defs>{['blue', 'orange', 'green'].map(color => <marker key={color} id={`${id}-${color}`} markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0L7 3.5L0 7Z" fill={`var(--learning-${color})`} /></marker>)}</defs>
    {Array.from({ length: 9 }, (_, i) => i - 4).map(i => <path key={i} d={`M${xy([-4, i])}L${xy([7, i])}M${xy([i, -4])}L${xy([i, 7])}`} fill="none" stroke="var(--learning-grid)" strokeOpacity=".2" />)}
    <path d={`M10 ${origin[1]}H385M${origin[0]} 15V265`} stroke="var(--learning-neutral)" strokeOpacity=".7" />
    <text x="382" y={origin[1] - 8} fill="var(--learning-neutral)" fontSize="12">x</text><text x={origin[0] + 8} y="20" fill="var(--learning-neutral)" fontSize="12">y</text>
    {kind === 'vectors-and-forces' && <>
      {arrow([0, 0], [3, 0], 'blue')}{text([1, -.5], 'F₁: 3 N', 'blue')}
      {step >= 1 && <>{arrow([0, 0], [0, 4], 'orange', true)}{arrow([3, 0], [3, 4], 'orange')}{text([3.2, 2], 'F₂: 4 N', 'orange')}</>}
      {step >= 2 && <>{arrow([0, 0], [3, 4], 'green')}{text([.6, 2.5], 'Fnet: 5 N', 'green')}{text([3, 4.3], '(3, 4)')}</>}
      {text([-.4, -.5], '0')}
    </>}
    {kind === 'linear-transforms' && <>
      <path d={`M${xy([0, 0])}L${xy([1, 0])}L${xy([1, 1])}L${xy([0, 1])}Z`} stroke="var(--learning-neutral)" strokeWidth="2" strokeDasharray="5 4" fill="none" />
      {dot([1, 2], 'neutral')}{text([.4, 2.25], 'x = (1, 2)')}
      {step === 0 ? <>{arrow([0, 0], [1, 0], 'blue')}{arrow([0, 0], [0, 1], 'orange')}{text([.45, -.4], 'e₁', 'blue')}{text([-.45, 1.2], 'e₂', 'orange')}</>
        : <><path d={`M${xy([0, 0])}L${xy(shear([1, 0]))}L${xy(shear([1, 1]))}L${xy(shear([0, 1]))}Z`} stroke="var(--learning-blue)" fill="var(--learning-blue)" fillOpacity=".08" strokeWidth="2" />{arrow([0, 0], shear([1, 0]), 'blue')}{arrow([0, 0], shear([0, 1]), 'orange')}{text([.35, -.4], 'Ae₁', 'blue')}{text([.45, 1.25], 'Ae₂', 'orange')}</>}
      {step >= 2 && <>{arrow([1, 0], [3, 2], 'orange')}{arrow([0, 0], [3, 2], 'green')}{dot([3, 2], 'green')}{text([2.4, 2.25], 'Ax = (3, 2)', 'green')}</>}
    </>}
    {kind === 'gradients' && <>
      {[.5, 2, 6].map(loss => <ellipse key={loss} cx={origin[0]} cy={origin[1]} rx={Math.sqrt(loss) * scale} ry={Math.sqrt(2 * loss) * scale} fill="none" stroke="var(--learning-green)" strokeOpacity=".65" />)}
      {dot([0, 0], 'green')}{text([.2, -.45], '(0, 0)', 'green')}{dot(start, 'orange')}{text([2.1, 2.2], 'p₀ = (2, 2)', 'orange')}
      {step >= 1 && <>{arrow(start, next, 'blue')}{dot(next, 'blue')}{text([.35, 1.1], 'p₁', 'blue')}</>}
      {step === 2 && <>{arrow(next, next2, 'blue')}{dot(next2, 'blue')}{text([-.5, .8], 'p₂', 'blue')}</>}
      {step === 3 && <>{arrow(start, large, 'orange')}{dot(large, 'orange')}{text([-3.2, -.95], 'η=1.25', 'orange')}{text([2, .7], 'η=0.25', 'blue')}</>}
    </>}
  </svg>;
}
export default function ConceptWalkthrough({ kind }: { kind: LearningUnitId }) {
  const { language } = useLanguage(); const ko = language === 'ko'; const guide = learningGuides[kind]; const [step, setStep] = useState(0); const current = guide.steps[step];
  return <section className="ai-intuition" id="visual-intuition" aria-labelledby="ai-intuition-title">
    <div className="ai-section-heading"><div><span className="ai-eyebrow">BUILD YOUR INTUITION</span><h2 id="ai-intuition-title">{ko ? '먼저 그림으로 이해하기' : 'Build the idea with a diagram'}</h2></div><p>{ko ? '한 번에 한 단계씩 살펴보세요.' : 'Explore one step at a time.'}</p></div>
    <p className="ai-intuition-lead">{guide.intuition[language]}</p>
    <div className="ai-concept-steps" role="group" aria-label={ko ? '개념도 단계 선택' : 'Choose a diagram step'}>{guide.steps.map((item, index) => <Button key={index} type="button" size="sm" variant="outline" aria-pressed={step === index} onClick={() => setStep(index)}>{index + 1}. {item.title[language]}</Button>)}</div>
    <div className="ai-concept-content"><figure><Diagram kind={kind} step={step} label={current.diagram[language]} /><figcaption>{isLogicKind(kind) ? (ko ? '고정한 유한 기준 모형입니다. 단계별 설명과 별도의 일반 논증을 구분해 읽으세요.' : 'A fixed finite reference model. Read the step explanations separately from any general proof.') : (isFoundationKind(kind)||isElementaryKind(kind)||isDiscreteKind(kind)||isMatrixStructureKind(kind)||isAnalysisKind(kind)||isComplexKind(kind)||isProbabilityKind(kind)||isAbstractAlgebraKind(kind)||isTensorKind(kind)) ? (ko ? '고정 기준 예제의 그림과 수치입니다. 각 단계에서 같은 그림의 의미를 차례로 살펴보세요.' : 'Fixed reference plot and readout. Each step interprets another aspect of the same example.') : isCalculusKind(kind) ? (ko ? '값과 민감도의 계산 그래프 · 작은 화면에서는 가로로 이동하세요. 선 길이는 크기를 나타내지 않습니다.' : 'Computation graph of values and sensitivities · Scroll horizontally on narrow screens. Edge lengths do not encode magnitude.') : kind === 'gradients' ? (ko ? '매개변수 평면의 등고선 · 높이는 표시하지 않습니다.' : 'Contours in parameter space · Height is not shown.') : (ko ? 'z=0인 평면의 개념도 · 3D 실험과 구분해서 읽으세요.' : 'Concept diagram in the plane z=0 · Read alongside the 3D experiment.')} {ko ? '고정된 기준 예제이며 아래 실험의 설정과 독립적입니다.' : 'This fixed reference example is independent of the experiment settings below.'}</figcaption></figure>
      <div className="ai-concept-explanation" aria-live="polite" aria-atomic="true"><span className="ai-eyebrow">{step + 1} / {guide.steps.length}</span><h3>{current.title[language]}</h3><p>{current.explanation[language]}</p><p className="ai-concept-equation">{current.formula}</p></div>
    </div>
    <div className="ai-concept-pagination"><Button type="button" variant="outline" size="sm" disabled={step === 0} onClick={() => setStep(n => n - 1)}>{ko ? '이전 단계' : 'Previous step'}</Button><Button type="button" variant="outline" size="sm" disabled={step === guide.steps.length - 1} onClick={() => setStep(n => n + 1)}>{ko ? '다음 단계' : 'Next step'}</Button><a href="#experiment">{ko ? '실험에서 확인하기 →' : 'Check it in the experiment →'}</a></div>
  </section>;
}
