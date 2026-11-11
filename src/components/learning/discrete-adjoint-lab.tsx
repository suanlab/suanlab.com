'use client';

import { useState } from 'react';
import { useLanguage } from '@/components/language-provider';
import { discreteAdjointGuide } from '@/data/discrete-adjoint-reader';
import { calculateDiscreteAdjoint } from '@/lib/learning-discrete-adjoint';
import { WavePlot, waveNumber } from './waves-plot';

export default function DiscreteAdjointLab() {
  const { language } = useLanguage();
  const ko = language === 'ko';
  const [persistence, setPersistence] = useState(.5);
  const [first, setFirst] = useState(0);
  const [second, setSecond] = useState(0);
  const [target, setTarget] = useState(0);
  const result = calculateDiscreteAdjoint(1, persistence, [first, second], target, .1);
  const points = (values: number[]) => values.map((value, i): [number, number] => [i, value]);
  const readings = [
    [ko ? '손실 J' : 'Loss J', result.loss],
    ['∂J/∂u₀', result.controlGradient[0]],
    ['∂J/∂u₁', result.controlGradient[1]],
    ['dJ/da', result.persistenceGradient],
  ] as const;
  return <section id="experiment" className="grid min-w-0 grid-cols-1 gap-4" aria-labelledby="discrete-adjoint-title">
    <h2 id="discrete-adjoint-title">{ko ? '두 단계의 순방향과 역방향' : 'A two-step forward and reverse pass'}</h2>
    <label>a = {persistence}
      <input className="block w-full" type="range" min={0} max={1} step={.05} value={persistence} onChange={e => setPersistence(Number(e.target.value))}/>
    </label>
    <label>u₀ = {first}
      <input className="block w-full" type="range" min={-.5} max={.5} step={.05} value={first} onChange={e => setFirst(Number(e.target.value))}/>
    </label>
    <label>u₁ = {second}
      <input className="block w-full" type="range" min={-.5} max={.5} step={.05} value={second} onChange={e => setSecond(Number(e.target.value))}/>
    </label>
    <label>{ko ? '최종 목표' : 'Terminal target'} y = {target}
      <input className="block w-full" type="range" min={-1} max={1} step={.05} value={target} onChange={e => setTarget(Number(e.target.value))}/>
    </label>
    <WavePlot marker="discrete-adjoint" title={ko ? '단계별 상태와 수반 변수' : 'States and adjoints by step'}
      xLabel="n" yLabel={ko ? '값' : 'Value'} xDomain={[0, 2]}
      series={[
        { name: ko ? '상태 xₙ' : 'State xₙ', color: 'blue', segments: [points(result.states)] },
        { name: ko ? '수반 변수 pₙ' : 'Adjoint pₙ', color: 'orange', segments: [points(result.adjoint)] },
      ]}
      caption={ko ? '초기 상태는 1, 입력 벌점 γ는 0.1로 고정합니다.' : 'The initial state is one and the control penalty γ is 0.1.'}/>
    <dl className="grid grid-cols-2 gap-2 tabular-nums">
      {readings.map(([key, value]) => <div className="contents" key={key}><dt>{key}</dt><dd>{waveNumber(value)}</dd></div>)}
    </dl>
    {discreteAdjointGuide.experiments.map(experiment => <details key={experiment.title.en}>
      <summary>{experiment.title[language]}</summary>
      <p>{experiment.predict[language]}</p><p>{experiment.action[language]}</p><p>{experiment.explain[language]}</p>
    </details>)}
  </section>;
}

export function DiscreteAdjointConcept({ language }: { language: 'ko' | 'en' }) {
  const ko = language === 'ko';
  return <section className="ai-intuition" id="visual-intuition" aria-labelledby="discrete-adjoint-concept-title">
    <h2 id="discrete-adjoint-concept-title">{ko ? '연쇄법칙을 역방향으로 적용하기' : 'Apply the chain rule backward'}</h2>
    <p>{discreteAdjointGuide.intuition[language]}</p>
    <p className="ai-concept-equation">{discreteAdjointGuide.steps[0].formula}</p>
    <p>{discreteAdjointGuide.steps[0].explanation[language]}</p>
    <a href="#experiment">{ko ? '두 입력의 민감도 비교하기' : 'Compare the two input sensitivities'}</a>
  </section>;
}
