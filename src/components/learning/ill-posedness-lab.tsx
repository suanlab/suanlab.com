'use client';

import { useState } from 'react';
import { useLanguage } from '@/components/language-provider';
import { illPosednessGuide } from '@/data/ill-posedness-reader';
import { calculateSingularModeInverse } from '@/lib/learning-ill-posedness';
import { WavePlot, waveNumber } from './waves-plot';

export default function IllPosednessLab() {
  const { language } = useLanguage();
  const ko = language === 'ko';
  const [strength, setStrength] = useState(.5);
  const [error, setError] = useState(.01);
  const [direction, setDirection] = useState<'strong' | 'weak'>('weak');
  const result = calculateSingularModeInverse(strength, error, direction);
  const weakCurve = Array.from({ length: 100 }, (_, i): [number, number] => {
    const value = (i + 1) / 100;
    return [value, Math.abs(error) / value];
  });
  const states = {
    unique: ko ? '유일한 정확한 해' : 'One exact solution',
    nonunique: ko ? '정확한 해가 무한히 많음' : 'Infinitely many exact solutions',
    inconsistent: ko ? '정확한 해가 없음' : 'No exact solution',
  };
  const readings = [
    ['y₁', result.firstReading], ['y₂', result.secondReading],
    ['x̂₁', result.firstEstimate], ['x̂₂', result.secondEstimate],
    [ko ? '조건수' : 'Condition number', result.condition],
    [ko ? '복원 오차 노름' : 'Reconstruction error norm', result.estimateError],
  ] as const;
  return <section id="experiment" className="grid min-w-0 grid-cols-1 gap-4" aria-labelledby="ill-posedness-title">
    <h2 id="ill-posedness-title">{ko ? '특잇값이 작을 때의 복원' : 'Inversion with a small singular value'}</h2>
    <label>s = {strength}
      <input className="block w-full" type="range" min={0} max={1} step={.01} value={strength} onChange={e => setStrength(Number(e.target.value))}/>
    </label>
    <label>δ = {error}
      <input className="block w-full" type="range" min={-.05} max={.05} step={.005} value={error} onChange={e => setError(Number(e.target.value))}/>
    </label>
    <label>{ko ? '관측 오차 방향' : 'Observation error direction'}
      <select className="ml-2 rounded border border-border bg-background p-1" value={direction}
        onChange={e => setDirection(e.target.value === 'strong' ? 'strong' : 'weak')}>
        <option value="strong">{ko ? '강한 방향' : 'Strong direction'}</option>
        <option value="weak">{ko ? '약한 방향' : 'Weak direction'}</option>
      </select>
    </label>
    <WavePlot marker="ill-posedness" title={ko ? '특잇값에 따른 복원 오차' : 'Reconstruction error versus singular value'}
      xLabel="s" yLabel={ko ? '오차 노름' : 'Error norm'} xDomain={[0, 1]}
      series={[
        { name: ko ? '약한 방향' : 'Weak direction', color: 'orange', segments: [weakCurve] },
        { name: ko ? '강한 방향' : 'Strong direction', color: 'blue', segments: [[[.01, Math.abs(error)], [1, Math.abs(error)]]] },
      ]}
      dots={result.estimateError === null ? [] : [{ color: 'neutral', point: [strength, result.estimateError] }]}
      caption={ko ? '양의 s에서만 두 모드를 유일하게 복원합니다.' : 'Both modes have a unique reconstruction only for positive s.'}/>
    <p role="status">{states[result.state]}</p>
    <dl className="grid grid-cols-2 gap-2 tabular-nums">
      {readings.map(([key, value]) => <div className="contents" key={key}><dt>{key}</dt><dd>{waveNumber(value)}</dd></div>)}
    </dl>
    {illPosednessGuide.experiments.map(experiment => <details key={experiment.title.en}>
      <summary>{experiment.title[language]}</summary>
      <p>{experiment.predict[language]}</p><p>{experiment.action[language]}</p><p>{experiment.explain[language]}</p>
    </details>)}
  </section>;
}

export function IllPosednessConcept({ language }: { language: 'ko' | 'en' }) {
  const ko = language === 'ko';
  return <section className="ai-intuition" id="visual-intuition" aria-labelledby="ill-posedness-concept-title">
    <h2 id="ill-posedness-concept-title">{ko ? '존재성, 유일성, 안정성' : 'Existence, uniqueness, and stability'}</h2>
    <p>{illPosednessGuide.intuition[language]}</p>
    <p className="ai-concept-equation">{illPosednessGuide.steps[0].formula}</p>
    <p>{illPosednessGuide.steps[0].explanation[language]}</p>
    <a href="#experiment">{ko ? '특잇값과 관측 오차 바꾸기' : 'Vary the singular value and observation error'}</a>
  </section>;
}
