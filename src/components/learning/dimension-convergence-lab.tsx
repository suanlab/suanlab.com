'use client';

import { useState } from 'react';
import { useLanguage } from '@/components/language-provider';
import { dimensionConvergenceGuide } from '@/data/dimension-convergence-reader';
import { calculateOscillatorConvergence } from '@/lib/learning-dimension-convergence';
import { WavePlot, waveNumber } from './waves-plot';

export default function DimensionConvergenceLab() {
  const { language } = useLanguage();
  const ko = language === 'ko';
  const [steps, setSteps] = useState(20);
  const result = calculateOscillatorConvergence(steps);
  const exact = Array.from({ length: 101 }, (_, i): [number, number] => [Math.cos(i / 25), -Math.sin(i / 25)]);
  const readings = [
    ['h', result.h],
    [ko ? '계산한 에너지 비율' : 'Computed energy ratio', result.energyRatio],
    [ko ? '유도한 에너지 비율' : 'Derived energy ratio', result.analyticalEnergyRatio],
    [ko ? '최종 상태 오차' : 'Final state error', result.stateError],
  ] as const;
  return <section id="experiment" className="grid min-w-0 grid-cols-1 gap-4" aria-labelledby="dimension-convergence-title">
    <h2 id="dimension-convergence-title">{ko ? '시간 간격과 보존량 검사' : 'Time-step and invariant checks'}</h2>
    <label>{ko ? '단계 수' : 'Step count'} N = {steps}
      <input className="block w-full" type="range" min={20} max={160} step={20} value={steps} onChange={e => setSteps(Number(e.target.value))}/>
    </label>
    <WavePlot marker="dimension-convergence" title={ko ? '정확한 궤적과 오일러 궤적' : 'Exact and Euler trajectories'}
      xLabel="x" yLabel="p = v/ω" xDomain={[-1.6, 1.6]} yDomain={[-1.6, 1.6]}
      series={[
        { name: ko ? '정확한 해' : 'Exact solution', color: 'neutral', segments: [exact] },
        { name: 'Euler', color: 'blue', segments: [result.trajectory] },
      ]}
      dots={[{ color: 'blue', point: [result.x, result.p] }]}
      caption={ko ? '최종 무차원 시간 T는 4로 고정합니다.' : 'Final dimensionless time T is fixed at four.'}/>
    <dl className="grid grid-cols-2 gap-2 tabular-nums">
      {readings.map(([key, value]) => <div className="contents" key={key}><dt>{key}</dt><dd>{waveNumber(value)}</dd></div>)}
    </dl>
    {dimensionConvergenceGuide.experiments.map(experiment => <details key={experiment.title.en}>
      <summary>{experiment.title[language]}</summary>
      <p>{experiment.predict[language]}</p><p>{experiment.action[language]}</p><p>{experiment.explain[language]}</p>
    </details>)}
  </section>;
}

export function DimensionConvergenceConcept({ language }: { language: 'ko' | 'en' }) {
  const ko = language === 'ko';
  return <section className="ai-intuition" id="visual-intuition" aria-labelledby="dimension-convergence-concept-title">
    <h2 id="dimension-convergence-concept-title">{ko ? '정확한 기준과 독립 검산' : 'An exact reference and an independent check'}</h2>
    <p>{dimensionConvergenceGuide.intuition[language]}</p>
    <p className="ai-concept-equation">{dimensionConvergenceGuide.steps[0].formula}</p>
    <p>{dimensionConvergenceGuide.steps[0].explanation[language]}</p>
    <a href="#experiment">{ko ? '단계 수를 바꿔 수렴 확인하기' : 'Change the step count and check convergence'}</a>
  </section>;
}
