'use client';

import { useState } from 'react';
import { useLanguage } from '@/components/language-provider';
import { modelDiscrepancyGuide } from '@/data/model-discrepancy-reader';
import { calculateModelDiscrepancy } from '@/lib/learning-model-discrepancy';
import { WavePlot, waveNumber } from './waves-plot';

export default function ModelDiscrepancyLab() {
  const { language } = useLanguage();
  const ko = language === 'ko';
  const [equilibrium, setEquilibrium] = useState(.25);
  const [time, setTime] = useState(1);
  const current = calculateModelDiscrepancy(equilibrium, 1, time);
  const curve = Array.from({ length: 81 }, (_, i) => {
    const t = i / 20;
    return { t, ...calculateModelDiscrepancy(equilibrium, 1, t) };
  });
  const readings = [
    [ko ? '적합한 감쇠율' : 'Fitted decay rate', current.fittedRate],
    [ko ? '데이터 생성값' : 'Data-generating value', current.truth],
    [ko ? '간단한 모형의 예측' : 'Simplified prediction', current.prediction],
    [ko ? '모형 불일치' : 'Model discrepancy', current.discrepancy],
  ] as const;
  return <section id="experiment" className="grid min-w-0 grid-cols-1 gap-4" aria-labelledby="model-discrepancy-title">
    <h2 id="model-discrepancy-title">{ko ? '교정 시각을 지나 외삽하기' : 'Extrapolate beyond calibration'}</h2>
    <label>{ko ? '평형값' : 'Equilibrium'} c = {equilibrium}
      <input className="block w-full" type="range" min={0} max={.5} step={.05} value={equilibrium} onChange={e => setEquilibrium(Number(e.target.value))}/>
    </label>
    <label>t = {time}
      <input className="block w-full" type="range" min={0} max={4} step={.25} value={time} onChange={e => setTime(Number(e.target.value))}/>
    </label>
    <WavePlot marker="model-discrepancy" title={ko ? '교정과 외삽에서의 두 곡선' : 'Curves at calibration and beyond'}
      xLabel="t" yLabel="y" xDomain={[0, 4]} yDomain={[0, 1.05]}
      series={[
        { name: ko ? '데이터 생성 모형' : 'Data-generating model', color: 'blue', segments: [curve.map(row => [row.t, row.truth])] },
        { name: ko ? '평형값 0의 적합 모형' : 'Fitted zero-equilibrium model', color: 'orange', segments: [curve.map(row => [row.t, row.prediction])] },
      ]}
      dots={[{ color: 'blue', point: [time, current.truth] }, { color: 'orange', point: [time, current.prediction] }]}
      vertical={[1]}
      caption={ko ? '점선은 교정 관측 시각 t=1입니다.' : 'The vertical dashed line marks the calibration reading at t=1.'}/>
    <dl className="grid grid-cols-2 gap-2 tabular-nums">
      {readings.map(([key, value]) => <div className="contents" key={key}><dt>{key}</dt><dd>{waveNumber(value)}</dd></div>)}
    </dl>
    {modelDiscrepancyGuide.experiments.map(experiment => <details key={experiment.title.en}>
      <summary>{experiment.title[language]}</summary>
      <p>{experiment.predict[language]}</p><p>{experiment.action[language]}</p><p>{experiment.explain[language]}</p>
    </details>)}
  </section>;
}

export function ModelDiscrepancyConcept({ language }: { language: 'ko' | 'en' }) {
  const ko = language === 'ko';
  return <section className="ai-intuition" id="visual-intuition" aria-labelledby="model-discrepancy-concept-title">
    <h2 id="model-discrepancy-concept-title">{ko ? '생략한 물리 효과와 추정값' : 'Omitted physics and fitted parameters'}</h2>
    <p>{modelDiscrepancyGuide.intuition[language]}</p>
    <p className="ai-concept-equation">{modelDiscrepancyGuide.steps[0].formula}</p>
    <p>{modelDiscrepancyGuide.steps[0].explanation[language]}</p>
    <a href="#experiment">{ko ? '교정점 밖에서 모형 비교하기' : 'Compare the models beyond calibration'}</a>
  </section>;
}
