'use client';

import { useState } from 'react';
import { useLanguage } from '@/components/language-provider';
import { dynamicsResidualsGuide } from '@/data/dynamics-residuals-reader';
import { calculateDynamicsResidual } from '@/lib/learning-dynamics-residuals';
import { WavePlot, waveNumber } from './waves-plot';

export default function DynamicsResidualsLab() {
  const { language } = useLanguage();
  const ko = language === 'ko';
  const [amplitude, setAmplitude] = useState(.2);
  const [time, setTime] = useState(1);
  const current = calculateDynamicsResidual(amplitude, time);
  const curve = Array.from({ length: 81 }, (_, i) => {
    const t = i / 40;
    return { t, ...calculateDynamicsResidual(amplitude, t) };
  });
  const readings = [
    [ko ? '기준값과의 차' : 'Difference from reference', current.observationResidual],
    [ko ? '이 시각의 방정식 잔차' : 'Equation residual at this time', current.equationResidual],
    [ko ? '전체 구간 방정식 RMS 잔차' : 'Equation residual RMS over the interval', current.equationRms],
  ] as const;
  return <section id="experiment" className="grid min-w-0 grid-cols-1 gap-4" aria-labelledby="dynamics-residuals-title">
    <h2 id="dynamics-residuals-title">{ko ? '같은 관측을 지나는 다른 곡선' : 'Different curves through the same measurements'}</h2>
    <label>b = {amplitude}
      <input className="block w-full" type="range" min={-.3} max={.3} step={.05} value={amplitude} onChange={e => setAmplitude(Number(e.target.value))}/>
    </label>
    <label>t = {time}
      <input className="block w-full" type="range" min={0} max={2} step={.05} value={time} onChange={e => setTime(Number(e.target.value))}/>
    </label>
    <WavePlot marker="dynamics-residuals" title={ko ? '기준 궤적과 후보 궤적' : 'Reference and candidate trajectories'}
      xLabel="t" yLabel="x" xDomain={[0, 2]} yDomain={[-.25, 1.25]}
      series={[
        { name: ko ? '기준' : 'Reference', color: 'neutral', segments: [curve.map(row => [row.t, row.baseline])] },
        { name: ko ? '후보' : 'Candidate', color: 'blue', segments: [curve.map(row => [row.t, row.candidate])] },
      ]}
      dots={[
        ...[0, 1, 2].map((t): { color: string; point: [number, number] } => ({ color: 'neutral', point: [t, Math.exp(-t)] })),
        { color: 'orange', point: [time, current.candidate] },
      ]}
      caption={ko ? '회색 점은 적합에 사용한 세 관측입니다.' : 'Gray points are the three measurements used for fitting.'}/>
    <dl className="grid grid-cols-2 gap-2 tabular-nums">
      {readings.map(([key, value]) => <div className="contents" key={key}><dt>{key}</dt><dd>{waveNumber(value)}</dd></div>)}
    </dl>
    {dynamicsResidualsGuide.experiments.map(experiment => <details key={experiment.title.en}>
      <summary>{experiment.title[language]}</summary>
      <p>{experiment.predict[language]}</p><p>{experiment.action[language]}</p><p>{experiment.explain[language]}</p>
    </details>)}
  </section>;
}

export function DynamicsResidualsConcept({ language }: { language: 'ko' | 'en' }) {
  const ko = language === 'ko';
  return <section className="ai-intuition" id="visual-intuition" aria-labelledby="dynamics-residuals-concept-title">
    <h2 id="dynamics-residuals-concept-title">{ko ? '점의 오차와 기울기의 오차' : 'Errors in values and in dynamics'}</h2>
    <p>{dynamicsResidualsGuide.intuition[language]}</p>
    <p className="ai-concept-equation">{dynamicsResidualsGuide.steps[0].formula}</p>
    <p>{dynamicsResidualsGuide.steps[0].explanation[language]}</p>
    <a href="#experiment">{ko ? '두 잔차 비교하기' : 'Compare both residuals'}</a>
  </section>;
}
