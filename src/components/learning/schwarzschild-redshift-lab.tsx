'use client';

import { useState } from 'react';
import { useLanguage } from '@/components/language-provider';
import { schwarzschildRedshiftGuide } from '@/data/schwarzschild-redshift-reader';
import { calculateSchwarzschildRedshift } from '@/lib/learning-schwarzschild-redshift';
import { WavePlot, waveNumber } from './waves-plot';

const controls = [
  { key: 'emitter', ko: '방출 반지름 rₑ/M', en: 'Emitter radius rₑ/M', min: 2.5, max: 10, step: .25 },
  { key: 'receiver', ko: '수신 반지름 rₒ/M', en: 'Receiver radius rₒ/M', min: 12, max: 30, step: .5 },
  { key: 'beta', ko: '방출기의 바깥 방향 속도 β', en: 'Outward emitter speed β', min: -.8, max: .8, step: .05 },
] as const;

export default function SchwarzschildRedshiftLab() {
  const { language } = useLanguage();
  const ko = language === 'ko';
  const [input, setInput] = useState({ emitter: 4, receiver: 12, beta: 0 });
  const observe = (beta: number) => calculateSchwarzschildRedshift(input.emitter, input.receiver, beta);
  const result = observe(input.beta);
  const points = Array.from({ length: 33 }, (_, i): [number, number] => {
    const beta = (i - 16) / 20;
    return [beta, observe(beta).ratio];
  });
  return <section id="experiment" className="grid min-w-0 grid-cols-1 gap-4" aria-labelledby="redshift-experiment-title">
    <h2 id="redshift-experiment-title">{ko ? '어떤 조건에서 적색편이가 상쇄될까?' : 'When does the redshift cancel?'}</h2>
    {controls.map(({ key, min, max, step, ...labels }) => <label key={key}>{labels[language]} = {input[key]}
      <input className="block w-full" type="range" min={min} max={max} step={step} value={input[key]}
        onChange={e => setInput(current => ({ ...current, [key]: Number(e.target.value) }))}/>
    </label>)}
    <WavePlot marker="schwarzschild-redshift" title={ko ? '수신 주파수와 방출 주파수의 비' : 'Received-to-emitted frequency ratio'}
      xLabel="β" yLabel="ωₒ/ωₑ" xDomain={[-.8, .8]}
      series={[
        { name: 'ωₒ/ωₑ', color: 'blue', segments: [points] },
        { name: ko ? '편이 없음' : 'No shift', color: 'neutral', segments: [[[-.8, 1], [.8, 1]]] },
      ]}
      dots={[{ color: 'orange', point: [input.beta, result.ratio] }]}
      caption={ko ? '수신기는 바깥 반지름에서 정지하고 방출기만 움직입니다.' : 'The receiver stays at its outer radius while the emitter moves.'}/>
    <dl className="grid grid-cols-2 gap-2 tabular-nums">
      <div className="contents"><dt>{ko ? '중력 인자' : 'Gravitational factor'}</dt><dd>{waveNumber(result.gravitationalFactor)}</dd></div>
      <div className="contents"><dt>{ko ? '도플러 인자' : 'Doppler factor'}</dt><dd>{waveNumber(result.dopplerFactor)}</dd></div>
      <div className="contents"><dt>ωₒ/ωₑ</dt><dd>{waveNumber(result.ratio)}</dd></div>
      <div className="contents"><dt>z</dt><dd>{waveNumber(result.redshift)}</dd></div>
      <div className="contents"><dt>{ko ? '상쇄 속도' : 'Cancellation speed'} β</dt><dd>{waveNumber(result.cancellationSpeed)}</dd></div>
    </dl>
    {schwarzschildRedshiftGuide.experiments.map(experiment => <details key={experiment.title.en}>
      <summary>{experiment.title[language]}</summary>
      <p>{experiment.predict[language]}</p><p>{experiment.action[language]}</p><p>{experiment.explain[language]}</p>
    </details>)}
  </section>;
}

export function SchwarzschildRedshiftConcept({ language }: { language: 'ko' | 'en' }) {
  const ko = language === 'ko';
  return <section className="ai-intuition" id="visual-intuition" aria-labelledby="redshift-concept-title">
    <h2 id="redshift-concept-title">{ko ? '주파수는 누가 재는가' : 'Who measures the frequency'}</h2>
    <p>{schwarzschildRedshiftGuide.intuition[language]}</p>
    <p className="ai-concept-equation">{schwarzschildRedshiftGuide.steps[0].formula}</p>
    <p>{schwarzschildRedshiftGuide.steps[0].explanation[language]}</p>
    <a href="#experiment">{ko ? '주파수비 비교하기' : 'Compare frequency ratios'}</a>
  </section>;
}
