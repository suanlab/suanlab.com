'use client';

import { useState } from 'react';
import { heavyModeMatchingGuide } from '@/data/heavy-mode-matching-reader';
import { useLanguage } from '@/components/language-provider';
import { calculateHeavyModeMatching, type HeavyModeInput } from '@/lib/learning-heavy-mode-matching';
import { WavePlot, waveNumber } from './waves-plot';

const fields = [
  { key: 'massRatio', ko: '질량비 M/m', en: 'Mass ratio M/m', min: 2, max: 16, step: 1 },
  { key: 'mixing', ko: '혼합비 g/(mM)', en: 'Mixing ratio g/(mM)', min: 0, max: .99, step: .01 },
  { key: 'frequencyRatio', ko: '주파수비 ω/M', en: 'Frequency ratio ω/M', min: 0, max: 2, step: .025 },
  { key: 'order', ko: '남길 최고 차수 p', en: 'Highest retained order p', min: 0, max: 6, step: 1 },
] as const;

export default function HeavyModeMatchingLab() {
  const { language } = useLanguage();
  const ko = language === 'ko';
  const [input, setInput] = useState<HeavyModeInput>({ massRatio: 4, mixing: .5, frequencyRatio: .5, order: 2 });
  const [plotLimit, setPlotLimit] = useState<1 | 2>(2);
  const sample = calculateHeavyModeMatching(input);
  const samples = Array.from({ length: 81 }, (_, i) => {
    const frequencyRatio = i * plotLimit / 80;
    return { frequencyRatio, ...calculateHeavyModeMatching({ ...input, frequencyRatio }) };
  });
  const fieldsToReport = [
    [ko ? '정확한 커널' : 'Exact kernel', sample.exactKernel],
    [ko ? '정합한 커널' : 'Matched kernel', sample.approximateKernel],
    [ko ? '커널 오차' : 'Kernel error', sample.kernelError],
    [ko ? '정확한 응답' : 'Exact response', sample.exactResponse],
    [ko ? '정합한 응답' : 'Matched response', sample.approximateResponse],
    [ko ? '응답 오차' : 'Response error', sample.responseError],
  ] as const;

  return <section id="experiment" className="grid min-w-0 grid-cols-1 gap-4" aria-labelledby="heavy-mode-experiment-title">
    <h2 id="heavy-mode-experiment-title">{ko ? '저주파 정합의 유효 범위' : 'The range of low-frequency matching'}</h2>
    {fields.map(({ key, min, max, step, ...labels }) => <label key={key}>
      {labels[language]} = {input[key]}
      <input className="block w-full" type="range" min={min} max={max} step={step}
        value={input[key]} onChange={e => setInput(current => ({ ...current, [key]: Number(e.target.value) }))}/>
    </label>)}
    <label className="flex items-center gap-2">
      <input type="checkbox" checked={plotLimit === 1} onChange={e => setPlotLimit(e.target.checked ? 1 : 2)}/>
      {ko ? '저주파 영역 확대 (ω/M ≤ 1)' : 'Zoom into the low-frequency range (ω/M ≤ 1)'}
    </label>
    <WavePlot marker="heavy-mode-matching" title={ko ? '가벼운 모드의 역응답 커널' : 'Inverse response kernel of the light mode'}
      xLabel="ω/M" yLabel="Q" xDomain={[0, plotLimit]} vertical={[1]}
      series={[
        { name: ko ? '정확한 커널' : 'Exact kernel', color: 'blue', segments: [samples.map(row => [row.frequencyRatio, row.exactKernel])] },
        { name: ko ? '정합한 커널' : 'Matched kernel', color: 'orange', segments: [samples.map(row => [row.frequencyRatio, row.approximateKernel])] },
      ]}
      dots={input.frequencyRatio > plotLimit ? [] : [
        { color: 'blue', point: [input.frequencyRatio, sample.exactKernel] },
        { color: 'orange', point: [input.frequencyRatio, sample.approximateKernel] },
      ]}
      caption={ko ? '점선은 전개 반경 ω/M = 1입니다. 커널은 m = 1 단위에서 비교합니다.' : 'The vertical line marks the expansion radius ω/M = 1. Kernels use units with m = 1.'}/>
    <dl className="grid grid-cols-2 gap-2 tabular-nums">
      {fieldsToReport.map(([label, value]) => <div className="contents" key={label}>
        <dt>{label}</dt><dd>{waveNumber(value)}</dd>
      </div>)}
    </dl>
    {sample.approximateResponse === null && <p role="status">
      {ko ? '정합한 커널이 양수가 아니므로 이 주파수의 가우시안 응답을 표시할 수 없습니다.' : 'The matched kernel is nonpositive, so a Gaussian response is undefined at this frequency.'}
    </p>}
    {heavyModeMatchingGuide.experiments.map(experiment => <details key={experiment.title.en}>
      <summary>{experiment.title[language]}</summary>
      <p>{experiment.predict[language]}</p><p>{experiment.action[language]}</p><p>{experiment.explain[language]}</p>
    </details>)}
  </section>;
}

export function HeavyModeMatchingConcept({ language }: { language: 'ko' | 'en' }) {
  const ko = language === 'ko';
  return <section className="ai-intuition" id="visual-intuition" aria-labelledby="heavy-mode-concept-title">
    <h2 id="heavy-mode-concept-title">{ko ? '두 좌표에서 한 유효 좌표로' : 'From two coordinates to one effective coordinate'}</h2>
    <p>{heavyModeMatchingGuide.intuition[language]}</p>
    <p className="ai-concept-equation">Qₗ = Q₁₁ − Q₁₂²/Q₂₂</p>
    <p>{heavyModeMatchingGuide.steps[0].explanation[language]}</p>
    <a href="#experiment">{ko ? '저주파 근사의 범위 확인하기' : 'Inspect the range of the low-frequency approximation'}</a>
  </section>;
}
