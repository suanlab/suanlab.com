'use client';

import { useState } from 'react';
import { useLanguage } from '@/components/language-provider';
import { physicalValidationGuide } from '@/data/physical-validation-reader';
import { calculatePhysicalValidation } from '@/lib/learning-physical-validation';
import { WavePlot, waveNumber } from './waves-plot';

const inputs = [[1, 0], [1, 1], [1, -1], [.5, .5]] as const;

export default function PhysicalValidationLab() {
  const { language } = useLanguage();
  const ko = language === 'ko';
  const [selection, setSelection] = useState(1);
  const [gain, setGain] = useState(1);
  const [limit, setLimit] = useState(.8);
  const result = calculatePhysicalValidation(inputs[selection], gain, limit);
  const inputNames = ko ? ['기저', '보강', '상쇄', '절반 크기'] : ['Basis', 'Constructive', 'Canceling', 'Half amplitude'];
  const measurements = [
    [ko ? '설정 출력' : 'Programmed output', result.target],
    [ko ? '교정 예측' : 'Calibrated prediction', result.fitted],
    [ko ? '측정 출력' : 'Measured output', result.measured],
    [ko ? '교정 후 잔차' : 'Calibrated residual', result.residual],
  ] as const;
  return <section id="experiment" className="grid min-w-0 grid-cols-1 gap-4" aria-labelledby="physical-validation-title">
    <h2 id="physical-validation-title">{ko ? '기저 교정과 중첩 입력 검증' : 'Basis calibration and superposition tests'}</h2>
    <label>{ko ? '검증 입력' : 'Validation input'}
      <select className="ml-2 rounded border border-border bg-background p-1" value={selection} onChange={e => setSelection(Number(e.target.value))}>
        {inputs.map((input, i) => <option key={i} value={i}>{inputNames[i]} ({input.join(', ')})</option>)}
      </select>
    </label>
    <label>{ko ? '공통 이득' : 'Common gain'} g = {gain}
      <input className="block w-full" type="range" min={.5} max={1.2} step={.05} value={gain} onChange={e => setGain(Number(e.target.value))}/>
    </label>
    <label>{ko ? '읽기 한계' : 'Readout limit'} L = {limit}
      <input className="block w-full" type="range" min={.5} max={1.2} step={.05} value={limit} onChange={e => setLimit(Number(e.target.value))}/>
    </label>
    <WavePlot marker="physical-validation" title={ko ? '설정 출력과 측정 출력' : 'Programmed and measured output'}
      xLabel={ko ? '채널' : 'Channel'} yLabel={ko ? '출력' : 'Output'} xDomain={[.5, 2.5]} yDomain={[-1.25, 1.25]}
      series={[]}
      dots={[
        ...result.target.map((value, i) => ({ color: 'blue' as const, point: [i + .95, value] as [number, number] })),
        ...result.measured.map((value, i) => ({ color: 'orange' as const, point: [i + 1.05, value] as [number, number] })),
      ]}
      caption={ko ? '각 채널의 파란 점은 설정 출력, 주황 점은 측정 출력입니다.' : 'In each channel, blue marks the programmed output and orange marks the measured output.'}/>
    <table className="w-full text-left text-sm tabular-nums">
      <caption>{ko ? '채널별 출력과 잔차' : 'Output and residual by channel'}</caption>
      <thead><tr><th scope="col">{ko ? '측정량' : 'Measure'}</th><th scope="col">1</th><th scope="col">2</th></tr></thead>
      <tbody>{measurements.map(([label, [a, b]]) => <tr key={label}>
        <th scope="row">{label}</th><td>{waveNumber(a)}</td><td>{waveNumber(b)}</td>
      </tr>)}</tbody>
    </table>
    {physicalValidationGuide.experiments.map(experiment => <details key={experiment.title.en}>
      <summary>{experiment.title[language]}</summary>
      <p>{experiment.predict[language]}</p><p>{experiment.action[language]}</p><p>{experiment.explain[language]}</p>
    </details>)}
  </section>;
}

export function PhysicalValidationConcept({ language }: { language: 'ko' | 'en' }) {
  const ko = language === 'ko';
  return <section className="ai-intuition" id="visual-intuition" aria-labelledby="physical-validation-concept-title">
    <h2 id="physical-validation-concept-title">{ko ? '교정한 장치를 새 입력으로 검증하기' : 'Validate a calibrated device on new inputs'}</h2>
    <p>{physicalValidationGuide.intuition[language]}</p>
    <p className="ai-concept-equation">{physicalValidationGuide.steps[0].formula}</p>
    <p>{physicalValidationGuide.steps[0].explanation[language]}</p>
    <a href="#experiment">{ko ? '읽기 한계에 따른 잔차 비교하기' : 'Compare residuals across readout limits'}</a>
  </section>;
}
