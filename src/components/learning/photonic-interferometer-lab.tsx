'use client';

import { useState } from 'react';
import { useLanguage } from '@/components/language-provider';
import { photonicInterferometerGuide } from '@/data/photonic-interferometer-reader';
import { calculatePhotonicProduct } from '@/lib/learning-photonic-interferometer';
import { WavePlot, waveNumber } from './waves-plot';

export default function PhotonicInterferometerLab() {
  const { language } = useLanguage();
  const ko = language === 'ko';
  const [angle, setAngle] = useState(45);
  const [transmission, setTransmission] = useState(.5);
  const [sign, setSign] = useState(1);
  const at = (degrees: number) => calculatePhotonicProduct(sign, -sign, degrees * Math.PI / 180, transmission);
  const result = at(angle);
  const curve = Array.from({ length: 46 }, (_, i) => ({ angle: i * 2, ...at(i * 2) }));
  const readings = [
    ['y₁', result.y1],
    ['y₂', result.y2],
    ['|y₁|²', result.power1],
    ['|y₂|²', result.power2],
    [ko ? '감쇠된 파워' : 'Removed power', result.dissipatedPower],
  ] as const;
  return <section id="experiment" className="grid min-w-0 grid-cols-1 gap-4" aria-labelledby="photonic-title">
    <h2 id="photonic-title">{ko ? '두 포트의 출력 진폭과 파워' : 'Amplitudes and powers at two ports'}</h2>
    <label>θ = {angle}°
      <input className="block w-full" type="range" min={0} max={90} value={angle} onChange={e => setAngle(Number(e.target.value))}/>
    </label>
    <label>d = {transmission}
      <input className="block w-full" type="range" min={0} max={1} step={.05} value={transmission} onChange={e => setTransmission(Number(e.target.value))}/>
    </label>
    <div className="flex flex-wrap items-center gap-2">
      <button type="button" className="rounded border border-border px-3 py-1" onClick={() => setSign(-sign)}>
        {ko ? '입력 부호 반전' : 'Flip input signs'}
      </button>
      <span aria-live="polite">x = ({sign}, {-sign})</span>
    </div>
    <WavePlot marker="photonic-interferometer" title={ko ? '각도에 따른 출력 진폭' : 'Output amplitudes versus rotation'}
      xLabel="θ (°)" yLabel={ko ? '진폭' : 'Amplitude'} xDomain={[0, 90]} yDomain={[-1.25, 1.25]}
      series={[
        { name: 'y₁', color: 'blue', segments: [curve.map(row => [row.angle, row.y1])] },
        { name: 'y₂', color: 'orange', segments: [curve.map(row => [row.angle, row.y2])] },
      ]}
      dots={[{ color: 'blue', point: [angle, result.y1] }, { color: 'orange', point: [angle, result.y2] }]}
      caption={ko ? '검출기의 두 파워는 진폭의 제곱으로 계산합니다.' : 'The two detector powers are the squared amplitudes.'}/>
    <dl className="grid grid-cols-2 gap-2 tabular-nums">
      {readings.map(([key, value]) => <div className="contents" key={key}><dt>{key}</dt><dd>{waveNumber(value)}</dd></div>)}
    </dl>
    {photonicInterferometerGuide.experiments.map(experiment => <details key={experiment.title.en}>
      <summary>{experiment.title[language]}</summary>
      <p>{experiment.predict[language]}</p><p>{experiment.action[language]}</p><p>{experiment.explain[language]}</p>
    </details>)}
  </section>;
}

export function PhotonicInterferometerConcept({ language }: { language: 'ko' | 'en' }) {
  const ko = language === 'ko';
  return <section className="ai-intuition" id="visual-intuition" aria-labelledby="photonic-concept-title">
    <h2 id="photonic-concept-title">{ko ? '광장에서 행렬 곱까지' : 'From optical fields to matrix products'}</h2>
    <p>{photonicInterferometerGuide.intuition[language]}</p>
    <p className="ai-concept-equation">{photonicInterferometerGuide.steps[0].formula}</p>
    <p>{photonicInterferometerGuide.steps[0].explanation[language]}</p>
    <a href="#experiment">{ko ? '진폭과 검출 파워 비교하기' : 'Compare amplitudes and detected powers'}</a>
  </section>;
}
