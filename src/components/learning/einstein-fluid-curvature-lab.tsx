'use client';

import { useState } from 'react';
import { useLanguage } from '@/components/language-provider';
import { einsteinFluidCurvatureGuide } from '@/data/einstein-fluid-curvature-reader';
import { calculateEinsteinFluidCurvature } from '@/lib/learning-einstein-fluid-curvature';
import { WavePlot, waveNumber } from './waves-plot';

const fields = [
  { key: 'density', ko: '에너지 밀도 ρ', en: 'Energy density ρ', min: 0, max: 6, step: .25 },
  { key: 'w', ko: '압력비 w = p/ρ', en: 'Pressure ratio w = p/ρ', min: -30, max: 30, step: 1 },
  { key: 'lambda', ko: '우주 상수 Λ', en: 'Cosmological constant Λ', min: -2, max: 2, step: .25 },
] as const;

export default function EinsteinFluidCurvatureLab() {
  const { language } = useLanguage();
  const ko = language === 'ko';
  const [input, setInput] = useState({ density: 3, w: 0, lambda: 0 });
  const { density, w, lambda } = input;
  const observe = (ratio: number) => calculateEinsteinFluidCurvature(density, ratio * density, lambda);
  const result = observe(w);
  const samples = Array.from({ length: 61 }, (_, i) => {
    const ratio = (i - 30) / 30;
    return { ratio, ...observe(ratio) };
  });
  const series = [
    { key: 'scalarCurvature', name: 'R', color: 'blue' },
    { key: 'timeRicci', name: 'R₀₀', color: 'orange' },
    { key: 'spatialRicci', name: 'Rᵢᵢ', color: 'green' },
  ] as const;
  return <section id="experiment" className="grid min-w-0 grid-cols-1 gap-4" aria-labelledby="einstein-fluid-title">
    <h2 id="einstein-fluid-title">{ko ? '완전 유체의 리치 곡률' : 'Ricci curvature of a perfect fluid'}</h2>
    <div className="flex flex-wrap gap-2">
      {[
        { density: 3, w: 0, lambda: 0, label: ko ? '먼지' : 'Dust' },
        { density: 3, w: 1 / 3, lambda: 0, label: ko ? '복사' : 'Radiation' },
        { density: 0, w: 0, lambda: 1, label: ko ? '진공' : 'Vacuum' },
      ].map(({ label, ...preset }) => <button key={label} type="button" className="rounded border border-border px-2 py-1"
        onClick={() => setInput(preset)}>{label}</button>)}
    </div>
    {fields.map(({ key, min, max, step, ...labels }) => <label key={key}>{labels[language]} · {waveNumber(input[key])}
      <input className="block w-full" type="range" min={min} max={max} step={step} value={key === 'w' ? input.w * 30 : input[key]} aria-valuetext={waveNumber(input[key])}
        onChange={e => setInput(current => ({ ...current, [key]: Number(e.target.value) / (key === 'w' ? 30 : 1) }))}/>
    </label>)}
    <WavePlot marker="einstein-fluid-curvature" title={ko ? '압력에 따른 곡률 성분' : 'Curvature components versus pressure'}
      xLabel="w" yLabel={ko ? '곡률' : 'Curvature'} xDomain={[-1, 1]}
      series={series.map(({ key, name, color }) => ({ name, color, segments: [samples.map(sample => [sample.ratio, sample[key]])] }))}
      dots={series.map(({ key, color }) => ({ color, point: [w, result[key]] }))}
      caption={ko ? 'κ=1인 단위의 유체 정지계 성분입니다. Rᵢᵢ는 공간 방향 하나의 값입니다.' : 'These are fluid rest-frame components in units with κ=1. Rᵢᵢ is for one spatial direction.'}/>
    <dl className="grid grid-cols-2 gap-2 tabular-nums">
      <div className="contents"><dt>T</dt><dd>{waveNumber(result.stressTrace)}</dd></div>
      {series.map(({ key, name }) => <div className="contents" key={key}><dt>{name}</dt><dd>{waveNumber(result[key])}</dd></div>)}
    </dl>
    {einsteinFluidCurvatureGuide.experiments.map(experiment => <details key={experiment.title.en}>
      <summary>{experiment.title[language]}</summary>
      <p>{experiment.predict[language]}</p><p>{experiment.action[language]}</p><p>{experiment.explain[language]}</p>
    </details>)}
  </section>;
}

export function EinsteinFluidCurvatureConcept({ language }: { language: 'ko' | 'en' }) {
  const ko = language === 'ko';
  return <section className="ai-intuition" id="visual-intuition" aria-labelledby="einstein-concept-title">
    <h2 id="einstein-concept-title">{ko ? '수축하기 전과 후의 곡률' : 'Curvature before and after contraction'}</h2>
    <p>{einsteinFluidCurvatureGuide.intuition[language]}</p>
    <p className="ai-concept-equation">R = −R₀₀ + 3Rᵢᵢ</p>
    <p>{einsteinFluidCurvatureGuide.steps[0].explanation[language]}</p>
    <a href="#experiment">{ko ? '세 가지 원천 비교' : 'Compare three sources'}</a>
  </section>;
}
