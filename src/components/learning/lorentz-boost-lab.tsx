'use client';

import { useState } from 'react';
import { lorentzBoostGuide } from '@/data/lorentz-boost-reader';
import { useLanguage } from '@/components/language-provider';
import { boostSpacetimeEvent, intervalSquared, type SpacetimeEvent } from '@/lib/learning-lorentz-boost';
import { WavePlot, waveNumber } from './waves-plot';

const origin = { ct: 0, x: 0 };
const controls = [
  { key: 'beta', label: 'β', min: -.9, max: .9, step: .05 },
  { key: 'ct', label: 'ct', min: -3, max: 3, step: .25 },
  { key: 'x', label: 'x', min: -3, max: 3, step: .25 },
] as const;

export default function LorentzBoostLab() {
  const { language } = useLanguage();
  const ko = language === 'ko';
  const [input, setInput] = useState({ beta: .6, ct: 0, x: 3 });
  const transformed = boostSpacetimeEvent(input, input.beta);
  const extent = Math.ceil(Math.max(4, Math.abs(transformed.x), Math.abs(transformed.ct)));
  const domain: [number, number] = [-extent, extent];
  const panel = (id: 's' | 's-prime', event: SpacetimeEvent) => <WavePlot marker={`lorentz-${id}`}
    title={id === 's' ? (ko ? '관성계 S' : 'Frame S') : (ko ? '관성계 S′' : 'Frame S′')}
    xLabel={id === 's' ? 'x' : 'x′'} yLabel={id === 's' ? 'ct' : 'ct′'} xDomain={domain} yDomain={domain}
    series={[
      { color: 'blue', name: ko ? '빛의 경로' : 'Light rays', segments: [[[-extent, -extent], [extent, extent]], [[-extent, extent], [extent, -extent]]] },
    ]}
    dots={[{ color: 'neutral', point: [0, 0] }, { color: 'orange', point: [event.x, event.ct] }]}
    caption={ko ? `주황색 점 B의 좌표는 (${waveNumber(event.ct)}, ${waveNumber(event.x)})입니다.` : `The orange point B has coordinates (${waveNumber(event.ct)}, ${waveNumber(event.x)}).`}/>;
  return <section id="experiment" className="grid min-w-0 grid-cols-1 gap-4" aria-labelledby="lorentz-experiment-title">
    <h2 id="lorentz-experiment-title">{ko ? '관성계를 바꿔 사건 읽기' : 'Read the event in another frame'}</h2>
    <p>{ko ? '사건 A를 공통 원점에 두고 사건 B의 좌표와 상대 속도를 바꿉니다.' : 'Keep event A at the shared origin and vary event B and the relative velocity.'}</p>
    <div className="flex flex-wrap gap-2">
      {[
        { ct: 0, x: 3, label: ko ? '동시 사건' : 'Simultaneous events' },
        { ct: 2, x: 2, label: ko ? '빛 사건' : 'Light event' },
        { ct: 2, x: 0, label: ko ? '정지 시계의 눈금' : 'Ticks of a stationary clock' },
      ].map(({ ct, x, label }) => <button key={label} type="button" className="rounded border border-border px-2 py-1"
        onClick={() => setInput(current => ({ ...current, ct, x }))}>{label}</button>)}
    </div>
    {controls.map(({ key, label, min, max, step }) => <label key={key}>{label} = {input[key]}
      <input className="block w-full" type="range" min={min} max={max} step={step} value={input[key]}
        onChange={e => setInput(current => ({ ...current, [key]: Number(e.target.value) }))}/>
    </label>)}
    {panel('s', input)}
    {panel('s-prime', transformed)}
    <dl className="grid grid-cols-2 gap-2 tabular-nums">
      <div className="contents"><dt>γ</dt><dd>{waveNumber(transformed.gamma)}</dd></div>
      <div className="contents"><dt>η</dt><dd>{waveNumber(transformed.rapidity)}</dd></div>
      <div className="contents"><dt>Δs² · S</dt><dd>{waveNumber(intervalSquared(origin, input))}</dd></div>
      <div className="contents"><dt>Δs² · S′</dt><dd>{waveNumber(intervalSquared(origin, transformed))}</dd></div>
    </dl>
    {lorentzBoostGuide.experiments.map(experiment => <details key={experiment.title.en}>
      <summary>{experiment.title[language]}</summary>
      <p>{experiment.predict[language]}</p><p>{experiment.action[language]}</p><p>{experiment.explain[language]}</p>
    </details>)}
  </section>;
}

export function LorentzBoostConcept({ language }: { language: 'ko' | 'en' }) {
  const ko = language === 'ko';
  const [beta, setBeta] = useState(.6);
  return <section className="ai-intuition" id="visual-intuition" aria-labelledby="lorentz-concept-title">
    <h2 id="lorentz-concept-title">{ko ? '움직이는 좌표계의 원점' : 'Origin of the moving frame'}</h2>
    <p>{lorentzBoostGuide.intuition[language]}</p>
    <label>β = {beta}
      <input className="block w-full" type="range" min={-.9} max={.9} step={.05} value={beta}
        onChange={e => setBeta(Number(e.target.value))}/>
    </label>
    <p className="ai-concept-equation" aria-live="polite">ct = 2, x = {waveNumber(2 * beta)} → x′ = 0</p>
    <p>{lorentzBoostGuide.steps[0].explanation[language]}</p>
    <a href="#experiment">{ko ? '다른 사건 비교하기' : 'Compare other events'}</a>
  </section>;
}
