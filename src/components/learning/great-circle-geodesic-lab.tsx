'use client';

import { useState } from 'react';
import { useLanguage } from '@/components/language-provider';
import { greatCircleGeodesicGuide } from '@/data/great-circle-geodesic-reader';
import { greatCircleGeodesic, polarStraightLine } from '@/lib/learning-great-circle-geodesic';
import { WavePlot, waveNumber } from './waves-plot';

const controls = [
  { key: 'angle', ko: '끝점 분리각 (°)', en: 'Endpoint separation (°)', min: 10, max: 180, step: 5 },
  { key: 'tilt', ko: '대원 평면 각도 (°)', en: 'Great-circle plane angle (°)', min: 0, max: 180, step: 5 },
  { key: 'fraction', ko: '진행률', en: 'Progress', min: 0, max: 1, step: .025 },
  { key: 'parameter', ko: '직선의 아핀 매개변수 λ', en: 'Affine parameter λ of the straight line', min: -3, max: 3, step: .25 },
] as const;

export default function GreatCircleGeodesicLab() {
  const { language } = useLanguage();
  const ko = language === 'ko';
  const [input, setInput] = useState({ angle: 60, tilt: 30, fraction: .5, parameter: 1 });
  const [longArc, setLongArc] = useState(false);
  const observe = (fraction: number) => greatCircleGeodesic(2, input.angle / 180 * Math.PI, input.tilt / 180 * Math.PI, longArc, fraction);
  const sample = observe(input.fraction);
  const points = Array.from({ length: 81 }, (_, i) => observe(i / 80).point);
  const polar = polarStraightLine(input.parameter);
  const panel = (axis: 1 | 2) => <WavePlot marker={`great-circle-${axis}`}
    title={axis === 1 ? (ko ? 'xy 투영' : 'xy projection') : (ko ? 'xz 투영' : 'xz projection')}
    xLabel="x" yLabel={axis === 1 ? 'y' : 'z'} xDomain={[-2, 2]} yDomain={[-2, 2]}
    series={[{ name: ko ? '선택한 대원 호' : 'Selected great-circle arc', color: 'blue', segments: [points.map(p => [p[0], p[axis]])] }]}
    dots={[{ color: 'orange', point: [sample.point[0], sample.point[axis]] }]}
    caption={ko ? '구면 위 같은 점을 두 평면에 투영합니다.' : 'Both plots project the same point on the sphere.'}/>;
  return <section id="experiment" className="grid min-w-0 grid-cols-1 gap-4" aria-labelledby="great-circle-title">
    <h2 id="great-circle-title">{ko ? '같은 끝점을 잇는 대원 호' : 'Great-circle arcs with common endpoints'}</h2>
    {controls.map(({ key, min, max, step, ...labels }) => <label key={key}>{labels[language]} = {input[key]}
      <input className="block w-full" type="range" min={min} max={max} step={step} value={input[key]}
        onChange={e => setInput(current => ({ ...current, [key]: Number(e.target.value) }))}/>
    </label>)}
    <label className="flex items-center gap-2">
      <input type="checkbox" checked={longArc} onChange={e => setLongArc(e.target.checked)}/>
      {ko ? '반대 방향의 호 선택' : 'Use the arc in the opposite direction'}
    </label>
    {panel(1)}
    {panel(2)}
    <dl className="grid grid-cols-2 gap-2 tabular-nums">
      <div className="contents"><dt>{ko ? '선택한 호의 길이' : 'Selected arc length'}</dt><dd>{waveNumber(sample.length)}</dd></div>
      <div className="contents"><dt>{ko ? '최단 호 길이' : 'Shortest arc length'}</dt><dd>{waveNumber(sample.shortestLength)}</dd></div>
      <div className="contents"><dt>r̈</dt><dd>{waveNumber(polar.radialAcceleration)}</dd></div>
      <div className="contents"><dt>rθ̇²</dt><dd>{waveNumber(polar.connectionCorrection)}</dd></div>
    </dl>
    {sample.antipodal && <p>{ko ? '대척점에서는 대원 평면을 바꾸어도 두 끝점과 길이가 유지됩니다.' : 'At antipodal separation, changing the great-circle plane preserves both endpoints and the length.'}</p>}
    {greatCircleGeodesicGuide.experiments.map(experiment => <details key={experiment.title.en}>
      <summary>{experiment.title[language]}</summary>
      <p>{experiment.predict[language]}</p><p>{experiment.action[language]}</p><p>{experiment.explain[language]}</p>
    </details>)}
  </section>;
}

export function GreatCircleGeodesicConcept({ language }: { language: 'ko' | 'en' }) {
  const ko = language === 'ko';
  return <section className="ai-intuition" id="visual-intuition" aria-labelledby="great-circle-concept-title">
    <h2 id="great-circle-concept-title">{ko ? '측지선의 국소 조건' : 'The local geodesic condition'}</h2>
    <p>{greatCircleGeodesicGuide.intuition[language]}</p>
    <p className="ai-concept-equation">∇Ẋ Ẋ = 0</p>
    <p>{greatCircleGeodesicGuide.steps[0].explanation[language]}</p>
    <a href="#experiment">{ko ? '두 호의 길이 비교' : 'Compare both arc lengths'}</a>
  </section>;
}
