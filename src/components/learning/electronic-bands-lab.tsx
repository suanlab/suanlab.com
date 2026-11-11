'use client';

import { useState } from 'react';
import { useLanguage } from '@/components/language-provider';
import { electronicBandsGuide } from '@/data/electronic-bands-reader';
import { calculateElectronicBands } from '@/lib/learning-electronic-bands';
import { WavePlot, waveNumber } from './waves-plot';

export default function ElectronicBandsLab() {
  const { language } = useLanguage();
  const ko = language === 'ko';
  const [hopping, setHopping] = useState(.6);
  const [delta, setDelta] = useState(0);
  const [wave, setWave] = useState(1);
  const result = calculateElectronicBands(1, hopping, delta, Math.PI * wave);
  const curve = Array.from({ length: 81 }, (_, i) => {
    const position = i / 40 - 1;
    return { position, ...calculateElectronicBands(1, hopping, delta, Math.PI * position) };
  });
  const readings = [
    [ko ? '아래쪽 에너지' : 'Lower energy', result.lower],
    [ko ? '위쪽 에너지' : 'Upper energy', result.upper],
    [ko ? '벌크 밴드 간격' : 'Bulk band gap', result.gap],
    [ko ? '위쪽 상태의 A 가중치' : 'Upper state weight on A', result.upperWeightA],
    [ko ? '위쪽 밴드의 기울기 dE/dq' : 'Upper band slope dE/dq', result.upperSlope],
  ] as const;
  return <section id="experiment" className="grid min-w-0 grid-cols-1 gap-4" aria-labelledby="electronic-bands-title">
    <h2 id="electronic-bands-title">{ko ? '호핑과 자리 에너지가 만드는 밴드' : 'Bands from hopping and on-site energies'}</h2>
    <label>t₂ = {hopping}
      <input className="block w-full" type="range" min={0} max={1} step={.05} value={hopping} onChange={e => setHopping(Number(e.target.value))}/>
    </label>
    <label>Δ = {delta}
      <input className="block w-full" type="range" min={-.8} max={.8} step={.05} value={delta} onChange={e => setDelta(Number(e.target.value))}/>
    </label>
    <label>q/π = {wave}
      <input className="block w-full" type="range" min={-1} max={1} step={.025} value={wave} onChange={e => setWave(Number(e.target.value))}/>
    </label>
    <WavePlot marker="electronic-bands" title={ko ? '첫 브릴루앙 영역의 밴드' : 'Bands in the first Brillouin zone'}
      xLabel="q/π" yLabel="E" xDomain={[-1, 1]}
      series={[
        { name: ko ? '아래쪽 밴드' : 'Lower band', color: 'blue', segments: [curve.map(row => [row.position, row.lower])] },
        { name: ko ? '위쪽 밴드' : 'Upper band', color: 'orange', segments: [curve.map(row => [row.position, row.upper])] },
      ]}
      dots={[{ color: 'blue', point: [wave, result.lower] }, { color: 'orange', point: [wave, result.upper] }]}
      caption={ko ? '셀 안 호핑 t₁을 1로 고정합니다.' : 'Intracell hopping t₁ is fixed at one.'}/>
    <dl className="grid grid-cols-2 gap-2 tabular-nums">
      {readings.map(([key, value]) => <div className="contents" key={key}>
        <dt>{key}</dt><dd>{value === null ? '—' : waveNumber(value)}</dd>
      </div>)}
    </dl>
    {result.upperWeightA === null && <p role="status">{ko ? '두 에너지가 같아 자리 가중치와 가지 기울기가 하나로 정해지지 않습니다.' : 'At the crossing, individual site weights and the sorted branch slope are undefined.'}</p>}
    {electronicBandsGuide.experiments.map(experiment => <details key={experiment.title.en}>
      <summary>{experiment.title[language]}</summary>
      <p>{experiment.predict[language]}</p><p>{experiment.action[language]}</p><p>{experiment.explain[language]}</p>
    </details>)}
  </section>;
}

export function ElectronicBandsConcept({ language }: { language: 'ko' | 'en' }) {
  const ko = language === 'ko';
  return <section className="ai-intuition" id="visual-intuition" aria-labelledby="electronic-bands-concept-title">
    <h2 id="electronic-bands-concept-title">{ko ? '두 가지와 하나의 간격' : 'Two branches and one gap'}</h2>
    <p>{electronicBandsGuide.intuition[language]}</p>
    <p className="ai-concept-equation">{electronicBandsGuide.steps[0].formula}</p>
    <p>{electronicBandsGuide.steps[0].explanation[language]}</p>
    <a href="#experiment">{ko ? '밴드 간격 조절하기' : 'Adjust the band gap'}</a>
  </section>;
}
