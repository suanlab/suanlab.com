'use client';

import { useState } from 'react';
import { useLanguage } from '@/components/language-provider';
import { fourMomentumGuide } from '@/data/four-momentum-reader';
import { compareCollisionGeometry } from '@/lib/learning-four-momentum';
import { WavePlot, waveNumber } from './waves-plot';

const controls = [
  { key: 'mass', ko: '각 입자의 질량 m', en: 'Each particle’s mass m', min: .5, max: 4, step: .25 },
  { key: 'kinetic', ko: '총 운동에너지 K', en: 'Total kinetic energy K', min: 0, max: 10, step: .25 },
  { key: 'finalMass', ko: '최종 질량의 합', en: 'Final mass sum', min: 1, max: 10, step: .25 },
] as const;

export default function FourMomentumLab() {
  const { language } = useLanguage();
  const ko = language === 'ko';
  const [input, setInput] = useState({ mass: 1, kinetic: 2.5, finalMass: 3 });
  const result = compareCollisionGeometry(input.mass, input.kinetic, input.finalMass);
  const samples = Array.from({ length: 41 }, (_, i) => {
    const k = i / 4;
    return { k, ...compareCollisionGeometry(input.mass, k, input.finalMass) };
  });
  const fixed = ko ? '고정 표적' : 'Fixed target';
  const beams = ko ? '마주 오는 빔' : 'Opposing beams';
  return <section id="experiment" className="grid min-w-0 grid-cols-1 gap-4" aria-labelledby="four-momentum-title">
    <h2 id="four-momentum-title">{ko ? '충돌 배치와 가용 에너지' : 'Collision geometry and available energy'}</h2>
    {controls.map(({ key, min, max, step, ...labels }) => <label key={key}>{labels[language]} = {input[key]}
      <input className="block w-full" type="range" min={min} max={max} step={step} value={input[key]}
        onChange={e => setInput(current => ({ ...current, [key]: Number(e.target.value) }))}/>
    </label>)}
    <WavePlot marker="four-momentum" title={ko ? '운동량 중심계 에너지' : 'Center-of-momentum energy'} xLabel="K" yLabel="W"
      series={[
        { name: fixed, color: 'blue', segments: [samples.map(row => [row.k, row.fixedTarget.centerEnergy])] },
        { name: beams, color: 'orange', segments: [samples.map(row => [row.k, row.opposingBeams.centerEnergy])] },
        { name: ko ? '필요 에너지' : 'Required energy', color: 'neutral', segments: [[[0, input.finalMass], [10, input.finalMass]]] },
      ]}
      dots={[
        { color: 'blue', point: [input.kinetic, result.fixedTarget.centerEnergy] },
        { color: 'orange', point: [input.kinetic, result.opposingBeams.centerEnergy] },
      ]}
      caption={ko ? '회색 선은 후보 최종 상태에 필요한 최소 에너지입니다.' : 'The gray line marks the minimum energy required by the proposed final state.'}/>
    <dl className="grid grid-cols-2 gap-2 tabular-nums">
      <div className="contents"><dt>{ko ? '실험실의 총에너지' : 'Total lab energy'}</dt><dd>{waveNumber(result.labEnergy)}</dd></div>
      {([[fixed, result.fixedTarget], [beams, result.opposingBeams]] as const).map(([label, geometry]) =>
        <div className="contents" key={label}><dt>{label} · W</dt>
          <dd>{waveNumber(geometry.centerEnergy)} · {geometry.allowed ? (ko ? '문턱 충족' : 'Threshold met') : (ko ? '문턱 미달' : 'Below threshold')}</dd>
        </div>)}
      <div className="contents"><dt>{fixed} · β<sub>cm</sub></dt><dd>{waveNumber(result.fixedTarget.centerSpeed)}</dd></div>
    </dl>
    {fourMomentumGuide.experiments.map(experiment => <details key={experiment.title.en}>
      <summary>{experiment.title[language]}</summary>
      <p>{experiment.predict[language]}</p><p>{experiment.action[language]}</p><p>{experiment.explain[language]}</p>
    </details>)}
  </section>;
}

export function FourMomentumConcept({ language }: { language: 'ko' | 'en' }) {
  const ko = language === 'ko';
  return <section className="ai-intuition" id="visual-intuition" aria-labelledby="four-momentum-concept-title">
    <h2 id="four-momentum-concept-title">{ko ? '충돌에 사용할 수 있는 에너지' : 'Energy available to the collision'}</h2>
    <p>{fourMomentumGuide.intuition[language]}</p>
    <p className="ai-concept-equation">{fourMomentumGuide.steps[0].formula}</p>
    <p>{fourMomentumGuide.steps[0].explanation[language]}</p>
    <a href="#experiment">{ko ? '충돌 배치 비교하기' : 'Compare collision arrangements'}</a>
  </section>;
}
