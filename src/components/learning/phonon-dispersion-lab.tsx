'use client';

import { useState } from 'react';
import { useLanguage } from '@/components/language-provider';
import { phononDispersionGuide } from '@/data/phonon-dispersion-reader';
import { calculatePhononModes } from '@/lib/learning-phonon-dispersion';
import { WavePlot, waveNumber } from './waves-plot';

export default function PhononDispersionLab() {
  const { language } = useLanguage();
  const ko = language === 'ko';
  const [mass, setMass] = useState(4);
  const [wave, setWave] = useState(0);
  const [phase, setPhase] = useState(0);
  const [branch, setBranch] = useState<'acoustic' | 'optical'>('acoustic');
  const q = wave * Math.PI;
  const result = calculatePhononModes(1, mass, q);
  const curve = Array.from({ length: 61 }, (_, i) => ({ position: i / 60, ...calculatePhononModes(1, mass, i / 60 * Math.PI) }));
  const amplitude = result.displacements?.[branch];
  const branchNames = { acoustic: ko ? '음향' : 'Acoustic', optical: ko ? '광학' : 'Optical' };
  return <section id="experiment" className="grid min-w-0 grid-cols-1 gap-4" aria-labelledby="phonon-title">
    <h2 id="phonon-title">{ko ? '두 원자 사슬의 분산과 변위' : 'Dispersion and displacement in the diatomic chain'}</h2>
    <label>mᵦ/mₐ = {mass}
      <input className="block w-full" type="range" min={1} max={6} step={.25} value={mass} onChange={e => setMass(Number(e.target.value))}/>
    </label>
    <label>q/π = {wave}
      <input className="block w-full" type="range" min={0} max={1} step={.025} value={wave} onChange={e => setWave(Number(e.target.value))}/>
    </label>
    <WavePlot marker="phonon-dispersion" title={ko ? '음향·광학 분산' : 'Acoustic and optical dispersion'}
      xLabel="q/π" yLabel="ω" xDomain={[0, 1]}
      series={[
        { name: branchNames.acoustic, color: 'blue', segments: [curve.map(row => [row.position, row.acoustic])] },
        { name: branchNames.optical, color: 'orange', segments: [curve.map(row => [row.position, row.optical])] },
      ]}
      dots={[{ color: 'blue', point: [wave, result.acoustic] }, { color: 'orange', point: [wave, result.optical] }]}
      caption={ko ? '용수철 강성과 셀 길이를 1로 둡니다.' : 'Spring stiffness and cell length are set to one.'}/>
    <label>{ko ? '변위를 볼 가지' : 'Branch to display'}
      <select className="ml-2 rounded border border-border bg-background p-1" value={branch}
        onChange={e => setBranch(e.target.value === 'acoustic' ? 'acoustic' : 'optical')}>
        <option value="acoustic">{branchNames.acoustic}</option><option value="optical">{branchNames.optical}</option>
      </select>
    </label>
    <label>{ko ? '진동 위상 (한 주기 중 비율)' : 'Oscillation phase (fraction of a cycle)'} = {phase}
      <input className="block w-full" type="range" min={0} max={1} step={.025} value={phase} onChange={e => setPhase(Number(e.target.value))}/>
    </label>
    {amplitude ? <figure>
      <div className="relative h-12 rounded bg-muted" aria-hidden="true">
        {Array.from({ length: 10 }, (_, i) => {
          const motion = amplitude[i % 2] * Math.cos(q * i / 2 - 2 * Math.PI * phase);
          return <span key={i} className={`absolute top-3 text-lg ${i % 2 ? 'text-orange-500' : 'text-blue-500'}`}
            style={{ left: `${7 + 9 * i + 3 * motion}%` }}>{i % 2 ? '●' : '○'}</span>;
        })}
      </div>
      <figcaption>{ko ? '○는 A, ●는 B입니다. 첫 셀의 변위' : '○ marks A and ● marks B. First-cell displacements'} · {amplitude.map((a, i) => waveNumber(a * Math.cos(q * i / 2 - 2 * Math.PI * phase))).join(', ')}</figcaption>
    </figure> : <p role="status">{ko ? '두 가지가 만나는 점에서는 가지별 변위 방향이 하나로 정해지지 않습니다.' : 'At the branch touching point, individual displacement directions are not uniquely determined.'}</p>}
    <dl className="grid grid-cols-2 gap-2 tabular-nums">
      <div className="contents"><dt>{branchNames.acoustic} ω</dt><dd>{waveNumber(result.acoustic)}</dd></div>
      <div className="contents"><dt>{branchNames.optical} ω</dt><dd>{waveNumber(result.optical)}</dd></div>
      <div className="contents"><dt>{ko ? '장파장 음속' : 'Long-wavelength sound speed'}</dt><dd>{waveNumber(result.soundSpeed)}</dd></div>
    </dl>
    {phononDispersionGuide.experiments.map(experiment => <details key={experiment.title.en}>
      <summary>{experiment.title[language]}</summary><p>{experiment.predict[language]}</p><p>{experiment.action[language]}</p><p>{experiment.explain[language]}</p>
    </details>)}
  </section>;
}

export function PhononDispersionConcept({ language }: { language: 'ko' | 'en' }) {
  const ko = language === 'ko';
  return <section className="ai-intuition" id="visual-intuition" aria-labelledby="phonon-concept-title">
    <h2 id="phonon-concept-title">{ko ? '한 셀에서 두 정상 모드' : 'Two normal modes from one cell'}</h2>
    <p>{phononDispersionGuide.intuition[language]}</p><p className="ai-concept-equation">D(q)v = ω²v</p>
    <p>{phononDispersionGuide.steps[0].explanation[language]}</p>
    <a href="#experiment">{ko ? '분산과 변위 확인' : 'Inspect dispersion and displacement'}</a>
  </section>;
}
