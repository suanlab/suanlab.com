'use client';

import { useState } from 'react';
import { rotationalCurvatureGuide } from '@/data/rotational-curvature-reader';
import { useLanguage } from '@/components/language-provider';
import { calculateRotationalCurvature, type RotationalSurface } from '@/lib/learning-rotational-curvature';
import { WavePlot, waveNumber } from './waves-plot';

const names = {
  plane: { ko: '평면', en: 'Plane' },
  sphere: { ko: '구면', en: 'Sphere' },
  hyperbolic: { ko: '쌍곡 평면', en: 'Hyperbolic plane' },
};

export default function RotationalCurvatureLab() {
  const { language } = useLanguage();
  const ko = language === 'ko';
  const [surface, setSurface] = useState<RotationalSurface>('sphere');
  const [scale, setScale] = useState(2);
  const [ratio, setRatio] = useState(.7);
  const [turn, setTurn] = useState(1);
  const result = calculateRotationalCurvature(surface, scale, ratio);
  const localAngle = result.frameAngleChange * turn;
  return <section id="experiment" className="grid min-w-0 grid-cols-1 gap-4" aria-labelledby="rotational-curvature-title">
    <h2 id="rotational-curvature-title">{ko ? '원을 따라 평행이동하기' : 'Parallel transport around a circle'}</h2>
    <label>{ko ? '곡면' : 'Surface'}
      <select className="ml-2 rounded border border-border bg-background p-1" value={surface}
        onChange={e => setSurface(e.target.value as RotationalSurface)}>
        {Object.entries(names).map(([key, label]) => <option key={key} value={key}>{label[language]}</option>)}
      </select>
    </label>
    <label>{ko ? '길이 척도' : 'Length scale'} R = {scale}
      <input className="block w-full" type="range" min={.5} max={4} step={.25} value={scale}
        onChange={e => setScale(Number(e.target.value))}/>
    </label>
    <label>{ko ? '반지름 비' : 'Radius ratio'} r/R = {ratio}
      <input className="block w-full" type="range" min={.1} max={1.5} step={.05} value={ratio}
        onChange={e => setRatio(Number(e.target.value))}/>
    </label>
    <label>{ko ? '한 바퀴 중 이동한 비율' : 'Fraction of a full loop'} = {turn}
      <input className="block w-full" type="range" min={0} max={1} step={.025} value={turn}
        onChange={e => setTurn(Number(e.target.value))}/>
    </label>
    <WavePlot marker="rotational-curvature" title={ko ? '움직이는 기저에서의 벡터 각도' : 'Vector angle in the moving basis'}
      xLabel="θ/(2π)" yLabel="α (rad)" xDomain={[0, 1]}
      series={[
        { name: names[surface][language], color: 'blue', segments: [[[0, 0], [1, result.frameAngleChange]]] },
        { name: ko ? '평면 기준' : 'Plane reference', color: 'neutral', segments: [[[0, 0], [1, -2 * Math.PI]]] },
      ]}
      dots={[{ color: 'orange', point: [turn, localAngle] }]}
      caption={ko ? '기저를 따라 기록한 각도는 한 바퀴 뒤 2π를 법으로 비교합니다. 평면 기준을 빼면 곡률에 의한 회전이 남습니다.' : 'Compare the angles modulo 2π after a full loop. Subtracting the plane reference gives the rotation due to curvature.'}/>
    <dl className="grid grid-cols-2 gap-2 tabular-nums">
      <div className="contents"><dt>Γʳ<sub>θθ</sub></dt><dd>{waveNumber(result.radialConnection)}</dd></div>
      <div className="contents"><dt>Γ<sup>θ</sup><sub>rθ</sub></dt><dd>{waveNumber(result.angularConnection)}</dd></div>
      <div className="contents"><dt>{ko ? '가우스 곡률' : 'Gaussian curvature'} K</dt><dd>{waveNumber(result.curvature)}</dd></div>
      <div className="contents"><dt>{ko ? '둘러싼 면적' : 'Enclosed area'}</dt><dd>{waveNumber(result.area)}</dd></div>
      <div className="contents"><dt>{ko ? '한 바퀴 뒤의 잔여 회전' : 'Residual rotation after a full loop'}</dt><dd>{waveNumber(result.holonomy)}</dd></div>
    </dl>
    {rotationalCurvatureGuide.experiments.map(experiment => <details key={experiment.title.en}>
      <summary>{experiment.title[language]}</summary>
      <p>{experiment.predict[language]}</p><p>{experiment.action[language]}</p><p>{experiment.explain[language]}</p>
    </details>)}
  </section>;
}

export function RotationalCurvatureConcept({ language }: { language: 'ko' | 'en' }) {
  const ko = language === 'ko';
  return <section className="ai-intuition" id="visual-intuition" aria-labelledby="rotational-concept-title">
    <h2 id="rotational-concept-title">{ko ? '움직이는 기저와 곡률' : 'A moving basis and curvature'}</h2>
    <p>{rotationalCurvatureGuide.intuition[language]}</p>
    <p className="ai-concept-equation">Γʳ<sub>θθ</sub> = −ff′,　K = −f″/f</p>
    <p>{rotationalCurvatureGuide.steps[0].explanation[language]}</p>
    <a href="#experiment">{ko ? '세 곡면에서 벡터 옮기기' : 'Transport a vector on three surfaces'}</a>
  </section>;
}
