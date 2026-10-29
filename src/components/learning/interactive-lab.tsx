'use client';
import { Button } from '@/components/ui/button';
import { useEffect, useMemo, useRef, useState } from 'react';
import { useLanguage } from '@/components/language-provider';
import { useSceneActivity } from '@/components/motion/use-scene-activity';
import { useSiteMotion } from '@/components/motion/motion-provider';
import { formatNumber as number, quadratic, type Mat3, type Vec2, type Vec3 } from '@/lib/learning-simulations';
import { buildLabScene, defaultLab, labValues, matrixPresets, validLab, type LabKind, type LabSettings } from './lab-model';
import { useSavedState } from './use-saved-state';
import LabViewport from './lab-viewport';
import GuidedExperiments from './guided-experiments';
import { NumberField, VectorFields, Tuple } from './lab-controls';

function LossChart({ values, ko }: { values: number[]; ko: boolean }) {
  const max = Math.max(1e-9, ...values); const width = 300; const height = 100;
  const points = values.map((v, i) => `${10 + i / Math.max(1, values.length - 1) * (width - 20)},${height - 15 - v / max * 70}`).join(' ');
  return <figure className="ai-loss-chart"><svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={ko ? `반복별 손실: ${values.map(number).join(', ')}` : `Loss per step: ${values.map(number).join(', ')}`}>
    <path d="M10 10V85H290" fill="none" stroke="var(--learning-neutral)" /><polyline points={points} fill="none" stroke="var(--learning-orange)" strokeWidth="2.5" />
    <circle cx={points.split(' ').at(-1)?.split(',')[0]} cy={points.split(' ').at(-1)?.split(',')[1]} r="3.5" fill="var(--learning-orange)" />
    <text x="14" y="13" fontSize="10" fill="hsl(var(--muted-foreground))">L: {number(max)}</text><text x="250" y="98" fontSize="10" fill="hsl(var(--muted-foreground))">k = {values.length - 1}</text>
  </svg><figcaption>{ko ? '반복별 손실 · 세로축은 현재 최댓값에 맞춥니다.' : 'Loss per step · The vertical scale follows the current maximum.'}</figcaption></figure>;
}
export default function InteractiveLab({ kind }: { kind: LabKind }) {
  const { language } = useLanguage(); const ko = language === 'ko'; const motion = useSiteMotion();
  const [s, set, ready, storage] = useSavedState(`suanlab-learning:${kind}:lab`, defaultLab, validLab);
  const [playing, setPlaying] = useState(false); const panel = useRef<HTMLDivElement>(null); const active = useSceneActivity(panel);
  const values = useMemo(() => labValues(s), [s]); const scene = useMemo(() => buildLabScene(kind, s), [kind, s]);
  function change(patch: Partial<LabSettings>, reset = true) {
    setPlaying(false); set(current => ({ ...current, ...(reset ? { time: 0, steps: 0 } : {}), ...patch }));
  }
  useEffect(() => { if (!active) setPlaying(false); }, [active]);
  useEffect(() => {
    if (!playing || !active) return;
    const timer = setInterval(() => set(current => kind === 'gradients'
      ? { ...current, steps: Math.min(50, current.steps + 1) }
      : { ...current, time: Math.min(5, Math.round((current.time + .1) * 10) / 10) }), kind === 'gradients' ? 700 : 100);
    return () => clearInterval(timer);
  }, [playing, active, kind, set]);
  const ended = kind === 'gradients' ? s.steps >= 50 || values.stopped || values.objective.loss < 1e-12 : s.time >= 5;
  useEffect(() => { if (ended) setPlaying(false); }, [ended]);
  const labels = kind === 'vectors-and-forces'
    ? (ko ? ['3–4–5 합력', '반대 힘 · 정지', '반대 힘 · 이동', '같은 방향', '음의 z 방향'] : ['3–4–5 resultant', 'Opposing · at rest', 'Opposing · moving', 'Same direction', 'Negative z'])
    : kind === 'linear-transforms'
      ? (ko ? ['항등', '비등방 확대', '90° 회전', '전단', '반사', '차원 축소'] : ['Identity', 'Anisotropic scale', '90° rotation', 'Shear', 'Reflection', 'Collapse'])
      : (ko ? ['안정 · 0.25', '왕복 수렴 · 0.75', '경계 · 1', '발산 · 1.25', '등방 곡면', '최솟값에서 시작'] : ['Stable · 0.25', 'Alternating · 0.75', 'Boundary · 1', 'Diverging · 1.25', 'Isotropic surface', 'Start at minimum']);
  function preset(index: number) {
    if (kind === 'vectors-and-forces') {
      const forces: [Vec3, Vec3][] = [[[3, 0, 0], [0, 4, 0]], [[3, 0, 0], [-3, 0, 0]], [[3, 0, 0], [-3, 0, 0]], [[3, 0, 0], [3, 0, 0]], [[0, 0, -2], [0, 0, 0]]];
      change({ force1: forces[index][0], force2: forces[index][1], velocity0: index === 2 ? [1, 0, 0] : [0, 0, 0], mass: index === 4 ? .5 : 2, mode: index === 2 ? 'motion' : 'forces' });
    } else if (kind === 'linear-transforms') change({ matrix: matrixPresets[index], point: index === 2 ? [1, 0, 0] : index === 3 ? [1, 2, 0] : [1, 2, 3] });
    else change({ a: index === 4 ? 1 : 2, b: 1, start: index === 5 ? [0, 0] : [2, 2], rate: [.25, .75, 1, 1.25, .25, .25][index] });
  }
  return <section ref={panel} className="ai-lab" id="experiment" aria-labelledby="ai-lab-title">
    <div className="ai-lab-heading"><div><span className="ai-eyebrow">INTERACTIVE LAB</span><h2 id="ai-lab-title">{ko ? '직접 바꿔보세요' : 'Make it your experiment'}</h2></div><span className="ai-live-dot">{ko ? '실시간 계산' : 'Live calculation'}</span></div>
    <GuidedExperiments kind={kind} ready={ready} onLoad={guide => {
      preset(guide.preset);
      if (guide.time !== undefined) change({ time: guide.time, mode: 'motion' }, false);
      if (guide.steps !== undefined) change({ steps: guide.steps }, false);
    }} />
    <fieldset disabled={!ready} className="ai-presets"><legend>{ko ? '비교할 예제' : 'Examples to compare'}</legend>{labels.map((label, i) => <Button variant="outline" size="sm" key={label} type="button" onClick={() => preset(i)}>{label}</Button>)}</fieldset>
    {kind === 'vectors-and-forces' && <div className="ai-mode-switch" role="group" aria-label={ko ? '그림의 물리량' : 'Physical quantity in view'}>
      <Button variant="outline" size="sm" type="button" aria-pressed={s.mode === 'forces'} onClick={() => change({ mode: 'forces' }, false)}>{ko ? '힘의 합 · N' : 'Force addition · N'}</Button>
      <Button variant="outline" size="sm" type="button" aria-pressed={s.mode === 'motion'} onClick={() => change({ mode: 'motion' }, false)}>{ko ? '입자 운동 · m' : 'Particle motion · m'}</Button>
    </div>}
    <LabViewport scene={scene} />
    <div className="ai-legend">{kind === 'vectors-and-forces' ? (s.mode === 'forces' ? <><span className="ai-blue">F₁</span><span className="ai-orange">F₂</span><span className="ai-green">{ko ? '합력' : 'Resultant'}</span><span>{ko ? '점선: F₂의 평행 이동' : 'Dashed: translated F₂'}</span></> : <><span className="ai-green">{ko ? '0–5 s 경로' : '0–5 s trajectory'}</span><span className="ai-orange">{ko ? '현재 위치' : 'Current position'}</span></>)
      : kind === 'linear-transforms' ? <><span>{ko ? '회색 점선: 원래 단위 정육면체' : 'Gray dashed: original unit cube'}</span><span className="ai-green">{ko ? '실선: 변환된 정육면체' : 'Solid: transformed cube'}</span><span className="ai-orange">Ax</span><span>{ko ? '기저: 파랑 e₁ · 주황 e₂ · 초록 e₃' : 'Basis: blue e₁ · orange e₂ · green e₃'}</span></>
      : <><span className="ai-green">L(x,y)</span><span className="ai-orange">{ko ? '갱신 경로' : 'Update path'}</span><span className="ai-blue">−∇L {ko ? '(방향만 표시)' : '(direction only)'}</span></>}</div>
    <fieldset disabled={!ready} className="ai-parameters"><legend>{ko ? '모델 설정' : 'Model settings'}</legend>
      {kind === 'vectors-and-forces' && <><VectorFields label="F₁ (N)" value={s.force1} change={force1 => change({ force1 })} /><VectorFields label="F₂ (N)" value={s.force2} change={force2 => change({ force2 })} />
        <div className="ai-two-fields"><NumberField label={ko ? '질량 (kg)' : 'Mass (kg)'} value={s.mass} min={.5} max={5} onChange={mass => change({ mass })} /><span className="ai-small">{ko ? '힘·질량·초기 속도를 바꾸면 시간을 0으로 돌립니다.' : 'Changing forces, mass or initial velocity resets time to zero.'}</span></div>
        <VectorFields label={ko ? '초기 속도 (m/s)' : 'Initial velocity (m/s)'} value={s.velocity0} change={velocity0 => change({ velocity0 })} />
        <label className="ai-slider">{ko ? '시간' : 'Time'}: {number(s.time)} s<input type="range" min="0" max="5" step=".1" value={s.time} onChange={e => change({ time: Number(e.target.value), mode: 'motion' }, false)} /></label></>}
      {kind === 'linear-transforms' && <><fieldset className="ai-matrix"><legend>A · {ko ? '행을 기준으로 입력' : 'Enter by row'}</legend>{s.matrix.map((row, i) => <div key={i} className="ai-three-fields">{row.map((value, j) => <NumberField key={j} label={`A${i + 1}${j + 1}`} value={value} min={-3} max={3} onChange={n => change({ matrix: s.matrix.map((r, ri) => r.map((x, ci) => ri === i && ci === j ? n : x)) as Mat3 })} />)}</div>)}</fieldset>
        <VectorFields label={ko ? '입력 벡터 x' : 'Input vector x'} value={s.point} min={-3} max={3} change={point => change({ point })} /></>}
      {kind === 'gradients' && <><div className="ai-two-fields"><NumberField label="a" value={s.a} min={.5} max={5} onChange={a => change({ a })} /><NumberField label="b" value={s.b} min={.5} max={5} onChange={b => change({ b })} /></div>
        <div className="ai-two-fields">{s.start.map((v, i) => <NumberField key={i} label={`${ko ? '시작' : 'Start'} ${['x', 'y'][i]}`} value={v} min={-3} max={3} onChange={n => change({ start: s.start.map((x, j) => i === j ? n : x) as Vec2 })} />)}</div>
        <label className="ai-slider">{ko ? '학습률' : 'Learning rate'} η = {number(s.rate)}<input type="range" min=".05" max="2" step=".05" value={s.rate} onChange={e => change({ rate: Number(e.target.value) })} /></label>
        <p className="ai-small">{ko ? '이 이차함수에서 모든 시작점의 수렴 조건' : 'Convergence from every starting point for this quadratic'}: 0 &lt; η &lt; {number(values.limit)}.</p></>}
    </fieldset>
    {kind !== 'linear-transforms' && <div className="ai-playback">
      <Button size="sm" type="button" disabled={!ready || !active || ended} onClick={() => { if (kind === 'vectors-and-forces') set(current => ({ ...current, mode: 'motion' })); setPlaying(!playing); }}>{playing ? (ko ? '일시정지' : 'Pause') : (ko ? '재생' : 'Play')}</Button>
      <Button variant="outline" size="sm" type="button" disabled={!ready || ended} onClick={() => change(kind === 'gradients' ? { steps: Math.min(50, s.steps + 1) } : { time: Math.min(5, Math.round((s.time + .1) * 10) / 10), mode: 'motion' }, false)}>{kind === 'gradients' ? (ko ? '한 단계' : 'One step') : '+0.1 s'}</Button>
      <Button variant="outline" size="sm" type="button" onClick={() => change({ time: 0, steps: 0 }, false)}>{ko ? '처음으로' : 'Restart'}</Button>
      {(!motion.canAnimate) && <p className="ai-small">{ko ? '모션 설정에 따라 자동 재생을 멈췄습니다. 한 단계씩 실행할 수 있습니다.' : 'Automatic playback is paused by motion preferences. You can still step manually.'}</p>}
    </div>}
    <div className="ai-readout" data-testid="lab-readout"><h3>{ko ? '수치로 확인하기' : 'Check the numbers'}</h3><p className="ai-small">{ko ? '화면값은 반올림해 표시합니다. 계산에는 반올림 전 모델 값을 사용합니다.' : 'Readouts are rounded. Calculations use the unrounded model values.'}</p><dl>
      {kind === 'vectors-and-forces' && <><div><dt>Fnet (N)</dt><dd><Tuple value={values.particle.force} /></dd></div><div><dt>|Fnet| (N)</dt><dd>{number(values.particle.magnitude)}</dd></div><div><dt>a (m/s²)</dt><dd><Tuple value={values.particle.acceleration} /></dd></div><div><dt>v (m/s)</dt><dd><Tuple value={values.particle.velocity} /></dd></div><div><dt>p (m)</dt><dd><Tuple value={values.particle.position} /></dd></div><div><dt>{ko ? '운동 에너지 (J)' : 'Kinetic energy (J)'}</dt><dd>{number(values.particle.kineticEnergy)}</dd></div><div><dt>{ko ? '힘이 한 일 (J)' : 'Work by forces (J)'}</dt><dd>{number(values.particle.work)}</dd></div></>}
      {kind === 'linear-transforms' && <><div><dt>Ax</dt><dd><Tuple value={values.transformed} /></dd></div><div><dt>det(A)</dt><dd>{number(values.determinant)}</dd></div><div><dt>{ko ? '부피 배율 |det(A)|' : 'Volume scale |det(A)|'}</dt><dd>{number(Math.abs(values.determinant))}</dd></div><div><dt>{ko ? '역행렬' : 'Inverse'}</dt><dd>{values.determinant === 0 ? (ko ? '없음 · 차원 축소' : 'None · dimensional collapse') : (ko ? '존재' : 'Exists')}</dd></div></>}
      {kind === 'gradients' && <><div><dt>{ko ? '반복 k' : 'Step k'}</dt><dd>{values.path.length - 1}</dd></div><div><dt>pₖ</dt><dd><Tuple value={values.current} /></dd></div><div><dt>∇L(pₖ)</dt><dd><Tuple value={values.objective.gradient} /></dd></div><div><dt>L(pₖ)</dt><dd>{number(values.objective.loss)}</dd></div></>}
    </dl></div>
    {kind === 'gradients' && <><LossChart values={values.path.map(p => quadratic(s.a, s.b, p).loss)} ko={ko} />{(ended || s.rate >= values.limit) && <p className="ai-model-note" role="status">{values.stopped ? (ko ? '좌표 절댓값이 1,000을 넘어 실험을 멈췄습니다. 값은 그대로 표시하며, 학습률을 낮추면 다시 시작합니다.' : 'Stopped after a coordinate exceeded 1,000. The actual values remain visible. Lower the learning rate to restart.') : values.objective.loss < 1e-12 ? (ko ? '손실이 10⁻¹²보다 작아 자동 실행을 멈췄습니다.' : 'Stopped automatically because loss is below 10⁻¹².') : s.steps >= 50 ? (ko ? '검토를 위한 50회 반복에 도달했습니다. 처음부터 다시 실행할 수 있습니다.' : 'Reached the 50-step preview limit. Restart to run again.') : (ko ? '현재 학습률은 전체 시작점의 수렴을 보장하지 않습니다. 경계에서는 왕복하고, 경계를 넘으면 일부 성분이 커집니다.' : 'This rate does not ensure convergence from every starting point. At the boundary, a component can oscillate; above it, a component can grow.')}</p>}</>}
    <p className="ai-model-note">{kind === 'vectors-and-forces' ? (ko ? `모델: 관성계, 일정한 힘과 질량, 중력·마찰·충돌 없음. 힘 그림은 N, 운동 그림은 m 단위이며 화살표 길이는 입자의 이동 거리가 아닙니다. 현재 격자 단위: ${s.mode === 'forces' ? 'N' : 'm'}.` : `Model: an inertial frame, constant forces and mass, no gravity, friction or collisions. Force views use N; motion views use m. Arrow length is not displacement. Current grid unit: ${s.mode === 'forces' ? 'N' : 'm'}.`)
      : kind === 'linear-transforms' ? (ko ? '열벡터 y=Ax 규약입니다. 원래 점은 회색, 변환된 점은 주황색입니다. 이동과 비선형 활성화는 포함하지 않습니다.' : 'Column-vector convention: y=Ax. The original point is gray; the transformed point is orange. Translation and nonlinear activation are not included.')
      : (ko ? '모델: L=(ax²+by²)/2, a,b>0인 무차원 이차함수. 물리적으로 굴러가는 공이 아니라 이산 경사하강법입니다. 축척을 바꿔도 실제 계산값은 바뀌지 않습니다.' : 'Model: the dimensionless positive quadratic L=(ax²+by²)/2. This is discrete gradient descent, not a physically rolling ball. Visual scaling never changes the calculations.')}</p>
    <Button variant="outline" size="sm" type="button" className="ai-reset" onClick={() => { setPlaying(false); set(defaultLab); }}>{ko ? '모든 실험 설정 초기화' : 'Reset all experiment settings'}</Button>
    <p className="ai-small">{storage ? (ko ? '실험 설정은 이 브라우저에 자동 저장됩니다.' : 'Experiment settings are saved in this browser.') : (ko ? '브라우저 저장소를 사용할 수 없어 현재 화면에서만 유지됩니다.' : 'Browser storage is unavailable. Settings remain in memory for this visit.')}</p>
  </section>;
}
