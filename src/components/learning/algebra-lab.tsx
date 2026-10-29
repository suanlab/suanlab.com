'use client';
import { useMemo } from 'react';
import { Button } from '@/components/ui/button';
import { useLanguage } from '@/components/language-provider';
import { formatNumber as number } from '@/lib/learning-simulations';
import { algebraPresets, algebraValues, buildAlgebraScene, defaultAlgebra, validAlgebra, type AlgebraKind, type AlgebraSettings } from './algebra-model';
import { useSavedState } from './use-saved-state';
import { NumberField, Tuple, VectorFields } from './lab-controls';
import LabViewport from './lab-viewport';
import GuidedExperiments from './guided-experiments';

function RegressionChart({ response, slope, ko }: { response: number[]; slope: number; ko: boolean }) {
  const x = (v: number) => 35 + v * 72; const y = (v: number) => 135 - v * 18;
  return <figure className="ai-regression-chart"><svg viewBox="0 0 300 250" role="img" aria-label={ko ? `관측값 ${response.join(', ')}, 기울기 ${number(slope)}인 원점을 지나는 회귀선. 점과 선 사이 세로 점선이 잔차입니다.` : `Observed values ${response.join(', ')}, regression through the origin with slope ${number(slope)}. Vertical dashed segments are residuals.`}>
    <path d="M35 15V235M20 135H285" fill="none" stroke="var(--learning-neutral)" />
    <path d={`M${x(0)},${y(0)}L${x(3)},${y(3 * slope)}`} stroke="var(--learning-green)" strokeWidth="2.5" fill="none" />
    {response.map((v, i) => <g key={i}><path d={`M${x(i + 1)},${y(v)}V${y((i + 1) * slope)}`} stroke="var(--learning-orange)" strokeWidth="2" strokeDasharray="4 3" /><circle cx={x(i + 1)} cy={y(v)} r="4.5" fill="var(--learning-blue)" /><text x={x(i + 1)} y="150" fill="var(--learning-neutral)" fontSize="11">{i + 1}</text></g>)}
    <text x="280" y="126" fill="var(--learning-neutral)" fontSize="12">x</text><text x="42" y="20" fill="var(--learning-neutral)" fontSize="12">y</text>
  </svg><figcaption>{ko ? '관측 평면: 가로축은 입력 x, 세로축은 응답 y입니다. 위의 3D 응답 공간과 축의 의미가 다릅니다. 선은 표시 창 바깥으로 이어질 수 있습니다.' : 'Observation plane: horizontal input x, vertical response y. These axes differ from the 3D response-space axes above. The line can extend outside the display window.'}</figcaption></figure>;
}
export default function AlgebraLab({ kind }: { kind: AlgebraKind }) {
  const { language } = useLanguage(); const ko = language === 'ko';
  const [s, set, ready, storage] = useSavedState(`suanlab-learning:${kind}:lab`, defaultAlgebra, validAlgebra);
  const values = useMemo(() => algebraValues(s), [s]); const scene = useMemo(() => buildAlgebraScene(kind, s), [kind, s]);
  const change = (patch: Partial<AlgebraSettings>) => set(current => ({ ...current, ...patch }));
  const preset = (index: number) => set({ ...defaultAlgebra, ...algebraPresets[kind][index] });
  const p = values.projection; const r = values.regression; const e = values.eigen;
  const names = kind === 'inner-products' ? (ko ? ['기준 투영', '직교', '반대 방향', '기준 영벡터'] : ['Reference projection', 'Orthogonal', 'Opposite direction', 'Zero reference'])
    : kind === 'least-squares' ? (ko ? ['기울기 0.5', '최소제곱 해', '정확한 직선', '큰 관측값'] : ['Slope 0.5', 'Least-squares solution', 'Exact line', 'Large observation'])
    : (ko ? ['고유방향', '공분산', '같은 고유값', '음의 고유값', '영행렬'] : ['Eigendirection', 'Covariance', 'Repeated eigenvalues', 'Negative eigenvalue', 'Zero matrix']);
  return <section className="ai-lab" id="experiment" aria-labelledby="ai-lab-title">
    <div className="ai-lab-heading"><div><span className="ai-eyebrow">INTERACTIVE LAB</span><h2 id="ai-lab-title">{ko ? '직접 바꿔보세요' : 'Make it your experiment'}</h2></div><span className="ai-live-dot">{ko ? '실시간 계산' : 'Live calculation'}</span></div>
    <GuidedExperiments kind={kind} ready={ready} onLoad={guide => preset(guide.preset)} />
    <fieldset disabled={!ready} className="ai-presets"><legend>{ko ? '비교할 예제' : 'Examples to compare'}</legend>{names.map((name, i) => <Button key={name} type="button" variant="outline" size="sm" onClick={() => preset(i)}>{name}</Button>)}</fieldset>
    {kind === 'eigenvalues' && <div className="ai-mode-switch" role="group" aria-label={ko ? '행렬 보기' : 'Matrix view'}><Button type="button" variant="outline" size="sm" aria-pressed={s.view === 'action'} onClick={() => change({ view: 'action' })}>{ko ? '행렬의 작용' : 'Matrix action'}</Button><Button type="button" variant="outline" size="sm" aria-pressed={s.view === 'covariance'} onClick={() => change({ view: 'covariance' })}>{ko ? '공분산 점 모형' : 'Covariance point model'}</Button></div>}
    <LabViewport scene={scene} />
    <div className="ai-legend">{kind === 'inner-products' ? <><span className="ai-blue">u</span><span className="ai-orange">v</span><span className="ai-green">{ko ? '투영 p' : 'Projection p'}</span><span>{ko ? '점선: 잔차 u−p' : 'Dashed: residual u−p'}</span></>
      : kind === 'least-squares' ? <><span className="ai-blue">{ko ? '관측 y' : 'Observed y'}</span><span className="ai-orange">x=(1,2,3)</span><span className="ai-green">{ko ? '예측 ax' : 'Prediction ax'}</span><span>{ko ? '점선: 잔차' : 'Dashed: residual'}</span></>
      : s.view === 'action' ? <><span className="ai-blue">v</span><span className="ai-orange">Av</span><span className="ai-green">{ko ? '변환된 단위원' : 'Transformed unit circle'}</span></> : <><span className="ai-blue">{ko ? '합성점 32개' : '32 synthetic points'}</span><span className="ai-green">PC₁ · √λ₁</span><span className="ai-orange">PC₂ · √λ₂</span></>}</div>
    <fieldset disabled={!ready} className="ai-parameters"><legend>{ko ? '모델 설정' : 'Model settings'}</legend>
      {kind === 'inner-products' && <><VectorFields label="u" value={s.u} change={u => change({ u })} /><VectorFields label="v" value={s.v} change={v => change({ v })} /></>}
      {kind === 'least-squares' && <><fieldset className="ai-vector-fields"><legend>{ko ? '관측 y' : 'Observed y'}</legend><div className="ai-three-fields">{s.response.map((value, i) => <NumberField key={i} label={`y${i + 1}`} value={value} min={-5} max={5} onChange={n => change({ response: s.response.map((x, j) => i === j ? n : x) as [number, number, number] })} />)}</div></fieldset>
        <p className="ai-small">{ko ? '입력은 x=(1,2,3)으로 고정합니다. y₁, y₂, y₃는 세 관측의 응답입니다.' : 'Inputs are fixed at x=(1,2,3). The fields y₁, y₂ and y₃ are the three observed responses.'}</p>
        <NumberField label={ko ? '기울기 a' : 'Slope a'} value={s.slope} min={-3} max={3} step="any" onChange={slope => change({ slope })} />
        <label className="ai-slider">{ko ? '기울기 조절' : 'Adjust slope'}: {number(s.slope)}<input type="range" min="-3" max="3" step=".001" value={s.slope} onChange={event => change({ slope: Number(event.target.value) })} /></label>
        <Button type="button" size="sm" onClick={() => change({ slope: r.optimalSlope })}>{ko ? '최소제곱 해 적용' : 'Apply least-squares solution'}</Button></>}
      {kind === 'eigenvalues' && <><p className="ai-small">A = [[a,b],[b,d]] · {ko ? '실대칭 2×2 행렬을 z=0에 표시합니다.' : 'A real symmetric 2×2 matrix displayed in z=0.'}</p><div className="ai-three-fields"><NumberField label="a = A11" value={s.a} min={0} max={4} step={.25} onChange={a => change({ a })} /><NumberField label="b = A12 = A21" value={s.b} min={-2} max={2} step={.25} onChange={b => change({ b })} /><NumberField label="d = A22" value={s.d} min={0} max={4} step={.25} onChange={d => change({ d })} /></div>
        <>{s.view === 'action' && <><label className="ai-slider">{ko ? '입력 방향' : 'Input direction'}: {number(s.angle)}°<input type="range" min="-90" max="180" step="1" value={s.angle} onChange={event => change({ angle: Number(event.target.value) })} /></label>
        <Button type="button" variant="outline" size="sm" onClick={() => change({ angle: e.angle, view: 'action' })}>{ko ? '첫 고유방향에 맞추기' : 'Align with first eigenvector'}</Button></>}</></>}
    </fieldset>
    <div className="ai-readout" data-testid="lab-readout"><h3>{ko ? '수치로 확인하기' : 'Check the numbers'}</h3><p className="ai-small">{ko ? '표시값은 반올림합니다. 계산에는 모델 값을 사용하며 매우 작은 잔여값은 부동소수점 오차일 수 있습니다.' : 'Readouts are rounded; calculations use model values. Tiny residuals can reflect floating-point roundoff.'}</p><dl>
      {kind === 'inner-products' && <><div><dt>u·v</dt><dd>{number(p.dot)}</dd></div><div><dt>cos θ</dt><dd>{p.cosine === null ? '—' : number(p.cosine)}</dd></div><div><dt>θ</dt><dd>{p.angle === null ? '—' : `${number(p.angle)}°`}</dd></div><div><dt>α = (u·v)/(v·v)</dt><dd>{p.coefficient === null ? '—' : number(p.coefficient)}</dd></div><div><dt>p = projᵥ u</dt><dd>{p.projected ? <Tuple value={p.projected} /> : '—'}</dd></div><div><dt>r = u−p</dt><dd>{p.residual ? <Tuple value={p.residual} /> : '—'}</dd></div><div><dt>v·r</dt><dd>{p.residualDot === null ? '—' : number(p.residualDot)}</dd></div></>}
      {kind === 'least-squares' && <><div><dt>{ko ? '현재 예측 ax' : 'Current prediction ax'}</dt><dd><Tuple value={r.predicted} /></dd></div><div><dt>r = y−ax</dt><dd><Tuple value={r.residual} /></dd></div><div><dt>L(a) = ½‖r‖²</dt><dd>{number(r.loss)}</dd></div><div><dt>dL/da</dt><dd>{number(r.gradient)}</dd></div><div><dt>x·r</dt><dd>{number(r.normalResidual)}</dd></div><div><dt>a*</dt><dd>{number(r.optimalSlope)}</dd></div><div><dt>L(a*)</dt><dd>{number(r.optimalLoss)}</dd></div></>}
      {kind === 'eigenvalues' && <><div><dt>λ₁, λ₂</dt><dd><Tuple value={e.values} /></dd></div><div><dt>q₁</dt><dd><Tuple value={e.first} /></dd></div>{s.view === 'action' && <><div><dt>v</dt><dd><Tuple value={values.direction} /></dd></div><div><dt>Av</dt><dd><Tuple value={values.mapped} /></dd></div><div><dt>vₓ(Av)ᵧ−vᵧ(Av)ₓ</dt><dd>{number(values.cross)}</dd></div><div><dt>vᵀAv</dt><dd>{number(values.rayleigh)}</dd></div></>}<div><dt>{ko ? '분산 비율 (A=C일 때)' : 'Variance ratio (if A=C)'}</dt><dd>{e.varianceRatio === null ? '—' : `${number(e.varianceRatio * 100)}%`}</dd></div></>}
      {kind === 'eigenvalues' && s.view === 'covariance' && values.sample && <><div><dt>{ko ? '합성점 평균' : 'Synthetic mean'}</dt><dd><Tuple value={values.sample.mean} /></dd></div><div><dt>C11, C12, C22 (1/n)</dt><dd><Tuple value={[values.sample.matrix[0][0], values.sample.matrix[0][1], values.sample.matrix[1][1]]} /></dd></div></>}
    </dl></div>
    {kind === 'inner-products' && (p.projected === null || p.angle === null) && <p className="ai-model-note" role="status">{p.projected === null ? (ko ? 'v가 영벡터이므로 투영 방향을 정의할 수 없습니다. 각도와 투영값을 계산한 것처럼 표시하지 않습니다.' : 'The reference v is zero, so its projection direction is undefined. Angle and projection are not reported as computed values.') : (ko ? 'u가 영벡터이므로 각도는 정의되지 않습니다. 영벡터의 투영은 0입니다.' : 'The angle of zero u is undefined. Its projection is zero.')}</p>}
    {kind === 'least-squares' && <><RegressionChart response={s.response} slope={s.slope} ko={ko} /><p className="ai-model-note">{ko ? '절편 없는 합성 데이터 모형 y≈ax입니다. 3D 좌표는 세 관측의 응답 성분이고, 데이터 점 (xᵢ,yᵢ)의 실제 공간 위치가 아닙니다. 입력이 0이 아니므로 이 한 계수 문제의 최소제곱 해는 유일합니다.' : 'Synthetic intercept-free model y≈ax. The 3D coordinates are response components for three observations, not physical locations of data points (xᵢ,yᵢ). Nonzero x makes the least-squares coefficient unique in this one-parameter problem.'}</p></>}
    {kind === 'eigenvalues' && <p className="ai-model-note" role="status">{!e.positiveSemidefinite ? (ko ? '음의 고유값이 있어 공분산 행렬로 해석할 수 없습니다. 공분산 보기에는 점을 만들지 않고 분산 비율도 표시하지 않습니다.' : 'A negative eigenvalue prevents a covariance interpretation. Covariance view generates no points and reports no variance ratio.') : e.varianceRatio === null ? (ko ? '총분산이 0이므로 분산 비율은 정의되지 않습니다.' : 'Total variance is zero, so variance ratios are undefined.') : e.repeated ? (ko ? '두 고유값이 같아 첫 고유방향은 유일하지 않습니다. 표시한 축은 가능한 직교 기저 하나입니다.' : 'Repeated eigenvalues do not define a unique first direction. The shown axes are one possible orthonormal basis.') : (ko ? '공분산 모형은 평균 0인 합성점 32개의 분모 n 공분산을 A에 맞춥니다. 정규분포 표본이 아닙니다. 초록 기준 타원의 반길이는 √λ이며 파랑 점들의 외곽은 그 √2배입니다. 행렬 작용의 배율 |λ|와 구분하세요.' : 'The covariance model uses 32 centered synthetic points whose denominator-n covariance matches A. They are not Gaussian samples. The green reference ellipse has half-axes √λ; blue points lie √2 times farther out. These differ from |λ| scaling in matrix-action view.')}</p>}
    <Button type="button" variant="outline" size="sm" className="ai-reset" onClick={() => set(defaultAlgebra)}>{ko ? '모든 실험 설정 초기화' : 'Reset all experiment settings'}</Button>
    <p className="ai-small">{storage ? (ko ? '실험 설정은 이 브라우저에 자동 저장됩니다.' : 'Experiment settings are saved in this browser.') : (ko ? '브라우저 저장소를 사용할 수 없어 현재 화면에서만 유지됩니다.' : 'Browser storage is unavailable. Settings remain in memory for this visit.')}</p>
  </section>;
}
