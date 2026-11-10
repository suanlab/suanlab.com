'use client';
import { useEffect, useId, useRef, useState } from 'react';
import { Button } from '@/components/ui/button';
import type { QubitSuccess } from '@/lib/learning-qubit-protocol';
import { QubitTable, QubitMatrix, QubitValue, QubitStatePanel, qubitCoordinateAt, type QubitTranslate } from './qubit-lab-view';
import QubitBloch from './qubit-bloch';

export default function QubitResult({ response, language }: { response: QubitSuccess; language: 'ko' | 'en' }) {
  const t: QubitTranslate = (ko, en) => language === 'ko' ? ko : en;
  if (response.kind === 'circuit') return <CircuitResult response={response} language={language} t={t} />;
  return <div data-qubit-result={response.kind}>
    {response.kind === 'channel' && <>
      <p>{t('항등은 그대로 두고, 감쇠는 상태를 비대칭으로 바꿉니다. 전치는 단독 입력에서는 유효할 수 있지만 Bell 참조계와 함께 검사하면 실패합니다. CP·TP와 이번 출력의 유효성은 서로 다른 질문입니다.', 'Identity leaves a state unchanged; damping changes it asymmetrically. Transposition can produce a valid isolated state while failing the Bell reference test. CP, TP and validity of this particular output answer different questions.')}</p>
      <QubitTable title={t('사상의 물리 조건', 'Physical conditions of the map')} headers={[t('조건', 'Condition'), t('결과', 'Result')]} rows={Object.entries(response.result.properties).map(([key, value]) => [({ hermiticityPreserving: 'Hermiticity preserving', completelyPositive: 'CP', tracePreserving: 'TP', traceNonincreasing: 'TNI', unital: 'Unital', quantumChannel: 'CPTP' } as Record<string, string>)[key], value ? t('예', 'Yes') : t('아니오', 'No')])} />
      <p>Tr Φ(ρ) = <QubitValue key="value-1" exact={response.result.outputTrace} display={qubitCoordinateAt(response.observations.coordinates, ['outputTrace'])} t={t} /></p>
    </>}
    {response.kind === 'environment' && <><p>t = {response.input.parameter} · γ = (2t / (1 + t²))²</p><p>{t('먼저 입력·계·환경·결합 상태의 순도를 비교하세요. 결합 입력이 순수할 때만 두 축약 상태의 순도가 같습니다. 혼합 입력 예제를 선택하면 이 조건의 중요성을 확인할 수 있습니다. t=1은 이 감쇠 모델에서 계의 정보를 환경으로 모두 옮기는 끝점입니다.', 'Compare input, system, environment and joint purity first. The two reduced purities are equal when the joint state is pure. Select the mixed-input example to see why that condition matters. In this damping model, t=1 transfers the system information to the environment.')}</p></>}
    {response.kind === 'instrument' && <>
      <p>{t('결과1에 조건부 X 보정', 'Conditional X correction on outcome1')}: {response.input.correctZ ? t('켜짐', 'Enabled') : t('꺼짐', 'Disabled')}</p>
      <p>{t('결과가 알려진 조건부 상태와 결과를 잊은 평균 상태를 구별하세요. 보정을 켠 입력에서는 결과1에 X를 적용합니다. 전/후 조건부 상태와 기록–계 상태를 비교하면 기록과 후속 제어의 역할을 볼 수 있습니다. 하나의 결과에 여러 Kraus 연산자가 들어가면 조건부 상태도 혼합일 수 있습니다.', 'Distinguish a state conditioned on a known outcome from the state obtained by forgetting the outcome. With correction enabled, outcome1 receives X. Compare conditional and record–system states before and after feedback. An outcome grouping several Kraus operators can retain a mixed conditional state.')}</p>
      <QubitTable title={t('기록 확률', 'Outcome probabilities')} headers={[t('결과', 'Outcome'), t('확률', 'Probability')]} rows={response.result.branches.map((branch, i) => [String(branch.label), <QubitValue key="value-2" exact={branch.probability} display={qubitCoordinateAt(response.observations.coordinates, ['branches', i, 'probability'])} t={t} />])} />
    </>}
    {response.kind === 'deferred' && <p>{t('결맞는 기록과 탈위상된 기록의 Y 성분을 비교하세요. 계의 마지막 읽기는 같아도 기록의 결맞음은 다를 수 있습니다. 이는 지정한 측정·제어와 마지막 읽기의 동등성이며, 모든 중간 관측이 같다는 뜻은 아닙니다. 이 모델의 결합 기저는 계–기록 순서입니다.', 'Compare Y in the coherent and dephased records. The final system readout can agree while record coherence differs. Equivalence concerns the specified measurement, control and final readout, not every intermediate observable. The joint basis is ordered system–record.')}</p>}
    <QubitStatePanel key={response.kind} states={response.observations.states} language={language} t={t} />
    <details><summary>{t('연산자와 원래 행렬 확인', 'Inspect operators and original matrices')}</summary>
      {response.kind === 'channel' && <><p>{t('Choi 행렬은 참조–출력 순서이며 정규화하지 않았습니다. 채널이면 trace는2입니다. Bell 출력은 J/2이며 물리적 양성 조건을 따로 검사합니다. Φ†(I)는 TP, Φ(I)는 unital 조건을 확인합니다.', 'The Choi matrix is unnormalized and ordered reference–output. A channel has Choi trace2. Its Bell output is J/2, with positivity checked separately. Φ†(I) tests TP; Φ(I) tests unitality.')}</p><QubitMatrix title="Choi J" matrix={response.result.choi} t={t} /><QubitMatrix title="Φ†(I)" matrix={response.result.dualIdentity} t={t} /><QubitMatrix title="Φ(I)" matrix={response.result.unitalImage} t={t} /></>}
      {response.kind === 'environment' && <><p>{t('V는 계를 계–환경 공간으로 보냅니다. U는 환경을 |0〉으로 준비했을 때 V와 같은 출력을 만듭니다. 다른 환경 기저의 Kraus 표현은 채널을 바꾸지 않지만 기록 해석은 바꿉니다.', 'V embeds the system into system–environment space. U agrees with V when the environment starts in |0〉. Kraus operators from another environment basis describe the same channel but different outcome records.')}</p><QubitMatrix title="V" matrix={response.result.isometry} t={t} /><QubitMatrix title="U" matrix={response.result.unitaryCompletion} t={t} />{response.result.mixedKraus.map((matrix, i) => <QubitMatrix key={i} title={`L${i}`} matrix={matrix} t={t} />)}<p>{t('축약 출력 = Kraus 출력', 'Reduced output = Kraus output')}: {response.result.reducedEqualsKraus ? t('확인', 'Verified') : t('불일치', 'Mismatch')}</p></>}
      {response.kind === 'instrument' && response.result.branches.map(branch => <div key={branch.label}><QubitMatrix title={`E${branch.label}`} matrix={branch.effect} t={t} /><QubitMatrix title={t(`결과${branch.label}의 정규화 전 출력`, `Unnormalized output for outcome${branch.label}`)} matrix={branch.unnormalized} t={t} /></div>)}
      {response.kind === 'deferred' && <QubitMatrix title={t('기록–계 순서의 고전 상태', 'Classical state in record–system order')} matrix={response.result.recordSystem} t={t} />}
    </details>
  </div>;
}

function CircuitResult({ response, language, t }: { response: Extract<QubitSuccess, { kind: 'circuit' }>; language: 'ko' | 'en'; t: QubitTranslate }) {
  const r = response.result, [step, setStep] = useState(r.gates.length), [site, setSite] = useState(0), id = useId();
  const frame = useRef<HTMLDivElement>(null), [playing, setPlaying] = useState(false), [reducedMotion, setReducedMotion] = useState(true);
  const actualStep = Math.min(step, r.gates.length), actualSite = Math.min(site, r.sites - 1), prefix = r.prefixes[actualStep], local = prefix.singleQubits[actualSite];
  useEffect(() => {
    const media = matchMedia('(prefers-reduced-motion: reduce)');
    const change = () => { setReducedMotion(media.matches); if (media.matches) setPlaying(false); };
    const visibility = () => { if (document.hidden) setPlaying(false); };
    change(); media.addEventListener('change', change); document.addEventListener('visibilitychange', visibility);
    const observer = typeof IntersectionObserver === 'undefined' ? null : new IntersectionObserver(([entry]) => { if (!entry.isIntersecting) setPlaying(false); });
    if (frame.current) observer?.observe(frame.current);
    return () => { media.removeEventListener('change', change); document.removeEventListener('visibilitychange', visibility); observer?.disconnect(); };
  }, []);
  useEffect(() => {
    if (!playing || reducedMotion) return;
    const interval = setInterval(() => setStep(value => Math.min(value + 1, r.gates.length)), 700);
    return () => clearInterval(interval);
  }, [playing, reducedMotion, r.gates.length]);
  useEffect(() => { if (actualStep === r.gates.length) setPlaying(false); }, [actualStep, r.gates.length]);
  const coordinate = (path: (string | number)[]) => qubitCoordinateAt(response.observations.coordinates, path);
  const vector = local.bloch.map((_, axis) => coordinate(['prefixes', actualStep, 'singleQubits', actualSite, 'bloch', axis]).real.coordinate);
  return <div ref={frame} data-qubit-result="circuit" data-qubit-prefix={actualStep} data-qubit-playback={playing ? 'playing' : 'stopped'}>
    <p>{t('게이트는 적힌 순서대로 적용하며 site0이 기저 비트열의 가장 왼쪽입니다. 단계 슬라이더로 이미 계산한 중간 상태를 살펴보세요. 확률이 같더라도 위상·밀도행렬·상관은 다를 수 있습니다.', 'Gates act in written chronological order; site0 is the leftmost basis bit. The step slider inspects precomputed intermediate states. Equal probabilities can hide different phases, density matrices and correlations.')}</p>
    <label className="qubit-field" htmlFor={`${id}-step`}>{t('회로 단계', 'Circuit step')}: {actualStep} / {r.gates.length}<input id={`${id}-step`} aria-label={t('회로 단계', 'Circuit step')} type="range" min={0} max={r.gates.length} step={1} value={actualStep} onChange={e => { setPlaying(false); setStep(Number(e.target.value)); }} /></label>
    <div className="qubit-actions"><Button variant="outline" disabled={actualStep === 0} onClick={() => { setPlaying(false); setStep(actualStep - 1); }}>{t('이전 게이트', 'Previous gate')}</Button><Button variant="outline" disabled={actualStep === r.gates.length} onClick={() => { setPlaying(false); setStep(actualStep + 1); }}>{t('다음 게이트', 'Next gate')}</Button><Button variant="outline" disabled={reducedMotion || r.gates.length === 0} onClick={() => { if (playing) setPlaying(false); else { if (actualStep === r.gates.length) setStep(0); setPlaying(true); } }}>{playing ? t('단계 재생 멈춤', 'Pause steps') : t('단계 재생', 'Play steps')}</Button></div>
    <p>{t('재생은 이산 게이트 순서를 보여주며 물리적 연속 시간이나 시간에 따른 Hamiltonian을 뜻하지 않습니다. 동작 줄이기 설정에서는 이전·다음 버튼과 슬라이더로 직접 단계를 선택하세요.', 'Playback shows discrete gate order, not continuous physical time or a time-dependent Hamiltonian. With reduced motion enabled, use the previous/next buttons and slider to choose steps manually.')}</p>
    <p>{actualStep === 0 ? t('게이트 적용 전 입력', 'Input before any gate') : (() => { const gate = r.gates[actualStep - 1]; return 'control' in gate ? `${gate.gate} · site${gate.control} → site${gate.target}` : 'other' in gate ? `${gate.gate} · site${gate.target} ↔ site${gate.other}` : `${gate.gate} · site${gate.target}`; })()}</p>
    <div className="qubit-scroll" role="region" tabIndex={0} aria-label={t('기저 측정 확률 그림', 'Basis measurement probability diagram')}>
      <svg viewBox="0 0 560 260" role="img" aria-label={t('선택 단계의 기저 확률. 정확 값은 다음 표에 있습니다.', 'Basis probabilities at this step. Exact values are in the following table.')}>
        <path d="M35 20V215H550" stroke="var(--learning-neutral)" fill="none" />
        <text x={15} y={28} fill="currentColor" fontSize={18}>1</text><text x={15} y={219} fill="currentColor" fontSize={18}>0</text>
        {prefix.probabilities.map((_, i) => { const x = 45 + i * 490 / prefix.probabilities.length, width = 400 / prefix.probabilities.length, value = coordinate(['prefixes', actualStep, 'probabilities', i]).real.coordinate!; return <g key={i}><rect x={x} y={215 - 190 * value} width={width} height={190 * value} fill="var(--learning-blue)" /><text x={x + width / 2} y={245} textAnchor="middle" fill="currentColor" fontSize={18}>{i.toString(2).padStart(r.sites, '0')}</text></g>; })}
      </svg>
    </div>
    <QubitTable title={t('기저별 원시 진폭과 정규화 확률', 'Raw amplitudes and normalized basis probabilities')} headers={[t('기저', 'Basis'), t('원시 진폭', 'Raw amplitude'), t('확률', 'Probability')]} rows={prefix.probabilities.map((probability, i) => [i.toString(2).padStart(r.sites, '0'), <QubitValue key="value-3" exact={prefix.rawAmplitudes[i]} display={coordinate(['prefixes', actualStep, 'rawAmplitudes', i])} t={t} />, <QubitValue key="value-4" exact={probability} display={coordinate(['prefixes', actualStep, 'probabilities', i])} t={t} />])} />
    <p>{t('원시 진폭은 입력 노름을 유지합니다. 확률과 밀도행렬만 노름으로 정규화하며, 원시 진폭의 절댓값이1보다 클 수도 있습니다.', 'Raw amplitudes preserve the input norm. Probabilities and density matrices are normalized by that norm; a raw amplitude can exceed one in magnitude.')}</p>
    <label className="qubit-field" htmlFor={`${id}-site`}>{t('축약 상태를 볼 큐비트', 'Qubit whose reduced state to inspect')}<select id={`${id}-site`} aria-label={t('축약 상태를 볼 큐비트', 'Qubit whose reduced state to inspect')} value={actualSite} onChange={e => setSite(Number(e.target.value))}>{prefix.singleQubits.map(q => <option key={q.site} value={q.site}>site{q.site}</option>)}</select></label>
    <QubitTable title={t('선택 큐비트의 축약 관측', 'Reduced observations for the selected qubit')} headers={[t('관측량', 'Observation'), t('정확 값', 'Exact value')]} rows={[
      [t('순도', 'Purity'), <QubitValue key="value-5" exact={local.purity} display={coordinate(['prefixes', actualStep, 'singleQubits', actualSite, 'purity'])} t={t} />],
      ...local.bloch.map((value, axis) => [['X', 'Y', 'Z'][axis], <QubitValue key="value-6" exact={value} display={coordinate(['prefixes', actualStep, 'singleQubits', actualSite, 'bloch', axis])} t={t} />]),
    ]} />
    {vector.every(v => v !== null) && <QubitBloch vector={vector as number[]} language={language} scope={`site${actualSite}`} />}
    <QubitMatrix title={t('선택 큐비트의 축약 밀도행렬', 'Reduced density matrix of the selected qubit')} matrix={local.density} t={t} />
    {prefix.pauliPairs.length > 0 && <QubitTable title={t('큐비트 쌍의 상관과 연결 상관', 'Pair correlations and connected correlations')} headers={[t('쌍 · 축', 'Pair · axes'), '〈σa σb〉', '〈σa σb〉 − 〈σa〉〈σb〉']} rows={prefix.pauliPairs.flatMap((pair, k) => pair.expectations.flatMap((row, a) => row.map((value, b) => [`${pair.first},${pair.second} · ${['X', 'Y', 'Z'][a]}${['X', 'Y', 'Z'][b]}`, <QubitValue key="value-7" exact={value} display={coordinate(['prefixes', actualStep, 'pauliPairs', k, 'expectations', a, b])} t={t} />, <QubitValue key="value-8" exact={pair.connected[a][b]} display={coordinate(['prefixes', actualStep, 'pauliPairs', k, 'connected', a, b])} t={t} />])))} />}
    <details><summary>{t('선택 단계의 전체 밀도행렬과 노름', 'Full density matrix and norm at this step')}</summary><QubitMatrix title={`ρ · ${actualStep}`} matrix={prefix.density} t={t} /><p>‖ψ‖² = <QubitValue key="value-9" exact={prefix.normSquared} display={coordinate(['prefixes', actualStep, 'normSquared'])} t={t} /></p></details>
    <QubitTable title={t('전체 회로의 최종 검산 · 단계 선택과 별개', 'Final checks for the complete circuit · independent of the step selection')} headers={[t('검산', 'Check'), t('결과', 'Result')]} rows={[
      ['U†U = I', r.unitaryVerified ? t('확인', 'Verified') : t('불일치', 'Mismatch')],
      [t('입력과 같은 원시 진폭', 'Same raw amplitudes as input'), r.sameAmplitudesAsInput ? t('예', 'Yes') : t('아니오', 'No')],
      [t('입력과 같은 물리 상태', 'Same physical state as input'), r.samePhysicalStateAsInput ? t('예', 'Yes') : t('아니오', 'No')],
      ['F = |〈ψin|ψout〉|² / (‖ψin‖² ‖ψout‖²)', <QubitValue key="value-10" exact={r.fidelityWithInput} display={coordinate(['fidelityWithInput'])} t={t} />],
      [t('역회로의 원시 진폭 복원', 'Raw amplitude recovery by inverse circuit'), r.inverseRecovered ? t('확인', 'Verified') : t('불일치', 'Mismatch')],
    ]} />
    <details><summary>{t('전체 회로 U와 역회로 잔차', 'Full circuit U and inverse residual')}</summary><QubitMatrix title="U" matrix={r.unitary} t={t} /><QubitTable title={t('역회로 잔차', 'Inverse residual')} headers={[t('기저', 'Basis'), t('잔차', 'Residual')]} rows={r.inverseResidual.map((value, i) => [String(i), <QubitValue key="value-11" exact={value} t={t} />])} /></details>
  </div>;
}
