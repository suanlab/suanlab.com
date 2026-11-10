'use client';
import { useMemo,useRef,useState,useEffect,type ReactNode } from 'react';
import type { PartitionObservation as Q } from '@/lib/learning-partition-exact-core';
import { useSceneActivity } from '@/components/motion/use-scene-activity';
import { MasterButton as Button,MasterTable as Table,MasterObservations as Observations } from './master-equation-view';
import SpinGraph,{type SpinTeachingResult} from './many-body-spin-graph';
export const spinCoordinate=(q:Q|null|undefined)=>q&&q.value!==null&&q.resolution!=='unresolved-at-precision-ceiling'&&(q.value!==0||q.sign===0)?q.value:null;
export const spinNumber=(q:Q|null|undefined)=>{const v=spinCoordinate(q);return v===null?'—':v===0?'0':Math.abs(v)<.001||Math.abs(v)>=10000?v.toExponential(3):String(Number(v.toPrecision(6)));};
function Lazy({title,children}:{title:string;children:ReactNode}){const[open,setOpen]=useState(false);return <details onToggle={e=>setOpen(e.currentTarget.open)}><summary>{title}</summary>{open&&children}</details>;}
function Certificates({result,language}:{result:SpinTeachingResult;language:'ko'|'en'}){
 const t=(ko:string,en:string)=>language==='ko'?ko:en;
 const rows=useMemo(()=>{const out:[string,Q][]=[];function walk(v:unknown,path:string){if(v&&typeof v==='object'&&'resolution'in v&&'bounds'in v){out.push([path,v as Q]);return;}if(Array.isArray(v))v.forEach((x,i)=>walk(x,`${path}[${i}]`));else if(v&&typeof v==='object')Object.entries(v).filter(([k])=>k!=='parameters').forEach(([k,x])=>walk(x,path?`${path}.${k}`:k));}walk(result,'');return out;},[result]);
 const[page,setPage]=useState(0),last=Math.max(0,Math.ceil(rows.length/20)-1),actual=Math.min(page,last);
 return <><div className="ai-lab-actions"><Button variant="outline" disabled={!actual} onClick={()=>setPage(actual-1)}>{t('이전 근거','Previous certificates')}</Button><output aria-label={t('근거 페이지','Certificate page')}>{actual+1}/{last+1}</output><Button variant="outline" disabled={actual===last} onClick={()=>setPage(actual+1)}>{t('다음 근거','Next certificates')}</Button></div><Observations title={t('모든 관측의 정확·구간 근거','Exact and interval certificates for every observation')} rows={rows.slice(actual*20,actual*20+20)} t={t}/></>;
}
export default function SpinResultView({result:r,language}:{result:SpinTeachingResult;language:'ko'|'en'}){
 const t=(ko:string,en:string)=>language==='ko'?ko:en,[site,setSite]=useState(0),[playing,setPlaying]=useState(false),ref=useRef<HTMLDivElement>(null),active=useSceneActivity(ref),selected=site%r.sites;
 useEffect(()=>{if(!playing||!active)return;const timer=setInterval(()=>setSite(v=>(v+1)%r.sites),1000);return()=>clearInterval(timer);},[playing,active,r.sites]);
 const metrics:[string,Q|null][]=[[t('원시 제곱 노름 N','Raw squared norm N'),r.rawNormSquared],[t('평균 에너지 E','Mean energy E'),r.energy],[t('두 번째 모멘트 〈H²〉','Second moment 〈H²〉'),r.secondMoment],[t('에너지 분산','Energy variance'),r.variance]];
 return <div ref={ref} data-spin-result data-spin-energy={r.energy.value} data-spin-variance={r.variance.value} data-spin-selected={selected}>
  <h3>{t('현재 입력에서 검증한 결과','Verified results for the current input')}</h3>
  <p>{t(`spin ${r.sites}개 · 복소 진폭 ${r.dimension}개 · 조밀 H 원소 ${r.cost.denseComplexMatrixEntries}개. 이 계수는 Pauli σ를 곱하며 물리적 S=ℏσ/2의 계수가 아닙니다.`,` ${r.sites} spins · ${r.dimension} complex amplitudes · ${r.cost.denseComplexMatrixEntries} dense H entries. Coefficients multiply Pauli σ, not physical S=ℏσ/2.`)}</p>
  <Observations title={t('에너지와 고유값 잔차','Energy and eigenvalue residual')} rows={metrics} t={t}/>
  <p data-spin-classification>{r.classification.exactEigenstate?t('분산0: 현재 상태는 H의 고유상태입니다.','Zero variance: the current state is an eigenstate of H.'):t('분산이 양수: 현재 상태는 H의 고유상태가 아닙니다.','Positive variance: the current state is not an eigenstate of H.')} {' '}{r.diagonalSpectrum?(r.diagonalSpectrum.selectedStateGround?t('전 기저 열거에서 바닥 공간에 속함을 확인했습니다.','Full basis enumeration certifies ground-space membership.'):t('전 기저 열거에서 바닥 상태가 아님을 확인했습니다.','Full basis enumeration certifies that this is not a ground state.')):t('비대각 H의 바닥·gap·스펙트럼은 이 계산에서 제공하지 않습니다.','Ground state, gap and spectrum of a nondiagonal H are not provided by this calculation.')}</p>
  <div className="ai-lab-actions" role="group" aria-label={t('국소 위치 살펴보기','Inspect local sites')}>
   {r.localStates.map(s=><Button key={s.site} variant="outline" aria-pressed={selected===s.site} onClick={()=>{setPlaying(false);setSite(s.site);}}>{t(`위치 ${s.site}`,`Site ${s.site}`)}</Button>)}
   <Button variant="outline" onClick={()=>{setPlaying(false);setSite(v=>(v+1)%r.sites);}}>{t('다음 위치','Next site')}</Button><Button variant="outline" disabled={!active||r.sites<2} aria-pressed={playing&&active} onClick={()=>setPlaying(v=>!v)}>{playing&&active?t('안내 재생 일시정지','Pause guided highlighting'):t('위치 순서 안내 재생','Play site highlighting')}</Button>
  </div>
  <p className="ai-small">{t('재생은 위치를 차례로 강조할 뿐 상태·Hamiltonian·물리 시간을 바꾸지 않습니다. 모션을 줄이거나 화면을 벗어나면 진행을 멈춥니다.','Playback only highlights sites; it does not change the state, Hamiltonian or physical time. It pauses with reduced motion or while off screen.')}</p>
  <SpinGraph result={r} selected={selected} language={language}/>
  <Table title={t('국소 Bloch 벡터와 순도','Local Bloch vectors and purities')} headers={[t('위치','Site'),'〈X〉','〈Y〉','〈Z〉',t('순도','Purity')]} rows={r.localStates.map(s=>[String(s.site),...s.bloch.map(spinNumber),spinNumber(s.purity)])}/>
  <Table title={t('모든 쌍·모든 축의 연결 상관','Connected correlations for every pair and axis')} headers={[t('쌍','Pair'),t('축','Axes'),t('쌍 기대값','Pair expectation'),t('국소 평균을 뺀 값','Connected value')]} rows={r.correlations.flatMap(pair=>pair.values.flatMap((row,a)=>row.map((v,b)=>[pair.sites.join('–'),`${'XYZ'[a]}${'XYZ'[b]}`,spinNumber(v.value),spinNumber(v.connected)])))}/>
  <Lazy title={t('Pauli 항별 에너지와 보존 대칭','Pauli-term energies and conserved symmetries')}>
   <p>{t('단어의 왼쪽부터 위치0,1,…이며 I는 항등 연산입니다. 계수의 부호를 그래프의 연결 유무와 혼동하지 마세요.','A word reads sites0,1,… from the left; I is identity. The sign of a coefficient differs from whether a graph edge is present.')}</p>
   <Table title={t('Pauli 항의 에너지 수지','Pauli-term energy account')} headers={[t('단어','Word'),t('계수','Coefficient'),t('기대값','Expectation'),t('에너지 기여','Energy contribution')]} rows={r.terms.map(v=>[v.word,spinNumber(v.coefficient),spinNumber(v.expectation),spinNumber(v.energyContribution)])}/>
   <Table title={t('보존과 현재 상태의 날카로움','Conservation and sharpness in the selected state')} headers={[t('관측량','Observable'),t('H와 교환','Commutes with H'),t('현재 상태의 분산','Selected-state variance')]} rows={[[ 'Σ Z',r.totalZ.conserved?t('예','Yes'):t('아니오','No'),spinNumber(r.totalZ.variance)],['⊗ Z',r.globalZParity.conserved?t('예','Yes'):t('아니오','No'),spinNumber(r.globalZParity.variance)]]}/>
  </Lazy>
  <Lazy title={t('전체 복소 진폭·Hψ·정규화 잔차','Complete complex amplitudes, Hψ and normalized residuals')}>
   <p>{t('표는 공동 상태 전체를 나타냅니다. 3D의 국소 화살표만으로 이 진폭들을 복원할 수 없습니다. 수치는 반올림 표시이며 아래 근거에서 인증 구간을 확인할 수 있습니다.','This table describes the complete joint state. Local 3D arrows do not determine these amplitudes. Numbers are rounded; certified bounds are available below.')}</p>
   <Table title={t('기저 순서와 전체 상태','Basis order and complete state')} headers={[t('기저','Basis'),'Re ψ','Im ψ','Re ψ/√N','Im ψ/√N','Re Hψ','Im Hψ','Re r/√N','Im r/√N']} rows={r.basisLabels.map((label,i)=>[`|${label}〉`,spinNumber(r.state[i].real),spinNumber(r.state[i].imaginary),spinNumber(r.normalizedState[i].real),spinNumber(r.normalizedState[i].imaginary),spinNumber(r.hamiltonianAction[i].real),spinNumber(r.hamiltonianAction[i].imaginary),spinNumber(r.normalizedResidual[i].real),spinNumber(r.normalizedResidual[i].imaginary)])}/>
  </Lazy>
  {r.diagonalSpectrum&&<Lazy title={t('대각 H의 전체 기저 에너지와 바닥 인증','All basis energies and ground certificate of diagonal H')}>
   <Observations title={t('대각 열거 인증','Diagonal enumeration certificate')} t={t} rows={[[t('바닥 에너지','Ground energy'),r.diagonalSpectrum.groundEnergy],[t('바닥 공간 확률','Ground-space probability'),r.diagonalSpectrum.groundProbability],[t('바닥 밖 누출','Leakage'),r.diagonalSpectrum.leakage],[t('들뜸 gap','Excitation gap'),r.diagonalSpectrum.excitationGap]]}/>
   <Table title={t('모든 기저의 에너지','Energy of every basis state')} headers={[t('기저','Basis'),t('에너지','Energy')]} rows={r.diagonalSpectrum.basisEnergies.map((q,i)=>[`|${r.basisLabels[i]}〉`,spinNumber(q)])}/>
   <Table title={t('합친 에너지 준위','Grouped energy levels')} headers={[t('에너지','Energy'),t('중복도','Multiplicity'),t('현재 상태 확률','Selected-state probability')]} rows={r.diagonalSpectrum.levels.map(v=>[spinNumber(v.energy),v.multiplicity,spinNumber(v.probability)])}/>
  </Lazy>}
  <Lazy title={t('전체 Hamiltonian 행렬','Complete Hamiltonian matrix')}><Table title="H" headers={[t('행/열','Row/column'),...r.basisLabels]} rows={r.hamiltonian.map((row,i)=>[r.basisLabels[i],...row.map(v=>`${spinNumber(v.real)} + (${spinNumber(v.imaginary)})i`)])}/></Lazy>
  <Lazy title={t('모든 정확·구간 수치 근거','Every exact and interval certificate')}><Certificates result={r} language={language}/></Lazy>
 </div>;
}
