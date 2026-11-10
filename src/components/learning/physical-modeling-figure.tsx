'use client';
import IdentifiabilityObservabilityFigure from './identifiability-observability-figure';
import ModelSymmetryFigure from './model-symmetry-figure';
import ConservationConstraintFigure from './conservation-constraint-figure';
import NondimensionalFigure from './nondimensional-figure';
import ApproximationValidityFigure from './approximation-validity-figure';
import InitialBoundaryFigure from './initial-boundary-figure';
import SystemBoundaryFigure from './system-boundary-figure';
import MeasurementFigure from './measurement-figure';
import {useLanguage} from '@/components/language-provider';
import {baseDimensionSymbols,baseUnitSymbols,referenceQuantities,type PhysicalModelingResult} from '@/lib/learning-physical-modeling';
const fmt=(v:number)=>Math.abs(v)>=10000||(v!==0&&Math.abs(v)<.001)?v.toExponential(1):Number(v.toPrecision(4)).toString();
export default function PhysicalModelingFigure({result,label,axis}:{result:PhysicalModelingResult;label:string;axis:string}){
 const {language}=useLanguage(),ko=language==='ko',d=result.dimensions,u=result.units;
 if(result.forceMass||result.oscillatorProbes||result.sensorObservability)return <IdentifiabilityObservabilityFigure result={result} label={label} axis={axis}/>;
 if(result.galilean||result.rotationModel)return <ModelSymmetryFigure result={result} label={label} axis={axis}/>;
 if(result.impulseConstraints||result.circleConstraints)return <ConservationConstraintFigure result={result} label={label} axis={axis}/>;
 if(result.nondimensionalOscillator||result.buckingham)return <NondimensionalFigure result={result} label={label} axis={axis}/>;
 if(result.pendulum||result.dragRegimes)return <ApproximationValidityFigure result={result} label={label} axis={axis}/>;
 if(result.initialData||result.heatBoundary||result.poissonBoundary)return <InitialBoundaryFigure result={result} label={label} axis={axis}/>;
 if(result.balance||result.observation)return <SystemBoundaryFigure result={result} label={label} axis={axis}/>;
 if(result.measurementCalibration||result.propagation)return <MeasurementFigure result={result} label={label} axis={axis}/>;
 const [x0,x1,y0,y1]=result.bounds!,x=(v:number)=>60+255*(v-x0)/(x1-x0),y=(v:number)=>260-200*(v-y0)/(y1-y0),colors=['blue','orange','green'];
 const rows=d?[
  `${ko?'기준':'Reference'} ${d.referenceCode}: ${referenceQuantities[d.referenceCode][language]}`,
  `${ko?'후보 지수':'Candidate exponents'}: (${d.candidate.join(', ')})`,
  `${ko?'기준 지수':'Reference exponents'}: (${d.reference.join(', ')})`,
  `${ko?'몫의 지수 차이':'Quotient exponent defect'}: (${d.defect.join(', ')})`,
  `${ko?'차원 일치':'Dimension match'}: ${Number(d.dimensionsMatch)}`,
  ko?'일치해도 물리적 의미나 법칙은 별도입니다.':'A match does not prove meaning or a law.',
 ]:u?[
  `${ko?'새 단위':'New units'}: (${u.lengthUnit.label} m, ${u.massUnit.label} kg, ${u.timeUnit.label} s)`,
  `${ko?'조회 시각':'Query time'}: ${u.t} s → ${fmt(u.timePrime.value)}`,
  `${ko?'위치':'Position'}: ${fmt(u.position.value)} m → ${fmt(u.positionPrime.value)}`,
  `${ko?'힘':'Force'}: ${fmt(u.force.value)} N → ${fmt(u.forcePrime.value)}`,
  `${ko?'운동에너지':'Kinetic energy'}: ${fmt(u.energy.value)} J → ${fmt(u.energyPrime.value)}`,
  `${ko?'정확 복원 x,v,F,E':'Exact reconstruction x,v,F,E'}: ${[u.exactPosition,u.exactVelocity,u.exactForce,u.exactEnergy].map(Number).join(', ')}`,
 ]:[];
 return <div className="ai-game-figure"><figure className="ai-foundation-figure"><div className="ai-foundation-scroll" tabIndex={0} role="region" aria-label={ko?'차원 지수와 단위 변환 그래프 · 가로 이동 가능':'Dimension exponents and unit conversion plot · Horizontal scrolling available'}><svg data-physical-modeling-plot viewBox="0 0 350 560" role="img" aria-label={label}>
 <path d="M60 60V260H315" fill="none" stroke="var(--learning-neutral)"/>
 {d?d.candidate.map((v,i)=><g key={i}><path d={`M${x(i)-4} ${y(0)}V${y(v)}`} stroke="var(--learning-blue)" strokeWidth="5"/><path d={`M${x(i)+4} ${y(0)}V${y(d.reference[i])}`} stroke="var(--learning-orange)" strokeWidth="5"/><circle cx={x(i)-4} cy={y(v)} r="3" fill="var(--learning-blue)"/><circle cx={x(i)+4} cy={y(d.reference[i])} r="3" fill="var(--learning-orange)"/><text x={x(i)} y="282" textAnchor="middle" fontSize="10" fill="currentColor">{baseDimensionSymbols[i]}</text></g>):result.curves.map((c,i)=><path key={i} d={c.map(([a,b],j)=>`${j?'L':'M'}${x(a)} ${y(b)}`).join(' ')} fill="none" stroke={`var(--learning-${colors[i]})`} strokeWidth={i===0?3:2} strokeDasharray={i===1?'5 4':i===2?'2 4':undefined}/>)}
 <g fill="currentColor" fontSize="9" textAnchor="end">{[y0,(y0+y1)/2,y1].map(v=><text key={v} style={{fontSize:9}} x="53" y={y(v)+3}>{fmt(v)}</text>)}</g>
 {u&&<><g fill="currentColor" fontSize="10"><text x="60" y="282">0</text><text x="315" y="282" textAnchor="end">5 s</text></g><circle cx={x(u.t)} cy={y(u.position.value)} r="4" fill="var(--learning-blue)"/><circle cx={x(u.t)} cy={y(u.wrongPosition.value)} r="4" fill="var(--learning-green)"/></>}
 {rows.map((v,i)=><text key={i} x="175" y={325+34*i} fill="currentColor" fontSize="10" textAnchor="middle">{v}</text>)}
 <text x="175" y="540" fill="currentColor" fontSize="10" textAnchor="middle">{d?(ko?'파랑 후보 · 주황 기준':'Blue candidate · orange reference'):(ko?'파랑 기준 · 주황 변환 · 초록 숫자 유지':'Blue reference · orange coherent · green retained')}</text>
 </svg></div><figcaption>{axis}</figcaption></figure>
 {d&&<div className="ai-model-note"><p>{ko?'기본 차원과 SI 기본 단위':'Base dimensions and SI base units'}: {baseDimensionSymbols.map((s,i)=>`${s}: ${baseUnitSymbols[i]}`).join(' · ')}</p><p>{ko?'기준 물리량 코드':'Reference quantity codes'}: {referenceQuantities.map((q,i)=>`${i} ${q[language]}`).join(' · ')}</p><p>{ko?'에너지와 토크는 같은 지수를 가지지만 서로 다른 물리량입니다. 차원 1인 각도와 확률도 같은 의미가 아닙니다.':'Energy and torque share exponents while remaining different quantities. Dimension-one angles and probabilities also have different meanings.'}</p></div>}
 <div className="ai-discrete-table" tabIndex={0} role="region" aria-label={ko?'차원·단위 계산 표 · 가로 이동 가능':'Dimension and unit calculation table · Horizontal scrolling available'}><table><caption>{ko?'현재 설정의 계산 결과':'Calculation results for current settings'}</caption><thead><tr>{result.table.headers.map(h=><th key={h} scope="col">{h}</th>)}</tr></thead><tbody>{result.table.rows.map((row,i)=><tr key={i}>{row.map((v,j)=><td key={j}>{typeof v==='number'?fmt(v):v}</td>)}</tr>)}</tbody></table></div>
 </div>;
}
