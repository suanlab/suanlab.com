'use client';
import{isContinuumConservativeKind}from '@/lib/learning-continuum-conservative-lab';
import{isContinuumTurbulenceKind}from '@/lib/learning-continuum-turbulence-lab';
import{isContinuumMultiphysicsKind}from '@/lib/learning-continuum-multiphysics-lab';
import{isContinuumDimensionlessKind}from '@/lib/learning-continuum-dimensionless-lab';
import{isContinuumConstitutiveKind}from '@/lib/learning-continuum-constitutive-lab';
import {isContinuumReactionKind}from '@/lib/learning-continuum-reaction-transport-lab';
import {continuumDiffusionKinds,isContinuumDiffusionKind}from '@/lib/learning-continuum-diffusion-lab';
import {continuumFlowKinds,isContinuumFlowKind}from '@/lib/learning-continuum-euler-navier-stokes-lab';
import {continuumStressStrainKinds,isContinuumStressStrainKind}from '@/lib/learning-continuum-stress-strain-lab';
import {continuumAssumptionKinds,isContinuumAssumptionKind}from '@/lib/learning-continuum-assumptions-lab';
import {isContinuumKind,evaluateContinuum,continuumKinds,type ContinuumKind,type ContinuumResult}from '@/lib/learning-continuum';
import ContinuumFigure from './continuum-figure';
import {isElectromagnetismKind,evaluateElectromagnetism,type ElectromagnetismKind,type ElectromagnetismResult}from '@/lib/learning-electromagnetism';
import ElectromagnetismFigure from './electromagnetism-figure';
import ElectroMagnetostaticsFigure from './electro-magnetostatics-figure';
import ElectromagneticEffectiveFigure from './electromagnetic-effective-models-figure';
import {electromagneticEffectiveKinds}from '@/lib/learning-electromagnetic-effective-models-lab';
import CircuitSensorsFigure from './circuit-sensors-figure';
import {circuitSensorsKinds}from '@/lib/learning-circuit-sensors-lab';
import GaugeFreedomFigure from './gauge-freedom-figure';
import {gaugeFreedomKinds} from '@/lib/learning-gauge-freedom-lab';
import MaterialInterfacesFigure from './material-interfaces-figure';
import {materialInterfacesKinds} from '@/lib/learning-material-interfaces-lab';
import PoyntingEnergyFigure from './poynting-energy-figure';
import {poyntingEnergyKinds} from '@/lib/learning-poynting-energy-lab';
import InductionWavesFigure from './induction-waves-figure';
import {inductionWavesKinds} from '@/lib/learning-induction-waves-lab';
import MaxwellFigure from './maxwell-figure';
import {maxwellKinds} from '@/lib/learning-maxwell-lab';
import PotentialsFigure from './potentials-figure';
import {potentialsKinds} from '@/lib/learning-potentials-lab';
import {electroMagnetostaticsKinds} from '@/lib/learning-electro-magnetostatics-lab';
import {isWavesKind,evaluateWaves,type WavesKind,type WavesResult} from '@/lib/learning-waves';
import WavesFigure from './waves-figure';
import {isAnalyticalMechanicsKind,evaluateAnalyticalMechanics,type AnalyticalMechanicsKind,type AnalyticalMechanicsResult}from '@/lib/learning-analytical-mechanics';
import AnalyticalMechanicsFigure from './analytical-mechanics-figure';
import{isClassicalMechanicsKind,evaluateClassicalMechanics,type ClassicalMechanicsKind,type ClassicalMechanicsResult}from '@/lib/learning-classical-mechanics';
import ClassicalMechanicsFigure from './classical-mechanics-figure';
import {isPhysicalModelingKind,evaluatePhysicalModeling,type PhysicalModelingKind,type PhysicalModelingResult} from '@/lib/learning-physical-modeling';
import PhysicalModelingFigure from './physical-modeling-figure';
import {isPrivacyKind,evaluatePrivacy,type PrivacyKind,type PrivacyResult} from '@/lib/learning-privacy';
import PrivacyFigure from './privacy-figure';
import {isGameKind,evaluateGame,type GameKind,type GameResult} from '@/lib/learning-games';
import GameFigure from './game-figure';
import {isCombinatorialKind,evaluateCombinatorial,type CombinatorialKind,type CombinatorialResult} from '@/lib/learning-combinatorial';
import CombinatorialFigure from './combinatorial-figure';
import { isControlKind, evaluateControl, type ControlKind, type ControlResult } from '@/lib/learning-control';
import { isCausalKind, evaluateCausal, type CausalKind, type CausalResult } from '@/lib/learning-causality';
import { isTheoryKind, evaluateTheory, type TheoryKind } from '@/lib/learning-theory';
import { isAlgorithmKind, evaluateAlgorithm, type AlgorithmKind } from '@/lib/learning-algorithms';
import { isScientificKind, evaluateScientific, type ScientificKind } from '@/lib/learning-scientific';
import { isStochasticKind, evaluateStochastic, type StochasticKind } from '@/lib/learning-stochastic';
import { isDifferentialKind, evaluateDifferential, type DifferentialKind } from '@/lib/learning-differential';
import { isNumericalKind, evaluateNumerical, type NumericalKind } from '@/lib/learning-numerical';
import { isTransportKind, evaluateTransport, type TransportKind } from '@/lib/learning-transport';
import { isVariationalKind, evaluateVariational, type VariationalKind } from '@/lib/learning-variational';
import { isOptimizationKind, evaluateOptimization, type OptimizationKind } from '@/lib/learning-optimization';
import { isConvexKind, evaluateConvex, type ConvexKind } from '@/lib/learning-convex';
import { isSymmetryKind, evaluateSymmetry, type SymmetryKind } from '@/lib/learning-symmetry';
import { isGeometryKind, evaluateGeometry, type GeometryKind } from '@/lib/learning-geometry';
import { isHarmonicKind, evaluateHarmonic, type HarmonicKind } from '@/lib/learning-harmonic';
import { isApproximationKind, evaluateApproximation, type ApproximationKind } from '@/lib/learning-approximation';
import { isFunctionalKind, evaluateFunctional, type FunctionalKind } from '@/lib/learning-functional';
import { isCodingKind, evaluateCoding, type CodingKind } from '@/lib/learning-coding';
import { isGraphicalKind, evaluateGraphical, type GraphicalKind, type GraphicalResult } from '@/lib/learning-graphical';
import { isHighDimensionalKind, evaluateHighDimensional, type HighDimensionalKind } from '@/lib/learning-high-dimensional';
import { isInformationKind, evaluateInformation, type InformationKind } from '@/lib/learning-information';
import { isBayesKind, evaluateBayes, type BayesKind } from '@/lib/learning-bayes';
import { isStatisticsKind, evaluateStatistics, type StatisticsKind } from '@/lib/learning-statistics';

import { isMeasureKind, evaluateMeasure, type MeasureKind } from '@/lib/learning-measure';

import { isProbabilityKind, evaluateProbability, type ProbabilityKind } from '@/lib/learning-probability';

import { isAbstractAlgebraKind, evaluateAbstractAlgebra, type AbstractAlgebraKind } from '@/lib/learning-abstract-algebra';
import { useId, useMemo } from 'react';
import { useLanguage } from '@/components/language-provider';
import { Button } from '@/components/ui/button';
import { continuumLabs, electromagnetismLabs, wavesLabs, analyticalMechanicsLabs, classicalMechanicsLabs, physicalModelingLabs, privacyLabs, gameLabs, combinatorialLabs, controlLabs, causalLabs, theoryLabs, algorithmLabs, scientificLabs, stochasticLabs, differentialLabs, numericalLabs, transportLabs, variationalLabs, optimizationLabs, convexLabs, symmetryLabs, geometryLabs, harmonicLabs, approximationLabs, functionalLabs, codingLabs, graphicalLabs, highDimensionalLabs, informationLabs, bayesLabs, statisticsLabs, measureLabs, probabilityLabs, abstractAlgebraLabs, tensorLabs, complexLabs, analysisLabs, matrixStructureLabs, discreteLabs, elementaryLabs, foundationLabs, learningUnits } from '@/data/interactive-book';
import { evaluateFoundation, type FoundationKind, type FoundationResult } from '@/lib/learning-foundations';
import { evaluateElementary, isElementaryKind, type ElementaryKind } from '@/lib/learning-elementary';
import { formatNumber } from '@/lib/learning-simulations';
import { useSavedState } from './use-saved-state';
import { NumberField } from './lab-controls';
import GuidedExperiments from './guided-experiments';
import { evaluateDiscrete, isDiscreteKind, type DiscreteKind } from '@/lib/learning-discrete';
import DiscreteFigure from './discrete-figure';
import { evaluateMatrixStructure, isMatrixStructureKind, type MatrixStructureKind } from '@/lib/learning-matrix-structures';
import { evaluateAnalysis, isAnalysisKind, type AnalysisKind } from '@/lib/learning-analysis';
import { evaluateComplex, isComplexKind, type ComplexKind } from '@/lib/learning-complex';
import { evaluateTensor, isTensorKind, type TensorKind } from '@/lib/learning-tensors';
import TensorFigure from './tensor-figure';
import ProbabilityFigure from './probability-figure';
import MeasureFigure from './measure-figure';
import GraphicalFigure from './graphical-figure';
import GeometryFigure from './geometry-figure';
import SymmetryFigure from './symmetry-figure';
import ConvexFigure from './convex-figure';
import OptimizationFigure from './optimization-figure';
import VariationalFigure from './variational-figure';
import TransportFigure from './transport-figure';
import NumericalFigure from './numerical-figure';
import DifferentialFigure from './differential-figure';
import TheoryFigure from './theory-figure';
import CausalFigure from './causal-figure';
import ControlFigure from './control-figure';
import type {TheoryResult} from '@/lib/learning-theory';
import AlgorithmFigure,{AlgorithmBenchmark} from './algorithm-figure';
import type {AlgorithmResult} from '@/lib/learning-algorithms';
import ScientificFigure from './scientific-figure';
import type { ScientificResult } from '@/lib/learning-scientific';
import StochasticFigure from './stochastic-figure';
import type { StochasticResult } from '@/lib/learning-stochastic';
import type { DifferentialResult } from '@/lib/learning-differential';
import type { NumericalResult } from '@/lib/learning-numerical';
import type { TransportResult } from '@/lib/learning-transport';
import type { VariationalResult } from '@/lib/learning-variational';
import type { OptimizationResult } from '@/lib/learning-optimization';
import type { ConvexResult } from '@/lib/learning-convex';
import type { SymmetryResult } from '@/lib/learning-symmetry';
import type { GeometryResult } from '@/lib/learning-geometry';
type PlotKind=ContinuumKind|ElectromagnetismKind|WavesKind|AnalyticalMechanicsKind|ClassicalMechanicsKind|PhysicalModelingKind|PrivacyKind|GameKind|CombinatorialKind|ControlKind|CausalKind|FoundationKind|ElementaryKind|DiscreteKind|MatrixStructureKind|AnalysisKind|ComplexKind|TensorKind|AbstractAlgebraKind|ProbabilityKind|MeasureKind|StatisticsKind|BayesKind|InformationKind|HighDimensionalKind|GraphicalKind|CodingKind|FunctionalKind|ApproximationKind|HarmonicKind|GeometryKind|SymmetryKind|ConvexKind|OptimizationKind|VariationalKind|TransportKind|NumericalKind|DifferentialKind|TheoryKind|AlgorithmKind|ScientificKind|StochasticKind;
const specFor=(kind:PlotKind)=>isContinuumKind(kind)?continuumLabs[kind]:isElectromagnetismKind(kind)?electromagnetismLabs[kind]:isWavesKind(kind)?wavesLabs[kind]:isAnalyticalMechanicsKind(kind)?analyticalMechanicsLabs[kind]:isClassicalMechanicsKind(kind)?classicalMechanicsLabs[kind]:isPhysicalModelingKind(kind)?physicalModelingLabs[kind]:isPrivacyKind(kind)?privacyLabs[kind]:isGameKind(kind)?gameLabs[kind]:isCombinatorialKind(kind)?combinatorialLabs[kind]:isControlKind(kind)?controlLabs[kind]:isCausalKind(kind)?causalLabs[kind]:isTheoryKind(kind)?theoryLabs[kind]:isAlgorithmKind(kind)?algorithmLabs[kind]:isScientificKind(kind)?scientificLabs[kind]:isStochasticKind(kind)?stochasticLabs[kind]:isDifferentialKind(kind)?differentialLabs[kind]:isNumericalKind(kind)?numericalLabs[kind]:isTransportKind(kind)?transportLabs[kind]:isVariationalKind(kind)?variationalLabs[kind]:isOptimizationKind(kind)?optimizationLabs[kind]:isConvexKind(kind)?convexLabs[kind]:isSymmetryKind(kind)?symmetryLabs[kind]:isGeometryKind(kind)?geometryLabs[kind]:isHarmonicKind(kind)?harmonicLabs[kind]:isApproximationKind(kind)?approximationLabs[kind]:isFunctionalKind(kind)?functionalLabs[kind]:isCodingKind(kind)?codingLabs[kind]:isGraphicalKind(kind)?graphicalLabs[kind]:isHighDimensionalKind(kind)?highDimensionalLabs[kind]:isInformationKind(kind)?informationLabs[kind]:isBayesKind(kind)?bayesLabs[kind]:isStatisticsKind(kind)?statisticsLabs[kind]:isMeasureKind(kind)?measureLabs[kind]:isProbabilityKind(kind)?probabilityLabs[kind]:isAbstractAlgebraKind(kind)?abstractAlgebraLabs[kind]:isTensorKind(kind)?tensorLabs[kind]:isComplexKind(kind)?complexLabs[kind]:isAnalysisKind(kind)?analysisLabs[kind]:isMatrixStructureKind(kind)?matrixStructureLabs[kind]:isDiscreteKind(kind)?discreteLabs[kind]:isElementaryKind(kind)?elementaryLabs[kind]:foundationLabs[kind];
const modelFor=(kind:PlotKind,values:number[])=>isContinuumKind(kind)?evaluateContinuum(kind,values):isElectromagnetismKind(kind)?evaluateElectromagnetism(kind,values):isWavesKind(kind)?evaluateWaves(kind,values):isAnalyticalMechanicsKind(kind)?evaluateAnalyticalMechanics(kind,values):isClassicalMechanicsKind(kind)?evaluateClassicalMechanics(kind,values):isPhysicalModelingKind(kind)?evaluatePhysicalModeling(kind,values):isPrivacyKind(kind)?evaluatePrivacy(kind,values):isGameKind(kind)?evaluateGame(kind,values):isCombinatorialKind(kind)?evaluateCombinatorial(kind,values):isControlKind(kind)?evaluateControl(kind,values):isCausalKind(kind)?evaluateCausal(kind,values):isTheoryKind(kind)?evaluateTheory(kind,values):isAlgorithmKind(kind)?evaluateAlgorithm(kind,values):isScientificKind(kind)?evaluateScientific(kind,values):isStochasticKind(kind)?evaluateStochastic(kind,values):isDifferentialKind(kind)?evaluateDifferential(kind,values):isNumericalKind(kind)?evaluateNumerical(kind,values):isTransportKind(kind)?evaluateTransport(kind,values):isVariationalKind(kind)?evaluateVariational(kind,values):isOptimizationKind(kind)?evaluateOptimization(kind,values):isConvexKind(kind)?evaluateConvex(kind,values):isSymmetryKind(kind)?evaluateSymmetry(kind,values):isGeometryKind(kind)?evaluateGeometry(kind,values):isHarmonicKind(kind)?evaluateHarmonic(kind,values):isApproximationKind(kind)?evaluateApproximation(kind,values):isFunctionalKind(kind)?evaluateFunctional(kind,values):isCodingKind(kind)?evaluateCoding(kind,values):isGraphicalKind(kind)?evaluateGraphical(kind,values):isHighDimensionalKind(kind)?evaluateHighDimensional(kind,values):isInformationKind(kind)?evaluateInformation(kind,values):isBayesKind(kind)?evaluateBayes(kind,values):isStatisticsKind(kind)?evaluateStatistics(kind,values):isMeasureKind(kind)?evaluateMeasure(kind,values):isProbabilityKind(kind)?evaluateProbability(kind,values):isAbstractAlgebraKind(kind)?evaluateAbstractAlgebra(kind,values):isTensorKind(kind)?evaluateTensor(kind,values):isComplexKind(kind)?evaluateComplex(kind,values):isAnalysisKind(kind)?evaluateAnalysis(kind,values):isMatrixStructureKind(kind)?evaluateMatrixStructure(kind,values):isDiscreteKind(kind)?evaluateDiscrete(kind,values):isElementaryKind(kind)?evaluateElementary(kind,values):evaluateFoundation(kind,values);
const n=(value:number)=>Number.isFinite(value)?formatNumber(value):value===Infinity?'∞':value===-Infinity?'−∞':'—';
export function FoundationPlot({kind,result,label}:{kind:PlotKind;result:FoundationResult;label:string}){
 const {language}=useLanguage();const arrow=useId().replace(/:/g,'');const ko=language==='ko';const spec=specFor(kind);const points=result.curves.flat();
 const xs=points.map(v=>v[0]),ys=points.map(v=>v[1]);const geometry=['svd','real-complex-numbers','trigonometry-coordinates','vector-spaces-rank','determinants-traces','implicit-inverse-functions','vector-fields-divergence-curl','line-surface-volume-integrals','gauss-stokes-theorems','complex-polar-maps','cauchy-riemann-holomorphic','complex-contour-integrals','residues-winding','analytic-continuation-branches','complex-linear-algebra','wirtinger-gradients'].includes(kind);const radius=Math.max(.01,...xs.map(Math.abs),...ys.map(Math.abs));const minX=geometry?-radius:Math.min(0,...xs),maxX=geometry?radius:Math.max(.01,...xs),minY=geometry?-radius:Math.min(0,...ys),maxY=geometry?radius:Math.max(.01,...ys);const dx=maxX-minX,dy=maxY-minY;
 const xy=([x,y]:number[])=>[45+260*(x-minX)/dx,(geometry?300:220)-(geometry?260:170)*(y-minY)/dy];const path=(curve:number[][])=>curve.map((v,i)=>`${i?'L':'M'}${xy(v)}`).join('');
 if(isContinuumKind(kind))return <ContinuumFigure result={result as ContinuumResult}/>;
 if(isElectromagnetismKind(kind))return <ElectromagnetismFigure result={result as ElectromagnetismResult}/>;
 if(isWavesKind(kind))return <WavesFigure result={result as WavesResult}/>;
 if(isAnalyticalMechanicsKind(kind))return <AnalyticalMechanicsFigure result={result as AnalyticalMechanicsResult} label={label} axis={spec.axis[language]}/>;
 if(isClassicalMechanicsKind(kind))return <ClassicalMechanicsFigure result={result as ClassicalMechanicsResult} label={label} axis={spec.axis[language]}/>;
 if(isPhysicalModelingKind(kind))return <PhysicalModelingFigure result={result as PhysicalModelingResult} label={label} axis={spec.axis[language]}/>;
 if(isPrivacyKind(kind))return <PrivacyFigure result={result as PrivacyResult} label={label} axis={spec.axis[language]}/>;
 if(isGameKind(kind))return <GameFigure result={result as GameResult} label={label} axis={spec.axis[language]}/>;
 if(isCombinatorialKind(kind))return <CombinatorialFigure result={result as CombinatorialResult} label={label} axis={spec.axis[language]}/>;
 if(isControlKind(kind))return <ControlFigure result={result as ControlResult} label={label} axis={spec.axis[language]}/>;
 if(isCausalKind(kind))return <CausalFigure result={result as CausalResult} label={label} axis={spec.axis[language]}/>;
 if(isTheoryKind(kind))return <TheoryFigure result={result as TheoryResult} label={label} axis={spec.axis[language]}/>;
 if(isAlgorithmKind(kind))return <AlgorithmFigure result={result as AlgorithmResult} label={label} axis={spec.axis[language]}/>;
 if(isScientificKind(kind))return <ScientificFigure result={result as ScientificResult} label={label} axis={spec.axis[language]}/>;
 if(isStochasticKind(kind))return <StochasticFigure result={result as StochasticResult} label={label} axis={spec.axis[language]}/>;
 if(isDifferentialKind(kind))return <DifferentialFigure result={result as DifferentialResult} label={label} axis={spec.axis[language]}/>;
 if(isNumericalKind(kind))return <NumericalFigure result={result as NumericalResult} label={label} axis={spec.axis[language]}/>;
 if(isTransportKind(kind))return <TransportFigure result={result as TransportResult} label={label} axis={spec.axis[language]}/>;
 if(isVariationalKind(kind))return <VariationalFigure result={result as VariationalResult} label={label} axis={spec.axis[language]}/>;
 if(isOptimizationKind(kind))return <OptimizationFigure result={result as OptimizationResult} label={label} axis={spec.axis[language]}/>;
 if(isConvexKind(kind))return <ConvexFigure result={result as ConvexResult} label={label} axis={spec.axis[language]}/>;
 if(isSymmetryKind(kind))return <SymmetryFigure result={result as SymmetryResult} label={label} axis={spec.axis[language]}/>;
 if(isGeometryKind(kind))return <GeometryFigure result={result as GeometryResult} label={label} axis={spec.axis[language]}/>;
 if(isHarmonicKind(kind))return <MeasureFigure result={result} label={label} axis={spec.axis[language]} variant="harmonic"/>;
 if(isApproximationKind(kind))return <MeasureFigure result={result} label={label} axis={spec.axis[language]} variant="approximation"/>;
 if(isFunctionalKind(kind))return <MeasureFigure result={result} label={label} axis={spec.axis[language]} variant="functional"/>;
 if(isCodingKind(kind))return <MeasureFigure result={result} label={label} axis={spec.axis[language]} variant="information"/>;
 if(isGraphicalKind(kind))return <GraphicalFigure result={result as GraphicalResult} label={label} axis={spec.axis[language]}/>;
 if(isHighDimensionalKind(kind))return <MeasureFigure result={result} label={label} axis={spec.axis[language]} variant="high-dimensional"/>;
 if(isInformationKind(kind))return <MeasureFigure result={result} label={label} axis={spec.axis[language]} variant="information"/>;
 if(isBayesKind(kind))return <MeasureFigure result={result} label={label} axis={spec.axis[language]} variant="bayes"/>;
 if(isStatisticsKind(kind))return <MeasureFigure result={result} label={label} axis={spec.axis[language]} variant="statistics"/>;
 if(isMeasureKind(kind))return <MeasureFigure result={result} label={label} axis={spec.axis[language]}/>;
 if(isProbabilityKind(kind))return <ProbabilityFigure result={result} label={label} axis={spec.axis[language]}/>;
 if(isAbstractAlgebraKind(kind))return <DiscreteFigure result={result} label={label} axis={spec.axis[language]} multicolor/>;
 if(isTensorKind(kind))return <TensorFigure result={result} label={label} axis={spec.axis[language]} legend={spec.legend}/>;
 if(isDiscreteKind(kind))return <DiscreteFigure result={result} label={label} axis={spec.axis[language]}/>;
 const marker=(curve:number[][],i:number)=>{
  if(curve.length<2)return undefined;const first=curve[0],last=curve[curve.length-1];
  if(first[0]===last[0]&&first[1]===last[1])return undefined;
  return kind==='vector-fields-divergence-curl'?`url(#${arrow}-0)`:(kind==='line-surface-volume-integrals'||kind==='complex-contour-integrals')&&i>0?`url(#${arrow}-1)`:undefined;
 };
 const bars=kind==='probability-distributions'||kind==='expectation-variance';
 return <figure className="ai-foundation-figure"><div className="ai-foundation-scroll" tabIndex={0} role="region" aria-describedby={kind==='thermodynamics'?`${arrow}-hint`:undefined} aria-label={ko?'실험 그림 · 작은 화면에서 가로로 이동':'Experiment plot · Scroll horizontally on narrow screens'}><svg data-equal-scale={geometry?'true':undefined} viewBox={geometry?'0 0 350 360':'0 0 350 280'} role="img" aria-label={label} style={{minWidth:kind==='thermodynamics'?400:undefined}}>
 <defs>{['blue','orange'].map((color,i)=><marker key={color} id={`${arrow}-${i}`} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L10 5L0 10Z" fill={`var(--learning-${color})`}/></marker>)}</defs>
 <path d={`M35 ${xy([0,0])[1]}H315M${xy([0,0])[0]} 35V${geometry?310:230}`} stroke="var(--learning-neutral)" fill="none"/>
 {result.curves.map((curve,i)=><g key={i}>{bars&&i===0?curve.map(([x,y],j)=><path key={j} d={`M${xy([x,0])}L${xy([x,y])}`} stroke="var(--learning-blue)" strokeWidth={Math.min(18,130/curve.length)}/>):<path markerEnd={marker(curve,i)} d={path(curve)} fill="none" stroke={`var(--learning-${i&&kind!=='vector-fields-divergence-curl'?'orange':'blue'})`} strokeWidth="2.5" strokeDasharray={i&&kind!=='vector-fields-divergence-curl'?'5 3':undefined}/>}{curve.length<=6&&curve.map((v,j)=><circle key={j} cx={xy(v)[0]} cy={xy(v)[1]} r="3.5" fill={`var(--learning-${i&&kind!=='vector-fields-divergence-curl'?'orange':'blue'})`}/>)}</g>)}
 <g fill="var(--learning-neutral)" fontSize="12"><text style={{fontSize:kind==='thermodynamics'?14:undefined}} x="45" y={geometry?325:245}>{n(minX)}</text><text style={{fontSize:kind==='thermodynamics'?14:undefined}} x="305" y={geometry?325:245} textAnchor="end">{n(maxX)}</text><text style={{fontSize:kind==='thermodynamics'?14:undefined}} x="45" y="28">{n(maxY)}</text><text style={{fontSize:kind==='thermodynamics'?14:undefined}} x="45" y={geometry?345:265}>{geometry?(ko?'가로·세로의 단위 길이가 같습니다.':'Equal unit length on both axes.'):(ko?'축 범위는 현재 값에 맞춰 조절합니다.':'Axis ranges adapt to current values.')}</text></g>
 {result.curves.length===0&&<text style={{fontSize:kind==='thermodynamics'?14:undefined}} x="175" y="140" textAnchor="middle" fill="var(--learning-neutral)" fontSize="13">{ko?'이 설정에서 그릴 값이 없습니다.':'No plotted value at this setting.'}</text>}
 {result.holes?.map((point,i)=><circle key={i} cx={xy(point)[0]} cy={xy(point)[1]} r="5" fill="hsl(var(--card))" stroke="var(--learning-orange)" strokeWidth="2" data-excluded-point="true"/>)}
 </svg></div>{kind==='thermodynamics'&&<p id={`${arrow}-hint`} className="ai-small">{ko?'좁은 화면에서는 그림 영역에 초점을 두고 좌우 방향키로 이동하세요.':'On narrow screens, focus the plot region and use the left and right arrow keys.'}</p>}<figcaption>{spec.axis[language]}</figcaption><div className="ai-legend">{result.curveLabels?result.curveLabels.map((name,i)=><span key={i} className={i?'ai-orange':'ai-blue'}>{name}</span>):spec.legend.slice(0,result.curves.length).map((name,i)=><span key={i} className={i?'ai-orange':'ai-blue'}>{name[language]}</span>)}</div>{kind==='kronecker-products'&&<DiscreteFigure result={{...result,curves:[]}} label={label} axis={ko?'표: A⊗B의 실제 원소':'Table: actual entries of A⊗B'}/>}</figure>;
}
export function FoundationConcept({kind,label,step}:{kind:PlotKind;label:string;step:number}){
 const spec=specFor(kind);const continuumKind=isContinuumKind(kind),assumptionKind=isContinuumAssumptionKind(kind),stressStrainKind=isContinuumStressStrainKind(kind),flowKind=isContinuumFlowKind(kind),diffusionKind=isContinuumDiffusionKind(kind);const staticKind=(electroMagnetostaticsKinds as readonly string[]).includes(kind),potentialKind=(potentialsKinds as readonly string[]).includes(kind),maxwellKind=(maxwellKinds as readonly string[]).includes(kind),inductionKind=(inductionWavesKinds as readonly string[]).includes(kind);const energyKind=(poyntingEnergyKinds as readonly string[]).includes(kind);const effectiveKind=(electromagneticEffectiveKinds as readonly string[]).includes(kind);const circuitKind=(circuitSensorsKinds as readonly string[]).includes(kind);const gaugeKind=(gaugeFreedomKinds as readonly string[]).includes(kind);const materialKind=(materialInterfacesKinds as readonly string[]).includes(kind);const preset=isContinuumConservativeKind(kind)?[0,1,2,3][step]:isContinuumTurbulenceKind(kind)?[0,1,2,3][step]:isContinuumMultiphysicsKind(kind)?[0,1,2,3][step]:isContinuumDimensionlessKind(kind)?[0,1,2,3][step]:isContinuumConstitutiveKind(kind)?[0,1,2,3][step]:isContinuumReactionKind(kind)?[0,1,2,3][step]:diffusionKind?(kind===continuumDiffusionKinds[0]?[0,0,2,3]:[0,1,2,3])[step]:flowKind?(kind===continuumFlowKinds[0]?[0,1,2,3]:[0,1,2,3])[step]:stressStrainKind?(kind===continuumStressStrainKinds[0]?[0,3,1,2]:[1,0,3,6])[step]:assumptionKind?(kind===continuumAssumptionKinds[0]?[0,2,3,6]:[0,1,2,6])[step]:continuumKind?(kind===continuumKinds[0]?[0,1,2,6]:[0,1,3,5])[step]:effectiveKind?(kind===electromagneticEffectiveKinds[0]?[0,0,1,3]:[0,1,1,3])[step]:circuitKind?(kind===circuitSensorsKinds[0]?[0,1,0,3]:[0,1,2,3])[step]:gaugeKind?(kind===gaugeFreedomKinds[0]?[0,1,2,3]:[0,1,2,4])[step]:materialKind?(kind===materialInterfacesKinds[0]?[0,1,2,7]:[0,2,3,12])[step]:energyKind?(kind===poyntingEnergyKinds[0]?[0,1,2,0]:[0,1,4,7])[step]:inductionKind?(kind===inductionWavesKinds[0]?[0,0,2,7]:[0,1,3,6])[step]:maxwellKind?(kind===maxwellKinds[0]?[0,2,4,0]:[3,0,0,1])[step]:potentialKind?(kind===potentialsKinds[0]?[0,2,0,3]:[0,0,5,0])[step]:step;const result=useMemo(()=>modelFor(kind,continuumKind||staticKind||potentialKind||maxwellKind||inductionKind||energyKind||materialKind||gaugeKind||circuitKind||effectiveKind?spec.presets[preset].values:spec.controls.map(c=>c.initial)),[kind,spec,continuumKind,staticKind,potentialKind,maxwellKind,inductionKind,energyKind,materialKind,gaugeKind,circuitKind,effectiveKind,preset]);const entries=Object.entries(result.stats);const entry=entries[Math.min(isContinuumConservativeKind(kind)?[0,1,2,3][step]:isContinuumTurbulenceKind(kind)?[0,1,2,3][step]:isContinuumMultiphysicsKind(kind)?[0,1,2,3][step]:isContinuumDimensionlessKind(kind)?[0,1,2,3][step]:isContinuumConstitutiveKind(kind)?[0,1,2,3][step]:isContinuumReactionKind(kind)?[0,1,2,3][step]:diffusionKind?(kind===continuumDiffusionKinds[0]?[0,1,3,4]:[0,4,2,3])[step]:flowKind?(kind===continuumFlowKinds[0]?[0,2,1,3]:[0,1,2,3])[step]:stressStrainKind?(kind===continuumStressStrainKinds[0]?[0,4,2,2]:[2,1,3,2])[step]:assumptionKind?(kind===continuumAssumptionKinds[0]?[0,1,1,3]:[0,3,2,4])[step]:continuumKind?(kind===continuumKinds[0]?[0,2,1,3]:[2,3,0,4])[step]:effectiveKind?(kind===electromagneticEffectiveKinds[0]?[0,2,3,0]:[0,1,2,3])[step]:circuitKind?(kind===circuitSensorsKinds[0]?[0,2,3,0]:[1,0,2,3])[step]:gaugeKind?(kind===gaugeFreedomKinds[0]?[0,2,1,3]:[0,2,1,3])[step]:materialKind?step:energyKind?(kind===poyntingEnergyKinds[0]?[0,1,2,3]:[0,2,0,1])[step]:inductionKind?(kind===inductionWavesKinds[0]?[0,1,2,3]:[1,2,3,4])[step]:potentialKind&&kind===potentialsKinds[0]?[0,1,0,0][step]:kind==='thermodynamics'?[0,0,2,4][step]:step,entries.length-1)];
 return <div>{(isElectromagnetismKind(kind)||isWavesKind(kind)||isAnalyticalMechanicsKind(kind)||isComplexKind(kind)||isClassicalMechanicsKind(kind)||isPhysicalModelingKind(kind)||isPrivacyKind(kind)||isGameKind(kind)||isCombinatorialKind(kind)||isControlKind(kind)||isCausalKind(kind)||isTheoryKind(kind)||isAlgorithmKind(kind)||isScientificKind(kind)||isStochasticKind(kind)||isDifferentialKind(kind)||isNumericalKind(kind)||isTransportKind(kind)||isVariationalKind(kind)||isOptimizationKind(kind)||isConvexKind(kind)||isSymmetryKind(kind)||isGeometryKind(kind)||isHarmonicKind(kind)||isApproximationKind(kind)||isFunctionalKind(kind)||isCodingKind(kind)||isGraphicalKind(kind)||isHighDimensionalKind(kind)||isInformationKind(kind)||isBayesKind(kind)||isStatisticsKind(kind)||isMeasureKind(kind)||isProbabilityKind(kind)||isAbstractAlgebraKind(kind)||isTensorKind(kind))&&result.statement&&<p className="ai-logic-statement">{result.statement}</p>}{continuumKind?<ContinuumFigure result={result as ContinuumResult} conceptStep={step}/>:effectiveKind?<ElectromagneticEffectiveFigure result={result as ElectromagnetismResult} conceptStep={step}/>:circuitKind?<CircuitSensorsFigure result={result as ElectromagnetismResult} conceptStep={step}/>:gaugeKind?<GaugeFreedomFigure result={result as ElectromagnetismResult} conceptStep={step}/>:materialKind?<MaterialInterfacesFigure result={result as ElectromagnetismResult} conceptStep={step}/>:energyKind?<PoyntingEnergyFigure result={result as ElectromagnetismResult} conceptStep={step}/>:inductionKind?<InductionWavesFigure result={result as ElectromagnetismResult} conceptStep={step}/>:maxwellKind?<MaxwellFigure result={result as ElectromagnetismResult} conceptStep={step}/>:potentialKind?<PotentialsFigure result={result as ElectromagnetismResult} conceptStep={step}/>:staticKind&&(result as ElectromagnetismResult).statics?<ElectroMagnetostaticsFigure lab={(result as ElectromagnetismResult).statics!} conceptStep={step}/>:<FoundationPlot kind={kind} result={result} label={label}/>}<p className="ai-concept-equation">{entry[0]} = {entry[1]===null?'—':Array.isArray(entry[1])?(entry[1].length?`(${entry[1].map(n).join(', ')})`:'∅'):n(entry[1])}</p></div>;
}
export default function FoundationsLab({kind}:{kind:PlotKind}){
 const {language}=useLanguage();const ko=language==='ko';const spec=specFor(kind);
 const initial=useMemo(()=>({values:spec.controls.map(c=>c.initial)}),[spec]);
 const valid=useMemo(()=>(v:unknown):v is {values:number[]}=>{if(!v||typeof v!=='object')return false;const arr=(v as {values:unknown}).values;return Array.isArray(arr)&&arr.length===spec.controls.length&&arr.every((x,i)=>Number.isFinite(x)&&x>=spec.controls[i].min&&x<=spec.controls[i].max&&(spec.controls[i].step==='any'||Number.isInteger(x)));},[spec]);
 const [s,set,ready,storage]=useSavedState(`suanlab-learning:${kind}:lab`,initial,valid);const result=useMemo(()=>modelFor(kind,s.values),[kind,s.values]);
 const load=(index:number)=>set({values:[...spec.presets[index].values]});
 const warnings={'numeric-range':ko?'수학적 값이 현재 숫자 표현 범위를 벗어났습니다. 이를 무한값이나 정의역 오류로 해석하지 마세요.':'The mathematical value is outside the current numeric representation range. This is not mathematical infinity or a domain failure.','zero-divisor':ko?'0으로 나누는 값은 정의하지 않습니다. 분모·시간·역함수의 배율을 확인하세요.':'Division by zero is undefined. Check the denominator, time or inverse-map scale.','undefined-angle':ko?'영점에는 고유한 방향각이 없습니다. 크기 0과 정의되지 않는 각을 구분하세요.':'Zero has no unique direction angle. Distinguish zero magnitude from an undefined angle.','degenerate-equation':ko?'일차 계수가 0입니다. 0x=0이면 모든 실수, 0x=b≠0이면 해가 없습니다.':'The linear coefficient is zero. 0x=0 has all real solutions; 0x=b≠0 has none.','function-domain':ko?'선택 입력이나 밑이 이 함수의 정의 조건을 만족하지 않습니다. 주변 극한을 함수값으로 바꾸지 마세요.':'The selected input or base violates this function’s domain conditions. A nearby limit is not a value at the excluded point.','infinite-sum':ko?'이 설정에는 유한한 보통 의미의 무한 합이 없습니다. 부분합과 급수의 합을 구분하세요.':'There is no finite ordinary infinite sum at this setting. Distinguish a partial sum from a series sum.','asymptotic-bound':ko?'선택한 C로는 유한 시작점이 없습니다. 이 후보의 실패가 Big-O 분류 자체의 실패는 아닙니다.':'No finite threshold works for the chosen C. Failure of this candidate does not disprove the Big-O class.',support:ko?'모형이 관측 가능한 사건에 확률 0을 줍니다. 무한 로그 비용을 유한한 값으로 숨기지 않습니다.':'The model assigns zero probability to a possible observation. Infinite log cost is not replaced by a finite value.',rank:ko?'특이하거나 식별되지 않는 방향이 있습니다. 표시된 최소 노름 선택과 데이터의 유일한 결론을 구분하세요.':'A direction is singular or unidentified. Distinguish the displayed minimum-norm choice from a uniquely data-determined conclusion.',stationary:ko?'전이가 없어 모든 분포가 정상입니다. 유일한 정상값은 정의되지 않습니다.':'With no transitions every distribution is stationary. No unique stationary occupancy is defined.',stability:ko?'안정성 또는 수렴 조건의 경계·바깥입니다. 실제 연속 모형의 성질과 수치 경로를 구분하세요.':'This setting lies on or outside a stability or convergence condition. Distinguish the continuous model from its numerical trajectory.',uncertainty:ko?'가정에 따른 구간이 0을 포함합니다. 이는 효과 없음의 증명도, 평가 설계가 타당하다는 보장도 아닙니다.':'The assumption-dependent interval includes zero. This neither proves no effect nor validates the evaluation design.'};
 const title=learningUnits.find(u=>u.id===kind)!.title[language];
 return <section className="ai-lab" id="experiment" aria-labelledby="ai-lab-title"><div className="ai-lab-heading"><div><span className="ai-eyebrow">INTERACTIVE LAB</span><h2 id="ai-lab-title">{ko?'직접 바꿔보세요':'Make it your experiment'}</h2></div><span className="ai-live-dot">{ko?'실시간 계산':'Live calculation'}</span></div>
 <GuidedExperiments kind={kind} ready={ready} onLoad={g=>load(g.preset)}/><fieldset disabled={!ready} className="ai-presets"><legend>{ko?'비교할 예제':'Examples to compare'}</legend>{spec.presets.map((p,i)=><Button key={i} variant="outline" size="sm" onClick={()=>load(i)}>{p.name[language]}</Button>)}</fieldset>
 {result.statement&&<p className="ai-logic-statement" aria-live="polite">{result.statement}</p>}<FoundationPlot kind={kind} result={result} label={`${title}. ${spec.axis[language]}. ${spec.legend.map(t=>t[language]).join('; ')}`}/>{isAlgorithmKind(kind)&&(result as AlgorithmResult).benchmark&&<AlgorithmBenchmark config={(result as AlgorithmResult).benchmark!}/>}
 <fieldset disabled={!ready} className="ai-parameters"><legend>{ko?'모델 설정':'Model settings'}</legend><div className="ai-three-fields">{spec.controls.map((control,i)=><NumberField resetKey={s} integer={control.step!=='any'} key={i} label={control.label[language]} min={control.min} max={control.max} step={control.step} value={s.values[i]} onChange={x=>{if(control.step!=='any'&&!Number.isInteger(x))return;set(v=>({values:v.values.map((y,j)=>j===i?x:y)}));}}/>)}</div></fieldset>
 <div className="ai-readout" data-testid="lab-readout"><h3>{ko?'수치로 확인하기':'Check the numbers'}</h3><p className="ai-small">{ko?'표시값은 반올림합니다. —는 정의되지 않거나 표시할 유한값이 없는 경우입니다.':'Readouts are rounded. — means undefined or no finite display value.'}</p><dl>{Object.entries(result.stats).map(([key,value])=><div key={key}><dt>{key}</dt><dd>{value===null?'—':Array.isArray(value)?(value.length?`(${value.map(n).join(', ')})`:'∅'):n(value)}</dd></div>)}</dl></div>
 {result.warning&&<p className="ai-model-note" role="status">{warnings[result.warning]}</p>}<p className="ai-model-note">{spec.note[language]}</p>
 <Button className="ai-reset" size="sm" variant="outline" onClick={()=>set({values:[...initial.values]})}>{ko?'모든 실험 설정 초기화':'Reset all experiment settings'}</Button><p className="ai-small">{storage?(ko?'실험 설정은 이 브라우저에 자동 저장됩니다.':'Experiment settings are saved in this browser.'):(ko?'브라우저 저장소를 사용할 수 없어 현재 방문에서만 유지됩니다.':'Browser storage is unavailable. Settings remain in memory for this visit.')}</p>
 </section>;
}
