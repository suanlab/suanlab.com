'use client';
import { photonicInterferometerId } from '@/data/photonic-interferometer-reader';
import { PhotonicInterferometerConcept } from './photonic-interferometer-lab';
import { photodiodeBandwidthId } from '@/data/photodiode-bandwidth-reader';
import { PhotodiodeBandwidthConcept } from './photodiode-bandwidth-lab';
import { electronicBandsId } from '@/data/electronic-bands-reader';
import { ElectronicBandsConcept } from './electronic-bands-lab';
import { phononDispersionId } from '@/data/phonon-dispersion-reader';
import { PhononDispersionConcept } from './phonon-dispersion-lab';
import { weakFieldClockId } from '@/data/weak-field-clock-reader';
import { WeakFieldClockConcept } from './weak-field-clock-lab';
import { rindlerObservablesId } from '@/data/rindler-observables-reader';
import { RindlerObservablesConcept } from './rindler-observables-lab';
import { schwarzschildRedshiftId } from '@/data/schwarzschild-redshift-reader';
import { SchwarzschildRedshiftConcept } from './schwarzschild-redshift-lab';
import { einsteinFluidCurvatureId } from '@/data/einstein-fluid-curvature-reader';
import { EinsteinFluidCurvatureConcept } from './einstein-fluid-curvature-lab';
import { greatCircleGeodesicId } from '@/data/great-circle-geodesic-reader';
import { GreatCircleGeodesicConcept } from './great-circle-geodesic-lab';
import { rotationalCurvatureId } from '@/data/rotational-curvature-reader';
import { RotationalCurvatureConcept } from './rotational-curvature-lab';
import { fourMomentumId } from '@/data/four-momentum-reader';
import { FourMomentumConcept } from './four-momentum-lab';
import { causalRelayId } from '@/data/causal-relay-reader';
import { CausalRelayConcept } from './causal-relay-lab';
import { minkowskiPathId } from '@/data/minkowski-path-reader';
import { MinkowskiPathConcept } from './minkowski-path-lab';
import { lorentzBoostId } from '@/data/lorentz-boost-reader';
import { LorentzBoostConcept } from './lorentz-boost-lab';
import { slowFastMemoryId } from '@/data/slow-fast-memory-reader';
import { SlowFastMemoryConcept } from './slow-fast-memory-lab';
import { heavyModeMatchingId } from '@/data/heavy-mode-matching-reader';
import { HeavyModeMatchingConcept } from './heavy-mode-matching-lab';
import { isingDecimationId } from '@/data/ising-decimation-reader';
import { IsingDecimationConcept } from './ising-decimation-lab';
import { isingCriticalityId } from '@/data/ising-criticality-reader';
import { IsingCriticalityConcept } from './ising-criticality-lab';
import { legendreResponseUnitId } from '@/data/legendre-response-reader';
import { LegendreResponseConcept } from './legendre-response-lab';
import { scalarGibbsUnitId } from '@/lib/learning-scalar-gibbs-input';
import { ScalarGibbsConcept } from './scalar-gibbs-lab';
import {isBlockOrderTeachingId} from '@/lib/learning-block-order-teaching';
import BlockOrderReaderConcept from './block-order-reader-concept';
import {isThermalFieldTeachingId} from '@/lib/learning-thermal-field-teaching';
import ThermalFieldConcept from './thermal-field-concept';

import {isRealTimePhaseTeachingId} from '@/lib/learning-real-time-phase-teaching';
import RealTimePhaseConcept from './real-time-phase-concept';
import {isSaddleFluctuationTeachingId} from '@/lib/learning-saddle-fluctuation-teaching';
import SaddleFluctuationConcept from './saddle-fluctuation-concept';
import {isEuclideanKernelTeachingId} from '@/lib/learning-euclidean-kernel-teaching';
import EuclideanKernelConcept from './euclidean-kernel-concept';
import {isBrownianBridgeTeachingId} from '@/lib/learning-brownian-bridge-teaching';
import BrownianBridgeConcept from './brownian-bridge-concept';
import {isSpatialSpectrumTeachingId} from '@/lib/learning-spatial-spectrum-teaching';
import SpatialSpectrumConcept from './spatial-spectrum-concept';
import {isNongaussianFieldTeachingId} from '@/lib/learning-nongaussian-field-teaching';
import NongaussianFieldConcept from './nongaussian-field-concept';
import {isGaussianFieldTeachingId} from '@/lib/learning-gaussian-field-teaching';
import GaussianFieldConcept from './gaussian-field-concept';
import {isDuffingSecularTeachingId} from '@/lib/learning-duffing-secular-teaching';
import DuffingSecularConcept from './duffing-secular-concept';
import {isBoundaryLayerTeachingId} from '@/lib/learning-boundary-layer-teaching';
import BoundaryLayerConcept from './boundary-layer-concept';

import {isQuarticAsymptoticTeachingId} from '@/lib/learning-quartic-asymptotic-teaching';
import QuarticAsymptoticConcept from './quartic-asymptotic-concept';

import {isRegularPerturbationTeachingId} from '@/lib/learning-regular-perturbation-teaching';
import RegularPerturbationConcept from './regular-perturbation-concept';

import {isGraphGaussTeachingId} from '@/lib/learning-graph-gauss-teaching';
import GraphGaussConcept from './graph-gauss-concept';

import {isGaugeOrbitTeachingId} from '@/lib/learning-gauge-orbits-teaching';
import GaugeOrbitConcept from './gauge-orbits-concept';

import {isGraphGaugeTeachingId} from '@/lib/learning-graph-gauge-teaching';
import GraphGaugeConcept from './graph-gauge-concept';
import {isFieldModesTeachingId} from '@/lib/learning-field-modes-teaching';
import FieldModesConcept from './field-modes-concept';
import {isFieldBalanceTeachingId} from '@/lib/learning-field-balance-teaching';
import FieldBalanceConcept from './field-balance-concept';
import {isScalarFieldTeachingId} from '@/lib/learning-scalar-field-teaching';
import ScalarFieldConcept from './scalar-field-concept';
import {isInformationTeachingId} from '@/lib/learning-information-teaching';
import InformationConcept from './information-concept';
import {isQubitTeachingId} from '@/lib/learning-qubit-teaching-input';
import QubitConcept from './qubit-concept';
import ManyBodyConcept from './many-body-concept';
import {isManyBodyTeachingId} from '@/lib/learning-many-body-teaching';
import QuantumVariationalConceptPanel, { QuantumVariationalConceptView } from './quantum-variational-concept';
import { isQuantumVariationalKind } from '@/lib/learning-quantum-variational';
import TensorEntanglementConceptPanel,{TensorConceptView as TensorEntanglementConceptView} from './tensor-concept';
import {isTensorKind as isEntanglementKind} from '@/lib/learning-tensor';
import UncertaintyConceptPanel,{UncertaintyConceptView} from './uncertainty-concept';
import {isUncertaintyKind} from '@/lib/learning-uncertainty';
import DensityConceptPanel,{DensityConceptView} from './density-concept';
import {isDensityKind} from '@/lib/learning-density';
import SpinConceptPanel,{SpinConceptView} from './spin-concept';
import {isSpinKind} from '@/lib/learning-spin';
import UnitaryConceptPanel,{UnitaryConceptView} from './unitary-concept';
import {isUnitaryKind} from '@/lib/learning-unitary';
import ObservableConceptPanel,{ObservableConceptView} from './observable-concept';
import {isObservableKind} from '@/lib/learning-observable';
import HilbertConceptPanel,{HilbertConceptView} from './hilbert-concept';
import {isHilbertKind} from '@/lib/learning-hilbert';
import CoarseConceptPanel,{CoarseConceptView} from './coarse-concept';
import {isCoarseKind} from '@/lib/learning-coarse';
import SynchronizationConceptPanel,{SynchronizationConceptView} from './synchronization-concept';
import {isSynchronizationKind} from '@/lib/learning-synchronization';

import ThermalConceptPanel,{ThermalConceptView} from './thermal-concept';
import {isThermalKind} from '@/lib/learning-thermal';
import EntropyConceptPanel,{EntropyConceptView}from './entropy-production-concept';
import {isEntropyKind}from '@/lib/learning-entropy-production';
import LrConceptPanel,{LrConceptView}from './linear-response-concept';
import {isLrKind}from '@/lib/learning-linear-response';
import FdtConceptPanel,{FdtConceptView}from './fluctuation-dissipation-concept';
import {isFdtKind}from '@/lib/learning-fluctuation-dissipation';

import ReversalConceptPanel,{ReversalConceptView}from './cycle-reversal-concept';
import {isReversalKind}from '@/lib/learning-cycle-reversal';

import CurrentConceptPanel,{CurrentConceptView}from './probability-current-concept';
import {isCurrentKind}from '@/lib/learning-probability-current';
import FokkerConceptPanel,{FokkerConceptView}from './fokker-planck-concept';
import {isFokkerKind}from '@/lib/learning-fokker-planck';
import MasterConceptPanel,{MasterConceptView}from './master-equation-concept';
import {isMasterKind}from '@/lib/learning-master-equation';
import FiniteSizeConceptPanel,{FiniteSizeConceptView} from './finite-size-concept';
import {isFiniteSizeKind} from '@/lib/learning-finite-size';
import DetailedBalanceConceptPanel,{DetailedBalanceConceptView} from './detailed-balance-concept';
import {isDetailedBalanceKind} from '@/lib/learning-detailed-balance';
import CorrelationConceptPanel, {CorrelationConceptView} from './correlations-concept';
import {isCorrelationKind} from '@/lib/learning-correlations';
import PhaseConceptPanel,{PhaseConceptView} from './phase-transitions-concept';
import {isPhaseKind} from '@/lib/learning-phase-transitions';
import { BoltzmannConceptView } from './boltzmann-gibbs-concept';
import { isBoltzmannGibbsKind } from '@/lib/learning-boltzmann-gibbs';
import {PartitionConceptView} from './partition-concept';
import {isPartitionKind} from '@/lib/learning-partition';
import {EnsembleConceptView} from './ensemble-concept';
import {isEnsembleKind} from '@/lib/learning-ensembles';
import {MicroMacroConceptView} from './micro-macro-concept';
import {isMicroMacroKind} from '@/lib/learning-micro-macro';
import {FreeEnergyConceptView} from './free-energy-lab';
import {isFreeEnergyKind} from '@/lib/learning-free-energy';
import {ThermodynamicLawConcept} from './thermodynamic-laws-lab';
import {isThermodynamicLawKind} from '@/lib/learning-thermodynamic-laws';
import {TemperatureHeatWorkConcept} from './temperature-heat-work-lab';
import {isTemperatureHeatWorkKind} from '@/lib/learning-temperature-heat-work';
import{isContinuumConservativeKind}from '@/lib/learning-continuum-conservative-lab';
import{isContinuumTurbulenceKind}from '@/lib/learning-continuum-turbulence-lab';
import{isContinuumMultiphysicsKind}from '@/lib/learning-continuum-multiphysics-lab';
import{isContinuumDimensionlessKind}from '@/lib/learning-continuum-dimensionless-lab';
import{isContinuumConstitutiveKind}from '@/lib/learning-continuum-constitutive-lab';
import {isContinuumReactionKind}from '@/lib/learning-continuum-reaction-transport-lab';
import {isContinuumDiffusionKind}from '@/lib/learning-continuum-diffusion-lab';
import {isContinuumFlowKind}from '@/lib/learning-continuum-euler-navier-stokes-lab';
import {isContinuumStressStrainKind}from '@/lib/learning-continuum-stress-strain-lab';
import {isContinuumAssumptionKind}from '@/lib/learning-continuum-assumptions-lab';
import {isContinuumKind}from '@/lib/learning-continuum';
import {isElectromagnetismKind}from '@/lib/learning-electromagnetism';
import {isWavesKind} from '@/lib/learning-waves';
import{isAnalyticalMechanicsKind}from '@/lib/learning-analytical-mechanics';
import{isClassicalMechanicsKind}from '@/lib/learning-classical-mechanics';
import {isPhysicalModelingKind} from '@/lib/learning-physical-modeling';
import {isPrivacyKind} from '@/lib/learning-privacy';
import {isGameKind} from '@/lib/learning-games';
import {isCombinatorialKind} from '@/lib/learning-combinatorial';
import { isControlKind } from '@/lib/learning-control';
import { isCausalKind } from '@/lib/learning-causality';
import { isTheoryKind } from '@/lib/learning-theory';
import { isAlgorithmKind } from '@/lib/learning-algorithms';
import { isScientificKind } from '@/lib/learning-scientific';
import { isStochasticKind } from '@/lib/learning-stochastic';
import { isDifferentialKind } from '@/lib/learning-differential';
import { isNumericalKind } from '@/lib/learning-numerical';
import { isTransportKind } from '@/lib/learning-transport';
import { isVariationalKind } from '@/lib/learning-variational';
import { isOptimizationKind } from '@/lib/learning-optimization';
import { isConvexKind } from '@/lib/learning-convex';
import { isSymmetryKind } from '@/lib/learning-symmetry';
import { isGeometryKind } from '@/lib/learning-geometry';
import { isHarmonicKind } from '@/lib/learning-harmonic';
import { isApproximationKind } from '@/lib/learning-approximation';
import { isFunctionalKind } from '@/lib/learning-functional';
import { isCodingKind } from '@/lib/learning-coding';
import { isGraphicalKind } from '@/lib/learning-graphical';
import { isHighDimensionalKind } from '@/lib/learning-high-dimensional';
import { isInformationKind } from '@/lib/learning-information';
import { isBayesKind } from '@/lib/learning-bayes';
import { isStatisticsKind } from '@/lib/learning-statistics';

import { isMeasureKind } from '@/lib/learning-measure';

import { isProbabilityKind } from '@/lib/learning-probability';

import { isAbstractAlgebraKind } from '@/lib/learning-abstract-algebra';
import { isTensorKind } from '@/lib/learning-tensors';
import { isComplexKind } from '@/lib/learning-complex';
import { isAnalysisKind } from '@/lib/learning-analysis';
import { isMatrixStructureKind } from '@/lib/learning-matrix-structures';
import { isDiscreteKind } from '@/lib/learning-discrete';
import { useId, useState } from 'react';
import { Button } from '@/components/ui/button';
import { useLanguage } from '@/components/language-provider';
import { learningGuides, type LearningUnitId } from '@/data/interactive-book';
import AlgebraConcept from './algebra-concept';
import { LogicConcept } from './logic-lab';
import { isLogicKind } from '@/lib/learning-logic';
import { FoundationConcept } from './foundations-lab';
import { isElementaryKind } from '@/lib/learning-elementary';
import { isFoundationKind } from '@/lib/learning-foundations';
import CalculusGraph from './calculus-graph';
import { isCalculusKind } from './calculus-model';
import { isAlgebraKind } from './algebra-model';
import { gradientStep, transform3, type Vec2 } from '@/lib/learning-simulations';

function Diagram({ kind, step, label, language }: { kind: LearningUnitId; step: number; label: string; language: 'ko' | 'en' }) {
  const id = useId();
  if (isReversalKind(kind)) return <ReversalConceptView key={`${kind}:${step}`} kind={kind} step={step} language={language}/>;
  if (isQuantumVariationalKind(kind)) return <QuantumVariationalConceptView key={`${kind}:${step}`} kind={kind} step={step} language={language}/>;
  if (isEntanglementKind(kind)) return <TensorEntanglementConceptView key={`${kind}:${step}`} kind={kind} step={step} language={language}/>;
  if (isUncertaintyKind(kind)) return <UncertaintyConceptView key={`${kind}:${step}`} kind={kind} step={step} language={language}/>;
  if (isDensityKind(kind)) return <DensityConceptView key={`${kind}:${step}`} kind={kind} step={step} language={language}/>;
  if (isSpinKind(kind)) return <SpinConceptView key={`${kind}:${step}`} kind={kind} step={step} language={language}/>;
  if (isUnitaryKind(kind)) return <UnitaryConceptView key={`${kind}:${step}`} kind={kind} step={step} language={language}/>;
  if (isObservableKind(kind)) return <ObservableConceptView key={`${kind}:${step}`} kind={kind} step={step} language={language}/>;
  if (isHilbertKind(kind)) return <HilbertConceptView key={`${kind}:${step}`} kind={kind} step={step} language={language}/>;
  if (isCoarseKind(kind)) return <CoarseConceptView key={`${kind}:${step}`} kind={kind} step={step} language={language}/>;
  if (isSynchronizationKind(kind)) return <SynchronizationConceptView key={`${kind}:${step}`} kind={kind} step={step} language={language}/>;
  if (isThermalKind(kind)) return <ThermalConceptView key={`${kind}:${step}`} kind={kind} step={step} language={language}/>;
  if (isEntropyKind(kind)) return <EntropyConceptView key={`${kind}:${step}`} kind={kind} step={step} language={language}/>;
  if (isLrKind(kind)) return <LrConceptView key={`${kind}:${step}`} kind={kind} step={step} language={language}/>;
  if (isFdtKind(kind)) return <FdtConceptView key={`${kind}:${step}`} kind={kind} step={step} language={language}/>;
  if (isCurrentKind(kind)) return <CurrentConceptView key={`${kind}:${step}`} kind={kind} step={step} language={language}/>;
  if (isFokkerKind(kind)) return <FokkerConceptView key={`${kind}:${step}`} kind={kind} step={step} language={language}/>;
  if (isMasterKind(kind)) return <MasterConceptView key={`${kind}:${step}`} kind={kind} step={step} language={language}/>;
  if (isFiniteSizeKind(kind)) return <FiniteSizeConceptView key={`${kind}:${step}`} kind={kind} step={step} language={language}/>;
  if (isDetailedBalanceKind(kind)) return <DetailedBalanceConceptView key={`${kind}:${step}`} kind={kind} step={step} language={language}/>;
  if (isCorrelationKind(kind)) return <CorrelationConceptView key={`${kind}:${step}`} kind={kind} step={step} language={language}/>;
  if (isPhaseKind(kind)) return <PhaseConceptView key={`${kind}:${step}`} kind={kind} step={step} language={language}/>;
  if (isBoltzmannGibbsKind(kind)) return <BoltzmannConceptView kind={kind} step={step} language={language}/>;
  if (isPartitionKind(kind)) return <PartitionConceptView kind={kind} step={step} language={language}/>;
  if (isEnsembleKind(kind)) return <EnsembleConceptView kind={kind} step={step} language={language}/>;
  if (isMicroMacroKind(kind)) return <MicroMacroConceptView kind={kind} step={step} language={language}/>;
  if (isFreeEnergyKind(kind)) return <FreeEnergyConceptView kind={kind} step={step} language={language}/>;
  if (isThermodynamicLawKind(kind)) return <ThermodynamicLawConcept kind={kind} step={step}/>;
  if (isTemperatureHeatWorkKind(kind)) return <TemperatureHeatWorkConcept kind={kind} step={step}/>;
  if (isLogicKind(kind)) return <LogicConcept kind={kind} step={step} description={label}/>;
  if (isFoundationKind(kind)||isElementaryKind(kind)||isDiscreteKind(kind)||isMatrixStructureKind(kind)||isAnalysisKind(kind)||isComplexKind(kind)||((((((isContinuumKind(kind)||isElectromagnetismKind(kind))||isWavesKind(kind))||isAnalyticalMechanicsKind(kind))||isClassicalMechanicsKind(kind))||isPhysicalModelingKind(kind))||isPrivacyKind(kind)||isGameKind(kind)||isCombinatorialKind(kind))||isControlKind(kind)||isCausalKind(kind)||isTheoryKind(kind)||isAlgorithmKind(kind)||isScientificKind(kind)||isStochasticKind(kind)||isDifferentialKind(kind)||isNumericalKind(kind)||isTransportKind(kind)||isVariationalKind(kind)||isOptimizationKind(kind)||isConvexKind(kind)||isSymmetryKind(kind)||isGeometryKind(kind)||isHarmonicKind(kind)||isApproximationKind(kind)||isFunctionalKind(kind)||isCodingKind(kind)||isGraphicalKind(kind)||isHighDimensionalKind(kind)||isInformationKind(kind)||isBayesKind(kind)||isStatisticsKind(kind)||isMeasureKind(kind)||isProbabilityKind(kind)||isAbstractAlgebraKind(kind)||isTensorKind(kind)) return <FoundationConcept kind={kind} label={label} step={step} />;
  if (isCalculusKind(kind)) return <CalculusGraph kind={kind} stage={step} label={label} reference />;
  if (isAlgebraKind(kind)) return <AlgebraConcept kind={kind} step={step} label={label} />;
  const origin: Vec2 = kind === 'vectors-and-forces' ? [60, 225] : kind === 'linear-transforms' ? [55, 225] : [200, 140];
  const scale = kind === 'vectors-and-forces' ? 45 : kind === 'linear-transforms' ? 72 : 32;
  const xy = (p: Vec2): Vec2 => [origin[0] + p[0] * scale, origin[1] - p[1] * scale];
  const arrow = (from: Vec2, to: Vec2, color: string, dashed = false) => <path d={`M${xy(from)}L${xy(to)}`} stroke={`var(--learning-${color})`} strokeWidth="3" strokeDasharray={dashed ? '5 4' : undefined} fill="none" markerEnd={`url(#${id}-${color})`} />;
  const dot = (p: Vec2, color: string) => <circle cx={xy(p)[0]} cy={xy(p)[1]} r="5" fill={`var(--learning-${color})`} />;
  const text = (p: Vec2, value: string, color = 'neutral') => <text x={xy(p)[0]} y={xy(p)[1]} fill={`var(--learning-${color})`} fontSize="13">{value}</text>;
  const shear = (p: Vec2): Vec2 => transform3([[1, 1, 0], [0, 1, 0], [0, 0, 1]], [...p, 0]).slice(0, 2) as Vec2;
  const start: Vec2 = [2, 2]; const next = gradientStep(2, 1, start, .25); const next2 = gradientStep(2, 1, next, .25); const large = gradientStep(2, 1, start, 1.25);
  return <svg viewBox="0 0 400 280" role="img" aria-label={label} className="ai-concept-diagram">
    <defs>{['blue', 'orange', 'green'].map(color => <marker key={color} id={`${id}-${color}`} markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0L7 3.5L0 7Z" fill={`var(--learning-${color})`} /></marker>)}</defs>
    {Array.from({ length: 9 }, (_, i) => i - 4).map(i => <path key={i} d={`M${xy([-4, i])}L${xy([7, i])}M${xy([i, -4])}L${xy([i, 7])}`} fill="none" stroke="var(--learning-grid)" strokeOpacity=".2" />)}
    <path d={`M10 ${origin[1]}H385M${origin[0]} 15V265`} stroke="var(--learning-neutral)" strokeOpacity=".7" />
    <text x="382" y={origin[1] - 8} fill="var(--learning-neutral)" fontSize="12">x</text><text x={origin[0] + 8} y="20" fill="var(--learning-neutral)" fontSize="12">y</text>
    {kind === 'vectors-and-forces' && <>
      {arrow([0, 0], [3, 0], 'blue')}{text([1, -.5], 'F₁: 3 N', 'blue')}
      {step >= 1 && <>{arrow([0, 0], [0, 4], 'orange', true)}{arrow([3, 0], [3, 4], 'orange')}{text([3.2, 2], 'F₂: 4 N', 'orange')}</>}
      {step >= 2 && <>{arrow([0, 0], [3, 4], 'green')}{text([.6, 2.5], 'Fnet: 5 N', 'green')}{text([3, 4.3], '(3, 4)')}</>}
      {text([-.4, -.5], '0')}
    </>}
    {kind === 'linear-transforms' && <>
      <path d={`M${xy([0, 0])}L${xy([1, 0])}L${xy([1, 1])}L${xy([0, 1])}Z`} stroke="var(--learning-neutral)" strokeWidth="2" strokeDasharray="5 4" fill="none" />
      {dot([1, 2], 'neutral')}{text([.4, 2.25], 'x = (1, 2)')}
      {step === 0 ? <>{arrow([0, 0], [1, 0], 'blue')}{arrow([0, 0], [0, 1], 'orange')}{text([.45, -.4], 'e₁', 'blue')}{text([-.45, 1.2], 'e₂', 'orange')}</>
        : <><path d={`M${xy([0, 0])}L${xy(shear([1, 0]))}L${xy(shear([1, 1]))}L${xy(shear([0, 1]))}Z`} stroke="var(--learning-blue)" fill="var(--learning-blue)" fillOpacity=".08" strokeWidth="2" />{arrow([0, 0], shear([1, 0]), 'blue')}{arrow([0, 0], shear([0, 1]), 'orange')}{text([.35, -.4], 'Ae₁', 'blue')}{text([.45, 1.25], 'Ae₂', 'orange')}</>}
      {step >= 2 && <>{arrow([1, 0], [3, 2], 'orange')}{arrow([0, 0], [3, 2], 'green')}{dot([3, 2], 'green')}{text([2.4, 2.25], 'Ax = (3, 2)', 'green')}</>}
    </>}
    {kind === 'gradients' && <>
      {[.5, 2, 6].map(loss => <ellipse key={loss} cx={origin[0]} cy={origin[1]} rx={Math.sqrt(loss) * scale} ry={Math.sqrt(2 * loss) * scale} fill="none" stroke="var(--learning-green)" strokeOpacity=".65" />)}
      {dot([0, 0], 'green')}{text([.2, -.45], '(0, 0)', 'green')}{dot(start, 'orange')}{text([2.1, 2.2], 'p₀ = (2, 2)', 'orange')}
      {step >= 1 && <>{arrow(start, next, 'blue')}{dot(next, 'blue')}{text([.35, 1.1], 'p₁', 'blue')}</>}
      {step === 2 && <>{arrow(next, next2, 'blue')}{dot(next2, 'blue')}{text([-.5, .8], 'p₂', 'blue')}</>}
      {step === 3 && <>{arrow(start, large, 'orange')}{dot(large, 'orange')}{text([-3.2, -.95], 'η=1.25', 'orange')}{text([2, .7], 'η=0.25', 'blue')}</>}
    </>}
  </svg>;
}
export default function ConceptWalkthrough({ kind }: { kind: LearningUnitId }) {
  const { language } = useLanguage(); const ko = language === 'ko'; const guide = learningGuides[kind]; const [step, setStep] = useState(0); const current = guide.steps[step];
  if (isReversalKind(kind)) return <ReversalConceptPanel key={kind} kind={kind} language={language}/>;
  if (kind === photonicInterferometerId) return <PhotonicInterferometerConcept language={language}/>;
  if (kind === photodiodeBandwidthId) return <PhotodiodeBandwidthConcept language={language}/>;
  if (kind === electronicBandsId) return <ElectronicBandsConcept language={language}/>;
  if (kind === phononDispersionId) return <PhononDispersionConcept language={language}/>;
  if (kind === weakFieldClockId) return <WeakFieldClockConcept language={language}/>;
  if (kind === rindlerObservablesId) return <RindlerObservablesConcept language={language}/>;
  if (kind === schwarzschildRedshiftId) return <SchwarzschildRedshiftConcept language={language}/>;
  if (kind === einsteinFluidCurvatureId) return <EinsteinFluidCurvatureConcept language={language}/>;
  if (kind === greatCircleGeodesicId) return <GreatCircleGeodesicConcept language={language}/>;
  if (kind === rotationalCurvatureId) return <RotationalCurvatureConcept language={language}/>;
  if (kind === fourMomentumId) return <FourMomentumConcept language={language}/>;
  if (kind === causalRelayId) return <CausalRelayConcept language={language}/>;
  if (kind === minkowskiPathId) return <MinkowskiPathConcept language={language}/>;
  if (kind === lorentzBoostId) return <LorentzBoostConcept language={language}/>;
  if (kind === slowFastMemoryId) return <SlowFastMemoryConcept language={language}/>;
  if (kind === heavyModeMatchingId) return <HeavyModeMatchingConcept language={language}/>;
  if (kind === isingDecimationId) return <IsingDecimationConcept language={language}/>;
  if (kind === isingCriticalityId) return <IsingCriticalityConcept language={language}/>;
  if (kind === legendreResponseUnitId) return <LegendreResponseConcept language={language}/>;
  if (kind === scalarGibbsUnitId) return <ScalarGibbsConcept language={language}/>;
  if(isBlockOrderTeachingId(kind)) return <BlockOrderReaderConcept language={language}/>;
  if(isThermalFieldTeachingId(kind)) return <ThermalFieldConcept language={language}/>;
  if(isRealTimePhaseTeachingId(kind)) return <RealTimePhaseConcept language={language}/>;
  if(isSaddleFluctuationTeachingId(kind)) return <SaddleFluctuationConcept language={language}/>;
  if(isEuclideanKernelTeachingId(kind)) return <EuclideanKernelConcept language={language}/>;
  if(isBrownianBridgeTeachingId(kind)) return <BrownianBridgeConcept language={language}/>;
  if(isSpatialSpectrumTeachingId(kind)) return <SpatialSpectrumConcept language={language}/>;
  if(isNongaussianFieldTeachingId(kind)) return <NongaussianFieldConcept language={language}/>;
  if(isGaussianFieldTeachingId(kind)) return <GaussianFieldConcept language={language}/>;
  if (isDuffingSecularTeachingId(kind)) return <DuffingSecularConcept key={kind} language={language}/>;
  if (isBoundaryLayerTeachingId(kind)) return <BoundaryLayerConcept key={kind} language={language}/>;
  if (isQuarticAsymptoticTeachingId(kind)) return <QuarticAsymptoticConcept key={kind} language={language}/>;
  if (isRegularPerturbationTeachingId(kind)) return <RegularPerturbationConcept key={kind} language={language}/>;
  if (isGraphGaussTeachingId(kind)) return <GraphGaussConcept key={kind} language={language}/>;
  if (isGaugeOrbitTeachingId(kind)) return <GaugeOrbitConcept key={kind} language={language}/>;
  if (isGraphGaugeTeachingId(kind)) return <GraphGaugeConcept key={kind} language={language}/>;
  if (isFieldModesTeachingId(kind)) return <FieldModesConcept key={kind} language={language}/>;
  if (isFieldBalanceTeachingId(kind)) return <FieldBalanceConcept key={kind} language={language}/>;
  if (isScalarFieldTeachingId(kind)) return <ScalarFieldConcept key={kind} language={language}/>;
  if (isInformationTeachingId(kind)) return <InformationConcept key={kind} kind={kind} language={language}/>;
  if (isQubitTeachingId(kind)) return <QubitConcept key={kind} kind={kind} language={language} experimentHref="#experiment"/>;
  if (isManyBodyTeachingId(kind)) return <ManyBodyConcept key={kind} kind={kind} language={language}/>;
  if (isQuantumVariationalKind(kind)) return <QuantumVariationalConceptPanel key={kind} kind={kind} language={language}/>;
  if (isEntanglementKind(kind)) return <TensorEntanglementConceptPanel key={kind} kind={kind} language={language}/>;
  if (isUncertaintyKind(kind)) return <UncertaintyConceptPanel key={kind} kind={kind} language={language}/>;
  if (isDensityKind(kind)) return <DensityConceptPanel key={kind} kind={kind} language={language}/>;
  if (isSpinKind(kind)) return <SpinConceptPanel key={kind} kind={kind} language={language}/>;
  if (isUnitaryKind(kind)) return <UnitaryConceptPanel key={kind} kind={kind} language={language}/>;
  if (isObservableKind(kind)) return <ObservableConceptPanel key={kind} kind={kind} language={language}/>;
  if (isHilbertKind(kind)) return <HilbertConceptPanel key={kind} kind={kind} language={language}/>;
  if (isCoarseKind(kind)) return <CoarseConceptPanel key={kind} kind={kind} language={language}/>;
  if (isSynchronizationKind(kind)) return <SynchronizationConceptPanel key={kind} kind={kind} language={language}/>;
  if (isThermalKind(kind)) return <ThermalConceptPanel key={kind} kind={kind} language={language}/>;
  if (isEntropyKind(kind)) return <EntropyConceptPanel key={kind} kind={kind} language={language}/>;
  if (isLrKind(kind)) return <LrConceptPanel key={kind} kind={kind} language={language}/>;
  if (isFdtKind(kind)) return <FdtConceptPanel key={kind} kind={kind} language={language}/>;
  if (isCurrentKind(kind)) return <CurrentConceptPanel key={kind} kind={kind} language={language}/>;
  if (isFokkerKind(kind)) return <FokkerConceptPanel key={kind} kind={kind} language={language}/>;
  if (isMasterKind(kind)) return <MasterConceptPanel key={kind} kind={kind} language={language}/>;
  if (isFiniteSizeKind(kind)) return <FiniteSizeConceptPanel key={kind} kind={kind} language={language}/>;
  if (isDetailedBalanceKind(kind)) return <DetailedBalanceConceptPanel key={kind} kind={kind} language={language}/>;
  if (isCorrelationKind(kind)) return <CorrelationConceptPanel key={kind} kind={kind} language={language}/>;
  if (isPhaseKind(kind)) return <PhaseConceptPanel key={kind} kind={kind} language={language}/>;
  return <section className="ai-intuition" id="visual-intuition" aria-labelledby="ai-intuition-title">
    <div className="ai-section-heading"><div><span className="ai-eyebrow">BUILD YOUR INTUITION</span><h2 id="ai-intuition-title">{ko ? '먼저 그림으로 이해하기' : 'Build the idea with a diagram'}</h2></div><p>{ko ? '한 번에 한 단계씩 살펴보세요.' : 'Explore one step at a time.'}</p></div>
    <p className="ai-intuition-lead">{guide.intuition[language]}</p>
    <div className="ai-concept-steps" role="group" aria-label={ko ? '개념도 단계 선택' : 'Choose a diagram step'}>{guide.steps.map((item, index) => <Button key={index} type="button" size="sm" variant="outline" aria-pressed={step === index} onClick={() => setStep(index)}>{index + 1}. {item.title[language]}</Button>)}</div>
    <div className="ai-concept-content"><figure><Diagram kind={kind} step={step} label={current.diagram[language]} language={language} /><figcaption>{isPhaseKind(kind) ? (ko ? '선은 반올림된 표본 그림입니다. 근의 개수·역할은 인증된 계산으로 읽고, 유한 평균과 극한의 최소 집합을 구분하세요.' : 'Lines are rounded sampled drawings. Read certified calculations for root counts and roles; distinguish finite means from limiting minimum sets.') : isBoltzmannGibbsKind(kind) ? (ko ? '고정 기준 측도·조건부 갱신·연속 정규화의 준비를 worker에서 비교합니다. 양자화 경로와 이론적 법칙, 밀도와 확률 질량을 구별하세요.' : 'Workers compare fixed reference measures, conditional updates and continuous normalization. Distinguish quantized paths from theoretical laws, and densities from probability masses.') : isPartitionKind(kind) ? (ko ? '각 단계의 고정 준비를 별도 worker에서 계산합니다. 가중치·모멘트·기준 이동,결합·지원·source 응답,무한 사다리의 절단과 경계를 비교합니다.' : 'Fixed preparations are calculated in separate workers. Compare weights, moments and offsets; coupling, support and source response; and infinite-ladder cutoffs and boundaries.') : isEnsembleKind(kind) ? (ko ? '각 단계의 고정 준비를 worker에서 계산합니다. 저장고 크기·온도·화학 편향을 비교하는 막대는 시간 진화가 아닙니다.' : 'Fixed preparations for each step are calculated in workers. Bars comparing reservoir size, temperature or chemical bias are not time evolution.') : isMicroMacroKind(kind) ? (ko ? '각 단계는 별도의 고정 준비와 관측을 계산합니다. 아래 실험의 저장 설정이나 실제 시간 진화와 구별하세요.' : 'Each step calculates a separate fixed preparation and observation. Distinguish it from saved experiment settings and physical time evolution.') : isFreeEnergyKind(kind) ? (ko ? '각 단계의 고정 자유에너지 예제를 계산합니다. 실제 시간 경로와 구별하고 표의 값·부호·로그·상쇄 상태를 확인하세요.' : 'Each step calculates its fixed free-energy example. Distinguish it from a time trajectory and inspect values, signs, logs and cancellation status.') : isThermodynamicLawKind(kind) ? (ko ? '각 단계의 명시적인 고정 순환·저장고·스펙트럼 예제를 실제로 계산합니다. 표에서 단위·부호·표현 범위를 확인하세요.' : 'Each step calculates its stated fixed cycle, reservoir or spectrum example. Read units, signs and representable ranges in the table.') : isTemperatureHeatWorkKind(kind) ? (ko ? '각 단계는 가중 평형·에너지 보존·엔트로피 생성·연결 없음 또는 두 경로·고정 부피·같은 엔트로피·작은 열의 고정 계산 예제입니다.' : 'Steps use fixed calculated examples of weighted equilibrium, energy conservation, entropy production and disconnection, or two paths, fixed volume, equal entropy and small heat.') : isContinuumConservativeKind(kind) ? (ko ? '각 단계는 실제 셀 폭·저장 체적 대조·시간 보정·작은 변화 또는 실제 충격파·sonic 면 결손·정지entropy·경계통과의 고정 예제입니다.' : 'Steps use fixed examples of physical widths, storage controls, time correction and tiny updates, or actual shocks, sonic face defects, stationary entropy and boundary crossing.') : isContinuumTurbulenceKind(kind) ? (ko ? '각 단계는 실제 위상 공분산·같은 평균의 외력·중앙 점성 특이점·평균 일 수지 또는 양의 분산·두 번 필터·부호 전달·외력 에너지 예산의 고정 예제입니다.' : 'Steps use fixed examples of phase covariance, forcing at the same mean, centre viscosity singularity and mean work, or positive variance, repeated filtering, signed transfer and forced energy budgets.') : isContinuumMultiphysicsKind(kind) ? (ko ? '각 단계는 양방향 열 작용·단열 교환·전도 소산·일방향 근사 또는 단자 조건·냉각 열 수송·국소 전기 일·발전 포트의 고정 예제입니다.' : 'Steps use fixed examples of reciprocal thermal action, adiabatic exchange, conduction loss and a one-way approximation, or terminal conditions, cooling transport, local electric work and generating ports.') : isContinuumDimensionlessKind(kind) ? (ko ? '각 단계는 두 시간·하류 경계층·실제 플럭스·닮음 또는 Da·밀폐면 농도·실제 질량 수지·효과도의 고정 예제입니다.' : 'Steps use fixed examples of transport clocks, the downstream layer, actual flux and similarity, or Da, sealed-wall concentration, mass budgets and effectiveness.') : isContinuumConstitutiveKind(kind) ? (ko ? '각 단계는 병렬/직렬·초기 기억·첫 주기·작은3차 변화 또는 실제 변형·두 응력과 면적·관측 법선·작은 신장 에너지의 고정 예제입니다.' : 'Steps use fixed examples of parallel/series response, initial memory, first-cycle work and a tiny cubic change, or deformation, two stress/area measures, observation normals and small-stretch energy.') : isContinuumReactionKind(kind) ? (ko ? '각 단계는 비평형 완화·양방향 평형 전환·종별 창 수지·작은 생성량 또는 손계산 전선·실험실 정지·원천 누락·포화 여량의 고정 예제입니다.' : 'Steps use fixed examples of relaxation, equilibrium turnover, species window budgets and tiny production, or a worked front, laboratory stationarity, source omission and plateau deficits.') : isContinuumDiffusionKind(kind) ? (ko ? '각 단계는 온도/열유속·유한 창·현열/노름·작은 감쇠 또는 대칭 퍼짐·이류·창의 출입·꼬리 적분의 고정 예제입니다.' : 'Steps use fixed examples of temperature/flux, a finite window, sensible energy/norm and tiny decay, or symmetric spreading, advection, window exchange and tail integration.') : isContinuumFlowKind(kind) ? (ko ? '각 단계는 단일 음향·접촉·정지 특성·비선형 잔차 또는 네 가속도·감쇠·점성 영·작은 비영 변화의 고정 예제입니다.' : 'Steps use fixed examples of single sound, contact, a stationary characteristic and nonlinear residuals, or four accelerations, decay, zero viscosity and a tiny nonzero change.') : isContinuumStressStrainKind(kind) ? (ko ? '각 단계는 기울어진 면·평면 내 압축·균형 체적력·누락 대조 또는 정확한 회전·전단·별도 skew 지도·작은 보정의 고정 예제입니다.' : 'Steps use fixed examples of a tilted face, in-plane compression, balanced body force and an omission control, or exact rotation, shear, a distinct skew map and a tiny correction.') : isContinuumAssumptionKind(kind) ? (ko ? '각 단계는 반대 속도·가중 평균·빈 지지·변하는 관측 폭 또는 후보 구간·구간 소멸·작은 현재 창·빈 저장소의 고정 예제를 사용합니다.' : 'Steps use fixed examples of opposing streams, mass weighting, empty support and changing width, or feasible intervals, absent intervals, small selected windows and an empty reservoir.') : isContinuumKind(kind) ? (ko ? '각 단계는 고정·물질·이동 관측과 작은 변화 또는 팽창·압축·고정 구간·작은 압력 일의 고정 예제를 사용합니다.' : 'Steps use fixed examples of stationary/material/moving observations and tiny changes,or expansion/compression/fixed intervals and tiny pressure work.') : (['transmission-line-lumped-limit-energy','layered-medium-bloch-effective-limit'].includes(kind)) ? (ko ? '각 단계는 단자 근사·저장 비교·가까운 유한 공진·작은 보정 또는 장파장·금지대·유한 적층·작은 위상의 고정 예제를 사용합니다.' : 'Steps use fixed examples of terminal approximation,storage,finite near-resonance and tiny corrections,or long waves,gaps,finite stacks and small phase.') : (['rc-circuit-field-closure-approximation','finite-area-coil-sensor-observability'].includes(kind)) ? (ko ? '각 단계는 회로 수지·빠른 장 결함·저항 스윕·작은 변화 또는 면적 보정·이동·중복 교정·근접 기하의 고정 예제를 사용합니다.' : 'Steps use fixed examples of circuit balance,fast-field defects,resistance sweep and tiny changes,or area correction,motion,repeated calibration and nearby geometry.') : (['time-dependent-gauge-field-invariance','lorenz-gauge-source-initial-compatibility'].includes(kind)) ? (ko ? '각 단계는 동시·부분 변환·Lorenz·고정 루프 또는 호환 원천·공명 반례·초기 제약·일정 G의 고정 예제를 사용합니다.' : 'Steps use fixed examples of coordinated/partial changes,Lorenz and the fixed loop,or compatible sources,resonant failure,initial constraints and constant G.') : (['dielectric-interface-free-bound-charge','normal-incidence-impedance-interface'].includes(kind)) ? (ko ? '각 단계는 법선 장·표면 전하·접선 점프·작은 전하 또는 파동 성분·진폭과 전력·필요한 시트·작은 임피던스 차이의 고정 예제를 사용합니다.' : 'Steps use fixed examples of normal fields, surface charge, tangential mismatch and tiny charge, or traveling components, amplitude/power, required sheets and tiny impedance contrast.') : (['poynting-source-work-local-balance','transverse-pulse-energy-flux-balance'].includes(kind)) ? (ko ? '각 단계는 저장·경계 흐름, 일정 전기 저장, 전류 누락 또는 초기 에너지·단방향 구간·영자기장 반례·작은 흐름의 고정 예제를 사용합니다.' : 'Steps use fixed examples of storage and boundary flow, constant electric storage and omitted current, or initial energy, a one-way slab, a zero-B control and tiny flow.') : (['faraday-induction-fixed-moving-loop','electromagnetic-wave-equation-transverse-initial-data'].includes(kind)) ? (ko ? '각 단계는 장·이동 경로·정적 자기장의 운동 기여·작은 구간 변화 또는 초기 자료·단방향 전파·영자기장 반례·작은 시간의 고정 예제를 사용합니다.' : 'Steps use fixed examples of fields, moving paths, static-B motion and small interval changes, or initial data, one-way propagation, a zero-B control and tiny time.') : (['maxwell-plane-wave-local-integral-laws','maxwell-charge-displacement-continuity'].includes(kind)) ? (ko ? '각 단계는 파동·주파수 음성 대조·측정 방향 또는 원점·전류·축적·초기 제약의 고정 기준 예제를 사용합니다.' : 'Steps use fixed reference examples for fields, detuning and orientation, or the origin, currents, accumulation and initial constraints.') : (['electrostatic-potential-reference-path-work','magnetic-vector-potential-curl-gauge'].includes(kind)) ? (ko ? '각 단계는 경로·시험 전하·기준 또는 게이지·경계·측정 방향의 고정 기준 예제를 사용합니다.' : 'Steps use fixed path, test-charge and reference examples, or fixed gauge, interface and orientation examples.') : (['gauss-symmetry-electrostatic-sphere','ampere-symmetry-steady-current-cylinder'].includes(kind)) ? (ko ? '각 단계는 반지름 절반·경계·축 또는 중심·배경장의 고정 기준 예제를 사용합니다.' : 'Steps use fixed half-radius, interface, axis or center, and background reference examples.') : isLogicKind(kind) ? (ko ? '고정한 유한 기준 모형입니다. 단계별 설명과 별도의 일반 논증을 구분해 읽으세요.' : 'A fixed finite reference model. Read the step explanations separately from any general proof.') : (isFoundationKind(kind)||isElementaryKind(kind)||isDiscreteKind(kind)||isMatrixStructureKind(kind)||isAnalysisKind(kind)||isComplexKind(kind)||((((((isContinuumKind(kind)||isElectromagnetismKind(kind))||isWavesKind(kind))||isAnalyticalMechanicsKind(kind))||isClassicalMechanicsKind(kind))||isPhysicalModelingKind(kind))||isPrivacyKind(kind)||isGameKind(kind)||isCombinatorialKind(kind))||isControlKind(kind)||isCausalKind(kind)||isTheoryKind(kind)||isAlgorithmKind(kind)||isScientificKind(kind)||isStochasticKind(kind)||isDifferentialKind(kind)||isNumericalKind(kind)||isTransportKind(kind)||isVariationalKind(kind)||isOptimizationKind(kind)||isConvexKind(kind)||isSymmetryKind(kind)||isGeometryKind(kind)||isHarmonicKind(kind)||isApproximationKind(kind)||isFunctionalKind(kind)||isCodingKind(kind)||isGraphicalKind(kind)||isHighDimensionalKind(kind)||isInformationKind(kind)||isBayesKind(kind)||isStatisticsKind(kind)||isMeasureKind(kind)||isProbabilityKind(kind)||isAbstractAlgebraKind(kind)||isTensorKind(kind)) ? (ko ? '고정 기준 예제의 그림과 수치입니다. 각 단계에서 같은 그림의 의미를 차례로 살펴보세요.' : 'Fixed reference plot and readout. Each step interprets another aspect of the same example.') : isCalculusKind(kind) ? (ko ? '값과 민감도의 계산 그래프 · 작은 화면에서는 가로로 이동하세요. 선 길이는 크기를 나타내지 않습니다.' : 'Computation graph of values and sensitivities · Scroll horizontally on narrow screens. Edge lengths do not encode magnitude.') : kind === 'gradients' ? (ko ? '매개변수 평면의 등고선 · 높이는 표시하지 않습니다.' : 'Contours in parameter space · Height is not shown.') : (ko ? 'z=0인 평면의 개념도 · 3D 실험과 구분해서 읽으세요.' : 'Concept diagram in the plane z=0 · Read alongside the 3D experiment.')} {ko ? '고정된 기준 예제이며 아래 실험의 설정과 독립적입니다.' : 'This fixed reference example is independent of the experiment settings below.'}</figcaption></figure>
      <div className="ai-concept-explanation" aria-live="polite" aria-atomic="true"><span className="ai-eyebrow">{step + 1} / {guide.steps.length}</span><h3>{current.title[language]}</h3><p>{current.explanation[language]}</p><p className="ai-concept-equation">{current.formula}</p></div>
    </div>
    <div className="ai-concept-pagination"><Button type="button" variant="outline" size="sm" disabled={step === 0} onClick={() => setStep(n => n - 1)}>{ko ? '이전 단계' : 'Previous step'}</Button><Button type="button" variant="outline" size="sm" disabled={step === guide.steps.length - 1} onClick={() => setStep(n => n + 1)}>{ko ? '다음 단계' : 'Next step'}</Button><a href="#experiment">{ko ? '실험에서 확인하기 →' : 'Check it in the experiment →'}</a></div>
  </section>;
}
