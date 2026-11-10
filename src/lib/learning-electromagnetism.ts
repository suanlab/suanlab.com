import {electromagneticEffectiveKinds,evaluateElectromagneticEffective,type ElectromagneticEffectiveKind,type ElectromagneticEffectiveResult}from './learning-electromagnetic-effective-models-lab';
import {circuitSensorsKinds,evaluateCircuitSensors,type CircuitSensorsKind,type CircuitSensorsResult}from './learning-circuit-sensors-lab';
import {gaugeFreedomKinds,evaluateGaugeFreedom,type GaugeFreedomKind,type GaugeFreedomResult}from './learning-gauge-freedom-lab';
/** Electromagnetism's own lesson family; optical wave IDs remain unchanged. */
import {chargeCurrentFieldKinds,evaluateChargeCurrentFields,type ChargeCurrentFieldKind,type ChargeCurrentFieldResult}from './learning-charge-current-fields-lab';
import {electroMagnetostaticsKinds,evaluateElectroMagnetostatics,type ElectroMagnetostaticsKind,type ElectroMagnetostaticsResult}from './learning-electro-magnetostatics-lab';
import {potentialsKinds,evaluatePotentials,type PotentialsKind,type PotentialsResult}from './learning-potentials-lab';
import {maxwellKinds,evaluateMaxwell,type MaxwellKind,type MaxwellResult}from './learning-maxwell-lab';
import {inductionWavesKinds,evaluateInductionWaves,type InductionWavesKind,type InductionWavesResult}from './learning-induction-waves-lab';
import {poyntingEnergyKinds,evaluatePoyntingEnergy,type PoyntingEnergyKind,type PoyntingEnergyResult}from './learning-poynting-energy-lab';
import {materialInterfacesKinds,evaluateMaterialInterfaces,type MaterialInterfacesKind,type MaterialInterfacesResult}from './learning-material-interfaces-lab';
export const electromagnetismKinds=[...chargeCurrentFieldKinds,...electroMagnetostaticsKinds,...potentialsKinds,...maxwellKinds,...inductionWavesKinds,...poyntingEnergyKinds,...materialInterfacesKinds,...gaugeFreedomKinds,...circuitSensorsKinds,...electromagneticEffectiveKinds]as const;
export type ElectromagnetismKind=ChargeCurrentFieldKind|ElectroMagnetostaticsKind|PotentialsKind|MaxwellKind|InductionWavesKind|PoyntingEnergyKind|MaterialInterfacesKind|GaugeFreedomKind|CircuitSensorsKind|ElectromagneticEffectiveKind;
export type ElectromagnetismResult=Omit<ChargeCurrentFieldResult,'kind'>&{kind:ElectromagnetismKind;statics?:ElectroMagnetostaticsResult['statics'];scalarPotential?:PotentialsResult['scalarPotential'];vectorPotential?:PotentialsResult['vectorPotential'];planeMaxwell?:MaxwellResult['planeMaxwell'];radialMaxwell?:MaxwellResult['radialMaxwell'];inductionWave?:InductionWavesResult['inductionWave'];transversePulse?:InductionWavesResult['transversePulse'];sourceEnergy?:PoyntingEnergyResult['sourceEnergy'];pulseEnergy?:PoyntingEnergyResult['pulseEnergy'];dielectricBoundary?:MaterialInterfacesResult['dielectricBoundary'];normalInterface?:MaterialInterfacesResult['normalInterface'];gaugeTransform?:GaugeFreedomResult['gaugeTransform'];lorenzConstraint?:GaugeFreedomResult['lorenzConstraint'];rcCircuit?:CircuitSensorsResult['rcCircuit'];coilSensor?:CircuitSensorsResult['coilSensor'];transmissionLine?:ElectromagneticEffectiveResult['transmissionLine'];layeredMedium?:ElectromagneticEffectiveResult['layeredMedium']};
export const isElectromagnetismKind=(kind:string):kind is ElectromagnetismKind=>(electromagnetismKinds as readonly string[]).includes(kind);
export function evaluateElectromagnetism(kind:ElectromagnetismKind,values:number[]):ElectromagnetismResult{
 if(!isElectromagnetismKind(kind))throw new RangeError('Unknown electromagnetism lesson');
 if((electromagneticEffectiveKinds as readonly string[]).includes(kind))return evaluateElectromagneticEffective(kind as ElectromagneticEffectiveKind,values);
 if((circuitSensorsKinds as readonly string[]).includes(kind))return evaluateCircuitSensors(kind as CircuitSensorsKind,values);
 if((gaugeFreedomKinds as readonly string[]).includes(kind))return evaluateGaugeFreedom(kind as GaugeFreedomKind,values);
 if((materialInterfacesKinds as readonly string[]).includes(kind))return evaluateMaterialInterfaces(kind as MaterialInterfacesKind,values);
 if((poyntingEnergyKinds as readonly string[]).includes(kind))return evaluatePoyntingEnergy(kind as PoyntingEnergyKind,values);
 if((inductionWavesKinds as readonly string[]).includes(kind))return evaluateInductionWaves(kind as InductionWavesKind,values);
 if((maxwellKinds as readonly string[]).includes(kind))return evaluateMaxwell(kind as MaxwellKind,values);
 if((potentialsKinds as readonly string[]).includes(kind))return evaluatePotentials(kind as PotentialsKind,values);
 return (electroMagnetostaticsKinds as readonly string[]).includes(kind)?evaluateElectroMagnetostatics(kind as ElectroMagnetostaticsKind,values):evaluateChargeCurrentFields(kind as ChargeCurrentFieldKind,values);
}
