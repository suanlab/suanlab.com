import { thermalContactState, thermalContactWindow, type ThermalContactParameters } from './learning-thermal-contact';
import { idealGasPathState, idealGasPathComparison, type IdealGasPathParameters } from './learning-ideal-gas-paths';
export const temperatureHeatWorkKinds = ['finite-body-thermal-contact-heat-entropy', 'ideal-gas-two-path-heat-work'] as const;
export type TemperatureHeatWorkKind = typeof temperatureHeatWorkKinds[number];
export const isTemperatureHeatWorkKind = (id: string): id is TemperatureHeatWorkKind => (temperatureHeatWorkKinds as readonly string[]).includes(id);
export const thermalContactBounds = [[.5,4],[.5,4],[250,350],[250,350],[0,2],[0,8],[0,8],[0,8]];
export const gasPathBounds = [[.25,2],[250,350],[250,350],[.005,.04],[.005,.04],[0,1]];
function valid(v: unknown, bounds: number[][]): v is number[] {
  return Array.isArray(v) && v.length === bounds.length && v.every((x,i) => typeof x === 'number' && Number.isFinite(x) && x >= bounds[i][0] && x <= bounds[i][1]);
}
export const validThermalContactValues = (v: unknown): v is number[] => valid(v, thermalContactBounds);
export const validGasPathValues = (v: unknown): v is number[] => valid(v, gasPathBounds);
export function evaluateThermalContact(values: number[]) {
  if (!validThermalContactValues(values)) throw new RangeError('Invalid thermal contact controls');
  const p: ThermalContactParameters = { capacity1:values[0], capacity2:values[1], initialTemperature1:values[2], initialTemperature2:values[3], conductance:values[4] };
  return { parameters:p, query:thermalContactState(p,values[5]), window:values[6] <= values[7] ? thermalContactWindow(p,values[6],values[7]) : null,
    history:Array.from({length:65},(_,i)=>thermalContactState(p,i/8)), windowStart:values[6], windowEnd:values[7] };
}
export function evaluateGasPaths(values: number[]) {
  if (!validGasPathValues(values)) throw new RangeError('Invalid gas path controls');
  const p: IdealGasPathParameters = { moles:values[0], initialTemperature:values[1], finalTemperature:values[2], initialVolume:values[3], finalVolume:values[4] };
  return { parameters:p, queryA:idealGasPathState(p,'A',values[5]), queryB:idealGasPathState(p,'B',values[5]), comparison:idealGasPathComparison(p),
    history:Array.from({length:65},(_,i)=>({progress:i/64,a:idealGasPathState(p,'A',i/64),b:idealGasPathState(p,'B',i/64)})) };
}
