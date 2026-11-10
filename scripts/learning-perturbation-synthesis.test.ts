import test from 'node:test';import assert from 'node:assert/strict';
import {calculateRegularPerturbation} from '../src/lib/learning-regular-perturbation';
import {calculateQuarticAsymptotic} from '../src/lib/learning-quartic-asymptotic';
import {calculateBoundaryLayer} from '../src/lib/learning-boundary-layer';
import {calculateDuffingSecular} from '../src/lib/learning-duffing-secular';
import {readRegularPerturbationInput} from '../src/lib/learning-regular-perturbation-input';
import {readBoundaryLayerInput,boundaryLayerDefault} from '../src/lib/learning-boundary-layer-input';
import {readQuarticAsymptoticInput} from '../src/lib/learning-quartic-asymptotic-input';
import {duffingSecularDefault} from '../src/lib/learning-duffing-secular-input';
import {getLearningContent} from '../src/lib/learning-content';
const less=(a:string,b:string)=>{const[an,ad='1']=a.split('/').map(BigInt),[bn,bd='1']=b.split('/').map(BigInt);return an*BigInt(bd)<bn*BigInt(ad);};
test('capstone equilibrium and integral numbers have distinct correctly bounded errors',()=>{
 const regular=calculateRegularPerturbation({stiffness:'1',coupling:'1/10',source:'1',order:2}).approximations[2];assert.equal(regular.approximation,'93/100');assert.equal(regular.residual,'104357/10000000');assert.equal(regular.errorBound,regular.residual);assert(less(regular.rootDistance.upper,regular.errorBound));
 const integral=calculateQuarticAsymptotic({coupling:'1/10',order:2}),q=integral.approximations[2];assert.equal(q.partial,'613/640');assert.equal(q.firstOmittedMagnitude,'693/25600');assert.equal(integral.series,'divergent-asymptotic');assert(less(q.quadratureDistance.upper,q.firstOmittedMagnitude));assert(less('0',integral.quadratureInterval.lower)&&less(integral.quadratureInterval.upper,'1'));
});
test('zero interior boundary residual and a small Duffing residual cannot replace the missing diagnostic',()=>{
 const layer=calculateBoundaryLayer(boundaryLayerDefault);assert(layer.layer);assert.equal(layer.layer.outerUniformError,'1');assert(less('1/25000',layer.layer.compositeUniformError.lower));assert(less(layer.layer.compositeUniformError.upper,'1/20000'));
 // u=1 has u′=u″=0, hence zero interior residual, but violates u(0)=0.
 assert.equal(0*(1/10)+0*Number(boundaryLayerDefault.drift),Number(boundaryLayerDefault.forcing));assert.equal(1-Number(boundaryLayerDefault.left),1);
 const duffing=calculateDuffingSecular({...duffingSecularDefault,horizon:'1000000'}),phase=duffing.horizonDiagnostics.phaseDrift!;assert(less('204/25',phase.first.lower));assert(less(phase.first.upper,'817/100'));assert(less(duffing.approximations[0].normalizedResidualUniformBound,'1/1000'));assert.equal(duffing.horizonDiagnostics.fullGridResolvesEightPointsPerCycle,false);
});
test('excluded degenerations remain excluded; a stationary orbit does not acquire a period',()=>{
 assert.equal(readRegularPerturbationInput({stiffness:'0',coupling:'1',source:'1',order:2}),null);assert.equal(readBoundaryLayerInput({...boundaryLayerDefault,epsilon:'0'}),null);assert.equal(readQuarticAsymptoticInput({coupling:'-1',order:2}),null);
 const equilibrium=calculateDuffingSecular({...duffingSecularDefault,amplitude:'0'});assert.equal(equilibrium.frequency,null);assert.equal(equilibrium.period,null);assert.equal(equilibrium.horizonDiagnostics.phaseDrift,null);
});
test('both synthesis tables render under the existing final section with three valid review links',async()=>{
 const content=await getLearningContent('duffing-secular-frequency-long-time');for(const language of ['ko','en']as const){assert.equal(content.outline[language].length,20);assert(!content[language].includes('katex-error'));for(const id of ['regular-perturbation-residual-certificates','quartic-integral-asymptotic-truncation','singular-perturbation-boundary-layer-matching'])assert(content[language].includes(`href="../${id}/"`));assert(content[language].includes('0.0270703125'));assert(content[language].includes('8.16382'));}assert(!/[가-힣]/.test(content.en));
});
