import type {LearningUnit,LearningGuide} from './interactive-book';
import {manyBodySpinUnit,manyBodySpinGuide} from './many-body-spin-teaching';
import {manyBodyFermionUnit,manyBodyFermionGuide} from './many-body-fermion-teaching';
import {manyBodyBosonUnit,manyBodyBosonGuide} from './many-body-boson-teaching';
import {manyBodyMpsUnit,manyBodyMpsGuide} from './many-body-mps-teaching';
import {manyBodySchmidtUnit,manyBodySchmidtGuide} from './many-body-schmidt-teaching';
export const manyBodyTeaching=[
 {unit:manyBodySpinUnit,guide:manyBodySpinGuide},
 {unit:manyBodyFermionUnit,guide:manyBodyFermionGuide},
 {unit:manyBodyBosonUnit,guide:manyBodyBosonGuide},
 {unit:manyBodyMpsUnit,guide:manyBodyMpsGuide},
 {unit:manyBodySchmidtUnit,guide:manyBodySchmidtGuide},
] satisfies {unit:LearningUnit;guide:LearningGuide}[];
