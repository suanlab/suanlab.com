/** Equal-mass particles in units where c=1, at the same total lab kinetic energy. */
export function compareCollisionGeometry(mass: number, kinetic: number, finalMassSum: number) {
  if (![mass, kinetic, finalMassSum].every(Number.isFinite) || mass <= 0 || kinetic < 0 || finalMassSum < 0) {
    throw new RangeError('Expected positive particle mass and nonnegative energies');
  }
  const labEnergy = 2 * mass + kinetic;
  const fixedTargetMomentum = Math.sqrt(kinetic * (2 * mass + kinetic));
  const fixedTargetEnergy = Math.sqrt(2 * mass * labEnergy);
  const extraRestMass = Math.max(0, finalMassSum - 2 * mass);
  const fixedTargetThreshold = extraRestMass * (finalMassSum + 2 * mass) / (2 * mass);
  return {
    labEnergy,
    fixedTarget: {
      momentum: fixedTargetMomentum,
      centerSpeed: fixedTargetMomentum / labEnergy,
      centerEnergy: fixedTargetEnergy,
      threshold: fixedTargetThreshold,
      allowed: kinetic >= fixedTargetThreshold,
    },
    opposingBeams: {
      centerEnergy: labEnergy,
      threshold: extraRestMass,
      allowed: kinetic >= extraRestMass,
    },
  };
}
