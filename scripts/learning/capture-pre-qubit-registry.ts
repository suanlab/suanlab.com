import assert from 'node:assert/strict';
import { existsSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { learningUnits, learningGuides } from '../../src/data/interactive-book';
import { curriculumDomains } from '../../src/data/learning-curriculum';
import { qubitTeachingIds } from '../../src/lib/learning-qubit-teaching-input';
const destination = 'scripts/fixtures/pre-qubit-registry-hashes.json';
assert(!existsSync(destination), 'Historical registry evidence must not be overwritten');
assert.equal(learningUnits.length, 571);
assert.equal(learningUnits.reduce((n, unit) => n + learningGuides[unit.id].experiments.length, 0), 1713);
assert(qubitTeachingIds.every(id => !learningUnits.some(unit => String(unit.id) === id)));
const hash = (value: unknown) => createHash('sha256').update(JSON.stringify(value)).digest('hex');
writeFileSync(destination, JSON.stringify({
  unitOrder: learningUnits.map(unit => unit.id), guideOrder: Object.keys(learningGuides),
  units: Object.fromEntries(learningUnits.map(unit => [unit.id, hash(unit)])),
  guides: Object.fromEntries(learningUnits.map(unit => [unit.id, hash(learningGuides[unit.id])])),
  domains: Object.fromEntries(curriculumDomains.map(domain => [domain.id, { sha256: hash(domain), unitIds: domain.unitIds }])),
}, null, 2) + '\n');
console.log(JSON.stringify({ units: learningUnits.length, experiments: 1713, domains: curriculumDomains.length, destination }));
