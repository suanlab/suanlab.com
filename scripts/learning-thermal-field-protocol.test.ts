import assert from 'node:assert/strict';
import { test } from 'node:test';
import { spawnSync } from 'node:child_process';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import ts from 'typescript';
import { thermalFieldDefault, thermalFieldExamples, readThermalFieldInput, type ThermalFieldInput } from '../src/lib/learning-thermal-field-input';
import { evaluateThermalFieldRequest } from '../src/lib/learning-thermal-field-evaluator';
import { evaluateThermalFieldValidationRequest } from '../src/lib/learning-thermal-field-validation-evaluator';
import { validateThermalFieldResult } from '../src/lib/learning-thermal-field-validation';
import { thermalFieldInputKey, thermalFieldSuccessHeader, type ThermalFieldSuccess } from '../src/lib/learning-thermal-field-protocol';

const root = '.runtime/learning/thermal-field-protocol';
mkdirSync(root, { recursive: true });
const producer = (input: ThermalFieldInput, requestId = 1): ThermalFieldSuccess => {
  const result = evaluateThermalFieldRequest({ version: 1, requestId, input });
  if (!result.ok) assert.fail(JSON.stringify(result));
  return result;
};
const receipt = (input: ThermalFieldInput, response: unknown, requestId = 1) => evaluateThermalFieldValidationRequest({ version: 1, requestId, input, response });
const clone = <T>(x: T): T => structuredClone(x);
function python(args: string[], input?: string) {
  const run = spawnSync('python3', ['scripts/learning/verify-thermal-field-foundation.py', ...args], { input, encoding: 'utf8', timeout: 120000, maxBuffer: 12000000 });
  assert.equal(run.status, 0, run.stderr);
  return JSON.parse(run.stdout) as { cases: { input: ThermalFieldInput; expected: Record<string, unknown>; observationDiagnostic: Record<string, string> | null }[] };
}
const reference = python(['--observations']);
const exampleReferences = python(['--inputs', '--observations'], JSON.stringify(thermalFieldExamples.map(x => x.input)));
writeFileSync(`${root}/independent-references.json`, JSON.stringify({ foundation: reference, examples: exampleReferences }, null, 2) + '\n');
const all = [...reference.cases, ...exampleReferences.cases];

test('66 reference models pass the full independent response gate, including nonnormalizable states and tiny determinants', () => {
  let normalizable = 0, nonnormalizable = 0;
  const observations = [];
  for (const [index, row] of all.entries()) {
    const output = producer(row.input, index + 1), exact = output.result.exact;
    for (const [key, value] of Object.entries(row.expected)) assert.deepEqual(exact[key as keyof typeof exact], value, `reference ${index}: ${key}`);
    assert.equal(receipt(row.input, output, index + 1).phase, 'validated', `gate reference ${index}`);
    assert(Object.isFrozen(output) && Object.isFrozen(output.input.sourceDensity) && Object.isFrozen(exact));
    if (exact.kind === 'normalizable') {
      normalizable++; observations.push({ input: row.input, ranges: output.result.observations, diagnostic: row.observationDiagnostic });
    } else { nonnormalizable++; assert.equal(output.result.observations, null); }
  }
  // Decimal comparisons stay separate from the independent analytic interval gate.
  const diagnostic = spawnSync('python3', ['-c', `
import json,sys
from decimal import Decimal,localcontext
from fractions import Fraction
with localcontext() as c:
 c.prec=230
 for row in json.load(sys.stdin):
  for key,expected in row['diagnostic'].items():
   bounds=row['ranges'][key];lo,hi=Fraction(bounds['lower']),Fraction(bounds['upper'])
   l=Decimal(lo.numerator)/Decimal(lo.denominator);h=Decimal(hi.numerator)/Decimal(hi.denominator)
   assert l <= Decimal(expected) <= h, (key,str(l),expected,str(h))
print('Decimal diagnostics contained')
`], { input: JSON.stringify(observations), encoding: 'utf8', timeout: 30000 });
  assert.equal(diagnostic.status, 0, diagnostic.stderr);
  writeFileSync(`${root}/verification.json`, JSON.stringify({ status: 'complete-response-evaluator-verified', models: all.length, normalizable, nonnormalizable,
    decimalDiagnostics: normalizable * 4, analyticIntervalGate: 'independent 2048-bit arithmetic;512-bit dyadic bounds with at most6quanta width',
    actualWorkersVerified: false, browserVerified: false, unitRegistered: false }, null, 2) + '\n');
});

type Segment = string | number;
function leaves(value: unknown, prefix: Segment[] = []): { at: Segment[]; value: unknown }[] {
  if (!value || typeof value !== 'object') return [{ at: prefix, value }];
  return Object.entries(value).flatMap(([key, v]) => leaves(v, [...prefix, Array.isArray(value) ? Number(key) : key]));
}
function mutate(value: unknown, at: Segment[], replacement: unknown) {
  let cursor = value as Record<string | number, unknown>;
  for (const k of at.slice(0, -1)) cursor = cursor[k] as Record<string | number, unknown>;
  cursor[at[at.length - 1]] = replacement;
}
test('every exact-result leaf is independently checked, including derivative, Gram, variance and branch metadata', () => {
  let rejected = 0;
  for (const input of [thermalFieldDefault, thermalFieldExamples.find(x => x.id === 'two-dimensional')!.input,
    thermalFieldExamples.find(x => x.id === 'zero-pivot-indefinite')!.input]) {
    const original = producer(input);
    assert(validateThermalFieldResult(input, original.result));
    for (const leaf of leaves(original.result.exact)) {
      const altered = clone(original.result);
      mutate(altered.exact, leaf.at, typeof leaf.value === 'number' ? leaf.value + 1 : leaf.value === '0' ? '1' : '0');
      assert.equal(validateThermalFieldResult(input, altered), null, `unchecked exact leaf ${leaf.at.join('.')}`);
      rejected++;
    }
  }
  writeFileSync(`${root}/exact-leaf-mutations.json`, JSON.stringify({ rejected, scope: 'default,two-dimensional,andnonnormalizable-result' }, null, 2) + '\n');
});

test('logarithm observations reject false, reversed, wide, noncanonical, oversized and mislabeled ranges', () => {
  const original = producer(thermalFieldDefault);
  assert(original.result.observations);
  let rejected = 0;
  for (const key of ['logPartition', 'freeEnergy', 'thermodynamicEntropy', 'gibbsEntropy'] as const) {
    for (const replacement of [{ lower: '0', upper: '0' }, { lower: '100000', upper: '100001' }, { lower: '-100000', upper: '100000' },
      { lower: '1', upper: '-1' }, { lower: '0/2', upper: '1' }, { lower: '1/0', upper: '2' }, { lower: '9'.repeat(25000), upper: '1' },
      { lower: '0', upper: '1', hidden: true }]) {
      const output = clone(original); output.result.observations![key] = replacement;
      assert.equal(receipt(thermalFieldDefault, output).phase, 'rejected', key); rejected++;
    }
  }
  for (const patch of [{ precisionBits: 256 }, { interpretation: 'continuum' }, { gibbsEntropy: undefined }]) {
    const output = clone(original); Object.assign(output.result.observations!, patch);
    assert.equal(receipt(thermalFieldDefault, output).phase, 'rejected'); rejected++;
  }
  const nonnormalInput = thermalFieldExamples.find(x => x.id === 'massless-divergence')!.input;
  const nonnormal = clone(producer(nonnormalInput)); nonnormal.result.observations = clone(original.result.observations);
  assert.equal(receipt(nonnormalInput, nonnormal).phase, 'rejected');
  writeFileSync(`${root}/observation-mutations.json`, JSON.stringify({ rejected: rejected + 1 }, null, 2) + '\n');
});

test('ownership covers all thermal jets and request identity; equivalent bounded fractions normalize to one key', () => {
  const original = producer(thermalFieldDefault);
  for (const patch of [{ requestId: 2 }, { version: 2 }, { ok: false }, { kind: 'gaussian-field' }, { inputKey: 'wrong' },
    { extra: true }, { result: { ...original.result, extra: true } }]) {
    assert.equal(receipt(thermalFieldDefault, { ...original, ...patch }).phase, 'rejected');
  }
  for (const field of ['mass', 'stiffness', 'backgroundDensity'] as const) for (const key of ['value', 'first', 'second'] as const) {
    const changed = clone(thermalFieldDefault); changed[field][key] = '3';
    assert.equal(receipt(changed, original).phase, 'rejected');
  }
  const changedSource = clone(thermalFieldDefault); changedSource.sourceDensity[1].second = '1';
  assert.equal(receipt(changedSource, original).phase, 'rejected');
  assert.equal(thermalFieldSuccessHeader(original, changedSource, 1), false);
  const alias: ThermalFieldInput = clone(thermalFieldDefault); alias.mass.value = '2/2'; alias.temperature = '4/2'; alias.sourceDensity[1].first = '-0/2';
  assert.equal(thermalFieldInputKey(alias), thermalFieldInputKey(thermalFieldDefault));
  assert.equal(receipt(alias, original).phase, 'validated');
  const mutable = clone(thermalFieldDefault), owned = producer(mutable);
  mutable.shape[0] = 8; mutable.sourceDensity[0].value = '100';
  assert.equal(owned.input.shape[0], 2); assert.equal(owned.input.sourceDensity[0].value, '1');
  assert(Object.isFrozen(owned.result.exact));
});

test('descriptor-only input and wire snapshots reject hostile data without invoking getters', () => {
  let calls = 0;
  const getter = () => { calls++; throw Error('Getter executed'); };
  const cases: unknown[] = [null, {}, { ...thermalFieldDefault, extra: true }, { ...thermalFieldDefault, shape: Array(1) },
    { ...thermalFieldDefault, sourceDensity: Array(2) }, { ...thermalFieldDefault, temperature: '0' },
    { ...thermalFieldDefault, referenceScale: '1e-100' }, { ...thermalFieldDefault, mass: { ...thermalFieldDefault.mass, value: '9'.repeat(19) } }];
  const inputGetter = { ...thermalFieldDefault }; Object.defineProperty(inputGetter, 'temperature', { get: getter, enumerable: true }); cases.push(inputGetter);
  const nested = clone(thermalFieldDefault); Object.defineProperty(nested.mass, 'first', { get: getter, enumerable: true }); cases.push(nested);
  const arrayGetter = clone(thermalFieldDefault); Object.defineProperty(arrayGetter.shape, '0', { get: getter, enumerable: true }); cases.push(arrayGetter);
  const withSymbol = clone(thermalFieldDefault); Object.defineProperty(withSymbol, Symbol('extra'), { value: 1 }); cases.push(withSymbol);
  for (const input of cases) {
    assert.equal(readThermalFieldInput(input), null);
    const response = evaluateThermalFieldRequest({ version: 1, requestId: 1, input });
    assert.equal(response.ok, false);
  }
  const own = producer(thermalFieldDefault), accessor = clone(own);
  Object.defineProperty(accessor.result.exact, 'mean', { get: getter, enumerable: true });
  assert.equal(receipt(thermalFieldDefault, accessor).phase, 'rejected');
  const huge = clone(own) as unknown as { result: { exact: { mean: unknown } } };
  huge.result.exact.mean = Array(257).fill('0');
  assert.deepEqual(receipt(thermalFieldDefault, huge), { version: 1, requestId: 1, phase: 'rejected', error: 'resource-limit' });
  for (const id of [0, -1, 1.5, NaN, Infinity, Number.MAX_SAFE_INTEGER + 1, '1']) {
    assert.equal(evaluateThermalFieldRequest({ version: 1, requestId: id, input: thermalFieldDefault }).ok, false);
  }
  const requestGetter = { version: 1, requestId: 1, input: thermalFieldDefault }; Object.defineProperty(requestGetter, 'input', { get: getter });
  assert.equal(evaluateThermalFieldRequest(requestGetter).ok, false);
  assert.equal(calls, 0);
});

test('reader transport and independent gate have no runtime path to the producer engine or its interval implementation', () => {
  function imports(entry: string) {
    const visited = new Set<string>();
    function visit(file: string) {
      const absolute = path.resolve(file); if (visited.has(absolute)) return; visited.add(absolute);
      const source = ts.createSourceFile(absolute, readFileSync(absolute, 'utf8'), ts.ScriptTarget.Latest, true);
      for (const statement of source.statements) if (ts.isImportDeclaration(statement) && ts.isStringLiteral(statement.moduleSpecifier)) {
        const clause = statement.importClause;
        if (clause?.isTypeOnly || clause?.namedBindings && ts.isNamedImports(clause.namedBindings) && !clause.name && clause.namedBindings.elements.every(x => x.isTypeOnly)) continue;
        const specifier = statement.moduleSpecifier.text;
        if (specifier.startsWith('.')) visit(path.resolve(path.dirname(absolute), specifier + '.ts'));
      }
    }
    visit(entry); return [...visited].map(file => path.relative(process.cwd(), file));
  }
  for (const entry of ['src/lib/learning-thermal-field-input.ts', 'src/lib/learning-thermal-field-protocol.ts', 'src/lib/learning-thermal-field-validation-evaluator.ts']) {
    const dependency = imports(entry);
    for (const forbidden of ['src/lib/learning-thermal-field.ts', 'src/lib/learning-thermal-field-observations.ts', 'src/lib/learning-thermal-field-evaluator.ts',
      'src/lib/learning-quantum-hilbert-core.ts', 'src/lib/learning-real-time-interval.ts', 'src/lib/learning-interval-transcendentals.ts']) assert(!dependency.includes(forbidden), `${entry}: ${forbidden}`);
  }
});
