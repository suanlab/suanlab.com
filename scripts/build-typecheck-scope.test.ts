import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import { resolve } from 'node:path';
import ts from 'typescript';

const root = process.cwd();
function project(name: string) {
  const file = resolve(root, name);
  const config = ts.readConfigFile(file, ts.sys.readFile);
  assert.equal(config.error, undefined);
  const parsed = ts.parseJsonConfigFileContent(config.config, ts.sys, root, undefined, file);
  assert.deepEqual(parsed.errors, []);
  return { files: new Set(parsed.fileNames.map(name => resolve(name))), options: parsed.options };
}
const sources = (directory: string) => readdirSync(directory, { recursive: true })
  .filter((name): name is string => typeof name === 'string' && /\.tsx?$/.test(name))
  .map(name => resolve(directory, name));

test('production checking covers every source while full checking retains all test and CLI files', () => {
  const production = project('tsconfig.next.json'), full = project('tsconfig.json');
  const applicationFiles = sources('src'), verificationFiles = sources('scripts');
  assert(applicationFiles.length > 0 && verificationFiles.length > 0);
  for (const path of applicationFiles) {
    assert(production.files.has(path), `Production source omitted: ${path}`);
    assert(full.files.has(path), `Full source omitted: ${path}`);
  }
  for (const path of verificationFiles) {
    assert(!production.files.has(path), `Tooling leaked into build checker: ${path}`);
    assert(full.files.has(path), `Tooling lost its type check: ${path}`);
  }
  for (const config of [production, full]) {
    assert.equal(config.options.strict, true);
    assert.equal(config.options.noEmit, true);
  }
  const next = readFileSync('next.config.mjs', 'utf8');
  assert(next.includes("tsconfigPath: 'tsconfig.next.json'"));
  assert(!/ignoreBuildErrors\s*:\s*true/.test(next));
  const { scripts } = JSON.parse(readFileSync('package.json', 'utf8'));
  assert.match(scripts.typecheck, /^node --max-old-space-size=6144 node_modules\/typescript\/bin\/tsc --noEmit$/);
  assert(scripts.check.split(' && ').includes('npm run typecheck'));
});
