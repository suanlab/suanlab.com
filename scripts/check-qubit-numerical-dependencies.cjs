const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const esbuild = require('esbuild');

async function check() {
  const directory = path.resolve('.runtime/learning/quantum-circuit-response');
  fs.mkdirSync(directory, { recursive: true });
  const graphs = [];
  for (const entry of ['learning-qubit-circuit-validator.ts', 'learning-qubit-channel-validator.ts']) {
    const result = await esbuild.build({ entryPoints: [`src/lib/${entry}`], bundle: true, write: false, format: 'esm', platform: 'browser', metafile: true, logLevel: 'silent' });
    const inputs = Object.keys(result.metafile.inputs);
    for (const excluded of ['learning-qubit-circuit-exact.ts', 'learning-qubit-circuit-report.ts', 'learning-qubit-channel.ts']) {
      assert(!inputs.includes(`src/lib/${excluded}`), `${entry} imports product ${excluded}`);
    }
    assert(!inputs.some(p => p.startsWith('src/data/') || p.startsWith('src/components/')));
    graphs.push({ entry, productCircuitAndChannelModelsExcluded: true, inputs,
      dependencyHashes: Object.fromEntries(inputs.map(p => [p, crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex')])), metafile: result.metafile });
  }
  // New preparations must remain unreachable from the registered teaching UI.
  const walk = directory => fs.readdirSync(directory, { withFileTypes: true }).flatMap(entry => entry.isDirectory() ? walk(path.join(directory, entry.name)) : [path.join(directory, entry.name)]);
  const consumers = walk('src').filter(p => /\.(?:ts|tsx)$/.test(p) && !path.basename(p).startsWith('learning-qubit-') && /learning-qubit-/.test(fs.readFileSync(p, 'utf8')));
  assert.deepEqual(consumers, []);
  const report = { validatorGraphs: graphs, registeredSourceConsumers: consumers, heavyPreparationsNotRegistered: true };
  fs.writeFileSync(path.join(directory, 'numerical-dependency-graphs.json'), JSON.stringify(report, null, 2) + '\n');
  console.log('Two independent validator bundles exclude product circuit/channel models; preparations remain unregistered.');
}
check().catch(error => { console.error(error); process.exitCode = 1; });
