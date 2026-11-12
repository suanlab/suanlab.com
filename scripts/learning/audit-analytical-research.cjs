/* Run from the repository root, with the d486291 baseline available in Git. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { createHash } = require('node:crypto');
const { execFileSync } = require('node:child_process');
const sha = value => createHash('sha256').update(value).digest('hex');
const baseline = 'd486291';
const output = process.env.SUANLAB_ANALYTICAL_SOURCE_REPORT || '.runtime/learning/analytical-research-content.json';
const expected = [
  'polar-generalized-velocity-force-metric', 'double-pendulum-generalized-mass-metric',
  'free-particle-action-boundary-gauge', 'harmonic-action-stationary-saddle',
  'polar-lagrange-newton-equivalence', 'double-pendulum-lagrange-dynamics',
  'polar-hamiltonian-legendre-map', 'time-gauge-hamiltonian-energy',
  'poisson-observable-canonical-flow', 'angular-momentum-poisson-algebra',
  'four-dimensional-symplectic-maps', 'lagrangian-momentum-graphs',
  'noether-point-symmetry-defects', 'noether-boost-time-boundaries',
  'routh-cyclic-momentum-reduction', 'circle-dirac-constraint-geometry',
  'uniform-force-hamilton-jacobi-family', 'oscillator-hamilton-jacobi-caustic-atlas',
  'quartic-discrete-action-splitting', 'circle-rattle-projection-phase',
];
function nodes(tree, type) {
  const found = [];
  const walk = node => { if (node.type === type) found.push(node); node.children?.forEach(walk); };
  walk(tree);
  return found;
}

(async () => {
  const { unified } = await import('unified');
  const { default: parse } = await import('remark-parse');
  const { default: gfm } = await import('remark-gfm');
  const { default: math } = await import('remark-math');
  const parser = unified().use(parse).use(gfm).use(math);
  const data = JSON.parse(execFileSync('node_modules/.bin/tsx', ['-e', `
import {learningUnits,learningGuides,analyticalMechanicsLabs} from './src/data/interactive-book';
import {curriculumDomains} from './src/data/learning-curriculum';
import {getLearningContent} from './src/lib/learning-content';
(async()=>{
 const domain=curriculumDomains.find(d=>d.id==='analytical-mechanics')!;
 const closure=new Set<string>(),active=new Set<string>();
 function visit(id:string){if(active.has(id))throw Error('Prerequisite cycle '+id);if(closure.has(id))return;const u=learningUnits.find(u=>u.id===id);if(!u)throw Error('Missing '+id);active.add(id);for(const p of u.prerequisiteIds)visit(p);active.delete(id);closure.add(id);}
 domain.unitIds.forEach(visit);
 const units=[];
 for(const id of domain.unitIds){
  const unit=learningUnits.find(u=>u.id===id)!,guide=learningGuides[id],lab=analyticalMechanicsLabs[id];
  for(const p of unit.prerequisiteIds)if(learningUnits.findIndex(u=>u.id===p)>=learningUnits.indexOf(unit))throw Error('Prerequisite order '+id);
  const c=await getLearningContent(id);
  units.push({id,unit,guide,lab,render:Object.fromEntries(['en','ko'].map(lang=>[lang,{katexError:c[lang].includes('katex-error'),outline:c.outline[lang]}]))});
 }
 console.log(JSON.stringify({domain,units,closure:[...closure],domainIds:curriculumDomains.map(d=>d.id),registry:{units:learningUnits.length,guides:Object.values(learningGuides).reduce((n,g)=>n+g.experiments.length,0),domains:curriculumDomains.length,topics:curriculumDomains.reduce((n,d)=>n+d.topics.length,0)}}));
})().catch(e=>{console.error(e);process.exitCode=1});
`], { encoding: 'utf8', timeout: 120_000, maxBuffer: 20 * 1024 * 1024 }));
  assert.deepEqual(data.domain.unitIds, expected);
  assert.equal(data.domain.topics.length, 10);
  assert.deepEqual(data.registry, { units: 650, guides: 1969, domains: 51, topics: 509 });
  assert.ok(data.domain.prerequisiteIds.every(id => data.domainIds.includes(id)));
  const reviewed = JSON.parse(fs.readFileSync('scripts/fixtures/reviewed-learning-manuscripts.json', 'utf8'));
  const rows = [], assets = new Map(), external = new Set();
  let executions = 0, researchEquations = 0, exercises = 0;
  for (const { id, unit, guide, lab, render } of data.units) {
    assert.equal(unit.questions.length, 2, id);
    assert.equal(guide.steps.length, 4, id);
    assert.equal(guide.experiments.length, 3, id);
    for (const experiment of guide.experiments) {
      const preset = lab.presets[experiment.preset];
      assert.ok(preset, id);
      assert.equal(preset.values.length, lab.controls.length, id);
      preset.values.forEach((v, i) => assert.ok(Number.isInteger(v) && v >= lab.controls[i].min && v <= lab.controls[i].max, id));
    }
    const pair = [];
    for (const lang of ['en', 'ko']) {
      const filename = `content/learning/physics-math-for-ai/${id}.${lang}.md`;
      const source = fs.readFileSync(filename, 'utf8');
      const original = execFileSync('git', ['show', `${baseline}:${filename}`]);
      const start = source.search(/^## (Research extension:|연구 확장:)/m);
      assert.ok(start >= 0, filename);
      const headerEnd = source.indexOf('\n', start);
      const next = source.slice(headerEnd).search(/^## /m);
      assert.ok(next >= 0, filename);
      const end = headerEnd + next;
      assert.deepEqual(Buffer.from(source.slice(0, start) + source.slice(end)), original, filename);
      assert.equal(reviewed[filename].reviewedHash, sha(source), filename);
      assert.ok(reviewed[filename].previousHashes.includes(sha(original)), filename);
      assert.ok(fs.existsSync(reviewed[filename].review), filename);
      const inlineExercises = ['free-particle-action-boundary-gauge', 'harmonic-action-stationary-saddle'].includes(id);
      if (inlineExercises) {
        // These baseline lessons include the explanations in each numbered entry.
        const entries = [...source.matchAll(/^([1-6])\. (.+)$/gm)];
        const prior = [...original.toString('utf8').matchAll(/^([1-6])\. (.+)$/gm)];
        assert.deepEqual(entries.map(m => [m[1], m[2]]), prior.map(m => [m[1], m[2]]), filename);
        assert.deepEqual(entries.map(m => Number(m[1])), [1, 2, 3, 4, 5, 6], filename);
      } else {
        assert.equal((source.match(lang === 'en' ? /\*\*Solution:\*\*/g : /\*\*해설:\*\*/g) || []).length, 6, filename);
      }
      assert.doesNotMatch(source, /[\u0000-\u0008\u000b-\u001f]/, filename);
      if (lang === 'en') assert.doesNotMatch(source, /[가-힣]/, filename);
      const whole = parser.parse(source), research = parser.parse(source.slice(start, end));
      const cells = nodes(whole, 'code').filter(n => n.lang === 'python');
      assert.equal(cells.length, 1, filename);
      const codes = nodes(research, 'code').filter(n => n.lang === 'python').map(n => n.value);
      assert.equal(codes.length, 1, filename);
      assert.deepEqual(codes, cells.map(n => n.value), filename);
      const displayed = nodes(research, 'math').map(n => n.value);
      const stdout = execFileSync('python3', ['-c', codes[0]], { encoding: 'utf8', timeout: 20_000 });
      executions++;
      researchEquations += displayed.length;
      exercises += 6;
      assert.equal(render[lang].katexError, false, filename);
      const headings = render[lang].outline;
      assert.equal(new Set(headings.map(h => h.id)).size, headings.length, filename);
      const images = nodes(whole, 'image');
      assert.equal(images.length, 1, filename);
      const image = images[0];
      assert.ok(image.alt && image.url.endsWith(`.${lang}.svg`), filename);
      const asset = 'public' + image.url;
      assert.ok(fs.existsSync(asset), asset);
      assets.set(asset, sha(fs.readFileSync(asset)));
      for (const link of nodes(whole, 'link')) if (/^https?:/.test(link.url)) external.add(link.url);
      const tables = nodes(research, 'table');
      assert.equal(tables.length, 1, filename);
      pair.push({ displayed, codes, stdout });
      rows.push({ id, lang, path: filename, sha256: sha(source), originalSha256: sha(original),
        researchSha256: sha(source.slice(start, end)), displayedResearchEquations: displayed.length,
        pythonSha256: sha(codes[0]), stdoutSha256: sha(stdout), originalExercisesWithSolutions: 6,
        exerciseFormat: inlineExercises ? 'inline explanations preserved from baseline' : 'explicit solution labels',
        outlineCount: headings.length, image: image.url, review: reviewed[filename].review });
    }
    assert.deepEqual(pair[0], pair[1], `${id}: research math/code/output bilingual match`);
  }
  assert.equal(rows.length, 40);
  assert.equal(assets.size, 40);
  assert.equal(executions, 40);
  assert.equal(exercises, 240);
  const result = { schemaVersion: 1, status: 'source-and-executable-inventory-verified', baselineCommit: baseline,
    domain: data.domain, registry: data.registry, prerequisites: { domainIds: data.domain.prerequisiteIds, closure: data.closure },
    originalManuscriptsPreserved: 40, reviewedManuscripts: rows, pairedResearchUnits: 20,
    pythonCellsExecuted: executions, displayedResearchEquations: researchEquations, originalExercisesWithSolutions: exercises,
    guides: 60, quizQuestions: 40, conceptSteps: 80,
    assets: [...assets].map(([path, sha256]) => ({ path, sha256 })), externalReferences: [...external].sort(),
    readerVerification: 'Run scripts/check-analytical-domain.cjs against the final static export separately.' };
  fs.mkdirSync(path.dirname(output), { recursive: true });
  fs.writeFileSync(output, JSON.stringify(result, null, 2) + '\n');
  console.log(`Analytical research inventory passed: 10 topics, 20 paired units, 40 preserved originals, 40 Python executions, ${researchEquations} displayed research equations, 240 original explained exercises, 40 figures, ${data.closure.length} prerequisite units.`);
})().catch(error => { console.error(error); process.exitCode = 1; });
