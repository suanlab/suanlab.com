/* Run from the repository root, with the 6d2ff9de3308198973a0f46df5e53e9d2090eedb baseline available in Git. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { createHash } = require('node:crypto');
const { execFileSync } = require('node:child_process');
const sha = value => createHash('sha256').update(value).digest('hex');
const baseline = '6d2ff9de3308198973a0f46df5e53e9d2090eedb';
const output = process.env.SUANLAB_NONEQUILIBRIUM_SOURCE_REPORT || '.runtime/learning/nonequilibrium-research-audit/source.json';
const expected = [
  "two-state-continuous-time-generator-euler",
  "poisson-uniformization-potential-events",
  "state-dependent-holding-time-jump-occupation",
  "ornstein-uhlenbeck-density-euler-moments",
  "fokker-planck-cell-mass-boundaries",
  "stationary-gaussian-circulating-current",
  "continuous-cycle-path-reversal",
  "thermal-ou-fluctuation-response",
  "markov-perturbation-linear-response",
  "stationary-jump-path-entropy-production",
  "driven-two-state-heat-work",
  "reservoir-resolved-two-state-heat-flow",
  "hopf-amplitude-critical-relaxation",
  "two-oscillator-locking-phase-slips",
  "finite-kuramoto-collective-order",
  "markov-partition-lumpability-memory",
  "linear-hidden-memory-markov-limit"
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
import {learningUnits,learningGuides} from './src/data/interactive-book';
import {curriculumDomains} from './src/data/learning-curriculum';
import {getLearningContent} from './src/lib/learning-content';
(async()=>{
 const domain=curriculumDomains.find(d=>d.id==='nonequilibrium')!;
 const closure=new Set<string>(),active=new Set<string>();
 function visit(id:string){if(active.has(id))throw Error('Prerequisite cycle '+id);if(closure.has(id))return;const u=learningUnits.find(u=>u.id===id);if(!u)throw Error('Missing '+id);active.add(id);for(const p of u.prerequisiteIds)visit(p);active.delete(id);closure.add(id);}
 domain.unitIds.forEach(visit);
 const units=[];
 for(const id of domain.unitIds){
  const unit=learningUnits.find(u=>u.id===id)!,guide=learningGuides[id];
  for(const p of unit.prerequisiteIds)if(learningUnits.findIndex(u=>u.id===p)>=learningUnits.indexOf(unit))throw Error('Prerequisite order '+id);
  const c=await getLearningContent(id);
  const exported=(await import('node:fs')).readFileSync('out/book/online/physics-math-for-ai/'+id+'/index.html','utf8');
  // Korean SSR and the streamed Flight payload together carry both locales.
  const prefix='self.__next_f.push(';
  const flight=exported.split('<script>').filter(s=>s.startsWith(prefix+'[1,')).map(s=>JSON.parse(s.slice(prefix.length,s.indexOf('</script>')-1))[1]).join('');
  for(const lang of ['en','ko'])if(!exported.includes(c[lang])&&!flight.includes(c[lang]))throw Error('Export does not contain exact current '+lang+' rendered manuscript: '+id);
  units.push({id,unit,guide,render:Object.fromEntries(['en','ko'].map(lang=>[lang,{katexError:c[lang].includes('katex-error'),outline:c.outline[lang],exactCurrentRenderInExport:true}]))});
 }
 console.log(JSON.stringify({domain,units,closure:[...closure],domainIds:curriculumDomains.map(d=>d.id),registry:{units:learningUnits.length,guides:Object.values(learningGuides).reduce((n,g)=>n+g.experiments.length,0),domains:curriculumDomains.length,topics:curriculumDomains.reduce((n,d)=>n+d.topics.length,0)}}));
})().catch(e=>{console.error(e);process.exitCode=1});
`], { encoding: 'utf8', timeout: 120_000, maxBuffer: 20 * 1024 * 1024 }));
  assert.deepEqual(data.domain.unitIds, expected);
  assert.equal(data.domain.topics.length, 10);
  assert.deepEqual(data.registry, { units: 650, guides: 1969, domains: 51, topics: 509 });
  assert.ok(data.domain.prerequisiteIds.every(id => data.domainIds.includes(id)));
  const reviewed = JSON.parse(fs.readFileSync('scripts/fixtures/reviewed-learning-manuscripts.json', 'utf8'));
  const rows = [], assets = new Map(), external = new Set(), originalFormulaPairs = [];
  let executions = 0, researchEquations = 0, exercises = 0, originalPythonExecutions = 0, originalTypeScriptExecutions = 0;
  for (const { id, unit, guide, render } of data.units) {
    assert.ok(unit.questions.length >= 2, id);
    assert.ok(guide.steps.length >= 3, id);
    assert.equal(guide.experiments.length, 3, id);
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
      let restored=source.slice(0,start)+source.slice(end);
      const corrections={"hopf-amplitude-critical-relaxation": {"ko": {"previous": "아직 읽기 화면·Worker·시각화 등록을 하지 않았으며 두 진동자와 다수 진동자의 가족도 남아 있다.", "current": "현재는 읽기 화면·Worker·시각화와 두 진동자·다수 진동자 가족이 모두 등록되어 있으며 통합 검증 근거는 연구 진행표에 연결되어 있다."}, "en": {"previous": "Reader, Worker and visualization registration are still pending, as are the pairwise and collective families.", "current": "The reader, Worker, visualization and the pairwise and collective families are now registered; their integration evidence is linked in the research progress record."}}, "two-oscillator-locking-phase-slips": {"ko": {"previous": "현재 원고·수치 기반은 Worker·교수 화면·읽기 등록 전 단계다.", "current": "현재 원고·수치 기반과 Worker·교수 화면·reader가 등록되어 있으며 연구 진행표에 통합 검증 근거를 연결했다."}, "en": {"previous": "The present numerical foundation and draft still await Worker, teaching UI and reader registration.", "current": "The numerical foundation, Worker, teaching UI and reader are now registered; the research progress record links their integration evidence."}}};
      const correction=corrections[id]?.[lang];
      if(correction){assert.ok(restored.includes(correction.current),filename);restored=restored.replace(correction.current,correction.previous);}
      assert.deepEqual(Buffer.from(restored),original,filename);
      assert.equal(reviewed[filename].reviewedHash, sha(source), filename);
      assert.ok(reviewed[filename].previousHashes.includes(sha(original)), filename);
      assert.ok(fs.existsSync(reviewed[filename].review), filename);
      const originalText = original.toString('utf8');
      const numberedHeadings = [...originalText.matchAll(/^### (?:(?:Exercise|연습|문제) )?[1-6](?:\.|\b|\s)/gm)];
      const solutions = originalText.match(/\*\*(?:Answer|Solution|Explanation|해설|풀이|답)[.:]\*\*/g) || [];
      const inline=[...originalText.matchAll(/^\*\*[1-6]\.([^\n]+)$/gm)];
      const format=numberedHeadings.length===6?'six numbered H3 questions with labelled explanations':'six inline bold questions with explanations';
      if(numberedHeadings.length===6){assert.equal(solutions.length,6,filename);assert.equal(inline.length,0,filename);}
      else{assert.equal(numberedHeadings.length,0,filename);assert.equal(inline.length,6,filename);assert.ok(inline.every(item=>/\*\*.+/.test(item[1])),filename);}
      assert.doesNotMatch(source, /[\u0000-\u0008\u000b-\u001f]/, filename);
      if (lang === 'en') assert.doesNotMatch(source, /[가-힣]/, filename);
      const whole = parser.parse(source), research = parser.parse(source.slice(start, end));
      const cells = nodes(whole, 'code').filter(n => n.lang === 'python');
      const originalTree = parser.parse(originalText);
      const originalPython = nodes(originalTree, 'code').filter(n => n.lang === 'python');
      assert.equal(cells.length, originalPython.length + 1, filename);
      const codes = nodes(research, 'code').filter(n => n.lang === 'python').map(n => n.value);
      assert.equal(codes.length, 1, filename);
      assert.ok(cells.some(n => n.value === codes[0]), filename);
      for (const cell of originalPython) { execFileSync('python3', ['-c', cell.value], {encoding:'utf8', timeout:20000}); originalPythonExecutions++; }
      for (const cell of nodes(originalTree,'code').filter(n => n.lang === 'typescript')) { execFileSync('node_modules/.bin/tsx',['-e',cell.value],{encoding:'utf8',timeout:20000}); originalTypeScriptExecutions++; }
      const displayed = nodes(research, 'math').map(n => n.value);
      assert.equal(nodes(research, 'emphasis').length, 0, filename + ': no unintended emphasis in research symbols');
      assert.equal(nodes(research,'heading').filter(n=>n.depth===4).length,3,filename);
      const baselineMath = nodes(parser.parse(originalText), 'math').map(n => n.value.replace(/\s+/g, ' ').trim());
      const stdout = execFileSync('python3', ['-c', codes[0]], { encoding: 'utf8', timeout: 20_000 });
      executions++;
      researchEquations += displayed.length;
      exercises += 6;
      assert.equal(render[lang].katexError, false, filename);
      const headings = render[lang].outline;
      assert.equal(new Set(headings.map(h => h.id)).size, headings.length, filename);
      const images = nodes(research, 'image');
      assert.equal(images.length, 1, filename);
      assert.equal(nodes(whole,'image').length, nodes(originalTree,'image').length + 1, filename);
      const image = images[0];
      assert.ok(image.alt && image.url.endsWith(`.${lang}.svg`), filename);
      const asset = 'public' + image.url;
      assert.ok(fs.existsSync(asset), asset);
      assert.deepEqual(fs.readFileSync(asset), fs.readFileSync('out' + image.url), asset + ': exported SVG must match current source');
      assets.set(asset, sha(fs.readFileSync(asset)));
      for (const link of nodes(whole, 'link')) if (/^https?:/.test(link.url)) external.add(link.url);
      const tables = nodes(research, 'table');
      assert.equal(tables.length, 1, filename);
      const plain = n => 'value' in n ? n.value : (n.children || []).map(plain).join('');
      const tableRows = tables[0].children.slice(1).map(row => row.children.map(plain));
      assert.deepEqual(stdout.trim().split('\n').map(row => row.trim().split(/\s+/)), tableRows, filename + ': executed table parity');
      pair.push({ displayed, codes, stdout, baselineMath });
      rows.push({ id, lang, path: filename, sha256: sha(source), originalSha256: sha(original),
        researchSha256: sha(source.slice(start, end)), displayedResearchEquations: displayed.length,
        pythonSha256: sha(codes[0]), stdoutSha256: sha(stdout), originalExercisesWithSolutions: 6, exactCurrentRenderInExport: render[lang].exactCurrentRenderInExport,
        exerciseFormat: format, originalPythonCells: originalPython.length,
        outlineCount: headings.length, image: image.url, review: reviewed[filename].review });
    }
    const {baselineMath: enOriginal, ...enResearch} = pair[0], {baselineMath: koOriginal, ...koResearch} = pair[1];
    assert.deepEqual(enResearch, koResearch, `${id}: research math/code/output bilingual match`);
    assert.deepEqual(enOriginal.map(equation => equation.replace(/\s/g,'').replace(/[.,;]$/,'')), koOriginal.map(equation => equation.replace(/\s/g,'').replace(/[.,;]$/,'')), `${id}: original math differs beyond whitespace and terminal punctuation`);
    originalFormulaPairs.push({id, whitespaceAndTerminalPunctuationEquivalent: true, exactPair: JSON.stringify(enOriginal) === JSON.stringify(koOriginal), enCount: enOriginal.length, koCount: koOriginal.length, differences: enOriginal.map((equation,index) => equation === koOriginal[index] ? null : {index,en:equation,ko:koOriginal[index]}).filter(Boolean)});
  }
  assert.equal(rows.length, 34);
  assert.equal(assets.size, 34);
  assert.equal(executions, 34);
  assert.equal(exercises, 204);





  const result = { schemaVersion: 1, status: 'source-and-executable-inventory-verified', baselineCommit: baseline,
    domain: data.domain, registry: data.registry, prerequisites: { domainIds: data.domain.prerequisiteIds, closure: data.closure },
    originalManuscriptsPreserved: 34, editorialCorrections: {"hopf-amplitude-critical-relaxation": {"ko": {"previous": "아직 읽기 화면·Worker·시각화 등록을 하지 않았으며 두 진동자와 다수 진동자의 가족도 남아 있다.", "current": "현재는 읽기 화면·Worker·시각화와 두 진동자·다수 진동자 가족이 모두 등록되어 있으며 통합 검증 근거는 연구 진행표에 연결되어 있다."}, "en": {"previous": "Reader, Worker and visualization registration are still pending, as are the pairwise and collective families.", "current": "The reader, Worker, visualization and the pairwise and collective families are now registered; their integration evidence is linked in the research progress record."}}, "two-oscillator-locking-phase-slips": {"ko": {"previous": "현재 원고·수치 기반은 Worker·교수 화면·읽기 등록 전 단계다.", "current": "현재 원고·수치 기반과 Worker·교수 화면·reader가 등록되어 있으며 연구 진행표에 통합 검증 근거를 연결했다."}, "en": {"previous": "The present numerical foundation and draft still await Worker, teaching UI and reader registration.", "current": "The numerical foundation, Worker, teaching UI and reader are now registered; the research progress record links their integration evidence."}}}, reviewedManuscripts: rows, pairedResearchUnits: 17,
    pythonCellsExecuted: executions, originalPythonExecutions, originalTypeScriptExecutions, displayedResearchEquations: researchEquations, originalExercisesWithSolutions: exercises,
    guides: data.units.reduce((n,u)=>n+u.guide.experiments.length,0), quizQuestions: data.units.reduce((n,u)=>n+u.unit.questions.length,0), conceptSteps: data.units.reduce((n,u)=>n+u.guide.steps.length,0),
    originalFormulaPairs, researchQuestionsWithExplanations: 102,
    assets: [...assets].map(([path, sha256]) => ({ path, sha256 })), externalReferences: [...external].sort(),
    readerVerification: 'Whole-domain browser verification remains separate; run the eight research checker families against one frozen final static export.' };
  fs.mkdirSync(path.dirname(output), { recursive: true });
  fs.writeFileSync(output, JSON.stringify(result, null, 2) + '\n');
  console.log(`Nonequilibrium research source audit passed: ${data.domain.topics.length} topics, ${data.units.length} units, ${rows.length} preserved manuscripts, ${executions} research Python cells, ${researchEquations} research equations, ${exercises} original explained exercises, ${assets.size} figures.`);
})().catch(error => { console.error(error); process.exitCode = 1; });
