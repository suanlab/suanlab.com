/** Inventory signals are review hints, never automatic proof of comprehensibility. */
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import { learningUnits, learningGuides } from '../../src/data/interactive-book';
import { curriculumDomains } from '../../src/data/learning-curriculum';
import { learningLocation } from '../../src/data/learning-paths';
import { isFirstStepId } from '../../src/lib/learning-first-steps';
type Node = { type: string; value?: string; children?: Node[] };
const parser = unified().use(remarkParse).use(remarkGfm).use(remarkMath);
async function main() {
  const units = [];
  for (const unit of learningUnits) {
    const location = learningLocation(unit.id)!;
    const languages: Record<string, unknown> = {};
    for (const locale of ['ko', 'en']) {
      const source = await readFile(`content/learning/physics-math-for-ai/${unit.id}.${locale}.md`, 'utf8');
      const metrics = { inlineMath: 0, displayMath: 0, codeCells: 0, images: 0, plainMathCandidateNodes: 0, inlineCodeMathCandidateNodes: 0 };
      const walk = (node: Node) => {
        if (node.type === 'inlineMath') metrics.inlineMath++;
        if (node.type === 'math') metrics.displayMath++;
        if (node.type === 'code') metrics.codeCells++;
        if (node.type === 'image') metrics.images++;
        if (node.value && /[=≈∑∫∇α-ωΑ-Ω]|(?:\d|[a-z])\^[\d{]/u.test(node.value)) {
          if (node.type === 'text') metrics.plainMathCandidateNodes++;
          if (node.type === 'inlineCode') metrics.inlineCodeMathCandidateNodes++;
        }
        node.children?.forEach(walk);
      };
      walk(parser.parse(source) as unknown as Node);
      languages[locale] = { sha256: createHash('sha256').update(source).digest('hex'), ...metrics };
    }
    units.push({ id: unit.id, path: location.path.id, domain: location.domain.id, firstReading: isFirstStepId(unit.id) ? 'interactive-prototype' : 'pending-review', fullChapterPedagogyReview: 'pending', fullChapterMathMigration: 'pending-review', mathematicsSurfacesToReview: ['manuscript', 'concept-explanation', 'quiz', 'lab-results', 'diagram-labels'], registryFormulaReview: 'pending', guideFormulaCount: learningGuides[unit.id].steps.length, languages });
  }
  const inventory = { schemaVersion: 1, scope: 'Existing Physics and Mathematics for AI textbook; full pedagogy redesign is distinct from earlier research review', counts: { domains: curriculumDomains.length, topics: curriculumDomains.reduce((n, d) => n + d.topics.length, 0), units: learningUnits.length, manuscripts: learningUnits.length * 2, guidedExperiments: Object.values(learningGuides).reduce((n, g) => n + g.experiments.length, 0), firstReadingPrototypes: units.filter(u => u.firstReading === 'interactive-prototype').length, fullChapterRedesignsVerified: 0 }, candidateDetection: 'AST text and inline-code nodes containing equation-like signs; heuristic only. Code cells are excluded. A zero count does not certify that every mathematical expression uses LaTeX.', completionCriteria: ['Intuitive first explanation with prerequisite support', 'Authored LaTeX throughout the chapter and user-facing mathematics', 'At least two fully explained worked examples', 'Concept figures and manipulable examples tied to the mathematics', 'Meaningful staged motion with reduced-motion alternative', 'Bilingual semantic and mathematical agreement', 'Independent numerical and browser verification'], units };
  await writeFile('docs/online-learning/physics-math-for-ai/pedagogy-redesign-inventory.json', JSON.stringify(inventory, null, 2) + '\n');
  console.log(JSON.stringify(inventory.counts));
}
main().catch(error => { console.error(error); process.exitCode = 1; });
