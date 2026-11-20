import { curriculumDomains } from './learning-curriculum';
import { learningUnits, type BookText, type LearningUnitId } from './interactive-book';

export interface LearningPath {
  id: string;
  title: BookText;
  purpose: BookText;
  domainIds: string[];
}
const text = (ko: string, en: string): BookText => ({ ko, en });
export const learningPaths: LearningPath[] = [
  { id: 'language', title: text('수학의 언어와 기초', 'The language of mathematics'), purpose: text('기호를 읽고, 관계를 표현하고, 주장을 확인합니다.', 'Read symbols, describe relationships and check a claim.'), domainIds: ['logic-proof', 'elementary-math', 'discrete-math'] },
  { id: 'mathematics', title: text('AI 핵심 수학', 'Core mathematics for AI'), purpose: text('벡터·변화·공간을 그림과 계산으로 연결합니다.', 'Connect vectors, change and spaces through pictures and calculations.'), domainIds: ['linear-algebra', 'analysis', 'matrix-calculus', 'complex-analysis', 'tensor-methods', 'abstract-algebra', 'functional-analysis', 'approximation-kernels', 'harmonic-analysis', 'geometry-topology', 'symmetry-groups'] },
  { id: 'uncertainty', title: text('확률·통계·정보와 추론', 'Probability, information and inference'), purpose: text('불확실한 관측으로부터 무엇을 알 수 있는지 배웁니다.', 'Learn what uncertain observations can tell us.'), domainIds: ['probability', 'measure-probability', 'statistics', 'bayesian-inference', 'information-theory', 'high-dimensional', 'graphical-models', 'coding-algorithmic-information', 'stochastic-processes', 'causality'] },
  { id: 'computation', title: text('최적화·알고리즘·수치계산', 'Optimization, algorithms and computation'), purpose: text('좋은 해를 찾고, 계산 과정과 오차를 이해합니다.', 'Find useful solutions and understand computation and its errors.'), domainIds: ['convex-optimization', 'stochastic-optimization', 'variational-geometry', 'optimal-transport', 'numerical-computing', 'differential-equations', 'numerical-simulation', 'algorithms-complexity', 'discrete-optimization'] },
  { id: 'physics', title: text('물리 모델과 동역학', 'Physical models and dynamics'), purpose: text('힘·파동·흐름·열의 변화를 관찰하고 설명합니다.', 'Observe and explain changes in forces, waves, flows and heat.'), domainIds: ['physical-modeling', 'classical-mechanics', 'analytical-mechanics', 'waves-optics', 'electromagnetism', 'continuum-physics', 'thermo-statmech', 'nonequilibrium'] },
  { id: 'modern-physics', title: text('현대물리와 계산의 물리적 기반', 'Modern physics and physical computation'), purpose: text('양자·장·시공간과 계산 장치의 물리적 원리를 탐구합니다.', 'Explore quantum systems, fields, spacetime and computing devices.'), domainIds: ['quantum-physics', 'field-multiscale', 'relativity', 'matter-computation'] },
  { id: 'research', title: text('AI 연구와 문제 해결', 'AI research and problem solving'), purpose: text('모형·제어·실험을 연결해 연구 질문을 검증합니다.', 'Connect models, control and experiments to test research questions.'), domainIds: ['learning-theory', 'control-decision', 'game-theory', 'statistical-privacy', 'inverse-problems', 'research-methods'] },
];
export const unitById = new Map(learningUnits.map(unit => [unit.id, unit]));
export const domainById = new Map(curriculumDomains.map(domain => [domain.id, domain]));
// A lesson may have several relevant domains. Give it one home in the tree.
export const unitHomeDomain = new Map<LearningUnitId, string>();
for (const domain of curriculumDomains) for (const id of domain.unitIds) {
  if (!unitHomeDomain.has(id) || id === 'research-methods' && domain.id === 'research-methods') unitHomeDomain.set(id, domain.id);
}
export const domainUnits = (id: string) => domainById.get(id)!.unitIds.filter(unitId => unitHomeDomain.get(unitId) === id);
export const pathUnits = (path: LearningPath) => path.domainIds.flatMap(domainUnits);
export function learningLocation(id: LearningUnitId) {
  const domainId = unitHomeDomain.get(id);
  const domain = domainId ? domainById.get(domainId) : undefined;
  const path = learningPaths.find(path => path.domainIds.includes(domainId ?? ''));
  return domain && path ? { path, domain } : null;
}
const searchAliases: Partial<Record<LearningUnitId, string[]>> = {
  eigenvalues: ['PCA', 'principal component analysis', '주성분 분석'],
  svd: ['SVD', 'singular value decomposition', '특이값 분해'],
  gradients: ['gradient descent', '경사하강', '경사 하강'],
  'inner-products': ['dot product', 'projection', '직교 투영'],
};
export function searchLearningPath(query: string) {
  const needle = query.normalize('NFKC').trim().toLocaleLowerCase();
  const matches = (values: string[]) => values.some(value => value.normalize('NFKC').toLocaleLowerCase().includes(needle));
  const paths = learningPaths.flatMap(path => {
    const wholePath = !needle || matches([path.title.ko, path.title.en, path.purpose.ko, path.purpose.en]);
    const domains = path.domainIds.flatMap(id => {
      const domain = domainById.get(id)!;
      const wholeDomain = wholePath || matches([domain.title.ko, domain.title.en, ...domain.topics.flatMap(t => [t.ko, t.en])]);
      const units = domainUnits(id).filter(id => wholeDomain || matches([unitById.get(id)!.title.ko, unitById.get(id)!.title.en, unitById.get(id)!.subtitle.ko, unitById.get(id)!.subtitle.en, ...(searchAliases[id] ?? [])]));
      return wholeDomain || units.length ? [{ domain, unitIds: units }] : [];
    });
    return domains.length ? [{ path, domains }] : [];
  });
  return { paths, unitCount: paths.reduce((sum, p) => sum + p.domains.reduce((n, d) => n + d.unitIds.length, 0), 0) };
}
