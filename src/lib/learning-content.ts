import fs from 'node:fs/promises';
import path from 'node:path';
import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import remarkRehype from 'remark-rehype';
import rehypeSlug from 'rehype-slug';
import rehypeKatex from 'rehype-katex';
import rehypeStringify from 'rehype-stringify';
import { visit } from 'unist-util-visit';
import type { Root } from 'hast';

export interface LearningHeading { id: string; title: string; }
export interface LearningContent { ko: string; en: string; outline: { ko: LearningHeading[]; en: LearningHeading[] }; }

export async function getLearningContent(unitId: string): Promise<LearningContent> {
  if (!/^[a-z]+(?:-[a-z]+)*$/.test(unitId)) throw new Error('Invalid learning unit ID');
  const render = async (locale: 'ko' | 'en') => {
    const source = await fs.readFile(path.join(process.cwd(), 'content/learning/physics-math-for-ai', `${unitId}.${locale}.md`), 'utf8');
    const outline: LearningHeading[] = [];
    const file = await unified().use(remarkParse).use(remarkGfm).use(remarkMath).use(remarkRehype)
      .use(rehypeSlug).use(rehypeKatex, { strict: 'error' })
      .use(() => (tree: Root) => {
        let equation = 0; let table = 0;
        visit(tree, 'element', node => {
          if (node.tagName === 'h2' && typeof node.properties.id === 'string') {
            const parts: string[] = []; visit(node, 'text', text => { parts.push(text.value); });
            outline.push({ id: node.properties.id, title: parts.join('') });
          }
          if (node.tagName === 'pre') {
            node.properties.tabIndex = 0;
            node.properties.role = 'region';
            node.properties.ariaLabel = locale === 'ko' ? '코드 예제. 필요하면 좌우 화살표로 스크롤하세요.' : 'Code example. Use arrow keys to scroll if needed.';
          }
          if (node.tagName === 'table') {
            node.properties.tabIndex = 0;
            node.properties.ariaLabel = locale === 'ko' ? `표 ${++table}. 필요하면 좌우 화살표로 스크롤하세요.` : `Table ${++table}. Use arrow keys to scroll if needed.`;
          }
          if (Array.isArray(node.properties.className) && node.properties.className.includes('katex-display')) {
            node.properties.tabIndex = 0; node.properties.role = 'region';
            node.properties.ariaLabel = locale === 'ko' ? `수식 ${++equation}. 필요하면 좌우 화살표로 스크롤하세요.` : `Equation ${++equation}. Use arrow keys to scroll if needed.`;
          }
        });
      }).use(rehypeStringify).process(source);
    if (file.messages.length) throw new Error(`Invalid math in ${unitId}.${locale}: ${file.messages[0].reason}`);
    return { html: String(file), outline };
  };
  const [ko, en] = await Promise.all([render('ko'), render('en')]);
  return { ko: ko.html, en: en.html, outline: { ko: ko.outline, en: en.outline } };
}
