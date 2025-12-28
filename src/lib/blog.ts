import fs from 'fs';
import path from 'path';
import matter from 'gray-matter';
import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import remarkRehype from 'remark-rehype';
import rehypeHighlight from 'rehype-highlight';
import rehypeKatex from 'rehype-katex';
import rehypeSlug from 'rehype-slug';
import rehypeAutolinkHeadings from 'rehype-autolink-headings';
import rehypeStringify from 'rehype-stringify';
import { visit } from 'unist-util-visit';
import type { Root, Element } from 'hast';

// Custom rehype plugin to wrap block math in centered div
function rehypeWrapMath() {
  return (tree: Root) => {
    visit(tree, 'element', (node: Element, index, parent) => {
      // Find paragraphs that contain only a single katex span (block equations)
      if (
        node.tagName === 'p' &&
        node.children.length === 1 &&
        node.children[0].type === 'element' &&
        (node.children[0] as Element).tagName === 'span' &&
        (node.children[0] as Element).properties?.className &&
        ((node.children[0] as Element).properties.className as string[]).includes('katex')
      ) {
        // Wrap the paragraph in a div with math-block class
        const wrapper: Element = {
          type: 'element',
          tagName: 'div',
          properties: { className: ['math-block'] },
          children: [node],
        };
        if (parent && typeof index === 'number') {
          (parent as Element).children[index] = wrapper;
        }
      }
    });
  };
}

export interface BlogPost {
  slug: string;
  title: string;
  date: string;
  excerpt: string;
  category: string;
  tags: string[];
  thumbnail?: string;
  content: string;
  readingTime: number;
}

function calculateReadingTime(content: string): number {
  const wordsPerMinute = 200;
  const koreanCharsPerMinute = 500;

  // Count English words
  const englishWords = content.match(/[a-zA-Z]+/g)?.length || 0;
  // Count Korean characters
  const koreanChars = content.match(/[\uAC00-\uD7AF]/g)?.length || 0;

  const englishTime = englishWords / wordsPerMinute;
  const koreanTime = koreanChars / koreanCharsPerMinute;

  return Math.max(1, Math.ceil(englishTime + koreanTime));
}

export interface BlogPostMeta {
  slug: string;
  title: string;
  date: string;
  excerpt: string;
  category: string;
  tags: string[];
  thumbnail?: string;
  readingTime: number;
}

const postsDirectory = path.join(process.cwd(), 'content/blog');

export function getPostSlugs(): string[] {
  if (!fs.existsSync(postsDirectory)) {
    return [];
  }
  return fs.readdirSync(postsDirectory).filter((file) => file.endsWith('.md'));
}

export function getPostBySlug(slug: string): BlogPost | null {
  const realSlug = slug.replace(/\.md$/, '');
  const fullPath = path.join(postsDirectory, `${realSlug}.md`);

  if (!fs.existsSync(fullPath)) {
    return null;
  }

  const fileContents = fs.readFileSync(fullPath, 'utf8');
  const { data, content } = matter(fileContents);

  return {
    slug: realSlug,
    title: data.title || '',
    date: data.date || '',
    excerpt: data.excerpt || '',
    category: data.category || 'General',
    tags: data.tags || [],
    thumbnail: data.thumbnail || null,
    content,
    readingTime: calculateReadingTime(content),
  };
}

export async function getPostBySlugWithHtml(slug: string): Promise<(BlogPost & { contentHtml: string }) | null> {
  const post = getPostBySlug(slug);
  if (!post) return null;

  const processedContent = await unified()
    .use(remarkParse)
    .use(remarkGfm)
    .use(remarkMath)
    .use(remarkRehype)
    .use(rehypeSlug)
    .use(rehypeAutolinkHeadings, { behavior: 'wrap' })
    .use(rehypeKatex)
    .use(rehypeWrapMath)
    .use(rehypeHighlight, { detect: true })
    .use(rehypeStringify)
    .process(post.content);

  const contentHtml = processedContent.toString();

  return {
    ...post,
    contentHtml,
  };
}

export function getAllPosts(): BlogPostMeta[] {
  const slugs = getPostSlugs();
  const posts = slugs
    .map((slug) => {
      const post = getPostBySlug(slug.replace(/\.md$/, ''));
      if (!post) return null;
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
      const { content, ...meta } = post;
      return meta;
    })
    .filter((post): post is BlogPostMeta => post !== null)
    .sort((a, b) => (new Date(b.date) > new Date(a.date) ? 1 : -1));

  return posts;
}

export function getPostsByCategory(category: string): BlogPostMeta[] {
  return getAllPosts().filter((post) => post.category.toLowerCase() === category.toLowerCase());
}

export function getPostsByTag(tag: string): BlogPostMeta[] {
  return getAllPosts().filter((post) =>
    post.tags.some((t) => t.toLowerCase() === tag.toLowerCase())
  );
}

export function getAllCategories(): string[] {
  const posts = getAllPosts();
  const categories = new Set(posts.map((post) => post.category));
  return Array.from(categories).sort();
}

export function getAllTags(): string[] {
  const posts = getAllPosts();
  const tags = new Set(posts.flatMap((post) => post.tags));
  return Array.from(tags).sort();
}

export function searchPosts(query: string): BlogPostMeta[] {
  const lowerQuery = query.toLowerCase();
  return getAllPosts().filter((post) =>
    post.title.toLowerCase().includes(lowerQuery) ||
    post.excerpt.toLowerCase().includes(lowerQuery) ||
    post.tags.some((tag) => tag.toLowerCase().includes(lowerQuery)) ||
    post.category.toLowerCase().includes(lowerQuery)
  );
}
