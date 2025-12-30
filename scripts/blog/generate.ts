#!/usr/bin/env node
import { config } from 'dotenv';
import path from 'path';

// Load environment variables from .env.local
config({ path: path.join(process.cwd(), '.env.local') });
config({ path: path.join(process.cwd(), '.env') });

import { Command } from 'commander';
import { generateFromTopic, formatAsMarkdown, savePost as saveTopicPost } from './topic-generator';
import { generateFromPaper, savePost as savePaperPost } from './paper-summarizer';
import * as readline from 'readline';

const program = new Command();

program
  .name('blog-generate')
  .description('SuanLab 블로그 콘텐츠 자동 생성 도구')
  .version('1.0.0');

// Topic-based generation
program
  .command('topic')
  .description('주제/키워드 기반 블로그 포스트 생성')
  .option('-t, --topic <topic>', '주제 또는 키워드')
  .option('-c, --category <category>', '카테고리', 'General')
  .option('--tags <tags>', '태그 (쉼표로 구분)')
  .option('-o, --output <filename>', '출력 파일명 (slug)')
  .option('--preview', '미리보기만 (저장 안함)')
  .option('-y, --yes', '확인 없이 자동 저장')
  .option('-i, --image', 'DALL-E 3로 썸네일 이미지 생성')
  .action(async (options) => {
    try {
      // Interactive mode if topic not provided
      if (!options.topic) {
        options = await interactiveTopicMode(options);
      }

      const tags = options.tags
        ? options.tags.split(',').map((t: string) => t.trim())
        : [];

      console.log('\n🤖 OpenAI API로 콘텐츠 생성 중...\n');

      const post = await generateFromTopic({
        topic: options.topic,
        category: options.category,
        tags,
        generateImage: options.image,
      });

      // Show preview
      console.log('--- 미리보기 ---');
      console.log(`제목: ${post.title}`);
      console.log(`카테고리: ${post.category}`);
      console.log(`태그: ${post.tags.join(', ')}`);
      console.log(`\n${post.content.slice(0, 500)}...\n`);
      console.log('----------------\n');

      if (options.preview) {
        console.log('✅ 미리보기 완료 (저장하지 않음)');
        return;
      }

      // Confirm save (skip if --yes flag)
      let shouldSave = options.yes;
      if (!shouldSave) {
        shouldSave = await confirm('저장하시겠습니까?');
      }
      if (!shouldSave) {
        console.log('❌ 저장 취소됨');
        return;
      }

      const filepath = await saveTopicPost(post, options.output);
      console.log(`\n✅ 저장 완료: ${filepath}`);
    } catch (error) {
      console.error('❌ 오류:', error instanceof Error ? error.message : error);
      process.exit(1);
    }
  });

// Paper-based generation
program
  .command('paper')
  .description('논문 기반 블로그 포스트 생성')
  .option('-a, --arxiv <id>', 'arXiv 논문 ID 또는 URL')
  .option('-u, --url <url>', 'PDF URL')
  .option('-f, --file <path>', '로컬 PDF 파일 경로')
  .option('-o, --output <filename>', '출력 파일명 (slug)')
  .option('--preview', '미리보기만 (저장 안함)')
  .option('-y, --yes', '확인 없이 자동 저장')
  .option('-i, --image', 'DALL-E 3로 썸네일 이미지 생성')
  .action(async (options) => {
    try {
      // Interactive mode if no source provided
      if (!options.arxiv && !options.url && !options.file) {
        options = await interactivePaperMode(options);
      }

      console.log('\n🤖 논문 처리 및 요약 생성 중...\n');

      const post = await generateFromPaper({
        arxivId: options.arxiv,
        pdfUrl: options.url,
        localPath: options.file,
        generateImage: options.image,
      });

      // Show preview
      console.log('--- 미리보기 ---');
      console.log(`제목: ${post.title}`);
      console.log(`카테고리: ${post.category}`);
      console.log(`태그: ${post.tags.join(', ')}`);
      console.log(`\n${post.content.slice(0, 500)}...\n`);
      console.log('----------------\n');

      if (options.preview) {
        console.log('✅ 미리보기 완료 (저장하지 않음)');
        return;
      }

      // Confirm save (skip if --yes flag)
      let shouldSave = options.yes;
      if (!shouldSave) {
        shouldSave = await confirm('저장하시겠습니까?');
      }
      if (!shouldSave) {
        console.log('❌ 저장 취소됨');
        return;
      }

      const filepath = await savePaperPost(post, options.output);
      console.log(`\n✅ 저장 완료: ${filepath}`);
    } catch (error) {
      console.error('❌ 오류:', error instanceof Error ? error.message : error);
      process.exit(1);
    }
  });

// Default: interactive mode
program.action(async () => {
  console.log('🚀 SuanLab 블로그 콘텐츠 자동 생성 도구\n');

  const type = await select('생성 유형을 선택하세요:', [
    { value: 'topic', label: '주제 기반 - 키워드/주제로 새 포스트 생성' },
    { value: 'paper', label: '논문 요약 - 논문을 요약하여 포스트 생성' },
  ]);

  if (type === 'topic') {
    await program.parseAsync(['node', 'generate.ts', 'topic']);
  } else {
    await program.parseAsync(['node', 'generate.ts', 'paper']);
  }
});

// Helper functions for interactive mode
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

function prompt(question: string): Promise<string> {
  return new Promise((resolve) => {
    rl.question(question, (answer) => {
      resolve(answer.trim());
    });
  });
}

function confirm(question: string): Promise<boolean> {
  return new Promise((resolve) => {
    rl.question(`${question} (y/n): `, (answer) => {
      resolve(answer.toLowerCase() === 'y' || answer.toLowerCase() === 'yes');
    });
  });
}

function select(
  question: string,
  options: { value: string; label: string }[]
): Promise<string> {
  return new Promise((resolve) => {
    console.log(question);
    options.forEach((opt, i) => {
      console.log(`  ${i + 1}. ${opt.label}`);
    });
    rl.question('선택 (번호): ', (answer) => {
      const index = parseInt(answer) - 1;
      if (index >= 0 && index < options.length) {
        resolve(options[index].value);
      } else {
        resolve(options[0].value);
      }
    });
  });
}

async function interactiveTopicMode(options: Record<string, unknown>) {
  const topic = await prompt('주제를 입력하세요: ');

  const categories = [
    'Deep Learning',
    'Machine Learning',
    'NLP',
    'Computer Vision',
    'Data Science',
    'General',
  ];
  console.log('\n카테고리를 선택하세요:');
  categories.forEach((cat, i) => console.log(`  ${i + 1}. ${cat}`));
  const catIndex = parseInt(await prompt('선택 (번호): ')) - 1;
  const category = categories[catIndex] || 'General';

  const tagsInput = await prompt('태그를 입력하세요 (쉼표로 구분, 생략 가능): ');

  return {
    ...options,
    topic,
    category,
    tags: tagsInput || undefined,
  };
}

async function interactivePaperMode(options: Record<string, unknown>) {
  const sourceType = await select('논문 소스를 선택하세요:', [
    { value: 'arxiv', label: 'arXiv ID/URL' },
    { value: 'url', label: 'PDF URL' },
    { value: 'file', label: '로컬 PDF 파일' },
  ]);

  if (sourceType === 'arxiv') {
    const arxiv = await prompt('arXiv ID 또는 URL: ');
    return { ...options, arxiv };
  } else if (sourceType === 'url') {
    const url = await prompt('PDF URL: ');
    return { ...options, url };
  } else {
    const file = await prompt('PDF 파일 경로: ');
    return { ...options, file };
  }
}

// Run
program.parseAsync().finally(() => rl.close());
