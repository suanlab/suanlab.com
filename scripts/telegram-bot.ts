#!/usr/bin/env node
import { config } from 'dotenv';
import path from 'path';

// Load environment variables
config({ path: path.join(process.cwd(), '.env.local') });

import TelegramBot from 'node-telegram-bot-api';
import { execSync } from 'child_process';

const BOT_TOKEN = process.env.TELEGRAM_BOT_TOKEN;
const ALLOWED_USERS = process.env.TELEGRAM_ALLOWED_USERS?.split(',').map(id => parseInt(id.trim())) || [];

if (!BOT_TOKEN) {
  console.error('TELEGRAM_BOT_TOKEN is not set in .env.local');
  process.exit(1);
}

const bot = new TelegramBot(BOT_TOKEN, { polling: true });

console.log('🤖 SuanLab Blog Bot started!');
console.log(`Allowed users: ${ALLOWED_USERS.length > 0 ? ALLOWED_USERS.join(', ') : 'All users'}`);

// Check if user is authorized
function isAuthorized(userId: number): boolean {
  if (ALLOWED_USERS.length === 0) return true;
  return ALLOWED_USERS.includes(userId);
}

// Execute command and return output
function runCommand(command: string): string {
  try {
    return execSync(command, {
      encoding: 'utf-8',
      cwd: process.cwd(),
      timeout: 300000 // 5 minutes
    });
  } catch (error: any) {
    return error.stdout || error.message;
  }
}

// /start command
bot.onText(/\/start/, (msg) => {
  const chatId = msg.chat.id;
  const userId = msg.from?.id || 0;

  if (!isAuthorized(userId)) {
    bot.sendMessage(chatId, '⛔ 권한이 없습니다.');
    return;
  }

  bot.sendMessage(chatId, `
🤖 *SuanLab Blog Bot*에 오신 것을 환영합니다!

사용 가능한 명령어:

📝 *블로그 생성*
\`/topic <주제>\` - 주제 기반 블로그 생성
\`/paper <arXiv ID>\` - 논문 리뷰 생성

📋 *기타*
\`/status\` - 현재 블로그 상태
\`/help\` - 도움말

예시:
• \`/topic 트랜스포머 아키텍처\`
• \`/paper 2312.00752\`
`, { parse_mode: 'Markdown' });
});

// /help command
bot.onText(/\/help/, (msg) => {
  const chatId = msg.chat.id;
  const userId = msg.from?.id || 0;

  if (!isAuthorized(userId)) {
    bot.sendMessage(chatId, '⛔ 권한이 없습니다.');
    return;
  }

  bot.sendMessage(chatId, `
📖 *도움말*

*주제 기반 블로그 생성*
\`/topic <주제> [카테고리]\`

예시:
• \`/topic PyTorch 기초\`
• \`/topic RAG 시스템 NLP\`

*논문 리뷰 생성*
\`/paper <arXiv ID 또는 URL>\`

예시:
• \`/paper 2312.00752\`
• \`/paper https://arxiv.org/abs/2312.00752\`

생성된 블로그는 자동으로 GitHub에 푸시됩니다.
`, { parse_mode: 'Markdown' });
});

// /status command
bot.onText(/\/status/, async (msg) => {
  const chatId = msg.chat.id;
  const userId = msg.from?.id || 0;

  if (!isAuthorized(userId)) {
    bot.sendMessage(chatId, '⛔ 권한이 없습니다.');
    return;
  }

  try {
    const postCount = runCommand('ls -1 content/blog/*.md 2>/dev/null | wc -l').trim();
    const latestPosts = runCommand('ls -1t content/blog/*.md 2>/dev/null | head -5').trim();

    const postList = latestPosts.split('\n')
      .map(p => `• ${path.basename(p, '.md')}`)
      .join('\n');

    bot.sendMessage(chatId, `
📊 *블로그 상태*

총 포스트 수: *${postCount}*개

최근 포스트:
${postList}
`, { parse_mode: 'Markdown' });
  } catch (error) {
    bot.sendMessage(chatId, '❌ 상태 확인 중 오류가 발생했습니다.');
  }
});

// /topic command
bot.onText(/\/topic (.+)/, async (msg, match) => {
  const chatId = msg.chat.id;
  const userId = msg.from?.id || 0;

  if (!isAuthorized(userId)) {
    bot.sendMessage(chatId, '⛔ 권한이 없습니다.');
    return;
  }

  const input = match?.[1]?.trim();
  if (!input) {
    bot.sendMessage(chatId, '❌ 주제를 입력해주세요.\n예: `/topic PyTorch 기초`', { parse_mode: 'Markdown' });
    return;
  }

  // Parse topic and optional category
  const parts = input.split(/\s+/);
  let topic = input;
  let category = 'General';

  // Check if last word might be a category
  const knownCategories = ['NLP', 'Deep Learning', 'MLOps', 'Computer Vision', 'General'];
  const lastWord = parts[parts.length - 1];
  if (knownCategories.some(c => c.toLowerCase() === lastWord.toLowerCase())) {
    category = lastWord;
    topic = parts.slice(0, -1).join(' ');
  }

  bot.sendMessage(chatId, `
🔄 *블로그 생성 중...*

주제: ${topic}
카테고리: ${category}

⏳ 약 2-3분 소요됩니다.
`, { parse_mode: 'Markdown' });

  try {
    // Generate blog
    const output = runCommand(`npm run blog:topic -- -t "${topic}" -c "${category}" -i -y 2>&1`);

    // Extract saved file path
    const savedMatch = output.match(/저장 완료: (.+\.md)/);
    const savedPath = savedMatch ? savedMatch[1] : 'unknown';

    // Git commit and push
    const gitOutput = runCommand(`
      git add -A && \
      git commit -m "Add blog: ${topic}" && \
      git push origin master 2>&1
    `);

    const isGitSuccess = gitOutput.includes('master -> master') || gitOutput.includes('nothing to commit');

    bot.sendMessage(chatId, `
✅ *블로그 생성 완료!*

📄 파일: \`${path.basename(savedPath)}\`
🌐 GitHub: ${isGitSuccess ? '푸시 완료' : '푸시 실패'}

배포까지 약 1-2분 소요됩니다.
`, { parse_mode: 'Markdown' });

  } catch (error: any) {
    bot.sendMessage(chatId, `❌ 오류 발생: ${error.message}`);
  }
});

// /paper command
bot.onText(/\/paper (.+)/, async (msg, match) => {
  const chatId = msg.chat.id;
  const userId = msg.from?.id || 0;

  if (!isAuthorized(userId)) {
    bot.sendMessage(chatId, '⛔ 권한이 없습니다.');
    return;
  }

  const input = match?.[1]?.trim();
  if (!input) {
    bot.sendMessage(chatId, '❌ arXiv ID를 입력해주세요.\n예: `/paper 2312.00752`', { parse_mode: 'Markdown' });
    return;
  }

  // Extract arXiv ID from URL if needed
  let arxivId = input;
  const urlMatch = input.match(/arxiv\.org\/(?:abs|pdf)\/(\d+\.\d+)/);
  if (urlMatch) {
    arxivId = urlMatch[1];
  }

  bot.sendMessage(chatId, `
🔄 *논문 리뷰 생성 중...*

arXiv ID: ${arxivId}

⏳ 약 3-5분 소요됩니다.
`, { parse_mode: 'Markdown' });

  try {
    // Generate paper review
    const output = runCommand(`npm run blog:paper -- -a "${arxivId}" -i -y 2>&1`);

    // Extract title and saved path
    const titleMatch = output.match(/제목: (.+)/);
    const title = titleMatch ? titleMatch[1] : 'Unknown';

    const savedMatch = output.match(/저장 완료: (.+\.md)/);
    const savedPath = savedMatch ? savedMatch[1] : 'unknown';

    // Git commit and push
    const gitOutput = runCommand(`
      git add -A && \
      git commit -m "Add paper review: ${arxivId}" && \
      git push origin master 2>&1
    `);

    const isGitSuccess = gitOutput.includes('master -> master') || gitOutput.includes('nothing to commit');

    bot.sendMessage(chatId, `
✅ *논문 리뷰 생성 완료!*

📄 제목: ${title}
🆔 arXiv: ${arxivId}
📁 파일: \`${path.basename(savedPath)}\`
🌐 GitHub: ${isGitSuccess ? '푸시 완료' : '푸시 실패'}

배포까지 약 1-2분 소요됩니다.
`, { parse_mode: 'Markdown' });

  } catch (error: any) {
    bot.sendMessage(chatId, `❌ 오류 발생: ${error.message}`);
  }
});

// Handle unknown commands
bot.on('message', (msg) => {
  if (msg.text?.startsWith('/') &&
      !msg.text.startsWith('/start') &&
      !msg.text.startsWith('/help') &&
      !msg.text.startsWith('/status') &&
      !msg.text.startsWith('/topic') &&
      !msg.text.startsWith('/paper')) {
    bot.sendMessage(msg.chat.id, '❓ 알 수 없는 명령어입니다. `/help`를 입력해주세요.', { parse_mode: 'Markdown' });
  }
});

console.log('Bot is running. Press Ctrl+C to stop.');
