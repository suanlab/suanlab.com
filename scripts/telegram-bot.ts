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
      timeout: 300000, // 5 minutes
      env: {
        ...process.env,
        GIT_SSH_COMMAND: 'ssh -i /home/suanlab/.ssh/id_ed25519_bot -o StrictHostKeyChecking=no'
      }
    });
  } catch (error: unknown) {
    if (error instanceof Error && 'stdout' in error) {
      return (error as { stdout: string }).stdout || error.message;
    }
    return error instanceof Error ? error.message : 'Unknown error';
  }
}

// Detect input type
function detectInputType(input: string): 'arxiv' | 'pdf' | 'topic' {
  // Check for arXiv ID pattern (e.g., 2312.00752)
  if (/^\d{4}\.\d{4,5}(v\d+)?$/.test(input)) {
    return 'arxiv';
  }
  // Check for arXiv URL
  if (input.includes('arxiv.org')) {
    return 'arxiv';
  }
  // Check for PDF URL
  if (/^https?:\/\/.+\.pdf$/i.test(input)) {
    return 'pdf';
  }
  // Default to topic
  return 'topic';
}

// Extract arXiv ID from URL or return as-is
function extractArxivId(input: string): string {
  const match = input.match(/arxiv\.org\/(?:abs|pdf)\/(\d{4}\.\d{4,5})/);
  return match ? match[1] : input;
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
\`/blog <입력>\` - 자동 감지하여 블로그 생성

📋 *기타*
\`/status\` - 현재 블로그 상태
\`/help\` - 도움말

예시:
• \`/blog 2312.00752\` → 논문 리뷰
• \`/blog https://arxiv.org/abs/2312.00752\` → 논문 리뷰
• \`/blog https://example.com/paper.pdf\` → 논문 리뷰
• \`/blog 트랜스포머 아키텍처\` → 주제 블로그
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

*통합 블로그 생성 명령어*
\`/blog <입력>\`

입력 유형에 따라 자동으로 감지됩니다:
• arXiv ID (예: \`2312.00752\`) → 논문 리뷰
• arXiv URL → 논문 리뷰
• PDF URL (\`.pdf\`로 끝남) → 논문 리뷰
• 그 외 텍스트 → 주제 기반 블로그

*예시:*
• \`/blog 2312.00752\`
• \`/blog https://arxiv.org/abs/2312.00752\`
• \`/blog RAG 시스템 NLP\`

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
  } catch {
    bot.sendMessage(chatId, '❌ 상태 확인 중 오류가 발생했습니다.');
  }
});

// /blog command - unified command
bot.onText(/\/blog (.+)/, async (msg, match) => {
  const chatId = msg.chat.id;
  const userId = msg.from?.id || 0;

  if (!isAuthorized(userId)) {
    bot.sendMessage(chatId, '⛔ 권한이 없습니다.');
    return;
  }

  const input = match?.[1]?.trim();
  if (!input) {
    bot.sendMessage(chatId, '❌ 입력을 해주세요.\n예: `/blog 2312.00752` 또는 `/blog 트랜스포머`', { parse_mode: 'Markdown' });
    return;
  }

  const inputType = detectInputType(input);

  if (inputType === 'arxiv') {
    // Handle arXiv paper
    const arxivId = extractArxivId(input);

    bot.sendMessage(chatId, `
🔄 *논문 리뷰 생성 중...*

🆔 arXiv ID: ${arxivId}

⏳ 약 3-5분 소요됩니다.
`, { parse_mode: 'Markdown' });

    try {
      const output = runCommand(`npm run blog:paper -- -a "${arxivId}" -i -y 2>&1`);

      const titleMatch = output.match(/제목: (.+)/);
      const title = titleMatch ? titleMatch[1] : 'Unknown';

      const savedMatch = output.match(/저장 완료: (.+\.md)/);
      const savedPath = savedMatch ? savedMatch[1] : 'unknown';

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

    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      bot.sendMessage(chatId, `❌ 오류 발생: ${errorMessage}`);
    }

  } else if (inputType === 'pdf') {
    // Handle PDF URL
    bot.sendMessage(chatId, `
🔄 *논문 리뷰 생성 중...*

📎 PDF URL: ${input}

⏳ 약 3-5분 소요됩니다.
`, { parse_mode: 'Markdown' });

    try {
      const output = runCommand(`npm run blog:paper -- --url "${input}" -i -y 2>&1`);

      const titleMatch = output.match(/제목: (.+)/);
      const title = titleMatch ? titleMatch[1] : 'Unknown';

      const savedMatch = output.match(/저장 완료: (.+\.md)/);
      const savedPath = savedMatch ? savedMatch[1] : 'unknown';

      const gitOutput = runCommand(`
        git add -A && \
        git commit -m "Add paper review from PDF" && \
        git push origin master 2>&1
      `);

      const isGitSuccess = gitOutput.includes('master -> master') || gitOutput.includes('nothing to commit');

      bot.sendMessage(chatId, `
✅ *논문 리뷰 생성 완료!*

📄 제목: ${title}
📁 파일: \`${path.basename(savedPath)}\`
🌐 GitHub: ${isGitSuccess ? '푸시 완료' : '푸시 실패'}

배포까지 약 1-2분 소요됩니다.
`, { parse_mode: 'Markdown' });

    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      bot.sendMessage(chatId, `❌ 오류 발생: ${errorMessage}`);
    }

  } else {
    // Handle topic
    const parts = input.split(/\s+/);
    let topic = input;
    let category = 'General';

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
      const output = runCommand(`npm run blog:topic -- -t "${topic}" -c "${category}" -i -y 2>&1`);

      const savedMatch = output.match(/저장 완료: (.+\.md)/);
      const savedPath = savedMatch ? savedMatch[1] : 'unknown';

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

    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      bot.sendMessage(chatId, `❌ 오류 발생: ${errorMessage}`);
    }
  }
});

// Handle unknown commands
bot.on('message', (msg) => {
  if (msg.text?.startsWith('/') &&
      !msg.text.startsWith('/start') &&
      !msg.text.startsWith('/help') &&
      !msg.text.startsWith('/status') &&
      !msg.text.startsWith('/blog')) {
    bot.sendMessage(msg.chat.id, '❓ 알 수 없는 명령어입니다. `/help`를 입력해주세요.', { parse_mode: 'Markdown' });
  }
});

console.log('Bot is running. Press Ctrl+C to stop.');
