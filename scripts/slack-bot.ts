#!/usr/bin/env node
import { config } from 'dotenv';
import path from 'path';

// Load environment variables
config({ path: path.join(process.cwd(), '.env.local') });

import { App, LogLevel } from '@slack/bolt';
import { execSync } from 'child_process';

const SLACK_BOT_TOKEN = process.env.SLACK_BOT_TOKEN;
const SLACK_SIGNING_SECRET = process.env.SLACK_SIGNING_SECRET;
const SLACK_APP_TOKEN = process.env.SLACK_APP_TOKEN;
const ALLOWED_CHANNELS = process.env.SLACK_ALLOWED_CHANNELS?.split(',').map(c => c.trim()) || [];

if (!SLACK_BOT_TOKEN || !SLACK_SIGNING_SECRET || !SLACK_APP_TOKEN) {
  console.error('Missing Slack credentials in .env.local');
  console.error('Required: SLACK_BOT_TOKEN, SLACK_SIGNING_SECRET, SLACK_APP_TOKEN');
  process.exit(1);
}

const app = new App({
  token: SLACK_BOT_TOKEN,
  signingSecret: SLACK_SIGNING_SECRET,
  socketMode: true,
  appToken: SLACK_APP_TOKEN,
  logLevel: LogLevel.INFO,
});

// Check if channel is allowed
function isAllowedChannel(channelId: string): boolean {
  if (ALLOWED_CHANNELS.length === 0) return true;
  return ALLOWED_CHANNELS.includes(channelId);
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

// /suanblog slash command - unified command
app.command('/suanblog', async ({ command, ack, respond }) => {
  console.log('Received /suanblog command:', command.text);
  await ack();

  if (!isAllowedChannel(command.channel_id)) {
    await respond({
      response_type: 'ephemeral',
      text: ':no_entry: 이 채널에서는 사용할 수 없습니다.'
    });
    return;
  }

  const input = command.text?.trim();
  if (!input) {
    await respond({
      response_type: 'ephemeral',
      text: ':warning: 입력을 해주세요.\n예: `/blog 2312.00752` 또는 `/blog 트랜스포머 아키텍처`'
    });
    return;
  }

  const inputType = detectInputType(input);

  if (inputType === 'arxiv') {
    // Handle arXiv paper
    const arxivId = extractArxivId(input);

    await respond({
      response_type: 'in_channel',
      blocks: [
        {
          type: 'section',
          text: {
            type: 'mrkdwn',
            text: `:hourglass_flowing_sand: *논문 리뷰 생성 중...*\n:page_facing_up: arXiv ID: ${arxivId}\n\n약 3-5분 소요됩니다.`
          }
        }
      ]
    });

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

      await app.client.chat.postMessage({
        token: SLACK_BOT_TOKEN,
        channel: command.channel_id,
        blocks: [
          {
            type: 'section',
            text: {
              type: 'mrkdwn',
              text: `:white_check_mark: *논문 리뷰 생성 완료!*`
            }
          },
          {
            type: 'section',
            fields: [
              {
                type: 'mrkdwn',
                text: `*제목:*\n${title}`
              },
              {
                type: 'mrkdwn',
                text: `*arXiv:*\n${arxivId}`
              },
              {
                type: 'mrkdwn',
                text: `*파일:*\n\`${path.basename(savedPath)}\``
              },
              {
                type: 'mrkdwn',
                text: `*GitHub:*\n${isGitSuccess ? ':white_check_mark: 푸시 완료' : ':x: 푸시 실패'}`
              }
            ]
          }
        ]
      });
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      await app.client.chat.postMessage({
        token: SLACK_BOT_TOKEN,
        channel: command.channel_id,
        text: `:x: 오류 발생: ${errorMessage}`
      });
    }

  } else if (inputType === 'pdf') {
    // Handle PDF URL
    await respond({
      response_type: 'in_channel',
      blocks: [
        {
          type: 'section',
          text: {
            type: 'mrkdwn',
            text: `:hourglass_flowing_sand: *논문 리뷰 생성 중...*\n:link: PDF URL: ${input}\n\n약 3-5분 소요됩니다.`
          }
        }
      ]
    });

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

      await app.client.chat.postMessage({
        token: SLACK_BOT_TOKEN,
        channel: command.channel_id,
        blocks: [
          {
            type: 'section',
            text: {
              type: 'mrkdwn',
              text: `:white_check_mark: *논문 리뷰 생성 완료!*`
            }
          },
          {
            type: 'section',
            fields: [
              {
                type: 'mrkdwn',
                text: `*제목:*\n${title}`
              },
              {
                type: 'mrkdwn',
                text: `*파일:*\n\`${path.basename(savedPath)}\``
              },
              {
                type: 'mrkdwn',
                text: `*GitHub:*\n${isGitSuccess ? ':white_check_mark: 푸시 완료' : ':x: 푸시 실패'}`
              },
              {
                type: 'mrkdwn',
                text: `*배포:*\n약 1-2분 후 반영`
              }
            ]
          }
        ]
      });
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      await app.client.chat.postMessage({
        token: SLACK_BOT_TOKEN,
        channel: command.channel_id,
        text: `:x: 오류 발생: ${errorMessage}`
      });
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

    await respond({
      response_type: 'in_channel',
      blocks: [
        {
          type: 'section',
          text: {
            type: 'mrkdwn',
            text: `:hourglass_flowing_sand: *블로그 생성 중...*\n:memo: 주제: ${topic}\n:label: 카테고리: ${category}\n\n약 2-3분 소요됩니다.`
          }
        }
      ]
    });

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

      await app.client.chat.postMessage({
        token: SLACK_BOT_TOKEN,
        channel: command.channel_id,
        blocks: [
          {
            type: 'section',
            text: {
              type: 'mrkdwn',
              text: `:white_check_mark: *블로그 생성 완료!*`
            }
          },
          {
            type: 'section',
            fields: [
              {
                type: 'mrkdwn',
                text: `*주제:*\n${topic}`
              },
              {
                type: 'mrkdwn',
                text: `*파일:*\n\`${path.basename(savedPath)}\``
              },
              {
                type: 'mrkdwn',
                text: `*GitHub:*\n${isGitSuccess ? ':white_check_mark: 푸시 완료' : ':x: 푸시 실패'}`
              },
              {
                type: 'mrkdwn',
                text: `*배포:*\n약 1-2분 후 반영`
              }
            ]
          }
        ]
      });
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      await app.client.chat.postMessage({
        token: SLACK_BOT_TOKEN,
        channel: command.channel_id,
        text: `:x: 오류 발생: ${errorMessage}`
      });
    }
  }
});

// /suanblog-status slash command
app.command('/suanblog-status', async ({ command, ack, respond }) => {
  await ack();

  if (!isAllowedChannel(command.channel_id)) {
    await respond({
      response_type: 'ephemeral',
      text: ':no_entry: 이 채널에서는 사용할 수 없습니다.'
    });
    return;
  }

  try {
    const postCount = runCommand('ls -1 content/blog/*.md 2>/dev/null | wc -l').trim();
    const latestPosts = runCommand('ls -1t content/blog/*.md 2>/dev/null | head -5').trim();

    const postList = latestPosts.split('\n')
      .map(p => `• \`${path.basename(p, '.md')}\``)
      .join('\n');

    await respond({
      response_type: 'in_channel',
      blocks: [
        {
          type: 'section',
          text: {
            type: 'mrkdwn',
            text: `:bar_chart: *블로그 상태*\n\n총 포스트 수: *${postCount}*개`
          }
        },
        {
          type: 'section',
          text: {
            type: 'mrkdwn',
            text: `*최근 포스트:*\n${postList}`
          }
        }
      ]
    });
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    await respond({
      response_type: 'ephemeral',
      text: `:x: 상태 확인 중 오류: ${errorMessage}`
    });
  }
});

// /suanblog-help slash command
app.command('/suanblog-help', async ({ command, ack, respond }) => {
  await ack();

  await respond({
    response_type: 'ephemeral',
    blocks: [
      {
        type: 'header',
        text: {
          type: 'plain_text',
          text: ':robot_face: SuanLab Blog Bot'
        }
      },
      {
        type: 'section',
        text: {
          type: 'mrkdwn',
          text: '*사용 가능한 명령어:*'
        }
      },
      {
        type: 'section',
        text: {
          type: 'mrkdwn',
          text: '`/suanblog <입력>` - 자동 감지하여 블로그 생성\n`/suanblog-status` - 블로그 상태 확인\n`/suanblog-help` - 도움말'
        }
      },
      {
        type: 'divider'
      },
      {
        type: 'section',
        text: {
          type: 'mrkdwn',
          text: '*입력 유형 자동 감지:*\n• arXiv ID (예: `2312.00752`) → 논문 리뷰\n• arXiv URL → 논문 리뷰\n• PDF URL (`.pdf`로 끝남) → 논문 리뷰\n• 그 외 텍스트 → 주제 기반 블로그'
        }
      },
      {
        type: 'divider'
      },
      {
        type: 'section',
        text: {
          type: 'mrkdwn',
          text: '*예시:*\n• `/suanblog 2312.00752`\n• `/suanblog https://arxiv.org/abs/2312.00752`\n• `/suanblog 트랜스포머 아키텍처`\n• `/suanblog RAG 시스템 NLP`'
        }
      }
    ]
  });
});

// Start the app
(async () => {
  await app.start();
  console.log(':robot_face: SuanLab Slack Bot started!');
  console.log(`Allowed channels: ${ALLOWED_CHANNELS.length > 0 ? ALLOWED_CHANNELS.join(', ') : 'All channels'}`);
})();
