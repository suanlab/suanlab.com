#!/usr/bin/env node
import { config } from 'dotenv';
import path from 'path';

// Load environment variables
config({ path: path.join(process.cwd(), '.env.local') });

import { App, LogLevel } from '@slack/bolt';
import { execSync } from 'child_process';
import { generateFromPaper, savePost as savePaperPost } from './blog/paper-summarizer';
import { generateFromTopic, savePost as saveTopicPost } from './blog/topic-generator';

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
        GIT_SSH_COMMAND: 'ssh -i /home/suan/.ssh/id_ed25519 -o StrictHostKeyChecking=no'
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

// Parse multiple arXiv IDs from input (space, comma, or newline separated)
function parseMultipleArxivIds(input: string): string[] | null {
  const parts = input.split(/[\s,]+/).filter(Boolean);
  if (parts.length < 2) return null;

  const arxivIds: string[] = [];
  for (const part of parts) {
    if (/^\d{4}\.\d{4,5}(v\d+)?$/.test(part)) {
      arxivIds.push(part);
    } else if (part.includes('arxiv.org')) {
      arxivIds.push(extractArxivId(part));
    } else {
      return null; // Not all parts are arXiv IDs — fall through to normal logic
    }
  }
  return arxivIds;
}

// Process items sequentially with delay to avoid rate limiting
async function processBatch<T, R>(
  items: T[],
  fn: (item: T) => Promise<R>,
  delayMs: number = 5000
): Promise<PromiseSettledResult<R>[]> {
  const results: PromiseSettledResult<R>[] = [];
  for (let i = 0; i < items.length; i++) {
    if (i > 0) {
      console.log(`[Batch] Waiting ${delayMs / 1000}s before next item to avoid rate limiting...`);
      await new Promise(r => setTimeout(r, delayMs));
    }
    const result = await Promise.allSettled([fn(items[i])]);
    results.push(...result);
  }
  return results;
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

  // Check for multiple arXiv IDs (batch mode)
  const multipleIds = parseMultipleArxivIds(input);
  if (multipleIds && multipleIds.length > 1) {
    await respond({
      response_type: 'in_channel',
      blocks: [
        {
          type: 'section',
          text: {
            type: 'mrkdwn',
            text: `:hourglass_flowing_sand: *${multipleIds.length}개 논문 리뷰 일괄 생성 중...*\n:page_facing_up: arXiv IDs: ${multipleIds.join(', ')}\n\n약 ${multipleIds.length * 3}-${multipleIds.length * 5}분 소요됩니다.`
          }
        }
      ]
    });

    try {
      const results = await processBatch(multipleIds, async (arxivId) => {
        console.log(`[Batch] Generating paper review for arXiv: ${arxivId}`);
        const post = await generateFromPaper({ arxivId, generateImage: true });
        const filepath = await savePaperPost(post);
        console.log(`[Batch] Saved: ${filepath}`);
        return { arxivId, title: post.title, filepath };
      }, 5000);  // 5s delay between each paper to respect arXiv rate limits

      const succeeded: { arxivId: string; title: string; filepath: string }[] = [];
      const failed: { arxivId: string; reason: string }[] = [];

      results.forEach((result, index) => {
        if (result.status === 'fulfilled') {
          succeeded.push(result.value);
        } else {
          failed.push({
            arxivId: multipleIds[index],
            reason: result.reason instanceof Error ? result.reason.message : String(result.reason),
          });
        }
      });

      let isGitSuccess = false;
      if (succeeded.length > 0) {
        const gitOutput = runCommand(
          `git add -A && git commit -m "Add ${succeeded.length} paper reviews (batch)" && git push origin master 2>&1`
        );
        isGitSuccess = gitOutput.includes('master -> master') || gitOutput.includes('nothing to commit');
      }

      const resultLines = succeeded.map(
        s => `:white_check_mark: *${s.title}*\n\`${path.basename(s.filepath)}\``
      );
      const failedLines = failed.map(
        f => `:x: ${f.arxivId}: ${f.reason}`
      );

      const blocks: Array<{ type: string; text: { type: string; text: string } }> = [
        {
          type: 'section',
          text: {
            type: 'mrkdwn',
            text: `:clipboard: *일괄 논문 리뷰 생성 결과*\n성공: ${succeeded.length}개 | 실패: ${failed.length}개`
          }
        }
      ];

      if (resultLines.length > 0) {
        blocks.push({
          type: 'section',
          text: { type: 'mrkdwn', text: resultLines.join('\n\n') }
        });
      }

      if (failedLines.length > 0) {
        blocks.push({
          type: 'section',
          text: { type: 'mrkdwn', text: failedLines.join('\n') }
        });
      }

      blocks.push({
        type: 'section',
        text: {
          type: 'mrkdwn',
          text: `*GitHub:* ${isGitSuccess ? ':white_check_mark: 푸시 완료' : (succeeded.length === 0 ? ':warning: 생성된 포스트 없음' : ':x: 푸시 실패')}`
        }
      });

      await app.client.chat.postMessage({
        token: SLACK_BOT_TOKEN,
        channel: command.channel_id,
        blocks,
      });
    } catch (error: unknown) {
      console.error('Batch paper generation failed:', error);
      const errorMessage = error instanceof Error ? error.message : String(error);
      await app.client.chat.postMessage({
        token: SLACK_BOT_TOKEN,
        channel: command.channel_id,
        text: `:x: 일괄 생성 실패\n*오류:* ${errorMessage}`
      });
    }
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
      console.log(`Generating paper review for arXiv: ${arxivId}`);
      const post = await generateFromPaper({ arxivId, generateImage: true });
      const filepath = await savePaperPost(post);

      console.log(`Saved: ${filepath}`);

      const gitOutput = runCommand(
        `git add -A && git commit -m "Add paper review: ${arxivId}" && git push origin master 2>&1`
      );
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
                text: `*제목:*\n${post.title}`
              },
              {
                type: 'mrkdwn',
                text: `*arXiv:*\n${arxivId}`
              },
              {
                type: 'mrkdwn',
                text: `*파일:*\n\`${path.basename(filepath)}\``
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
      console.error('arXiv paper generation failed:', error);
      const errorMessage = error instanceof Error ? error.message : String(error);
      await app.client.chat.postMessage({
        token: SLACK_BOT_TOKEN,
        channel: command.channel_id,
        text: `:x: 논문 리뷰 생성 실패\n*오류:* ${errorMessage}`
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
      console.log(`Generating paper review from PDF URL: ${input}`);
      const post = await generateFromPaper({ pdfUrl: input, generateImage: true });
      const filepath = await savePaperPost(post);

      console.log(`Saved: ${filepath}`);

      const gitOutput = runCommand(
        `git add -A && git commit -m "Add paper review from PDF" && git push origin master 2>&1`
      );
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
                text: `*제목:*\n${post.title}`
              },
              {
                type: 'mrkdwn',
                text: `*파일:*\n\`${path.basename(filepath)}\``
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
      console.error('PDF paper generation failed:', error);
      const errorMessage = error instanceof Error ? error.message : String(error);
      await app.client.chat.postMessage({
        token: SLACK_BOT_TOKEN,
        channel: command.channel_id,
        text: `:x: 논문 리뷰 생성 실패\n*오류:* ${errorMessage}`
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
      console.log(`Generating blog post for topic: ${topic} (${category})`);
      const post = await generateFromTopic({ topic, category, generateImage: true });
      const filepath = await saveTopicPost(post);

      console.log(`Saved: ${filepath}`);

      const gitOutput = runCommand(
        `git add -A && git commit -m "Add blog: ${topic}" && git push origin master 2>&1`
      );
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
                text: `*제목:*\n${post.title}`
              },
              {
                type: 'mrkdwn',
                text: `*파일:*\n\`${path.basename(filepath)}\``
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
      console.error('Topic blog generation failed:', error);
      const errorMessage = error instanceof Error ? error.message : String(error);
      await app.client.chat.postMessage({
        token: SLACK_BOT_TOKEN,
        channel: command.channel_id,
        text: `:x: 블로그 생성 실패\n*오류:* ${errorMessage}`
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
          text: '*예시:*\n• `/suanblog 2312.00752`\n• `/suanblog https://arxiv.org/abs/2312.00752`\n• `/suanblog 2312.00752 2401.12345 2402.67890` (일괄 생성)\n• `/suanblog 트랜스포머 아키텍처`\n• `/suanblog RAG 시스템 NLP`'
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
