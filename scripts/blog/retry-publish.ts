import fs from 'node:fs';
import { publishPosts } from './publish';
import type { Job } from './job-queue';
const filename = '.runtime/slack-jobs.json';
const jobs: Record<string, Job> = JSON.parse(fs.readFileSync(filename, 'utf8'));
const job = jobs[process.argv[2]];
if (!job?.files?.length) throw new Error('Provide a saved job ID from /suanblog-status.');
if (['queued', 'running'].includes(job.status)) throw new Error('Wait for the job to finish.');
const result = publishPosts(job.files, `Publish saved blog job ${job.id}`);
console.log(result);
if (result !== 'master -> master') process.exitCode = 1;
// The running bot owns the job ledger. Do not overwrite it from a second process.
