import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
export interface JobAttempt { kind: 'generation' | 'publication'; started: string; finished?: string; outcome?: 'completed' | 'failed' | 'interrupted'; stage?: string; failureStage?: string; error?: string; }
export interface Job { id: string; status: 'queued' | 'running' | 'completed' | 'failed' | 'interrupted'; updated: string; stage?: string; failureStage?: string; files?: string[]; error?: string; generationError?: string; attempts?: JobAttempt[]; }
export function requestKey(input: string) {
  const normalized = input.trim().replace(/https?:\/\/(?:www\.)?arxiv\.org\/(?:abs|pdf)\//gi, '').replace(/(\d{4}\.\d{4,5})(?:v\d+)?(?:\.pdf)?/g, '$1').replace(/[\s,]+/g, ' ').toLowerCase();
  return createHash('sha256').update(normalized).digest('hex').slice(0, 16);
}
export class JobQueue {
  private tail: Promise<unknown> = Promise.resolve();
  readonly jobs: Record<string, Job>;
  constructor(private filename: string) {
    this.jobs = fs.existsSync(filename) ? JSON.parse(fs.readFileSync(filename, 'utf8')) : {};
    for (const job of Object.values(this.jobs)) if (['queued', 'running'].includes(job.status)) {
      job.status = 'interrupted'; job.error = 'Process stopped. Inspect saved files before retrying.';
      const attempt = job.attempts?.[job.attempts.length - 1];
      if (attempt && !attempt.finished) Object.assign(attempt, { finished: new Date().toISOString(), outcome: 'interrupted', stage: job.stage });
    }
    this.save();
  }
  private save() {
    fs.mkdirSync(path.dirname(this.filename), { recursive: true });
    fs.writeFileSync(`${this.filename}.tmp`, JSON.stringify(this.jobs, null, 2), { mode: 0o600 });
    fs.renameSync(`${this.filename}.tmp`, this.filename);
  }
  update(id: string, patch: Partial<Job>) {
    const job = this.jobs[id];
    if (patch.stage === 'publish-failed') patch.failureStage = job.stage;
    if (patch.status === 'failed' && patch.error && job.stage === 'saved') patch.generationError = patch.error;
    Object.assign(job, patch, { updated: new Date().toISOString() }); this.save();
  }
  private schedule(id: string, kind: JobAttempt['kind'], task: () => Promise<void>): Promise<void> {
    this.update(id, { status: 'queued', error: undefined, failureStage: undefined });
    const execution = this.tail.then(async () => {
      const attempt: JobAttempt = { kind, started: new Date().toISOString() };
      this.jobs[id].attempts = [...(this.jobs[id].attempts || []), attempt];
      this.update(id, { status: 'running', stage: kind === 'generation' ? 'generating' : 'publishing' });
      try {
        await task();
        if (this.jobs[id].status === 'failed' || this.jobs[id].stage === 'publish-failed') throw new Error(this.jobs[id].error || 'Publication failed; retry saved files without regenerating.');
        attempt.outcome = 'completed';
        this.update(id, { status: this.jobs[id].generationError ? 'failed' : 'completed', error: this.jobs[id].generationError });
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        attempt.outcome = 'failed'; attempt.error = message;
        this.jobs[id].failureStage ||= this.jobs[id].stage;
        if (kind === 'generation' && this.jobs[id].stage !== 'publish-failed') this.jobs[id].generationError = message;
        this.update(id, { status: 'failed', error: message });
        throw error;
      } finally {
        attempt.finished = new Date().toISOString(); attempt.stage = this.jobs[id].stage; attempt.failureStage = this.jobs[id].failureStage; this.save();
      }
    });
    this.tail = execution.catch(() => {});
    return execution;
  }
  run(input: string, task: (id: string) => Promise<void>): Promise<void> {
    const id = requestKey(input); const previous = this.jobs[id];
    if (previous && (['queued', 'running', 'completed'].includes(previous.status) || previous.files?.length)) return Promise.reject(new Error(`중복 작업 ${id}: ${previous.status}. 저장된 파일과 작업 기록을 확인하세요.`));
    this.jobs[id] = { id, status: 'queued', updated: new Date().toISOString(), attempts: previous?.attempts || [] };
    return this.schedule(id, 'generation', () => task(id));
  }
  async retryPublication(id: string, publish: (files: string[]) => string | Promise<string>): Promise<void> {
    const job = this.jobs[id];
    if (!job?.files?.length) throw new Error('Saved files are required for publication retry.');
    if (!job.attempts && job.status === 'failed' && job.stage !== 'publish-failed' && job.error) job.generationError = job.error;
    if (['queued', 'running'].includes(job.status)) throw new Error('Job is already queued or running.');
    if (job.stage === 'pushed') throw new Error('Saved files have already been pushed.');
    await this.schedule(id, 'publication', async () => {
      const result = await publish([...job.files!]);
      if (result === 'master -> master') this.update(id, { stage: 'pushed' });
      else if (result === 'Saved for review') this.update(id, { stage: 'review' });
      else { this.update(id, { stage: 'publish-failed' }); throw new Error(result); }
    });
  }
  metrics() {
    const jobs = Object.values(this.jobs);
    const attempts = jobs.flatMap(job => job.attempts || []);
    const finished = attempts.filter(a => a.finished && a.outcome !== 'interrupted');
    return {
      total: jobs.length, queued: jobs.filter(j => j.status === 'queued').length, running: jobs.filter(j => j.status === 'running').length,
      failed: jobs.filter(j => ['failed', 'interrupted'].includes(j.status)).length,
      review: jobs.filter(j => j.stage === 'review').length, pushed: jobs.filter(j => j.stage === 'pushed').length,
      retries: attempts.filter(a => a.kind === 'publication').length,
      averageSeconds: finished.length ? Math.round(finished.reduce((sum, a) => sum + Date.parse(a.finished!) - Date.parse(a.started), 0) / finished.length / 1000) : 0,
    };
  }
}
