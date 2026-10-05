import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
export interface Job { id: string; status: 'queued' | 'running' | 'completed' | 'failed' | 'interrupted'; updated: string; stage?: string; files?: string[]; error?: string; }
export function requestKey(input: string) {
  const normalized = input.trim().replace(/https?:\/\/(?:www\.)?arxiv\.org\/(?:abs|pdf)\//gi, '').replace(/(\d{4}\.\d{4,5})(?:v\d+)?(?:\.pdf)?/g, '$1').replace(/[\s,]+/g, ' ').toLowerCase();
  return createHash('sha256').update(normalized).digest('hex').slice(0, 16);
}
export class JobQueue {
  private tail: Promise<unknown> = Promise.resolve();
  readonly jobs: Record<string, Job>;
  constructor(private filename: string) {
    this.jobs = fs.existsSync(filename) ? JSON.parse(fs.readFileSync(filename, 'utf8')) : {};
    for (const job of Object.values(this.jobs)) if (['queued', 'running'].includes(job.status)) { job.status = 'interrupted'; job.error = 'Process stopped. Inspect saved files before retrying.'; }
    this.save();
  }
  private save() {
    fs.mkdirSync(path.dirname(this.filename), { recursive: true });
    fs.writeFileSync(`${this.filename}.tmp`, JSON.stringify(this.jobs, null, 2), { mode: 0o600 });
    fs.renameSync(`${this.filename}.tmp`, this.filename);
  }
  update(id: string, patch: Partial<Job>) { Object.assign(this.jobs[id], patch, { updated: new Date().toISOString() }); this.save(); }
  run(input: string, task: (id: string) => Promise<void>): Promise<void> {
    const id = requestKey(input); const previous = this.jobs[id];
    if (previous && (['queued', 'running', 'completed'].includes(previous.status) || previous.files?.length)) return Promise.reject(new Error(`중복 작업 ${id}: ${previous.status}. 저장된 파일과 작업 기록을 확인하세요.`));
    this.jobs[id] = { id, status: 'queued', updated: new Date().toISOString() }; this.save();
    const execution = this.tail.then(async () => {
      this.update(id, { status: 'running', stage: 'generating' });
      try {
        await task(id);
        if (this.jobs[id].status === 'failed' || this.jobs[id].stage === 'publish-failed') throw new Error(this.jobs[id].error || 'Publication failed; retry the saved files without regenerating.');
        this.update(id, { status: 'completed' });
      }
      catch (error) { this.update(id, { status: 'failed', error: error instanceof Error ? error.message : String(error) }); throw error; }
    });
    this.tail = execution.catch(() => {});
    return execution;
  }
}
