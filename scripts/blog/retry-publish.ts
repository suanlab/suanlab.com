import { config } from 'dotenv';
import { publishPosts } from './publish';
import { JobQueue } from './job-queue';
if (process.env.SUANLAB_QUEUE_LOCKED !== '1') throw new Error('Use npm run bot:retry -- JOB_ID to acquire the queue lock.');
config({ path: '.env.local', quiet: true });
const queue = new JobQueue('.runtime/slack-jobs.json');
const id = process.argv[2];
queue.retryPublication(id, files => publishPosts(files, `Publish saved blog job ${id}`, process.cwd(), stage => queue.update(id, { stage })))
  .then(() => console.log(`Saved job ${id}: ${queue.jobs[id].stage}`))
  .catch(error => { console.error(error.message); process.exitCode = 1; });
