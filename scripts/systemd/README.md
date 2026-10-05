# Slack bot service

The user service runs the bot from the repository root, where it reads `.env.local`.
The unit uses this machine's Node installation; update `ExecStart`, `WorkingDirectory`,
and `PATH` when moving the repository or upgrading Node.

Required environment values: `SLACK_BOT_TOKEN`, `SLACK_APP_TOKEN`, and
`SLACK_SIGNING_SECRET`. Content generation also needs the configured AI provider
keys. `SLACK_ALLOWED_CHANNELS` optionally restricts commands to comma-separated
channel IDs; when unset, commands are accepted in all channels available to the app.
Never commit `.env.local`.

Install or update the unit:

```bash
mkdir -p ~/.config/systemd/user
cp scripts/systemd/suanlab-slack.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now suanlab-slack.service
```

Operations:

```bash
systemctl --user status suanlab-slack.service
systemctl --user restart suanlab-slack.service
journalctl --user -u suanlab-slack.service -n 50 --no-pager
systemctl --user disable --now suanlab-slack.service
```

User lingering must be enabled for startup without an interactive login. It is
already enabled on the current host. The service restarts after failures.

Slack commands: `/suanblog`, `/suanblog-status`, `/suanblog-help` (these must also be
registered in the Slack app with Socket Mode enabled). A generation command saves
content, commits only generated Markdown and referenced blog images, then pushes
`master`. Unrelated site edits are excluded; existing staged changes or a different
branch prevent publication. Unpushed commits containing unrelated files and a
branch behind the remote also prevent automatic publication. A failed push leaves the local commit available for
manual review and retry. Do not start a second bot process alongside this service.

Connection startup does not test AI generation or GitHub publishing. The local
publication regression check uses only a disposable repository and local remote:

```bash
npx tsx --test scripts/blog/publish.test.ts
```

## Queue and recovery

The bot records jobs in `.runtime/slack-jobs.json` (gitignored, mode 0600).
Generation requests run one at a time. Repeated completed/queued requests are
rejected; arXiv URL and ID forms share a key. A restart marks in-flight work
interrupted instead of silently repeating billable generation. Partial batches
retain the paths of saved posts and the failed status.

Set `Environment=SUANLAB_PUBLISH_MODE=review` in a systemd override to require
review before pushing generated content. Restart the service after changing its
environment. Default mode retains automatic publishing. For saved jobs whose
publication failed, use `/suanblog retry JOB_ID` in an allowed channel. This shares
the running queue and requires no additional Slack command registration.

For offline recovery:

```bash
systemctl --user stop suanlab-slack.service
npm run bot:retry -- JOB_ID
systemctl --user start suanlab-slack.service
```

Both retry paths publish saved files without generating content. They persist
attempt start/end times, outcome and failure stage. Successful publication preserves
any earlier partial-generation error. Already-pushed jobs cannot be pushed again.
In review mode, a retry stays in review; it does not approve content or change mode.

Use `npm run bot:slack` for a foreground daemon. The wrapper and service use Linux
`flock` on `.runtime/slack-queue.lock`; an existing owner returns exit code 75.
The kernel releases the lock after exit/crash, so a stale file is harmless. Never
bypass the wrapper by setting its internal environment flag manually.

`/suanblog-status` reports job totals, waiting/running jobs, failed/interrupted jobs,
review/push counts, publication attempts and average finished-attempt duration.
Counters are job-based, not post-based. Inspect `.runtime/slack-jobs.json` for the
full history and `journalctl` for process diagnostics. Validation, synchronization,
staging, commit and push failures retain saved files for correction and retry.

## Publishing checkout decision

The current deployment retains the existing repository checkout. An isolated
worktree was considered, but would require separate credential/configuration and
asset synchronization while the user also maintains this checkout. The enforced
boundary is instead: one queue writer, master only, no pre-staged changes, no
unrelated unpushed commits, and only the current job's content/assets in the Git
commit. Local-remote integration tests exercise these refusal cases. A separate
publishing checkout remains an optional operational change if concurrent editing
frequently causes these deliberate refusals; it is not assumed to exist.
