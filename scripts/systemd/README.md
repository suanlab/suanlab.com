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
content, commits only the generated Markdown and local thumbnail, then pushes
`master`. Unrelated site edits are excluded; existing staged changes or a different
branch prevent publication. A failed push leaves the local commit available for
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
publication failed, stop new submissions and run:

```bash
npx tsx scripts/blog/retry-publish.ts JOB_ID
```

This retries Git publication only. It does not regenerate content or rewrite the
running bot's job history. Inspect the recorded filenames before retrying.
