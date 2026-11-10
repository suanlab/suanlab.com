# Build and full-project type checking

The GitHub Pages run for `c8dc07f` failed during Next.js type checking on Node.js20: V8 exhausted its approximately4GiB heap. [The failed build](https://github.com/suanlab/suanlab.com/actions/runs/42765366686/job/129660494014) reached `next build` after the earlier `npm run check` stages. The failure annotation identifies heap exhaustion, not a TypeScript diagnostic.

Next previously used the root `tsconfig.json`, which includes application sources and all test/CLI tooling. Cold full-project diagnostics now include2,466,489lines of importedJSON, largely verification fixtures. Checking those fixtures inside the production builder duplicates work already covered by the subsequent full-project check.

## Configuration

- `next.config.mjs` selects `tsconfig.next.json`. It inherits strict settings, aliases and the Next plugin from the root configuration, and includes every TypeScript source under`src/` plus generated Next types. Build errors remain enabled.
- `npm run typecheck` retains the original root configuration and all test/CLI files. Its dedicated Node process allows6144MiB of old-space heap. Other build/test subprocesses keep their existing memory settings.
- `npm run check` continues to run lint, content validation, tests, build, the full type check and export validation. No stage was removed.

`scripts/build-typecheck-scope.test.ts` parses both actual configurations and checks every current application and tooling source against their file sets. It prevents production source omissions, loss of tooling coverage, disabling build errors, or silently replacing the full check with the production configuration.

## Verification on Node.js20.20.2

Cold checks used`--incremental false --extendedDiagnostics`:

| Scope | Files | TypeScript lines | Reported compiler memory |
| --- | ---: | ---: | ---: |
| Production | 2726 | 288942 | 1766241K (about1.7GiB) |
| Full project, including the new scope test | 4325 | 345342 | 4340396K (about4.1GiB) |

Both completed without diagnostics. The scope regression and ESLint passed. An isolated, credential-free worktree also completed the actual Next production build on Node.js20 and validated1837exportedHTML pages, internal links/assets/fragments, sitemap routes and canonicals. The previously verified Euclidean calculation/validation Worker bytes remain identical. Existing local homepage output4547files is preserved.

Evidence is in`.runtime/learning/build-typecheck-memory/`: CI failure annotations, cold compiler diagnostics, scope/lint/build/export logs, Worker hashes and`verification.json`. One initial local build was launched from the root directory; that newly launched process group was stopped, the output was verified unchanged, and the successful build ran in the isolated worktree. Existing previews and worktrees were preserved.

This local evidence fixes the observed heap failure; public deployment still requires the updated GitHub run to finish. The pending/running older revision has the previous configuration and is not proof that this fix deployed.
