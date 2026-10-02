<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
Contributions require CLA — see CONTRIBUTING.md.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# Agent worktrees and the shared execution queue

## Contract and implementation status

This is the detailed operational contract referenced by [AGENTS.md Rules 34 and 35](../../../AGENTS.md). Several agents may analyze and edit simultaneously in separate worktrees. Only one agent at a time may own the machine's GL compilation, test and pipeline execution slot. At most one GL pipeline may run on the machine, across every checkout and agent.

**The controller is `.scripts/agent_queue.py`** (acceptance tests: `.scripts/tests/test_agent_queue.py`). There is no standing controller process to wait for: an agent that needs the slot starts the controller itself. An empty queue file is not an execution lock or a notification service; the controller supplies the locking, the queue-file watcher, process supervision and release below. Do not substitute timer-based polling. Documentation, source inspection, editing and ordinary Git operations do not require the execution slot.

| Command | Role |
|---|---|
| `claim --agent-id … --model … --created-at … --session-id … --worktree … --branch …` | Enqueue once and stay pending until this agent owns the slot; prints one `turn_granted` JSON line with the reservation token. A repeated claim reconnects to the same entry. |
| `launch --agent-id … --reservation … --plan PLAN.json` | Verify the idle reservation, run the one-time pre-launch execution check and start the detached supervisor; returns at once. |
| `supervise` | Started by `launch`: runs the plan's build/test steps and at most one final pipeline step inside a job object, waits for the complete process group, writes the completion record, preserves the listed evidence and removes the entry on success, failure or crash. |
| `release --agent-id …` | Remove a waiting or idle-reserved entry. |
| `status` | Print the queue and the GL executions visible on the machine. |

A plan is JSON: `descriptor`, absolute `cwd` and `completion` paths, `steps` (each `name`, `kind` = `build` / `test` / `pipeline`, `command`, `log`, optional `env` and `executable`) and optional `preserve` operations (`move` / `copy` / `copytree` of run artefacts to unique evidence names).

The maintainer's final crash rule is authoritative: every pipeline completion, including a crash, removes the owner from the queue. A retry requires a new entry at the bottom. Starting a pipeline releases the agent from active waiting, **not** the machine reservation.

## Identity and workspace

At session creation, every agent, including a delegated agent, assigns itself an identity containing the reported model name, creation timestamp in UTC and a random suffix. Store the unabridged model label and timestamp as separate fields, together with the session identifier, task branch and absolute worktree path. Do not guess an unavailable model variant. An existing session adopting this policy records its identity-registration time explicitly instead of inventing its earlier creation time. Keep this identity in the task ledger or a locally excluded identity record even while the agent is absent from the queue.

Each agent uses its own task branch and Git worktree for GL edits, builds, tests and runs. Agent worktrees are created under. The primary checkout belongs to the maintainer. Reading shared instructions, maintaining the central queue and local Git exclusions, and explicitly authorized landing/worktree administration are allowed there. Other work in that checkout requires an explicit maintainer instruction for the particular task; it does not grant a continuing exception to other agents.

Use the requested task base when creating the worktree. Keep configuration, executables, generated proof files, dumps, logs and temporary build products inside that worktree. Inspect configured output paths and junctions before execution: isolation of tracked source alone is insufficient if generated output still points to the primary checkout or another agent's worktree. Do not change the maintainer's branch outside an explicitly authorized operation or the primary-checkout synchronization required after a `main` update. Do not reuse another agent's working directory.

## One queue for the machine

The authoritative file is. Every worktree uses this absolute path; no per-worktree queue is permitted. Exclude `/.agent_queue.json` and its `/.agent_queue.json.*.tmp` transaction files through the primary repository's shared Git information directory, `info/exclude`. Do not commit queue state or use a tracked `.gitignore` entry as a replacement for the requested local exclusion.

Use the same machine-wide named Windows mutex, `Global\GL.AgentExecutionQueue.v1`, for every queue operation. A controller holds this mutex only for short read/check/change transactions, never throughout a build or run. Reservation ownership is recorded in the queue. Agent identity, reservation identity and process creation time must be checked before acting on an entry; process identifiers alone can be reused.

The initial JSON document is:

```json
{
  "schema_version": 1,
  "revision": 0,
  "next_ticket": 1,
  "entries": []
}
```

Each entry records:

| Field | Meaning |
|---|---|
| `agent_id`, `model`, `created_at_utc`, `session_id` | Stable agent identity; adoption time is identified when the session predates the policy. |
| `worktree`, `branch` | Absolute isolated checkout and owning task branch. |
| `ticket`, `enqueued_at_utc` | Monotonic queue order and time this turn was requested. |
| `state` | `waiting`, `reserved`, `building`, `testing`, `starting`, or `running`. |
| `reservation_id` | Fresh ownership token assigned on a successful claim. |
| `controller` | Supervising process identifier and creation time. |
| `job` | Null while idle; otherwise job kind, process identity, supervised child group, unique log paths and completion-record path. |

Keep at most one entry per agent. Only the first entry may have an owning state. Every mutation validates the existing document, preserves other agents' entries, increments `revision`, and publishes a complete UTF-8 document through same-directory atomic replacement. Queue order changes only by appending a new turn or removing an ended/cancelled turn. A malformed or unsupported document is an error, not permission to recreate an empty queue.

## Enqueue, wait, claim and notify

An agent joins only when ready to use its turn. It invokes **one persistent blocking wait operation** backed by an event-driven controller. The operation remains pending while the agent is waiting. Ending the agent session after appending its name would leave no delivery path for a turn notification and is not this protocol.

1. Arm a Windows directory-change watcher on the primary checkout before the initial queue transaction. Filter notifications for `.agent_queue.json`. Watch the directory because atomic replacement changes the file object and can invalidate a watcher attached only to the old file.
2. Acquire the shared mutex, read and validate the queue, and append this agent once with a new ticket if it has no entry. Check whether it is now first in the same transaction; do not wait for an event to discover an already-available turn.
3. If it is first and has no reservation yet, atomically mark it `reserved` with a fresh reservation token. If it is not first, release the mutex and wait on queue-file events.
4. A directory-change event wakes the controller, not the language model for each intermediate queue movement. The controller reacquires the mutex, rereads the authoritative document and rechecks eligibility. Multiple events may be coalesced; neither an event nor an old snapshot grants ownership. Duplicate events must not duplicate entries or claims.
5. When the claim succeeds, the pending helper operation returns a structured `turn_granted` result containing agent identity, ticket, reservation token, worktree and queue revision. Completing that tool operation notifies the waiting agent. The agent verifies that the result belongs to its own request before starting work. If it is no longer ready, it releases immediately and rejoins later.
6. The watcher monitors the queue file only, never GL processes: the queue is the single source of turn order. A GL execution started outside the queue is found by the one-time pre-launch check (next section) and reported; it is never waited on.

Arming the watcher before the first check prevents a release between checking and subscribing from being missed. Changes arriving during a transaction must remain pending for the next read. Notification overflow or watcher reconnection requires a fresh locked reconciliation; it never grants a slot by itself. An abandoned mutex similarly requires recovery of the existing state, not queue deletion.

No timer may repeatedly read the queue, query processes, inspect logs or send progress messages. A transport timeout is not loss of queue position: reconnect to the same request/controller and keep its event wait, rather than appending a second entry. If the environment cannot keep a wait pending and deliver its completion, do not claim that the agent will be awakened automatically; report the missing capability.

```mermaid
sequenceDiagram
    participant A as Waiting agent
    participant H as Its queue controller
    participant Q as Shared queue and mutex
    participant R as Current run supervisor
    A->>H: Enqueue and wait (one pending call)
    H->>H: Arm directory-change watcher
    H->>Q: Locked append and eligibility check
    Q-->>H: Another owner is first
    Note over H: Wait on queue-file change events; no polling
    R->>Q: On complete process-group exit, remove owner atomically
    Q-->>H: Directory-change event
    H->>Q: Lock, reread, claim if first
    Q-->>H: Reservation recorded
    H-->>A: turn_granted; pending tool call completes
```

## Execute one turn

Before **each** build, test or pipeline launch, the controller verifies the reservation under the mutex and checks once for conflicting execution across the entire machine; an overlap aborts that launch and is reported, never waited on. Include complete `main.py` pipelines, native prover processes, standalone verification/Lean jobs, and queued build/test jobs; do not filter only to this agent's worktree. No native prover process between pipeline batches does not mean that the pipeline has ended. If process state cannot be inspected, do not infer that the machine is idle. Any authorized WSL run participates in the same reservation and must expose its process lifetime to the supervisor.

The first agent may perform sequential rebuilds, focused compilation fixes, and unit-test runs while retaining its turn. Failed compilation or failed unit tests may be corrected and repeated within that turn. Never overlap those jobs. Release the idle reservation before extended analysis, substantial unrelated editing, waiting for user input, or when the required execution work is finished. Stop the current owned job normally before releasing; do not delete the reservation while a child is still running.

One turn permits **one pipeline launch attempt**. Record `starting` before launching it so another agent cannot claim the apparent gap before a process identifier exists. Launch under a controller that supervises the complete process group, records the process identity and logs, and attaches one completion watcher; then set `running`. If launch fails, finish the turn as a failure. The controller must reconcile a crash during `starting` by inspecting the supervised job before it removes the entry.

While the pipeline is running, its agent remains first and all other agents' compilation, tests and runs stay blocked. The agent can read logs after completion; it may do independent analysis meanwhile, but must not modify the source, inputs, executable or output files used by that live run. A user interruption or a disconnected agent does not release a still-running job. Never terminate another agent's or the maintainer's execution to obtain the slot.

The shared controller's reservation and launch discipline closes races between participating agents. An independent manual launch can ignore that discipline; a JSON file cannot prevent it. Maintainer-started jobs should use the same reservation protocol for a strict machine-wide guarantee. An existing external job blocks the launch that finds it, and the overlap is reported rather than concealed, waited on or forcibly terminated.

## Completion, release and crash recovery

The run supervisor waits for the complete pipeline process group, including verifier and Lean children. Root-process exit alone does not release the machine if a child survives. On success, failure or crash, it writes a unique completion record with the final status and evidence paths. Then, under the mutex, it verifies the agent/reservation identity and removes **that** first entry. Atomic publication wakes all queue controllers; only the newly eligible first agent may claim. Logs and post-run analysis must not keep the execution slot occupied.

The completed agent receives the completion notification and inspects the saved result. A retry joins the bottom with a new ticket; it does not keep or reclaim the old position. Several compilation/unit-test attempts before a pipeline are allowed, but every pipeline termination ends its turn, even if debugging is unfinished.

Cancelling a waiting request removes only that agent's entry under the mutex. Releasing an idle owner does the same. If the agent disappears during a live job, its supervisor retains the reservation and completes the normal release when the job ends. Recovery of a stale entry requires verifying that the agent/controller is inactive and no associated job or child remains. Inaccessibility, elapsed time, or a missing progress update is not proof of abandonment. Preserve recovery evidence and do not remove another agent's live reservation.

## Completion watcher lifetime and transport

A run supervisor and an agent notification watcher have different lifetimes. The supervisor owns the process group, preserves evidence and releases the execution slot even if the agent disconnects. A completion watcher only delivers the resulting completion record to the agent; losing its tool transport neither stops the job nor frees the slot. The existing controller remains unchanged by this notification procedure.

1. Start one persistent native helper through the shell tool and retain its returned session identity. A Windows directory-event helper, including PowerShell `FileSystemWatcher` with queued event subscriptions, is suitable. An in-process Node watcher whose tool call times out before the job ends is not a durable notification path. Increasing a requested tool timeout does not establish that the transport supports it.
2. Subscribe to creation, replacement/rename and change notifications for the unique completion filename before the initial existence check. Keep notifications queued while checking. Watch the directory because the supervisor publishes the complete JSON record atomically. Handle notification errors explicitly; do not spin, periodically inspect the file, or silently abandon the subscription. After each event, inspect only that completion path; once present, read its final status, exit codes and log paths and end the helper.
3. Keep one blocking consumption of that helper's output pending. A transport timeout reconnects to the same helper/session; it must not start another watcher, rerun the command, inspect live logs, or query process state. Reattaching to the completion wait is allowed by Rule 29. Forward the final helper output as a completion notification. An ordinary yielded shell session by itself does not promise that the language model will wake after its turn ends.
4. On notification, verify the descriptor and process/supervisor identity against the launched plan, then inspect the final evidence. Clean up that watcher's subscriptions and handles. A user-stopped run retains its actual nonzero exit status and incomplete timing; notification success does not imply pipeline success.

Live timing or log inspection still requires the maintainer's explicit request under Rule 29. That permission persists for the authorized investigation and should not be requested again at every observation. Without it, inspect only the final completion record and completed logs. A completion helper must not emit periodic unchanged-status messages.

## Landing and worktree cleanup

Commit complete milestones and push the agent branch under the existing repository rules. Before any landing, strip the branch's temporary traps and verify the stripped tree ([AGENTS.md Rule 36](../../../AGENTS.md)). A merge into `main` still needs the maintainer's authorization; this policy does not authorize automatic merges. Serialize authorized landing operations using the same short administrative mutex, verify the current destination head and cleanliness, and stop rather than overwriting concurrent maintainer changes. These administrative operations do not occupy a compilation/run turn.

After an authorized merge or squash into `main`, complete the primary-checkout synchronization below and verify that the intended committed work has landed; for a squash, record the content/tree comparison rather than relying on ancestry alone. **Every authorized landing — squash, merge or fast-forward, onto `main` or onto a parent branch — ends with the landing agent removing the worktree of the branch that landed, when that worktree serves that branch alone** (maintainer, 2026-09-26: a landed worktree is too big to keep). A worktree that also serves the destination branch or any other unlanded branch — a sub-branch worked in its parent's worktree, whose parent continues there — is not a landed worktree: switch it to the destination branch and keep it with its evidence (maintainer, 2026-09-30). Record the results in the task ledger first: the worktree's generated evidence (hashburst dumps, run logs, shortcut captures, build outputs) is deleted with it and is not preserved elsewhere. Require an empty job set, no queue entry, the landed commit pushed and a clean worktree. Resolve and verify the exact absolute worktree path and check it for junctions (`Get-ChildItem -Recurse -Force -Attributes ReparsePoint`; a recursive delete follows a junction out of the worktree), then remove only that agent's worktree with ordinary `git worktree remove`. Never force-remove dirty work, delete the primary checkout, discard unmerged work, or delete preserved source branches. No new build or pipeline run is implied by a landing or cleanup request.

## Keep the primary checkout at the main tip

The primary folder's purpose is to expose the latest `main` tip to the maintainer. Every agent that performs an authorized update to `main` owns the follow-through: leave checked out on branch `main`, with its index and tracked files matching the current `main` commit. This applies to merges, squashes, fast-forwards, direct commits and follow-up documentation or identifier-renumbering commits. An update is not complete merely because a branch reference or the remote advanced.

The maintainer explicitly authorizes this clean synchronization; no repeated approval is needed. It is an administrative exception to the agent-worktree rule and the no-unrequested-reset rule. It does not authorize rewriting history, losing local commits, discarding uncommitted work or changing the contents of untracked queue/evidence files.

1. Resolve the exact primary checkout path and acquire `Global\GL.AgentExecutionQueue.v1` for the administrative operation. Verify the current branch, main reference and worktree state before the authorized update. Record the pre-update commit and clean state so an expected checkout change can be distinguished from unrelated maintainer edits. Do not switch/reset a checkout being used by a live build or pipeline; preserve that execution and arrange synchronization after it ends.
2. If there are unexpected staged or unstaged changes, untracked-path collisions, or concurrent changes to the recorded branch/reference, stop and report them. Do not automatically stash, commit someone else's work, delete files or force a checkout. A missing clean-state record is not permission to assume that modifications are disposable.
3. Complete the authorized `main` update and resolve the full commit identifier of the current local `refs/heads/main` while holding the mutex. This verified identifier is the synchronization target. Do not use an old captured identifier after another update, or blindly use `origin/main`: a remote-tracking reference can be stale or behind unpublished local commits. Fetching a remote update is a separate part of the authorized synchronization; only a verified fast-forward may advance local main automatically. Divergence requires investigation, never a destructive reset.
4. Ensure the primary checkout is on `main` using ordinary `git switch main` from a clean state if necessary. Never run the maintenance reset while an agent/task branch is checked out, because it would move that branch. If `main` is unexpectedly checked out in another worktree, preserve it and resolve that ownership conflict; do not use checkout overrides.
5. Run `git reset --hard <verified-main-commit>` in the primary checkout. This is deliberately a hard reset, per Rule 4, to synchronize the tracked index and files. The previous steps must have established that it cannot discard unrelated work or obstructing untracked files. Do not run `git clean`; the shared queue, local exclusions and archived evidence remain untouched.
6. Verify that the checked-out branch is `main`, `HEAD` and `refs/heads/main` both equal the verified target, and both working-tree and staged diffs against `HEAD` are empty. When the task requires a push, verify that push and the resulting remote head too. If another final commit is added, repeat synchronization for that new tip. Record the final commit in the task evidence, release the administrative mutex, and only then report the main update complete or remove the landed agent worktree.

For a normal commit or merge performed directly in a clean primary checkout, HEAD and tracked files may already match. The same verification and explicitly authorized hard synchronization still complete the procedure. Do not mistake the requested final checkout reset for permission to reset the main branch backwards or discard work. These operations do not request a new build, test, proof run or determinism rerun.

## Controller acceptance before use

A controller implementation must demonstrate exclusive claims by competing agents, immediate notification when an empty queue is joined, reliable wake-up after atomic replacement, cancellation, transport reconnection without duplicate entries, a claim that ignores GL processes while the pre-launch check reports an external run, complete child-process supervision, and crash/failure dequeue followed by bottom re-entry. `.scripts/tests/test_agent_queue.py` demonstrates these on a private queue and mutex. It must also demonstrate that extended analysis releases an idle owner and that a running entry survives agent disconnection. Verify these coordination behaviors without launching simultaneous GL runs as a test technique.

This operational workflow changes neither mathematical inference nor prover storage architecture. The queue orders participating agents only; a manual launch outside the queue is not prevented, it is reported by the next pre-launch check.


## Open questions

### Execution-host lifetime of detached supervisors

A native `Start-Process` launch does not itself establish that the surrounding execution host lets its process group survive an ended agent turn. One observed run reached the first pipeline batch after a successful rebuild and native tests, then its supervisor and child identities disappeared without publishing completion, while the queue retained `running`. The termination cause is unknown. Until this host lifetime is verified, keep the agent turn and completion transport attached. A completion-file watcher alone cannot detect a supervisor that disappears before writing the file. Recovery still requires positive evidence that the recorded identities and associated GL executions are absent; preserve the interrupted attempt, record the unknown exit status, and remove only the matching stale reservation under the shared mutex.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
