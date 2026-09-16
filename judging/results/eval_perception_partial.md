# Evaluation -- randomised seeds

- Instruction: `Open the top drawer, set the plate, fork and spoon on the table, put the mug beside them and pour water into the mug.`
- Planner: `rules`
- Object positions from: **colour detector, driving ['bottle', 'mug'] only**
- Seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

**Task success: 70%** (7/10 seeds)

- Mean steps per episode: 10.2
- Mean wall clock per episode: 7.3 s execution + 0.0 s planning
- Parallelisable share of the plan: 64% (steps the dependency graph permits to overlap; the executor runs them one at a time)

## Which planner's output was executed

- `rules` -- 10/10 episodes

## What perception actually located

Every object the detector placed, and how far that was from the simulator's own answer. Objects it did not report kept their true position -- the cutlery starts inside a shut drawer and no camera can see it.

| Object | Readings used | Mean error | Worst |
| --- | --- | --- | --- |
| bottle | 111 | 10 mm | 16 mm |
| mug | 94 | 8 mm | 22 mm |

## Subgoal success

| Subgoal | Rate |
| --- | --- |
| mug_placed | 80% |
| spoon_placed | 80% |
| bottle_upright | 90% |
| drawer_open | 100% |
| fork_placed | 100% |
| mug_upright | 100% |
| plate_placed | 100% |

## Every failed seed

3 row(s) for 3 failed seed(s) — these two numbers must match, and the table below is keyed on seeds rather than on failing steps so that they do. A seed can run every step, have no step report a failure, and still not meet the task criterion; those are marked `end-state`.

| Seed | Kind | Where | Detail |
| --- | --- | --- | --- |
| 2 | step | [B] pick(object=mug) | lost the mug |
| 4 | end-state | end state (11 step(s) ran, none reported failure) | subgoals false at the end: bottle_upright |
| 8 | step | [B] pick(object=mug) | lost the mug |