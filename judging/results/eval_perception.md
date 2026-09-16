# Evaluation -- randomised seeds

- Instruction: `Open the top drawer, set the plate, fork and spoon on the table, put the mug beside them and pour water into the mug.`
- Planner: `rules`
- Object positions from: **colour detector**
- Seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

**Task success: 0%** (0/10 seeds)

- Mean steps per episode: 5.5
- Mean wall clock per episode: 2.4 s execution + 0.0 s planning
- Parallelisable share of the plan: 64% (steps the dependency graph permits to overlap; the executor runs them one at a time)

## Which planner's output was executed

- `rules` -- 10/10 episodes

## What perception actually located

Every object the detector placed, and how far that was from the simulator's own answer. Objects it did not report kept their true position -- the cutlery starts inside a shut drawer and no camera can see it.

| Object | Readings used | Mean error | Worst |
| --- | --- | --- | --- |
| bottle | 60 | 9 mm | 16 mm |
| fork | 45 | 87 mm | 562 mm |
| mug | 55 | 9 mm | 22 mm |
| plate | 45 | 141 mm | 460 mm |
| spoon | 50 | 157 mm | 406 mm |

## Subgoal success

| Subgoal | Rate |
| --- | --- |
| spoon_placed | 0% |
| mug_placed | 30% |
| fork_placed | 50% |
| plate_placed | 60% |
| bottle_upright | 100% |
| drawer_open | 100% |
| mug_upright | 100% |

## Every failed seed

10 row(s) for 10 failed seed(s) — these two numbers must match, and the table below is keyed on seeds rather than on failing steps so that they do. A seed can run every step, have no step report a failure, and still not meet the task criterion; those are marked `end-state`.

| Seed | Kind | Where | Detail |
| --- | --- | --- | --- |
| 0 | step | [A] pick(object=plate) | lost the plate |
| 1 | step | [A] pick(object=plate) | lost the plate |
| 2 | step | [B] pick(object=mug) | lost the mug |
| 3 | step | [B] place(object=mug target=mug) | mug landed 189 mm from its slot |
| 4 | step | [A] pick(object=plate) | could not build motion: arm A has no clearance above pre-pick-plate at [-0.131, 0.498, 0.009] (tried 2-8 cm) |
| 5 | step | [A] pick(object=spoon) | could not build motion: arm A cannot reach pick-spoon at [-0.049, 0.116, 0.023]: target outside the reachable workspace for this pitch |
| 6 | step | [A] pick(object=spoon) | could not build motion: arm A cannot reach pick-spoon at [-0.06, 0.124, 0.023]: target outside the reachable workspace for this pitch |
| 7 | step | [A] pick(object=fork) | could not build motion: arm A has no clearance above pre-pick-fork at [0.01, 0.487, 0.023] (tried 2-8 cm) |
| 8 | step | [A] pick(object=plate) | lost the plate |
| 9 | step | [A] pick(object=spoon) | could not build motion: arm A cannot reach pick-spoon at [-0.047, 0.116, 0.023]: target outside the reachable workspace for this pitch |