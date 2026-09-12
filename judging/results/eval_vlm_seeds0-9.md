# Evaluation -- randomised seeds

- Instruction: `Open the top drawer, set the plate, fork and spoon on the table, put the mug beside them and pour water into the mug.`
- Planner: `vlm+rules`
- Object positions from: **simulator state (no perception in the loop)**
- Seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

**Task success: 100%** (10/10 seeds)

- Mean steps per episode: 11.0
- Mean wall clock per episode: 4.2 s execution + 49.0 s planning
- Parallelisable share of the plan: 64% (steps the dependency graph permits to overlap; the executor runs them one at a time)

## Which planner's output was executed

- `rules` -- 10/10 episodes

Rejected upstream plans (the fallback chain recorded these):

- vlm failed: PlannerError: attempt 1: node n3 depends on unknown node(s) ['n2'] | attempt 2: unknown skill ''; known: ['handoff', 'home', 'open_drawer', 'pick', 'place', 'pour']

## Subgoal success

| Subgoal | Rate |
| --- | --- |
| bottle_upright | 100% |
| drawer_open | 100% |
| fork_placed | 100% |
| mug_placed | 100% |
| mug_upright | 100% |
| plate_placed | 100% |
| spoon_placed | 100% |