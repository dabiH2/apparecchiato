# Evaluation -- randomised seeds

- Instruction: `Open the top drawer, set the plate, fork and spoon on the table, put the mug beside them and pour water into the mug.`
- Planner: `rules`
- Object positions from: **simulator state (no perception in the loop)**
- Seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]

**Task success: 100%** (100/100 seeds)

- Mean steps per episode: 11.0
- Mean wall clock per episode: 3.0 s execution + 0.0 s planning
- Parallelisable share of the plan: 64% (steps the dependency graph permits to overlap; the executor runs them one at a time)

## Which planner's output was executed

- `rules` -- 100/100 episodes

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