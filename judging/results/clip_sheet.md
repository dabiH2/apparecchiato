# Clip sheet

Instruction: `Open the top drawer, set the plate, fork and spoon on the table, put the mug beside them and pour water into the mug.`

## seed 000  (`out/eval_clips/video/seed000.mp4`)

- **11 steps**, 64% parallelisable
- **hand-off:** plate A->B at [0.008, 0.15, 0.008]
- drawer at x=-0.160, cutlery reached by arm A
- randomisation: plate 0.98x 131 g mu=0.76, mug 0.92x 195 g mu=0.87, bottle 0.92x 158 g mu=0.64
- light at [0.16, -0.1, 0.93] intensity 1.02; table rgb [0.48, 0.3, 0.21], wall rgb [0.49, 0.35, 0.17]

| # | arms | action | why this arm |
| --- | --- | --- | --- |
| 1 | A | open_drawer() | explicitly requested |
| 2 | A | pick(object=plate) | only arm A can reach the plate |
| 3 | A+B | handoff(object=plate) | slot is out of arm A's reach, pass to arm B |
| 4 | B | place(object=plate target=plate) | arm B reaches the slot |
| 5 | A | pick(object=fork) | arm A reaches both the fork and its slot |
| 6 | A | place(object=fork target=fork) | same arm can finish the move |
| 7 | B | pick(object=mug) | arm B reaches both the mug and its slot |
| 8 | B | place(object=mug target=mug) | same arm can finish the move |
| 9 | A | pick(object=spoon) | arm A reaches both the spoon and its slot |
| 10 | A | place(object=spoon target=spoon) | same arm can finish the move |
| 11 | B+A | pour(source=bottle into=mug) | explicitly requested |

## seed 001  (`out/eval_clips/video/seed001.mp4`)

- **11 steps**, 64% parallelisable
- **hand-off:** plate A->B at [-0.011, 0.134, 0.009]
- drawer at x=-0.167, cutlery reached by arm A
- randomisation: plate 0.96x 148 g mu=1.20, mug 0.96x 103 g mu=0.58, bottle 0.93x 210 g mu=0.80
- light at [-0.03, 0.4, 1.14] intensity 0.90; table rgb [0.48, 0.28, 0.23], wall rgb [0.42, 0.22, 0.18]

| # | arms | action | why this arm |
| --- | --- | --- | --- |
| 1 | A | open_drawer() | explicitly requested |
| 2 | A | pick(object=plate) | only arm A can reach the plate |
| 3 | A+B | handoff(object=plate) | slot is out of arm A's reach, pass to arm B |
| 4 | B | place(object=plate target=plate) | arm B reaches the slot |
| 5 | A | pick(object=fork) | arm A reaches both the fork and its slot |
| 6 | A | place(object=fork target=fork) | same arm can finish the move |
| 7 | B | pick(object=mug) | arm B reaches both the mug and its slot |
| 8 | B | place(object=mug target=mug) | same arm can finish the move |
| 9 | A | pick(object=spoon) | arm A reaches both the spoon and its slot |
| 10 | A | place(object=spoon target=spoon) | same arm can finish the move |
| 11 | B+A | pour(source=bottle into=mug) | explicitly requested |

## seed 002  (`out/eval_clips/video/seed002.mp4`)

- **11 steps**, 64% parallelisable
- **hand-off:** plate A->B at [-0.035, 0.141, 0.008]
- drawer at x=-0.174, cutlery reached by arm A
- randomisation: plate 0.92x 257 g mu=1.00, mug 0.92x 111 g mu=0.91, bottle 0.96x 319 g mu=0.92
- light at [0.27, 0.16, 1.14] intensity 1.06; table rgb [0.43, 0.29, 0.22], wall rgb [0.36, 0.2, 0.14]

| # | arms | action | why this arm |
| --- | --- | --- | --- |
| 1 | A | open_drawer() | explicitly requested |
| 2 | A | pick(object=plate) | only arm A can reach the plate |
| 3 | A+B | handoff(object=plate) | slot is out of arm A's reach, pass to arm B |
| 4 | B | place(object=plate target=plate) | arm B reaches the slot |
| 5 | A | pick(object=fork) | arm A reaches both the fork and its slot |
| 6 | A | place(object=fork target=fork) | same arm can finish the move |
| 7 | B | pick(object=mug) | arm B reaches both the mug and its slot |
| 8 | B | place(object=mug target=mug) | same arm can finish the move |
| 9 | A | pick(object=spoon) | arm A reaches both the spoon and its slot |
| 10 | A | place(object=spoon target=spoon) | same arm can finish the move |
| 11 | B+A | pour(source=bottle into=mug) | explicitly requested |

## seed 003  (`out/eval_clips/video/seed003.mp4`)

- **11 steps**, 64% parallelisable
- **hand-off:** plate A->B at [-0.013, 0.143, 0.008]
- drawer at x=-0.197, cutlery reached by arm A
- randomisation: plate 0.89x 104 g mu=1.13, mug 0.92x 164 g mu=1.17, bottle 0.96x 229 g mu=0.94
- light at [-0.0, 0.16, 1.45] intensity 0.70; table rgb [0.45, 0.32, 0.14], wall rgb [0.62, 0.47, 0.23]

| # | arms | action | why this arm |
| --- | --- | --- | --- |
| 1 | A | open_drawer() | explicitly requested |
| 2 | A | pick(object=plate) | only arm A can reach the plate |
| 3 | A+B | handoff(object=plate) | slot is out of arm A's reach, pass to arm B |
| 4 | B | place(object=plate target=plate) | arm B reaches the slot |
| 5 | A | pick(object=fork) | arm A reaches both the fork and its slot |
| 6 | A | place(object=fork target=fork) | same arm can finish the move |
| 7 | B | pick(object=mug) | arm B reaches both the mug and its slot |
| 8 | B | place(object=mug target=mug) | same arm can finish the move |
| 9 | A | pick(object=spoon) | arm A reaches both the spoon and its slot |
| 10 | A | place(object=spoon target=spoon) | same arm can finish the move |
| 11 | B+A | pour(source=bottle into=mug) | explicitly requested |

## seed 004  (`out/eval_clips/video/seed004.mp4`)

- **11 steps**, 64% parallelisable
- **hand-off:** plate A->B at [-0.019, 0.159, 0.008]
- drawer at x=-0.200, cutlery reached by arm A
- randomisation: plate 1.03x 170 g mu=0.69, mug 0.93x 171 g mu=0.95, bottle 0.96x 161 g mu=0.66
- light at [-0.11, 0.35, 0.94] intensity 0.81; table rgb [0.42, 0.31, 0.25], wall rgb [0.59, 0.45, 0.17]

| # | arms | action | why this arm |
| --- | --- | --- | --- |
| 1 | A | open_drawer() | explicitly requested |
| 2 | A | pick(object=plate) | only arm A can reach the plate |
| 3 | A+B | handoff(object=plate) | slot is out of arm A's reach, pass to arm B |
| 4 | B | place(object=plate target=plate) | arm B reaches the slot |
| 5 | A | pick(object=fork) | arm A reaches both the fork and its slot |
| 6 | A | place(object=fork target=fork) | same arm can finish the move |
| 7 | B | pick(object=mug) | arm B reaches both the mug and its slot |
| 8 | B | place(object=mug target=mug) | same arm can finish the move |
| 9 | A | pick(object=spoon) | arm A reaches both the spoon and its slot |
| 10 | A | place(object=spoon target=spoon) | same arm can finish the move |
| 11 | B+A | pour(source=bottle into=mug) | explicitly requested |

## seed 005  (`out/eval_clips/video/seed005.mp4`)

- **11 steps**, 64% parallelisable
- **hand-off:** plate A->B at [-0.026, 0.152, 0.008]
- drawer at x=-0.180, cutlery reached by arm A
- randomisation: plate 1.01x 130 g mu=0.66, mug 0.95x 61 g mu=0.57, bottle 0.94x 296 g mu=1.16
- light at [0.06, -0.01, 0.96] intensity 1.07; table rgb [0.71, 0.43, 0.31], wall rgb [0.42, 0.21, 0.2]

| # | arms | action | why this arm |
| --- | --- | --- | --- |
| 1 | A | open_drawer() | explicitly requested |
| 2 | A | pick(object=plate) | only arm A can reach the plate |
| 3 | A+B | handoff(object=plate) | slot is out of arm A's reach, pass to arm B |
| 4 | B | place(object=plate target=plate) | arm B reaches the slot |
| 5 | A | pick(object=fork) | arm A reaches both the fork and its slot |
| 6 | A | place(object=fork target=fork) | same arm can finish the move |
| 7 | B | pick(object=mug) | arm B reaches both the mug and its slot |
| 8 | B | place(object=mug target=mug) | same arm can finish the move |
| 9 | A | pick(object=spoon) | arm A reaches both the spoon and its slot |
| 10 | A | place(object=spoon target=spoon) | same arm can finish the move |
| 11 | B+A | pour(source=bottle into=mug) | explicitly requested |

## seed 006  (`out/eval_clips/video/seed006.mp4`)

- **11 steps**, 64% parallelisable
- **hand-off:** plate A->B at [-0.003, 0.115, 0.009]
- drawer at x=-0.173, cutlery reached by arm A
- randomisation: plate 0.97x 136 g mu=0.95, mug 0.92x 140 g mu=0.74, bottle 0.94x 206 g mu=0.95
- light at [-0.5, 0.2, 1.11] intensity 1.29; table rgb [0.51, 0.31, 0.19], wall rgb [0.52, 0.38, 0.28]

| # | arms | action | why this arm |
| --- | --- | --- | --- |
| 1 | A | open_drawer() | explicitly requested |
| 2 | A | pick(object=plate) | only arm A can reach the plate |
| 3 | A+B | handoff(object=plate) | slot is out of arm A's reach, pass to arm B |
| 4 | B | place(object=plate target=plate) | arm B reaches the slot |
| 5 | A | pick(object=fork) | arm A reaches both the fork and its slot |
| 6 | A | place(object=fork target=fork) | same arm can finish the move |
| 7 | B | pick(object=mug) | arm B reaches both the mug and its slot |
| 8 | B | place(object=mug target=mug) | same arm can finish the move |
| 9 | A | pick(object=spoon) | arm A reaches both the spoon and its slot |
| 10 | A | place(object=spoon target=spoon) | same arm can finish the move |
| 11 | B+A | pour(source=bottle into=mug) | explicitly requested |

## seed 007  (`out/eval_clips/video/seed007.mp4`)

- **11 steps**, 64% parallelisable
- **hand-off:** plate A->B at [-0.02, 0.158, 0.009]
- drawer at x=-0.173, cutlery reached by arm A
- randomisation: plate 0.98x 91 g mu=0.81, mug 0.96x 77 g mu=1.16, bottle 0.95x 223 g mu=1.07
- light at [-0.07, 0.4, 1.3] intensity 0.96; table rgb [0.51, 0.38, 0.26], wall rgb [0.45, 0.33, 0.22]

| # | arms | action | why this arm |
| --- | --- | --- | --- |
| 1 | A | open_drawer() | explicitly requested |
| 2 | A | pick(object=plate) | only arm A can reach the plate |
| 3 | A+B | handoff(object=plate) | slot is out of arm A's reach, pass to arm B |
| 4 | B | place(object=plate target=plate) | arm B reaches the slot |
| 5 | A | pick(object=fork) | arm A reaches both the fork and its slot |
| 6 | A | place(object=fork target=fork) | same arm can finish the move |
| 7 | B | pick(object=mug) | arm B reaches both the mug and its slot |
| 8 | B | place(object=mug target=mug) | same arm can finish the move |
| 9 | A | pick(object=spoon) | arm A reaches both the spoon and its slot |
| 10 | A | place(object=spoon target=spoon) | same arm can finish the move |
| 11 | B+A | pour(source=bottle into=mug) | explicitly requested |

## seed 008  (`out/eval_clips/video/seed008.mp4`)

- **11 steps**, 64% parallelisable
- **hand-off:** plate A->B at [-0.004, 0.164, 0.008]
- drawer at x=-0.163, cutlery reached by arm A
- randomisation: plate 0.93x 216 g mu=0.91, mug 0.96x 81 g mu=0.63, bottle 0.93x 250 g mu=0.86
- light at [0.39, 0.38, 1.25] intensity 1.30; table rgb [0.56, 0.32, 0.28], wall rgb [0.36, 0.22, 0.18]

| # | arms | action | why this arm |
| --- | --- | --- | --- |
| 1 | A | open_drawer() | explicitly requested |
| 2 | A | pick(object=plate) | only arm A can reach the plate |
| 3 | A+B | handoff(object=plate) | slot is out of arm A's reach, pass to arm B |
| 4 | B | place(object=plate target=plate) | arm B reaches the slot |
| 5 | A | pick(object=fork) | arm A reaches both the fork and its slot |
| 6 | A | place(object=fork target=fork) | same arm can finish the move |
| 7 | B | pick(object=mug) | arm B reaches both the mug and its slot |
| 8 | B | place(object=mug target=mug) | same arm can finish the move |
| 9 | A | pick(object=spoon) | arm A reaches both the spoon and its slot |
| 10 | A | place(object=spoon target=spoon) | same arm can finish the move |
| 11 | B+A | pour(source=bottle into=mug) | explicitly requested |

## seed 009  (`out/eval_clips/video/seed009.mp4`)

- **11 steps**, 64% parallelisable
- **hand-off:** plate A->B at [-0.024, 0.155, 0.009]
- drawer at x=-0.191, cutlery reached by arm A
- randomisation: plate 1.02x 224 g mu=0.69, mug 0.92x 75 g mu=0.69, bottle 0.95x 254 g mu=0.64
- light at [0.35, -0.09, 1.31] intensity 0.85; table rgb [0.68, 0.46, 0.42], wall rgb [0.49, 0.31, 0.21]

| # | arms | action | why this arm |
| --- | --- | --- | --- |
| 1 | A | open_drawer() | explicitly requested |
| 2 | A | pick(object=plate) | only arm A can reach the plate |
| 3 | A+B | handoff(object=plate) | slot is out of arm A's reach, pass to arm B |
| 4 | B | place(object=plate target=plate) | arm B reaches the slot |
| 5 | A | pick(object=fork) | arm A reaches both the fork and its slot |
| 6 | A | place(object=fork target=fork) | same arm can finish the move |
| 7 | B | pick(object=mug) | arm B reaches both the mug and its slot |
| 8 | B | place(object=mug target=mug) | same arm can finish the move |
| 9 | A | pick(object=spoon) | arm A reaches both the spoon and its slot |
| 10 | A | place(object=spoon target=spoon) | same arm can finish the move |
| 11 | B+A | pour(source=bottle into=mug) | explicitly requested |
