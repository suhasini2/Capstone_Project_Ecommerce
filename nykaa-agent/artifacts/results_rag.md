### RAG comparison and calibration

| collection | mean P@3 | mean R@3 |
|---|---|---|
| fixed | 0.556 | 1.000 |
| sentence | 0.750 | 1.000 |

Recommendation: deploy the 'sentence' collection - mean P@3 0.750 / R@3 1.000 versus 0.556 / 1.000 for 'fixed'.

| collection | in-scope top-1 cosines | out-of-scope top-1 cosines | threshold |
|---|---|---|---|
| fixed | [0.518, 0.686, 0.746, 0.617] | [0.129, 0.024, 0.21] | 0.364 |
| sentence | [0.523, 0.8, 0.76, 0.589] | [0.096, 0.073, 0.234] | 0.378 |
