# step-counter

<p align="left">
    <a href="" alt="Interrogate">
        <img src="interrogate_badge.svg" /></a>
    <a href="" alt="Coverage">
        <img src="coverage_badge.svg" /></a>
</p>

# Metrics
| Path                       | test.f1-score_macro   | test.precision_macro   | test.recall_macro   | test.roc_auc   | test.step_count_mae   | test.support_macro   | train.f1_macro   | train.precision_macro   | train.recall_macro   | validation.f1_macro   | validation.precision_macro   | validation.recall_macro   |
|----------------------------|-----------------------|------------------------|---------------------|----------------|-----------------------|----------------------|------------------|-------------------------|----------------------|-----------------------|------------------------------|---------------------------|
| reports/metrics/train.json | -                     | -                      | -                   | -              | -                     | -                    | 0.7              | 0.68                    | 0.75                 | 0.69                  | 0.68                         | 0.74                      |
| reports/metrics/test.json  | 0.72                  | 0.7                    | 0.75                | 0.79           | 20.5                  | 1585.0               | -                | -                       | -                    | -                     | -                            | -                         |

# Steps Graph
```mermaid
flowchart TD
	node1["data/steps/test.dvc"]
	node2["data/steps/train.dvc"]
	node3["evaluate"]
	node4["predict_test"]
	node5["train"]
	node6["update-metrics"]
	node1-->node3
	node1-->node4
	node2-->node5
	node3-->node6
	node4-->node3
	node4-->node6
	node5-->node4
	node5-->node6
```
_graph_end_
