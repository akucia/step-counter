# step-counter

<p align="left">
    <a href="" alt="Interrogate">
        <img src="interrogate_badge.svg" /></a>
</p>

# Metrics
| Path                       | train.f1_macro   | train.precision_macro   | train.recall_macro   | validation.f1_macro   | validation.precision_macro   | validation.recall_macro   |
|----------------------------|------------------|-------------------------|----------------------|-----------------------|------------------------------|---------------------------|
| reports/metrics/train.json | 0.43             | 0.5                     | 0.49                 | 0.42                  | 0.49                         | 0.49                      |

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
	node4-->node6
	node5-->node4
	node5-->node6
```
_graph_end_
