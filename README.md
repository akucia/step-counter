# step-counter

<p align="left">
    <a href="" alt="Interrogate">
        <img src="interrogate_badge.svg" /></a>
    <a href="" alt="Coverage">
        <img src="coverage_badge.svg" /></a>
</p>

# Metrics
| Path                             | num_samples   | test.f1-score_macro   | test.precision_macro   | test.predicted_step_count   | test.recall_macro   | test.roc_auc   | test.support_macro   | test.target_step_count   | train.accuracy   | train.f1_macro   | train.precision_macro   | train.recall_macro   | validation.accuracy   | validation.f1_macro   | validation.precision_macro   | validation.recall_macro   | validation.threshold_mean   | validation.threshold_std   |
|----------------------------------|---------------|-----------------------|------------------------|-----------------------------|---------------------|----------------|----------------------|--------------------------|------------------|------------------|-------------------------|----------------------|-----------------------|-----------------------|------------------------------|---------------------------|-----------------------------|----------------------------|
| reports/metrics/train.json       | 7919          | -                     | -                      | -                           | -                   | -              | -                    | -                        | 0.87             | 0.79             | 0.76                    | 0.85                 | 0.9                   | 0.8                   | 0.8                          | 0.8                       | 0.7                         | 0.05                       |
| reports/metrics/test.json        | -             | 0.82                  | 0.83                   | 53                          | 0.81                | 0.92           | 1585.0               | 45                       | -                | -                | -                       | -                    | -                     | -                     | -                            | -                         | -                           | -                          |
| reports/metrics_tflite/test.json | -             | 0.82                  | 0.83                   | 53                          | 0.81                | 0.92           | 1585.0               | 45                       | -                | -                | -                       | -                    | -                     | -                     | -                            | -                         | -                           | -                          |

# Steps Graph
```mermaid
flowchart TD
	node1["data/steps/test.dvc"]
	node2["data/steps/train.dvc"]
	node3["evaluate"]
	node4["evaluate_tflite"]
	node5["export_tflite"]
	node6["predict_test"]
	node7["predict_tflite"]
	node8["train"]
	node9["update-metrics"]
	node1-->node3
	node1-->node4
	node1-->node6
	node1-->node7
	node2-->node5
	node2-->node8
	node3-->node9
	node4-->node9
	node5-->node6
	node5-->node7
	node6-->node3
	node6-->node9
	node7-->node4
	node7-->node9
	node8-->node5
	node8-->node6
	node8-->node9
```
_graph_end_
