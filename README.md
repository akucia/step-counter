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
	node1["data/steps/test/accelerometer-data-2023-09-01 07:49:52.631649.csv.dvc"]
	node2["data/steps/test/accelerometer-data-2023-09-01 07:54:26.715226.csv.dvc"]
	node3["data/steps/train/accelerometer-data-2023-08-25 20:31:42.558559.csv.dvc"]
	node4["data/steps/train/accelerometer-data-2023-09-01 07:51:30.084540.csv.dvc"]
	node5["data/steps/train/accelerometer-data-2023-09-01 07:54:07.963894.csv.dvc"]
	node6["data/steps/train/accelerometer-data-2023-09-01 07:54:17.508286.csv.dvc"]
	node7["data/steps/train/accelerometer-data-2023-09-01 07:54:56.218802.csv.dvc"]
	node8["data/steps/train/accelerometer-data-2023-09-01 07:56:40.404358.csv.dvc"]
	node9["data/steps/train/accelerometer-data-2023-09-01 07:56:53.696869.csv.dvc"]
	node10["data/steps/train/accelerometer-data-2023-09-01 08:01:10.489035.csv.dvc"]
	node11["predict_test"]
	node12["train"]
	node13["update-metrics"]
	node1-->node11
	node2-->node11
	node3-->node12
	node4-->node12
	node5-->node12
	node6-->node12
	node7-->node12
	node8-->node12
	node9-->node12
	node10-->node12
	node11-->node13
	node12-->node11
	node12-->node13
```
_graph_end_
