import subprocess  # nosec

if __name__ == "__main__":
    with open("README.md") as f:  # nosec
        readme_lines = f.readlines()

    metrics_start = readme_lines.index("# Metrics\n")
    metrics_end = readme_lines.index("# Steps Graph\n")

    graph_start = readme_lines.index("# Steps Graph\n")
    graph_end = readme_lines.index("_graph_end_\n")
    readme_before_metrics = readme_lines[:metrics_start]

    metrics_tables_md = []
    commands = [
        [
            "dvc",
            "metrics",
            "show",
            "--precision",
            "2",
            "--md",
        ],
    ]
    graph_md = [
        subprocess.run(
            ["dvc", "dag", "--md"], stdout=subprocess.PIPE  # nosec
        ).stdout.decode("utf-8")
    ]
    for cmd in commands:
        print(" ".join(cmd))
        metrics_tables_md.append(
            subprocess.run(cmd, stdout=subprocess.PIPE).stdout.decode("utf-8")  # nosec
        )

    readme_after_graph = readme_lines[graph_end:]

    graph_header = [readme_lines[graph_start]]
    metrics_header = [readme_lines[metrics_start]]
    new_readme = (
        readme_before_metrics
        + metrics_header
        + metrics_tables_md
        + graph_header
        + graph_md
        + readme_after_graph
    )

    with open("README.md", "w") as f:
        f.writelines(new_readme)
