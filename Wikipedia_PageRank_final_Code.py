"""
Wikipedia Page Ranking using PageRank
Complete code consolidated from Wikipedia_PageRank_Analysis.ipynb.

Before running: upload wikipedia_pagerank_prepared_data.zip to /content/
(or update zip_path below).
"""

# Notebook cell 1
import os
import json
import pickle
import zipfile
from pathlib import Path

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

print("Libraries imported successfully!")

# Notebook cell 2
import os
import zipfile
from pathlib import Path

zip_path = "/content/wikipedia_pagerank_prepared_data.zip"
output_dir = Path("/content/wikipedia_pagerank_prepared_data")

if not output_dir.exists() or not any(output_dir.iterdir()):
    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(output_dir)

print("Prepared dataset extracted!")

for file in output_dir.rglob("*"):
    if file.is_file():
        print(file.name, round(file.stat().st_size / (1024**2), 2), "MB")

# Notebook cell 3
graph_path = output_dir / "wikipedia_graph_100k.pkl"
metadata_path = output_dir / "wikipedia_pages_metadata.csv"
stats_path = output_dir / "graph_statistics.json"

with open(graph_path, "rb") as f:
    G_100k = pickle.load(f)

pages = pd.read_csv(metadata_path)

with open(stats_path, "r") as f:
    stats = json.load(f)

print("Graph and metadata loaded successfully!")
print("Metadata shape:", pages.shape)
print("Graph nodes:", G_100k.number_of_nodes())
print("Graph edges:", G_100k.number_of_edges())

# Notebook cell 4
print(json.dumps(stats, indent=4))

# Notebook cell 5
print("Directed graph:", G_100k.is_directed())
print("Nodes:", G_100k.number_of_nodes())
print("Edges:", G_100k.number_of_edges())
print("Self-loops:", nx.number_of_selfloops(G_100k))

zero_outdegree = sum(
    1 for _, d in G_100k.out_degree() if d == 0
)

zero_indegree = sum(
    1 for _, d in G_100k.in_degree() if d == 0
)

print("Pages with no outgoing links:", zero_outdegree)
print("Pages with no incoming links:", zero_indegree)

assert G_100k.number_of_nodes() == 100000
assert G_100k.is_directed()

# Notebook cell 6
in_degrees = dict(G_100k.in_degree())
out_degrees = dict(G_100k.out_degree())

graph_summary = {
    "nodes": G_100k.number_of_nodes(),
    "edges": G_100k.number_of_edges(),
    "self_loops": nx.number_of_selfloops(G_100k),
    "average_in_degree": sum(in_degrees.values()) / len(in_degrees),
    "average_out_degree": sum(out_degrees.values()) / len(out_degrees),
    "maximum_in_degree": max(in_degrees.values()),
    "maximum_out_degree": max(out_degrees.values()),
    "zero_in_degree_pages": zero_indegree,
    "zero_out_degree_pages": zero_outdegree
}

print(pd.Series(graph_summary).to_string())

# Notebook cell 7
top_in_degree = sorted(
    in_degrees.items(),
    key=lambda item: item[1],
    reverse=True
)[:10]

top_pages = pd.DataFrame(
    top_in_degree,
    columns=["title", "in_degree"]
)

display(top_pages)

# Notebook cell 8
plt.figure(figsize=(12, 6))

plt.barh(
    top_pages["title"][::-1],
    top_pages["in_degree"][::-1]
)

plt.xlabel("Number of Incoming Links")
plt.ylabel("Wikipedia Page")
plt.title("Top 10 Wikipedia Pages by In-Degree")
plt.tight_layout()
plt.show()

# Notebook cell 9
results_dir = Path("/content/graph_analysis_results")
results_dir.mkdir(exist_ok=True)

# Save top 10 pages by in-degree
top_pages.to_csv(
    results_dir / "top_10_in_degree.csv",
    index=False
)

# Save graph statistics
with open(results_dir / "graph_summary.json", "w") as f:
    json.dump(graph_summary, f, indent=4)

print("Graph analysis outputs saved successfully!")

print("Files saved:")
for file in results_dir.iterdir():
    print(file.name)

# Notebook cell 11
# This version retains the dataset's self-loops.
import time

damping_factor = 0.85
tolerance = 1e-6
max_iterations = 100

start_time = time.time()

pagerank_scores = nx.pagerank(
    G_100k,
    alpha=damping_factor,
    max_iter=max_iterations,
    tol=tolerance
)

elapsed_time = time.time() - start_time

print("PageRank calculated successfully!")
print("Pages ranked:", len(pagerank_scores))
print("Damping factor:", damping_factor)
print("Convergence tolerance:", tolerance)
print("Execution time (seconds):", round(elapsed_time, 2))
print("Sum of PageRank scores:", sum(pagerank_scores.values()))

# Notebook cell 12
# Check that the existing results are available
if "pagerank_scores" not in globals():
    raise NameError(
        "pagerank_scores is missing. Run Cell 10 first."
    )

if "pages" not in globals():
    raise NameError(
        "pages is missing. Run Cell 3 first."
    )

print("Number of PageRank scores:", len(pagerank_scores))
print("\nMetadata columns:", pages.columns.tolist())

# Sort all pages by PageRank score
pagerank_ranking = pd.DataFrame(
    pagerank_scores.items(),
    columns=["title", "pagerank_score"]
)

pagerank_ranking = pagerank_ranking.sort_values(
    by="pagerank_score",
    ascending=False
).reset_index(drop=True)

# Add rank numbers starting from 1
pagerank_ranking.insert(
    0, "pagerank_rank",
    range(1, len(pagerank_ranking) + 1)
)

print("\nTotal ranked pages:", len(pagerank_ranking))
print("\nTop 20 Wikipedia pages by PageRank:")

display(pagerank_ranking.head(20))

# Notebook cell 13
results_dir = Path("/content/graph_analysis_results")
results_dir.mkdir(parents=True, exist_ok=True)

# Save the full ranking of 100,000 pages
pagerank_ranking.to_csv(
    results_dir / "pagerank_ranking_100k.csv",
    index=False
)

# Save the top 10 pages separately
top_10_pagerank = pagerank_ranking.head(10).copy()

top_10_pagerank.to_csv(
    results_dir / "top_10_pagerank.csv",
    index=False
)

print("PageRank results saved successfully!")

print("\nSaved files:")
print(results_dir / "pagerank_ranking_100k.csv")
print(results_dir / "top_10_pagerank.csv")

print("\nTop 10 PageRank pages:")
display(top_10_pagerank)

# Notebook cell 14
import textwrap

# Check that Cell 12 has run
if "top_10_pagerank" not in globals():
    raise NameError("Run Cells 11 and 12 first.")

plot_data = top_10_pagerank.sort_values(
    "pagerank_score", ascending=True
).copy()

# Wrap long titles for readability
labels = [
    "\n".join(textwrap.wrap(str(title), width=28))
    for title in plot_data["title"]
]

fig, ax = plt.subplots(figsize=(12, 7))

ax.barh(labels, plot_data["pagerank_score"])

ax.set_xlabel("PageRank Score")
ax.set_ylabel("Wikipedia Page")
ax.set_title("Top 10 Wikipedia Pages by PageRank")
ax.grid(axis="x", linestyle="--", alpha=0.4)

plt.tight_layout()

# Save the figure
plt.savefig(
    results_dir / "top_10_pagerank.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Chart saved to:", results_dir / "top_10_pagerank.png")

# Notebook cell 15
# Compare PageRank and in-degree rankings

from scipy.stats import spearmanr

# Build a complete in-degree ranking from existing values
in_degree_ranking = pd.DataFrame(
    in_degrees.items(),
    columns=["title", "in_degree"]
)

in_degree_ranking["in_degree_rank"] = (
    in_degree_ranking["in_degree"]
    .rank(method="average", ascending=False)
)

# Merge the two rankings using their graph node identifiers
comparison = in_degree_ranking.merge(
    pagerank_ranking[["title", "pagerank_rank", "pagerank_score"]],
    on="title",
    how="inner",
    validate="one_to_one"
)

print("Pages matched across both rankings:", len(comparison))

# Top 10 from each method
top10_indegree_titles = set(
    in_degree_ranking.nsmallest(10, "in_degree_rank")["title"]
)

top10_pagerank_titles = set(
    pagerank_ranking.head(10)["title"]
)

overlap = top10_indegree_titles & top10_pagerank_titles

print("\nTop 10 overlap:", len(overlap), "pages")
print("Pages in both top 10 lists:", sorted(overlap))

print("\nPages in the top 10 of either method:")
top10_union = comparison[
    comparison["title"].isin(
        top10_indegree_titles | top10_pagerank_titles
    )
].sort_values("pagerank_rank")

display(top10_union[
    ["title", "in_degree", "in_degree_rank",
     "pagerank_score", "pagerank_rank"]
])

# Spearman correlation between in-degree and PageRank scores
correlation, p_value = spearmanr(
    comparison["in_degree"],
    comparison["pagerank_score"]
)

print("\nSpearman rank correlation:", round(correlation, 4))
print("P-value:", p_value)

# Save comparison outputs
comparison.to_csv(
    results_dir / "indegree_pagerank_comparison_100k.csv",
    index=False
)

top10_union.to_csv(
    results_dir / "top10_ranking_comparison.csv",
    index=False
)

print("\nComparison CSV files saved.")

# Notebook cell 16
#Analyze and visualize PageRank score distribution

import numpy as np

scores = np.array(list(pagerank_scores.values()))

print("PageRank score distribution")
print("----------------------------")
print("Minimum:", f"{scores.min():.10f}")
print("Median:", f"{np.median(scores):.10f}")
print("Mean:", f"{scores.mean():.10f}")
print("Maximum:", f"{scores.max():.10f}")
print("Standard deviation:", f"{scores.std():.10f}")

# Improved PageRank score distribution plot

scores = np.array(list(pagerank_scores.values()))

# Create bins evenly spaced in log space
log_bins = np.logspace(
    np.log10(scores.min()),
    np.log10(scores.max()),
    60
)

fig, ax = plt.subplots(figsize=(12, 6))

ax.hist(scores, bins=log_bins)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("PageRank Score (log scale)")
ax.set_ylabel("Number of Pages (log scale)")
ax.set_title("Distribution of PageRank Scores Across 100,000 Pages")
ax.grid(axis="both", linestyle="--", alpha=0.3)

plt.tight_layout()

plt.savefig(
    results_dir / "pagerank_score_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# Notebook cell 17
#Compare in-degree and PageRank visually

import numpy as np

# Ensure both dictionaries use the same graph nodes
nodes = list(pagerank_scores.keys())

degree_values = np.array([
    in_degrees[node] for node in nodes
])

pr_values = np.array([
    pagerank_scores[node] for node in nodes
])

fig, ax = plt.subplots(figsize=(10, 7))

ax.scatter(
    degree_values,
    pr_values,
    alpha=0.35,
    s=10,
    edgecolors="none"
)

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel("In-Degree (log scale)")
ax.set_ylabel("PageRank Score (log scale)")
ax.set_title("In-Degree vs. PageRank Across 100,000 Wikipedia Pages")

ax.grid(True, which="both", linestyle="--", alpha=0.3)

plt.tight_layout()

plt.savefig(
    results_dir / "indegree_vs_pagerank_scatter.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    "Scatter plot saved to:",
    results_dir / "indegree_vs_pagerank_scatter.png"
)

# Notebook cell 18
# Consolidate final results and verify saved files

import json
from pathlib import Path

# 1. Check required variables
required_variables = [
    "G_100k",
    "pagerank_scores",
    "pagerank_ranking",
    "top_10_pagerank",
    "comparison",
    "overlap",
    "elapsed_time",
    "damping_factor",
    "tolerance",
    "max_iterations",
    "graph_summary",
    "results_dir"
]

missing = [
    name for name in required_variables
    if name not in globals()
]

if missing:
    raise NameError(
        f"Missing variables: {missing}. "
        "Run the corresponding earlier cells first."
    )

results_dir = Path(results_dir)
results_dir.mkdir(parents=True, exist_ok=True)

# 2. Calculate summary statistics from existing results
scores = list(pagerank_scores.values())

from scipy.stats import spearmanr

correlation, p_value = spearmanr(
    comparison["in_degree"],
    comparison["pagerank_score"]
)

# 3. Create a reproducibility summary
experiment_summary = {
    "project": "Wikipedia PageRank Analysis",
    "dataset_pages": G_100k.number_of_nodes(),
    "directed_edges": G_100k.number_of_edges(),
    "self_loops": int(nx.number_of_selfloops(G_100k)),
    "pagerank_parameters": {
        "damping_factor": damping_factor,
        "tolerance": tolerance,
        "maximum_iterations": max_iterations,
        "execution_time_seconds": round(elapsed_time, 2),
        "number_of_pages_ranked": len(pagerank_scores),
        "sum_of_pagerank_scores": float(sum(scores))
    },
    "pagerank_distribution": {
        "minimum": float(min(scores)),
        "median": float(pd.Series(scores).median()),
        "mean": float(pd.Series(scores).mean()),
        "maximum": float(max(scores)),
        "standard_deviation": float(pd.Series(scores).std(ddof=0))
    },
    "ranking_comparison": {
        "pages_matched": len(comparison),
        "top_10_overlap_count": len(overlap),
        "top_10_overlap_pages": sorted(list(overlap)),
        "spearman_correlation": float(correlation),
        "spearman_p_value": float(p_value)
    }
}

summary_path = results_dir / "final_experiment_summary.json"

with open(summary_path, "w") as f:
    json.dump(experiment_summary, f, indent=4)

# 4. Verify expected result files
expected_files = [
    "top_10_in_degree.csv",
    "graph_summary.json",
    "pagerank_ranking_100k.csv",
    "top_10_pagerank.csv",
    "indegree_pagerank_comparison_100k.csv",
    "top10_ranking_comparison.csv",
    "top_10_pagerank.png",
    "pagerank_score_distribution.png",
    "indegree_vs_pagerank_scatter.png",
    "final_experiment_summary.json"
]

print("FINAL RESULTS VERIFICATION")
print("=" * 45)

for filename in expected_files:
    path = results_dir / filename
    status = "FOUND" if path.is_file() else "MISSING"
    print(f"{status:8} | {filename}")

print("\nExperiment summary:")
print(json.dumps(experiment_summary, indent=2))

print("\nResults folder:", results_dir)

# Notebook cell 19
# Final backup: package all analysis results into one ZIP

import shutil
from pathlib import Path

results_dir = Path("/content/graph_analysis_results")
backup_path = shutil.make_archive(
    "/content/wikipedia_pagerank_final_results",
    "zip",
    root_dir=results_dir
)

print("Backup created:", backup_path)
