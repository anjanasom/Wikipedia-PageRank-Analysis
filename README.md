# Wikipedia Page Ranking Using PageRank

## Project Overview

This project analyzes the structural importance of Wikipedia pages using the PageRank algorithm on a directed hyperlink graph.

It compares PageRank scores with in-degree rankings to understand how incoming links and the importance of linking pages influence page rankings.

## Dataset

- **Dataset:** Wikipedia Link Graph Dataset – 100K Pages
- **Source:** [Kaggle – Wikipedia Link Graph Dataset](https://www.kaggle.com/datasets/kutayahin/wikipedia-link-graph-100k)
- **Pages:** 100,000
- **Directed edges:** 8,577,044

The graph contains links between pages in the selected 100,000-page set.

## Technologies Used

- Python
- Pandas
- NumPy
- NetworkX
- SciPy
- Matplotlib
- Google Colab
- Streamlit
- Plotly

## Methodology

1. Load the prepared graph, metadata, and graph statistics.
2. Validate the directed graph and calculate degree statistics.
3. Identify the top 10 pages by in-degree.
4. Calculate PageRank scores using NetworkX.
5. Rank pages by PageRank score.
6. Compare in-degree and PageRank rankings.
7. Calculate top-10 overlap and Spearman correlation.
8. Visualize the rankings and score distributions.
9. Explore the saved analysis results through an interactive Streamlit dashboard.

## PageRank Parameters

- **Damping factor:** 0.85
- **Convergence tolerance:** 0.000001
- **Maximum iterations:** 100

## Results

- **Pages ranked:** 100,000
- **Directed edges:** 8,577,044
- **Top-10 overlap:** 9 pages
- **Spearman correlation:** Approximately 0.9077

These results indicate a strong relationship between incoming-link counts and PageRank scores in the selected graph.

## Interactive Dashboard

The Streamlit dashboard presents the existing analysis results through an interactive interface.

- **Overview:** Graph statistics, key metrics, top 10 PageRank pages, and PageRank score distribution.
- **Explore Rankings:** Search Wikipedia page titles, inspect ranks and scores, and download filtered ranking results.
- **Compare Algorithms:** Compare PageRank with in-degree rankings using interactive visualizations and top-page tables.
- **About the Project:** Review the methodology and saved PageRank configuration.

The dashboard reads saved output files; it does not rebuild the graph or rerun PageRank.

### Run the Dashboard

1. Clone this repository.
2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Ensure `app.py` and a `results/` folder are in the same project directory. Copy the following files from your existing analysis outputs into `results/`:

   - `graph_summary.json`
   - `final_experiment_summary.json`
   - `pagerank_ranking_100k.csv`
   - `top_10_pagerank.csv`
   - `top_10_in_degree.csv`
   - `indegree_pagerank_comparison_100k.csv`

4. Start the dashboard:

   ```bash
   streamlit run app.py
   ```

5. Open the local URL shown in the terminal, usually `http://localhost:8501`.

## Repository Structure

- `Wikipedia_PageRank_Analysis.ipynb` — Complete analysis notebook.
- `Wikipedia_PageRank_final_Code.py` — Consolidated Python implementation.
- `app.py` — Streamlit dashboard application.
- `requirements.txt` — Required Python libraries.
- `data/` — Dataset documentation and setup information, if included.
- `results/` — Ranking CSVs, graph statistics, summaries, and visualizations.

## Run the Original Analysis

1. Clone this repository.
2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset from the linked Kaggle source and prepare the required graph files.
4. Upload or place the prepared files at the paths expected by the notebook.
5. Open `Wikipedia_PageRank_Analysis.ipynb` in Google Colab or Jupyter and run the cells in order.

**Note:** The prepared graph and metadata are not included in the repository. The notebook requires these files to be available before execution.

## Limitations

- The analysis is restricted to the selected 100,000 pages.
- PageRank measures structural importance in the hyperlink graph, not article quality or factual accuracy.
- Execution time depends on the computing environment.