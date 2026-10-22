import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Wikipedia PageRank Explorer",
    page_icon="🔎",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"


@st.cache_data
def load_data():
    required = {
        "summary": RESULTS_DIR / "final_experiment_summary.json",
        "graph": RESULTS_DIR / "graph_summary.json",
        "pagerank": RESULTS_DIR / "pagerank_ranking_100k.csv",
        "top_indegree": RESULTS_DIR / "top_10_in_degree.csv",
        "top_pagerank": RESULTS_DIR / "top_10_pagerank.csv",
        "comparison": RESULTS_DIR / "indegree_pagerank_comparison_100k.csv",
    }
    missing = [str(path.relative_to(BASE_DIR)) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "These result files are missing: " + ", ".join(missing)
            + ". Make sure the results/ folder is beside app.py."
        )

    with open(required["summary"], "r", encoding="utf-8") as f:
        summary = json.load(f)
    with open(required["graph"], "r", encoding="utf-8") as f:
        graph_summary = json.load(f)

    pagerank = pd.read_csv(required["pagerank"])
    top_indegree = pd.read_csv(required["top_indegree"])
    top_pagerank = pd.read_csv(required["top_pagerank"])
    comparison = pd.read_csv(required["comparison"])
    return summary, graph_summary, pagerank, top_indegree, top_pagerank, comparison


st.title("🔎 Wikipedia PageRank Explorer")
st.caption(
    "Explore structural importance in a directed graph of 100,000 Wikipedia pages. "
    "This dashboard presents the results produced by the project notebook."
)

try:
    summary, graph_summary, pagerank, top_indegree, top_pagerank, comparison = load_data()
except Exception as exc:
    st.error(str(exc))
    st.stop()

# Overview
c1, c2, c3, c4 = st.columns(4)
c1.metric("Pages analysed", f'{summary.get("dataset_pages", summary.get("nodes", len(pagerank))):,}')
c2.metric("Directed edges", f'{summary.get("directed_edges", summary.get("edges", 0)):,}')
c3.metric("Top-10 overlap", f'{summary.get("top_10_overlap", 0)}/10')
c4.metric("Spearman correlation", f'{summary.get("spearman_correlation", 0):.4f}')

st.divider()

tab_overview, tab_search, tab_compare, tab_method = st.tabs(
    ["Overview & Top Pages", "Find a Page", "Ranking Comparison", "Methodology"]
)

with tab_overview:
    left, right = st.columns(2)
    with left:
        st.subheader("Top 10 by PageRank")
        chart_df = top_pagerank.sort_values("pagerank_score", ascending=True).set_index("title")
        st.bar_chart(chart_df["pagerank_score"], horizontal=True)
        st.dataframe(
            top_pagerank[["pagerank_rank", "title", "pagerank_score"]],
            use_container_width=True,
            hide_index=True,
        )
    with right:
        st.subheader("Top 10 by incoming links (in-degree)")
        degree_chart = top_indegree.sort_values("in_degree", ascending=True).set_index("title")
        st.bar_chart(degree_chart["in_degree"], horizontal=True)
        st.dataframe(top_indegree, use_container_width=True, hide_index=True)

    st.subheader("PageRank score distribution")
    st.write(
        "PageRank scores are highly skewed: a small number of pages receive much larger "
        "scores than most pages."
    )
    st.image(
        str(RESULTS_DIR / "pagerank_score_distribution.png"),
        caption="Distribution of PageRank scores across the selected graph",
        use_container_width=True,
    )

with tab_search:
    st.subheader("Look up a Wikipedia page")
    query = st.text_input(
        "Enter a page title or part of a title",
        placeholder="e.g. United States, ISBN, World War II",
    )
    if query.strip():
        matches = pagerank[
            pagerank["title"].astype(str).str.contains(query.strip(), case=False, na=False, regex=False)
        ].head(100)
        if matches.empty:
            st.info("No matching page title was found in this dataset.")
        else:
            st.caption(f"Showing up to 100 matches ({len(matches)} shown).")
            st.dataframe(matches, use_container_width=True, hide_index=True)
            exact = pagerank[pagerank["title"].astype(str).str.casefold() == query.strip().casefold()]
            if not exact.empty:
                rank_row = exact.iloc[0]
                st.success(
                    f'Exact match: **{rank_row["title"]}** — '
                    f'PageRank rank **#{int(rank_row["pagerank_rank"]):,}**, '
                    f'score **{rank_row["pagerank_score"]:.8g}**.'
                )
    else:
        st.write("Search the full ranking of 100,000 pages by title.")

with tab_compare:
    st.subheader("In-degree versus PageRank")
    st.write(
        "In-degree counts incoming links. PageRank also considers the scores of the pages "
        "that link to a page, so the two rankings are related but not identical."
    )
    c1, c2 = st.columns(2)
    c1.metric("Pages matched", f'{len(comparison):,}')
    c2.metric("Top-10 pages shared", f'{summary.get("top_10_overlap", 0)}')

    st.image(
        str(RESULTS_DIR / "indegree_vs_pagerank_scatter.png"),
        caption="Relationship between in-degree and PageRank (log-scaled axes)",
        use_container_width=True,
    )

    st.subheader("Compare individual rankings")
    sort_by = st.selectbox(
        "Sort comparison table by",
        ["pagerank_rank", "in_degree_rank", "in_degree", "pagerank_score"],
        index=0,
    )
    st.dataframe(
        comparison.sort_values(sort_by).head(100)[
            ["title", "in_degree", "in_degree_rank", "pagerank_score", "pagerank_rank"]
        ],
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Download full comparison CSV",
        data=comparison.to_csv(index=False).encode("utf-8"),
        file_name="indegree_pagerank_comparison_100k.csv",
        mime="text/csv",
    )

with tab_method:
    st.subheader("How this analysis was produced")
    st.markdown(
        """
        1. Load the prepared directed Wikipedia graph containing the selected 100,000 pages.
        2. Calculate in-degree and out-degree statistics.
        3. Calculate PageRank using NetworkX.
        4. Rank all pages by PageRank score.
        5. Compare the in-degree and PageRank rankings.
        6. Visualize the top pages, score distribution, and ranking relationship.
        """
    )
    a, b, c = st.columns(3)
    a.metric("Damping factor", f'{summary.get("damping_factor", 0.85):.2f}')
    b.metric("Tolerance", f'{summary.get("tolerance", 1e-6):g}')
    c.metric("Maximum iterations", f'{summary.get("max_iterations", 100)}')

    st.subheader("Graph statistics")
    st.dataframe(
        pd.DataFrame(
            [{"Metric": key.replace("_", " ").title(), "Value": value}
             for key, value in graph_summary.items()]
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Interpretation: PageRank represents structural importance in this filtered graph; "
        "it does not directly measure article quality or factual accuracy."
    )

st.divider()
st.caption("Wikipedia PageRank Analysis • Built with Streamlit • Results from the project notebook")
