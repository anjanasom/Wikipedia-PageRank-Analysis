
from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------
st.set_page_config(
    page_title="Wikipedia PageRank Explorer",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

# Update these two values before submitting the project.
PROJECT_NAME = "Web Analytics: Wikipedia PageRank Explorer"
TEAM_NAME = "UnSupervised"

# ------------------------------------------------------------
# VISUAL THEME
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .block-container { max-width: 1500px; padding-top: 1.5rem; padding-bottom: 2.5rem; }
    h1, h2, h3 { font-family: 'Manrope', sans-serif; letter-spacing: -0.025em; }
    .hero {
        padding: 2rem 2.2rem; border-radius: 20px; margin: .2rem 0 1.35rem 0;
        color: white; background: linear-gradient(115deg, #10233f 0%, #164e63 58%, #16817a 100%);
        box-shadow: 0 14px 34px rgba(16,35,63,.14);
    }
    .eyebrow { text-transform: uppercase; letter-spacing: .16em; font-size: .72rem;
        font-weight: 700; color: #a7e8df; margin-bottom: .7rem; }
    .hero h1 { color: #fff; font-size: clamp(2rem, 3.5vw, 3rem); line-height: 1.12; margin: 0 0 .75rem 0; }
    .hero p { color: #e0f2f3; font-size: 1.02rem; max-width: 850px; margin: 0; line-height: 1.65; }
    .hero-pills { display: flex; gap: .55rem; flex-wrap: wrap; margin-top: 1.25rem; }
    .hero-pill { border: 1px solid rgba(255,255,255,.25); background: rgba(255,255,255,.09);
        border-radius: 999px; padding: .38rem .75rem; font-size: .78rem; color: #f2ffff; }
    div[data-testid="stMetric"] { border: 1px solid rgba(100,116,139,.20); border-radius: 15px;
        padding: 1rem 1.05rem; background: var(--secondary-background-color); }
    div[data-testid="stMetricLabel"] { font-weight: 600; }
    div[data-testid="stMetricValue"] { font-family: 'Manrope', sans-serif; letter-spacing: -.035em; }
    .section-kicker { color: #0d9488; text-transform: uppercase; font-weight: 700;
        font-size: .72rem; letter-spacing: .14em; margin-bottom: .25rem; }
    .insight { border: 1px solid rgba(13,148,136,.23); border-left: 4px solid #0d9488;
        border-radius: 12px; padding: 1rem 1.1rem; background: rgba(13,148,136,.055); }
    [data-testid="stSidebar"] { border-right: 1px solid rgba(100,116,139,.16); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# LOAD SAVED RESULTS
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def read_csv(path):
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def read_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def csv_result(filename):
    path = RESULTS / filename
    if not path.exists():
        return None
    try:
        return read_csv(str(path))
    except Exception as exc:
        st.warning(f"Could not load results/{filename}: {exc}")
        return None


def json_result(filename):
    path = RESULTS / filename
    if not path.exists():
        return {}
    try:
        return read_json(str(path))
    except Exception as exc:
        st.warning(f"Could not load results/{filename}: {exc}")
        return {}


def fmt_int(value):
    try:
        return f"{int(float(value)):,}"
    except (TypeError, ValueError, OverflowError):
        return "—"


def fmt_num(value, digits=5):
    try:
        return f"{float(value):.{digits}g}"
    except (TypeError, ValueError, OverflowError):
        return "—"


def show_missing(filename):
    st.error(
        f"Missing results/{filename}. "
        "Make sure the results folder is beside app.py."
    )


graph = json_result("graph_summary.json")
experiment = json_result("final_experiment_summary.json")
rankings = csv_result("pagerank_ranking_100k.csv")
top_pr = csv_result("top_10_pagerank.csv")
top_in = csv_result("top_10_in_degree.csv")
comparison = csv_result("indegree_pagerank_comparison_100k.csv")
top_comparison = csv_result("top10_ranking_comparison.csv")

params = {
    "damping_factor": experiment.get("damping_factor", 0.85),
    "tolerance": experiment.get("tolerance", 1e-6),
    "max_iterations": experiment.get("max_iterations", 100),
}
overlap = experiment.get("top_10_overlap")
rho = experiment.get("spearman_correlation")
nodes = graph.get("nodes", experiment.get("nodes", 100000))
edges = graph.get("edges", experiment.get("edges", 8577044))

# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
with st.sidebar:
    st.markdown("# PageRank Explorer")
    st.caption("WIKIPEDIA LINK GRAPH")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Explore Rankings",
            "Compare Algorithms",
            "About the Project",
        ],
    )

    st.divider()
    st.markdown("**PageRank settings**")
    st.write(f"Damping factor: `{params['damping_factor']}`")
    st.write(f"Tolerance: `{params['tolerance']}`")
    st.write(f"Maximum iterations: `{params['max_iterations']}`")


# ============================================================
# 1. OVERVIEW
# ============================================================
if page == "Overview":
    st.markdown(
        f"""
        <section class="hero">
          <div class="eyebrow">Web Analytics · Graph Ranking</div>
          <h1>Discover what makes a Wikipedia page important.</h1>
          <p>Explore a directed network of Wikipedia pages and compare PageRank,
          a link-based importance measure, with simple incoming-link counts.</p>
          <div class="hero-pills">
            <span class="hero-pill">{fmt_int(nodes)} pages</span>
            <span class="hero-pill">{fmt_int(edges)} directed links</span>
            <span class="hero-pill">PageRank · damping factor {params['damping_factor']}</span>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-kicker">At a glance</div>',
        unsafe_allow_html=True,
    )
    st.subheader("The graph in numbers")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pages analyzed", fmt_int(nodes))
    c2.metric("Directed edges", fmt_int(edges))
    c3.metric(
        "Maximum in-degree",
        fmt_int(graph.get("maximum_in_degree")),
    )
    c4.metric(
        "Top-10 overlap",
        f"{fmt_int(overlap)}/10" if overlap is not None else "—",
    )

    c5, c6, c7, c8 = st.columns(4)
    c5.metric(
        "Average in-degree",
        fmt_num(graph.get("average_in_degree"), 6),
    )
    c6.metric("Self-loops", fmt_int(graph.get("self_loops")))
    c7.metric(
        "Zero out-degree pages",
        fmt_int(graph.get("zero_out_degree_pages")),
    )
    c8.metric("Spearman correlation", fmt_num(rho, 4))

    st.write("")
    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        st.markdown(
            '<div class="section-kicker">Rankings</div>',
            unsafe_allow_html=True,
        )
        st.subheader("Top 10 pages by PageRank")

        if top_pr is not None:
            display = top_pr.copy()
            display = display.rename(
                columns={
                    "pagerank_rank": "Rank",
                    "title": "Wikipedia page",
                    "pagerank_score": "PageRank score",
                }
            )

            st.dataframe(
                display,
                use_container_width=True,
                hide_index=True,
                height=390,
            )

            st.download_button(
                "Download top-10 PageRank CSV",
                data=top_pr.to_csv(index=False).encode("utf-8"),
                file_name="top_10_pagerank.csv",
                mime="text/csv",
            )
        else:
            show_missing("top_10_pagerank.csv")

    with right:
        st.markdown(
            '<div class="section-kicker">Score distribution</div>',
            unsafe_allow_html=True,
        )
        st.subheader("PageRank scores across pages")

        if rankings is not None and "pagerank_score" in rankings.columns:
            scores = pd.to_numeric(
                rankings["pagerank_score"],
                errors="coerce",
            ).dropna()
            scores = scores[scores > 0]

            if not scores.empty:
                scale = st.radio(
                    "Distribution scale",
                    options=["Normal scale", "Log scale"],
                    horizontal=True,
                    key="pagerank_distribution_scale",
                )

                if scale == "Log scale":
                    plot_values = np.log10(scores)
                    x_label = "log10(PageRank score)"
                else:
                    plot_values = scores
                    x_label = "PageRank score"

                fig = px.histogram(
                    x=plot_values,
                    nbins=40,
                    labels={
                        "x": x_label,
                        "count": "Number of pages",
                    },
                    color_discrete_sequence=["#0d9488"],
                )

                fig.update_layout(
                    margin=dict(l=8, r=8, t=18, b=8),
                    showlegend=False,
                    bargap=0.08,
                    xaxis_title=x_label,
                    yaxis_title="Number of pages",
                )
                fig.update_xaxes(showgrid=False)
                fig.update_yaxes(
                    gridcolor="rgba(148,163,184,.18)"
                )

                st.plotly_chart(fig, use_container_width=True)

                if scale == "Log scale":
                    st.caption(
                        "Log scale uses log10(PageRank score), making differences "
                        "between very small scores easier to inspect."
                    )
                else:
                    st.caption(
                        "Normal scale shows the original PageRank scores. Because "
                        "the distribution is highly skewed, many pages may appear "
                        "concentrated near zero."
                    )
            else:
                st.warning(
                    "No positive PageRank scores are available to plot."
                )
        else:
            show_missing("pagerank_ranking_100k.csv")

        st.markdown(
            f"""
            <div class="insight">
              <strong>Key finding</strong><br>
              The Spearman correlation is <strong>{fmt_num(rho, 4)}</strong>,
              indicating a strong positive association between in-degree and
              PageRank rankings. The methods can still rank individual pages differently.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown(
        '<div class="section-kicker">How to read the results</div>',
        unsafe_allow_html=True,
    )
    st.subheader("Two ways to measure page importance")

    expl1, expl2 = st.columns(2, gap="large")

    with expl1:
        st.markdown("#### In-degree")
        st.write(
            "Counts how many links point to a page. It identifies pages "
            "that receive many incoming links."
        )

    with expl2:
        st.markdown("#### PageRank")
        st.write(
            "Considers incoming links and the importance of the pages "
            "providing them. Links from important pages can contribute more."
        )

    st.info(
        "These results apply to the filtered 100,000-page graph. PageRank "
        "measures structural importance, not article quality, reliability, "
        "or factual accuracy."
    )


# ============================================================
# 2. EXPLORE RANKINGS
# ============================================================
elif page == "Explore Rankings":
    st.title("Explore PageRank Rankings")
    st.caption("Search, filter, inspect, and export the page rankings.")

    if rankings is None or not {
        "title",
        "pagerank_score",
    }.issubset(rankings.columns):
        show_missing("pagerank_ranking_100k.csv")
        st.stop()

    data = rankings.copy()
    data["title"] = data["title"].astype(str)
    data["pagerank_score"] = pd.to_numeric(
        data["pagerank_score"],
        errors="coerce",
    )

    # Bring in-degree details into the exploration table when available.
    if (
        comparison is not None
        and {"title", "in_degree", "in_degree_rank"}.issubset(comparison.columns)
    ):
        indegree_details = comparison[
            ["title", "in_degree", "in_degree_rank"]
        ].copy()
        indegree_details["title"] = indegree_details["title"].astype(str)
        data = data.merge(
            indegree_details.drop_duplicates("title"),
            on="title",
            how="left",
        )

    rank_col = (
        "pagerank_rank"
        if "pagerank_rank" in data.columns
        else None
    )

    if rank_col:
        data[rank_col] = pd.to_numeric(
            data[rank_col],
            errors="coerce",
        )

    query = st.text_input(
        "Search Wikipedia page title",
        placeholder="e.g. India, Mathematics, World War II",
    )

    st.markdown("**PageRank score range**")

    score_min = float(data["pagerank_score"].min())
    score_max = float(data["pagerank_score"].max())
    score_step = max((score_max - score_min) / 10000, 1e-12)

    s1, s2 = st.columns(2)

    selected_score_min = s1.number_input(
        "Minimum PageRank score",
        min_value=score_min,
        max_value=score_max,
        value=score_min,
        step=score_step,
        format="%.10f",
        help="Enter the exact lower score boundary.",
    )

    selected_score_max = s2.number_input(
        "Maximum PageRank score",
        min_value=score_min,
        max_value=score_max,
        value=score_max,
        step=score_step,
        format="%.10f",
        help="Enter the exact upper score boundary.",
    )

    selected_rank_min, selected_rank_max = 1, len(data)

    if rank_col:
        rank_min = int(data[rank_col].min())
        rank_max = int(data[rank_col].max())

        st.markdown("**Rank range**")
        r1, r2 = st.columns(2)

        selected_rank_min = r1.number_input(
            "Minimum rank",
            min_value=rank_min,
            max_value=rank_max,
            value=rank_min,
            step=1,
            help="Rank 1 is the highest PageRank score.",
        )

        selected_rank_max = r2.number_input(
            "Maximum rank",
            min_value=rank_min,
            max_value=rank_max,
            value=min(1000, rank_max),
            step=1,
        )

    if selected_score_min > selected_score_max:
        st.error(
            "Minimum score must be less than or equal to maximum score."
        )
        st.stop()

    if selected_rank_min > selected_rank_max:
        st.error(
            "Minimum rank must be less than or equal to maximum rank."
        )
        st.stop()

    view = data.copy()

    if query.strip():
        view = view[
            view["title"].str.contains(
                query.strip(),
                case=False,
                na=False,
                regex=False,
            )
        ]

    view = view[
        view["pagerank_score"].between(
            selected_score_min,
            selected_score_max,
        )
    ]

    if rank_col:
        view = view[
            view[rank_col].between(
                selected_rank_min,
                selected_rank_max,
            )
        ]

    sort_options = ["PageRank rank", "PageRank score"]
    if "in_degree" in view.columns:
        sort_options.append("In-degree")

    sort_choice = st.selectbox(
        "Sort results by",
        sort_options,
        index=0,
    )

    sort_columns = {
        "PageRank rank": rank_col,
        "PageRank score": "pagerank_score",
        "In-degree": "in_degree",
    }
    chosen_sort_col = sort_columns[sort_choice]

    if chosen_sort_col and chosen_sort_col in view.columns:
        view = view.sort_values(
            chosen_sort_col,
            ascending=sort_choice == "PageRank rank",
            na_position="last",
        )

    st.caption(f"Showing {len(view):,} of {len(data):,} pages")

    if not view.empty:
        selected = st.selectbox(
            "Select a page to inspect",
            view["title"].drop_duplicates().head(10000).tolist(),
        )

        row = view[view["title"] == selected].iloc[0]

        x, y, z, w = st.columns(4)
        x.metric("Selected page", selected)
        y.metric(
            "PageRank rank",
            fmt_int(row[rank_col]) if rank_col else "—",
        )
        z.metric(
            "PageRank score",
            fmt_num(row["pagerank_score"], 10),
        )
        w.metric(
            "In-degree",
            fmt_int(row.get("in_degree")),
            help="Number of incoming links from pages in the filtered graph.",
        )

        columns = [
            col
            for col in [
                rank_col,
                "title",
                "pagerank_score",
                "in_degree",
                "in_degree_rank",
            ]
            if col and col in view.columns
        ]

        labels = {
            "pagerank_rank": "PageRank rank",
            "title": "Wikipedia page",
            "pagerank_score": "PageRank score",
            "in_degree": "In-degree",
            "in_degree_rank": "In-degree rank",
        }

        st.dataframe(
            view[columns].rename(columns=labels),
            use_container_width=True,
            hide_index=True,
            height=460,
        )

        st.download_button(
            "Download filtered results as CSV",
            data=view[columns].to_csv(index=False).encode("utf-8"),
            file_name="filtered_pagerank_results.csv",
            mime="text/csv",
        )
    else:
        st.warning(
            "No pages match those filters. Widen the range or try another title."
        )


# ============================================================
# 3. COMPARE ALGORITHMS
# ============================================================
elif page == "Compare Algorithms":
    st.title("Compare PageRank and In-Degree")
    st.caption(
        "See how link counts and link-based importance agree and differ."
    )

    c1, c2 = st.columns(2)

    c1.metric(
        "Top-10 pages in common",
        f"{fmt_int(overlap)}/10" if overlap is not None else "—",
    )
    c2.metric(
        "Spearman rank correlation",
        fmt_num(rho, 4),
    )

    if comparison is None:
        show_missing("indegree_pagerank_comparison_100k.csv")
        st.stop()

    required = {"title", "in_degree", "pagerank_score"}

    if required.issubset(comparison.columns):
        plot_data = comparison.copy()

        plot_data["in_degree"] = pd.to_numeric(
            plot_data["in_degree"],
            errors="coerce",
        )
        plot_data["pagerank_score"] = pd.to_numeric(
            plot_data["pagerank_score"],
            errors="coerce",
        )

        plot_data = plot_data.dropna(
            subset=["in_degree", "pagerank_score"]
        )

        use_log = st.checkbox("Use logarithmic axes", value=False)

        max_points = st.select_slider(
            "Points to display",
            options=[1000, 2500, 5000, 10000],
            value=5000,
        )

        if len(plot_data) > max_points:
            plot_data = plot_data.sample(
                max_points,
                random_state=42,
            )

        fig = px.scatter(
            plot_data,
            x="in_degree",
            y="pagerank_score",
            hover_name="title",
            hover_data={
                "in_degree": True,
                "pagerank_score": ":.8g",
            },
            opacity=0.62,
            labels={
                "in_degree": "In-degree (incoming links)",
                "pagerank_score": "PageRank score",
            },
            color_discrete_sequence=["#0d9488"],
        )

        if use_log:
            # Zero values cannot be displayed on a logarithmic axis.
            fig.update_xaxes(type="log")
            fig.update_yaxes(type="log")

        fig.update_layout(
            margin=dict(l=10, r=10, t=20, b=10)
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(
            gridcolor="rgba(148,163,184,.18)"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            f"The chart displays a reproducible sample of up to "
            f"{max_points:,} pages. Hover over points to see page titles."
        )
    else:
        st.warning(
            "Comparison CSV must contain title, in_degree and "
            "pagerank_score columns."
        )

    left, right = st.columns(2, gap="large")

    with left:
        st.subheader("Top pages by in-degree")
        if top_in is not None:
            st.dataframe(
                top_in,
                use_container_width=True,
                hide_index=True,
            )
        else:
            show_missing("top_10_in_degree.csv")

    with right:
        st.subheader("Top pages by PageRank")
        if top_pr is not None:
            st.dataframe(
                top_pr,
                use_container_width=True,
                hide_index=True,
            )
        else:
            show_missing("top_10_pagerank.csv")

    if top_comparison is not None:
        st.subheader("Ranking comparison table")
        comparison_display = top_comparison.copy()

        if {
            "in_degree_rank",
            "pagerank_rank",
        }.issubset(comparison_display.columns):
            comparison_display["Rank difference"] = (
                pd.to_numeric(
                    comparison_display["in_degree_rank"],
                    errors="coerce",
                )
                - pd.to_numeric(
                    comparison_display["pagerank_rank"],
                    errors="coerce",
                )
            )

            st.caption(
                "Rank difference = in-degree rank minus PageRank rank. "
                "Positive values mean PageRank places the page higher; "
                "negative values mean in-degree places it higher. "
                "Larger absolute values indicate a bigger disagreement."
            )
            
            def highlight_rank_difference(row):
                styles = []

                for column in row.index:
                    if column != "Rank difference":
                        styles.append("")
                        continue

                    value = row[column]

                    if value > 0:
                        styles.append(
                            "background-color: #166534; color: #FFFFFF; font-weight: bold;"
                        )
                    elif value < 0:
                        styles.append(
                            "background-color: #991B1B; color: #FFFFFF; font-weight: bold;"
                        )
                    else:
                        styles.append(
                            "background-color: #374151; color: #FFFFFF; font-weight: bold;"
                        )

                return styles


            st.dataframe(
                comparison_display.style.apply(
                    highlight_rank_difference,
                    axis=1,
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.dataframe(
                comparison_display,
                use_container_width=True,
                hide_index=True,
            )

    st.info(
        "A high Spearman correlation means the rankings tend to move "
        "together overall. It does not mean every page has the same rank "
        "under both methods."
    )


# ============================================================
# 4. ABOUT THE PROJECT
# ============================================================
elif page == "About the Project":
    st.title("About the Project")

    st.write(
        "This project analyzes a directed Wikipedia link graph and compares "
        "in-degree with PageRank. It uses a filtered set of 100,000 pages and "
        "the links retained between those pages."
    )

    st.subheader("Methodology")

    st.markdown(
        "1. Prepare a directed graph from the Wikipedia link dataset.\n"
        "2. Keep links whose source and target are both in the selected "
        "100,000 pages.\n"
        "3. Calculate graph statistics and in-degree rankings.\n"
        "4. Run PageRank using the saved configuration.\n"
        "5. Compare rankings and visualize their relationship.\n\n"
        "The dashboard reads a graph prepared before the app runs; it does not "
        "remove duplicate links itself. Duplicate-edge handling therefore "
        "depends on the graph-preparation step and should be verified against "
        "that code."
    )

    st.subheader("Graph statistics")

    stats = {
        "Pages (nodes)": graph.get("nodes"),
        "Directed edges": graph.get("edges"),
        "Self-loops": graph.get("self_loops"),
        "Zero in-degree pages": graph.get("zero_in_degree_pages"),
        "Zero out-degree pages": graph.get("zero_out_degree_pages"),
        "Average in-degree": graph.get("average_in_degree"),
        "Average out-degree": graph.get("average_out_degree"),
        "Maximum in-degree": graph.get("maximum_in_degree"),
        "Maximum out-degree": graph.get("maximum_out_degree"),
    }

    st.dataframe(
        pd.DataFrame([stats]),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("PageRank parameters")
    st.json(params)

    st.subheader("Experiment summary")

    s1, s2, s3, s4 = st.columns(4)

    s1.metric(
        "Pages ranked",
        fmt_int(experiment.get("pages_ranked")),
    )
    s2.metric(
        "Recorded execution time",
        f"{fmt_num(experiment.get('execution_time_seconds'), 5)} s",
    )
    s3.metric(
        "Sum of scores",
        fmt_num(experiment.get("sum_pagerank_scores"), 12),
    )
    s4.metric(
        "Maximum score",
        fmt_num(experiment.get("max_pagerank"), 8),
    )

    st.write(
        f"Minimum score: **{fmt_num(experiment.get('min_pagerank'), 8)}**"
    )
    st.write(
        f"Median score: **{fmt_num(experiment.get('median_pagerank'), 8)}**"
    )
    st.write(
        f"Mean score: **{fmt_num(experiment.get('mean_pagerank'), 8)}**"
    )
    st.write(
        f"Standard deviation: **{fmt_num(experiment.get('std_pagerank'), 8)}**"
    )
    st.write(
        f"Spearman correlation: **{fmt_num(rho, 6)}**"
    )

    st.caption(
        "The saved p-value is 0.0 due to numeric underflow; "
        "interpret it as extremely small, not exactly zero."
    )

    st.warning(
        "PageRank reflects link structure in the filtered graph. It should "
        "not be interpreted as article quality or factual correctness."
    )


# ============================================================
# FOOTER
# ============================================================
st.divider()
st.markdown(
    f"**Wikipedia PageRank Explorer**  \n"
    f"{PROJECT_NAME}\n"
    f"Team members: {TEAM_NAME}"
)
