from pathlib import Path
import json
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Wikipedia PageRank Explorer", page_icon="🔎", layout="wide")

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; max-width: 1500px;}
.hero {padding: 1.5rem 1.7rem; border-radius: 18px; margin-bottom: 1rem;
background: linear-gradient(120deg,#10233f 0%,#164e63 55%,#256c73 100%); color: white;}
.hero h1 {font-size: 2.2rem; margin: 0 0 .35rem 0; color: white;}
.hero p {font-size: 1rem; margin: 0; color: #e4f4f7;}
div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.22); padding: 1rem;
border-radius: 14px; background: rgba(128,128,128,.04);}
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def read_csv(path):
    return pd.read_csv(path)

@st.cache_data(show_spinner=False)
def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def csv_result(filename):
    p = RESULTS / filename
    if not p.exists():
        return None
    try:
        return read_csv(str(p))
    except Exception as e:
        st.warning(f"Could not load {filename}: {e}")
        return None

def json_result(filename):
    p = RESULTS / filename
    if not p.exists():
        return {}
    try:
        return read_json(str(p))
    except Exception as e:
        st.warning(f"Could not load {filename}: {e}")
        return {}

def fmt_int(x):
    try: return f"{int(float(x)):,}"
    except (TypeError, ValueError): return "—"

def fmt_num(x, digits=5):
    try: return f"{float(x):.{digits}g}"
    except (TypeError, ValueError): return "—"

graph = json_result("graph_summary.json")
experiment = json_result("final_experiment_summary.json")
rankings = csv_result("pagerank_ranking_100k.csv")
top_pr = csv_result("top_10_pagerank.csv")
top_in = csv_result("top_10_in_degree.csv")
comparison = csv_result("indegree_pagerank_comparison_100k.csv")

with st.sidebar:
    st.markdown("## 🔎 PageRank Explorer")
    st.caption("Wikipedia link graph · 100K pages")
    page = st.radio("Navigate", ["Overview", "Explore Rankings", "Compare Algorithms", "About the Project"])
    st.divider()
    params = experiment.get("pagerank_parameters", {})
    st.markdown("**PageRank settings**")
    st.write(f"Damping factor: `{params.get('damping_factor', 0.85)}`")
    st.write(f"Tolerance: `{params.get('tolerance', '1e-6')}`")
    st.write(f"Max iterations: `{params.get('maximum_iterations', 100)}`")

def show_missing(filename):
    st.error(f"Missing `results/{filename}`. Copy it from your existing project results folder.")

if page == "Overview":
    st.markdown("""<div class="hero"><h1>Wikipedia PageRank Explorer</h1>
    <p>Explore how link structure influences page importance across a 100,000-page Wikipedia graph.</p></div>""",
    unsafe_allow_html=True)

    nodes = graph.get("nodes", experiment.get("dataset_pages", 100000))
    edges = graph.get("edges", experiment.get("directed_edges", 8577044))
    loops = graph.get("self_loops", experiment.get("self_loops", "—"))
    avg_in = graph.get("average_in_degree", "—")
    max_in = graph.get("maximum_in_degree", "—")
    comp = experiment.get("ranking_comparison", {})
    overlap = comp.get("top_10_overlap_count")
    rho = comp.get("spearman_correlation")

    a,b,c,d = st.columns(4)
    a.metric("Pages analyzed", fmt_int(nodes))
    b.metric("Directed edges", fmt_int(edges))
    c.metric("Self-loops", fmt_int(loops))
    d.metric("Top-10 overlap", f"{overlap}/10" if overlap is not None else "—")
    e,f,g = st.columns(3)
    e.metric("Average in-degree", fmt_num(avg_in))
    f.metric("Maximum in-degree", fmt_int(max_in))
    g.metric("Spearman correlation", fmt_num(rho, 4))

    left, right = st.columns([1.1, .9], gap="large")
    with left:
        st.subheader("🏆 Top 10 pages by PageRank")
        if top_pr is not None:
            st.dataframe(top_pr, use_container_width=True, hide_index=True)
        else:
            show_missing("top_10_pagerank.csv")
    with right:
        st.subheader("PageRank score distribution")
        if rankings is not None and "pagerank_score" in rankings:
            scores = pd.to_numeric(rankings["pagerank_score"], errors="coerce").dropna()
            fig = px.histogram(x=scores, nbins=45,
                               labels={"x":"PageRank score","count":"Number of pages"},
                               title="Distribution across ranked pages")
            fig.update_layout(margin=dict(l=10,r=10,t=55,b=10), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            show_missing("pagerank_ranking_100k.csv")

    st.divider()
    st.subheader("What does PageRank measure?")
    st.write("PageRank estimates page importance from the graph's link structure. It accounts not only "
             "for how many links point to a page, but also for the importance of the pages providing those links.")

elif page == "Explore Rankings":
    st.title("Explore PageRank Rankings")
    st.caption("Search the saved ranking of Wikipedia pages and inspect scores.")
    if rankings is None or not {"title", "pagerank_score"}.issubset(rankings.columns):
        show_missing("pagerank_ranking_100k.csv")
        st.stop()

    rankings["title"] = rankings["title"].astype(str)
    rankings["pagerank_score"] = pd.to_numeric(rankings["pagerank_score"], errors="coerce")
    rank_col = "pagerank_rank" if "pagerank_rank" in rankings.columns else None
    query = st.text_input("Search Wikipedia page title", placeholder="e.g. India, Mathematics, World War II")
    view = rankings[rankings["title"].str.contains(query.strip(), case=False, na=False, regex=False)] if query.strip() else rankings.copy()
    view = view.sort_values(rank_col if rank_col else "pagerank_score",
                            ascending=True if rank_col else False, na_position="last")
    st.caption(f"{len(view):,} matching page(s)")
    if not view.empty:
        cols = [c for c in ["pagerank_rank", "title", "pagerank_score"] if c in view.columns]
        labels = {"pagerank_rank":"Rank", "title":"Wikipedia page", "pagerank_score":"PageRank score"}
        selection = view["title"].head(5000).tolist()
        selected = st.selectbox("Select a page for details", selection)
        row = view[view["title"] == selected].iloc[0]
        x,y,z = st.columns(3)
        x.metric("Wikipedia page", selected)
        y.metric("PageRank rank", fmt_int(row[rank_col]) if rank_col else "—")
        z.metric("PageRank score", fmt_num(row["pagerank_score"], 8))
        st.dataframe(view[cols].rename(columns=labels), use_container_width=True, hide_index=True, height=480)
        st.download_button("Download filtered results as CSV", view[cols].to_csv(index=False).encode("utf-8"),
                           file_name="filtered_pagerank_results.csv", mime="text/csv")
    else:
        st.warning("No pages match that search. Try another title.")

elif page == "Compare Algorithms":
    st.title("PageRank vs. In-Degree")
    st.write("In-degree counts incoming links. PageRank also accounts for the importance of pages linking to a page.")
    if comparison is None:
        show_missing("indegree_pagerank_comparison_100k.csv")
        st.stop()
    required = {"title", "in_degree", "pagerank_score"}
    if required.issubset(comparison.columns):
        plot_data = comparison.copy()
        plot_data["in_degree"] = pd.to_numeric(plot_data["in_degree"], errors="coerce")
        plot_data["pagerank_score"] = pd.to_numeric(plot_data["pagerank_score"], errors="coerce")
        plot_data = plot_data.dropna(subset=["in_degree", "pagerank_score"])
        if len(plot_data) > 5000:
            plot_data = plot_data.sample(5000, random_state=42)
        fig = px.scatter(plot_data, x="in_degree", y="pagerank_score", hover_name="title",
                         opacity=.55, labels={"in_degree":"In-degree (incoming links)",
                         "pagerank_score":"PageRank score"}, title="Incoming links vs. PageRank score")
        fig.update_layout(margin=dict(l=10,r=10,t=55,b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Comparison CSV must contain title, in_degree and pagerank_score columns.")

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Top pages by in-degree")
        if top_in is not None: st.dataframe(top_in, use_container_width=True, hide_index=True)
        else: show_missing("top_10_in_degree.csv")
    with right:
        st.subheader("Top pages by PageRank")
        if top_pr is not None: st.dataframe(top_pr, use_container_width=True, hide_index=True)
        else: show_missing("top_10_pagerank.csv")

    summary = experiment.get("ranking_comparison", {})
    x,y = st.columns(2)
    overlap = summary.get("top_10_overlap_count")
    x.metric("Pages shared by both top-10 lists", f"{overlap}/10" if overlap is not None else "—")
    y.metric("Spearman rank correlation", fmt_num(summary.get("spearman_correlation"), 4))
    if summary.get("top_10_overlap_pages"):
        st.write("**Pages in both top-10 lists:** " + ", ".join(map(str, summary["top_10_overlap_pages"])))

elif page == "About the Project":
    st.title("About the Project")
    st.write("This project analyzes a directed Wikipedia link graph and compares two measures of page importance: in-degree and PageRank.")
    st.subheader("Methodology")
    st.markdown("1. Prepare a directed graph from the Wikipedia link dataset.\n"
                "2. Calculate graph statistics and degree measures.\n"
                "3. Run NetworkX PageRank on the prepared graph.\n"
                "4. Compare PageRank ordering with in-degree ordering.\n"
                "5. Explore the saved results through this dashboard.")
    st.subheader("PageRank configuration")
    st.json(experiment.get("pagerank_parameters", {
        "damping_factor": 0.85, "tolerance": 1e-6, "maximum_iterations": 100
    }))
    if experiment.get("ranking_comparison"):
        st.subheader("Saved comparison summary")
        st.json(experiment["ranking_comparison"])
    st.caption("The dashboard reads saved results. It does not rebuild the graph or rerun PageRank.")

st.divider()
st.caption("Wikipedia PageRank Explorer · Built with Streamlit, Pandas and Plotly")
