from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
from src.data_processing import load_goals, prepare_features
from src.dr_analysis import project, subspace_projection_matrix

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Messi Goals · Dimension Reduction", layout="wide")
st.title("Messi’s goals in context")
st.caption("Linked views of goal records. Select IDs once and the same goals highlight in both projections and original feature views.")
st.info("**Quick guide:** Use the sidebar to filter by season, competition, goal type, or venue. Select one or more goal IDs to highlight those records in every view. Use Reset selection to clear them. Expand the bonus matrix to compare context subspaces.")

@st.cache_data(show_spinner="Preparing the feature space and projections…")
def get_data():
    raw = load_goals(ROOT / "data" / "messi_goals.csv")
    clean, matrix, _ = prepare_features(raw)
    coords, metrics = project(matrix, clean)
    return clean, coords, metrics

try:
    df, coords, metrics = get_data()
except FileNotFoundError:
    st.error("The goal data file could not be found. From the project folder, check that `data/messi_goals.csv` is present.")
    st.code("streamlit run app.py")
    st.stop()
except Exception as exc:
    st.error("The dashboard could not prepare the data. Check that the CSV is unchanged and install the packages listed in requirements.txt.")
    with st.expander("Technical details"):
        st.code(f"{type(exc).__name__}: {exc}")
    st.stop()
with st.sidebar:
    st.header("Filter goals")
    st.caption("Leave a filter empty to include every value.")
    seasons = st.multiselect("Season", sorted(df.Season.dropna().unique()), help="Show goals from selected seasons.")
    competitions = st.multiselect("Competition", sorted(df.Competition.dropna().unique()), help="Limit the dashboard to selected competitions.")
    goal_types = st.multiselect("Goal type", sorted(df.Type.dropna().unique()), help="Filter by the recorded finishing type.")
    venues = st.multiselect("Venue", sorted(df.Venue.dropna().unique()), help="H means home and A means away.")
    st.button("Reset selection", on_click=lambda: st.session_state.update(selected_ids=[]))

mask = pd.Series(True, index=df.index)
for col, values in [("Season", seasons), ("Competition", competitions), ("Type", goal_types), ("Venue", venues)]:
    if values:
        mask &= df[col].isin(values)
filtered = coords.loc[mask].copy()
if filtered.empty:
    st.warning("No goals match these filters. Clear one or more sidebar filters to see records again.")
    st.stop()
valid_ids = set(filtered.goal_id)
if "selected_ids" not in st.session_state:
    st.session_state.selected_ids = []
st.session_state.selected_ids = [x for x in st.session_state.selected_ids if x in valid_ids]
labels = {r.goal_id: f"{r.goal_id} · {r.Season} · {r.Opponent} · {r.Minute}′" for r in filtered.itertuples()}
selected = st.multiselect("Select goals to link across every view", options=list(labels), format_func=lambda x: labels.get(x, x), key="selected_ids", help="Start typing an ID, season, or opponent. Choose multiple goals to highlight them together.")
show = filtered.copy()
show["selection"] = show.goal_id.map(lambda x: "Selected" if x in selected else "Other")
hover = ["goal_id", "Season", "Opponent", "Competition", "Minute", "Type", "Venue", "Club", "Playing_Position", "Goal_assist"]

@st.cache_data(show_spinner="Building context subspace matrix…")
def get_subspace_matrix(data):
    return subspace_projection_matrix(data)

def activate_subspace(key):
    st.session_state.active_subspace = key

with st.expander("Approved bonus: context projection matrix", expanded=True):
    st.markdown("Each cell applies PCA to one context group (diagonal) or a pair of groups (off diagonal). Click a cell label or any point in its small plot. The enlarged view appears above the matrix. Your selected goal IDs stay highlighted here and in the main dashboard views.")
    matrix_coords = get_subspace_matrix(show)
    groups = list(dict.fromkeys([key[0] for key in matrix_coords]))
    if "active_subspace" not in st.session_state or st.session_state.active_subspace not in matrix_coords:
        st.session_state.active_subspace = (groups[0], groups[0])

    active = st.session_state.active_subspace
    st.markdown(f"#### Enlarged cell: {active[0]} x {active[1]}")
    enlarged = matrix_coords[active].merge(show[["goal_id", "selection", *[col for col in hover if col != "goal_id"]]], on="goal_id", how="inner")
    fig = px.scatter(enlarged, x="PC1", y="PC2", color="selection",
                     color_discrete_map={"Selected":"#e63946", "Other":"#a8b3c4"}, hover_data=hover)
    st.plotly_chart(fig, use_container_width=True, key="enlarged_subspace")

    st.markdown("#### All context pairs")
    for row_group in groups:
        grid = st.columns(len(groups))
        for col_index, col_group in enumerate(groups):
            key = (row_group, col_group)
            title = f"{row_group} x {col_group}"
            with grid[col_index]:
                st.button(title, key=f"subspace_{groups.index(row_group)}_{col_index}",
                          use_container_width=True, on_click=activate_subspace, args=(key,))
                cell = matrix_coords[key].merge(show[["goal_id", "selection", "Season", "Opponent", "Competition", "Minute", "Type"]], on="goal_id", how="inner")
                fig = px.scatter(cell, x="PC1", y="PC2", color="selection",
                                 color_discrete_map={"Selected":"#e63946", "Other":"#a8b3c4"},
                                 hover_data=["goal_id", "Season", "Opponent", "Competition", "Minute", "Type"],
                                 custom_data=["goal_id"])
                fig.update_layout(height=190, margin=dict(l=4, r=4, t=4, b=4), showlegend=False)
                event = st.plotly_chart(fig, use_container_width=True,
                                        key=f"subspace_plot_{groups.index(row_group)}_{col_index}",
                                        on_select="rerun", selection_mode="points")
                points = event.selection.points
                signature = tuple(point.get("point_index", point.get("point_number")) for point in points)
                signature_key = f"subspace_signature_{groups.index(row_group)}_{col_index}"
                previous_signature = st.session_state.get(signature_key, ())
                st.session_state[signature_key] = signature
                if signature and signature != previous_signature:
                    st.session_state.active_subspace = key
                    st.rerun()

c1, c2 = st.columns(2)
with c1:
    fig = px.scatter(show, x="PC1", y="PC2", color="selection", color_discrete_map={"Selected":"#e63946", "Other":"#a8b3c4"}, hover_data=hover, title=f"PCA · first two components explain {metrics['pca_variance_total']:.1%}")
    st.plotly_chart(fig, use_container_width=True)
with c2:
    fig = px.scatter(show, x="TSNE1", y="TSNE2", color="selection", color_discrete_map={"Selected":"#e63946", "Other":"#a8b3c4"}, hover_data=hover, title="t-SNE · local neighborhood view")
    st.plotly_chart(fig, use_container_width=True)
st.caption(f"PCA trustworthiness={metrics['pca_trustworthiness']:.3f} · t-SNE trustworthiness={metrics['tsne_trustworthiness']:.3f} · k={metrics['trustworthiness_neighbors']} · perplexity={metrics['tsne_perplexity']:.1f}")

st.subheader("Original feature space")
selected_rows = show[show.goal_id.isin(selected)]
base1, base2 = st.columns(2)
with base1:
    fig = px.histogram(show, x="minute_num", nbins=20, title="Goal minute distribution")
    if not selected_rows.empty:
        for _, r in selected_rows.iterrows(): fig.add_vline(x=r.minute_num, line_color="#e63946", opacity=.35)
    st.plotly_chart(fig, use_container_width=True)
with base2:
    types = show.groupby("Type", dropna=False).size().rename("All filtered goals").to_frame()
    if not selected_rows.empty:
        types["Selected"] = selected_rows.groupby("Type", dropna=False).size()
        types = types.fillna(0).reset_index().melt("Type", var_name="Group", value_name="Goals")
        fig = px.bar(types, x="Type", y="Goals", color="Group", barmode="group", title="Goal type · selected vs filtered")
    else:
        fig = px.bar(types.reset_index(), x="Type", y="All filtered goals", title="Goal types · filtered goals")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Interpretable numeric profile")
parallel = show.copy()
parallel["selected_flag"] = parallel.goal_id.isin(selected).astype(int)
fig = px.parallel_coordinates(parallel, dimensions=["season_start", "minute_num", "selected_flag"], color="selected_flag", color_continuous_scale=[[0,"#a8b3c4"],[1,"#e63946"]], labels={"season_start":"Season starting year", "minute_num":"Minute", "selected_flag":"Selected (1=yes)"})
st.plotly_chart(fig, use_container_width=True)
st.subheader("Linked goal records")
table = show[show.goal_id.isin(selected)] if selected else show
st.dataframe(table[["goal_id", "Season", "Competition", "Opponent", "Minute", "Type", "Venue", "Playing_Position", "Goal_assist", "PC1", "PC2", "TSNE1", "TSNE2"]], use_container_width=True, hide_index=True)
st.download_button("Download current projection table", filtered.to_csv(index=False), "goal_projections.csv", "text/csv")
