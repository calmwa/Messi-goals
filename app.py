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

@st.cache_data(show_spinner="Preparing the feature space and projections…")
def get_data():
    raw = load_goals(ROOT / "data" / "messi_goals.csv")
    clean, matrix, _ = prepare_features(raw)
    coords, metrics = project(matrix, clean)
    return clean, coords, metrics

df, coords, metrics = get_data()
with st.sidebar:
    st.header("Filter goals")
    seasons = st.multiselect("Season", sorted(df.Season.dropna().unique()))
    competitions = st.multiselect("Competition", sorted(df.Competition.dropna().unique()))
    goal_types = st.multiselect("Goal type", sorted(df.Type.dropna().unique()))
    venues = st.multiselect("Venue", sorted(df.Venue.dropna().unique()))
    st.button("Reset selection", on_click=lambda: st.session_state.update(selected_ids=[]))

mask = pd.Series(True, index=df.index)
for col, values in [("Season", seasons), ("Competition", competitions), ("Type", goal_types), ("Venue", venues)]:
    if values:
        mask &= df[col].isin(values)
filtered = coords.loc[mask].copy()
valid_ids = set(filtered.goal_id)
if "selected_ids" not in st.session_state:
    st.session_state.selected_ids = []
st.session_state.selected_ids = [x for x in st.session_state.selected_ids if x in valid_ids]
labels = {r.goal_id: f"{r.goal_id} · {r.Season} · {r.Opponent} · {r.Minute}′" for r in filtered.itertuples()}
selected = st.multiselect("Select goals to link across every view", options=list(labels), format_func=lambda x: labels.get(x, x), key="selected_ids")
show = filtered.copy()
show["selection"] = show.goal_id.map(lambda x: "Selected" if x in selected else "Other")
hover = ["goal_id", "Season", "Opponent", "Competition", "Minute", "Type", "Venue", "Club", "Playing_Position", "Goal_assist"]

@st.cache_data(show_spinner="Building context subspace matrix…")
def get_subspace_matrix(data):
    return subspace_projection_matrix(data)

with st.expander("Approved bonus: context projection matrix", expanded=False):
    st.markdown("Each cell applies PCA to one context group (diagonal) or a pair of groups (off diagonal). This simplified matrix follows the paper's subspace comparison idea; it does not implement the projection tree. Select any cell to enlarge it. The same selected `goal_id` values stay highlighted here and in the main PCA, t-SNE, feature views, and table.")
    matrix_coords = get_subspace_matrix(show)
    groups = list(dict.fromkeys([key[0] for key in matrix_coords]))
    if "active_subspace" not in st.session_state or st.session_state.active_subspace not in matrix_coords:
        st.session_state.active_subspace = (groups[0], groups[0])
    for row_group in groups:
        grid = st.columns(len(groups))
        for col_index, col_group in enumerate(groups):
            key = (row_group, col_group)
            title = f"{row_group} × {col_group}"
            with grid[col_index]:
                if st.button(title, key=f"subspace_{groups.index(row_group)}_{col_index}", use_container_width=True):
                    st.session_state.active_subspace = key
                cell = matrix_coords[key].merge(show[["goal_id", "selection", "Season", "Opponent", "Competition", "Minute", "Type"]], on="goal_id", how="inner")
                fig = px.scatter(cell, x="PC1", y="PC2", color="selection",
                                 color_discrete_map={"Selected":"#e63946", "Other":"#a8b3c4"},
                                 hover_data=["goal_id", "Season", "Opponent", "Competition", "Minute", "Type"])
                fig.update_layout(height=190, margin=dict(l=4, r=4, t=4, b=4), showlegend=False)
                st.plotly_chart(fig, use_container_width=True, key=f"subspace_plot_{groups.index(row_group)}_{col_index}")
    active = st.session_state.active_subspace
    st.markdown(f"#### Enlarged cell: {active[0]} × {active[1]}")
    enlarged = matrix_coords[active].merge(show[["goal_id", "selection", *[col for col in hover if col != "goal_id"]]], on="goal_id", how="inner")
    fig = px.scatter(enlarged, x="PC1", y="PC2", color="selection",
                     color_discrete_map={"Selected":"#e63946", "Other":"#a8b3c4"}, hover_data=hover)
    st.plotly_chart(fig, use_container_width=True, key="enlarged_subspace")

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
