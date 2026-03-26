import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from PIL import Image
import os

st.set_page_config(page_title="LILA BLACK — Level Design Tool", layout="wide")

# ── Config ──────────────────────────────────────────────────────────────────
MINIMAP_PATHS = {
    'AmbroseValley': 'minimaps/AmbroseValley_Minimap.png',
    'GrandRift':     'minimaps/GrandRift_Minimap.png',
    'Lockdown':      'minimaps/Lockdown_Minimap.jpg',
}

EVENT_COLORS = {
    'Position':      '#4A90D9',
    'BotPosition':   '#888888',
    'Kill':          '#FF4136',
    'Killed':        '#FF851B',
    'BotKill':       '#FFDC00',
    'BotKilled':     '#B10DC9',
    'KilledByStorm': '#00CED1',
    'Loot':          '#2ECC40',
}

EVENT_SYMBOLS = {
    'Position':      'circle',
    'BotPosition':   'circle',
    'Kill':          'x',
    'Killed':        'x',
    'BotKill':       'diamond',
    'BotKilled':     'diamond',
    'KilledByStorm': 'star',
    'Loot':          'square',
}

# ── Data loading ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    import pyarrow.parquet as pq

    DATA_ROOT = r"C:\Users\Aashish S\OneDrive - Amrita Vishwa Vidyapeetham\Documents\lila-viz\player_data"
    DAYS = ["February_10", "February_11", "February_12", "February_13"]
    MAP_CONFIGS = {
        'AmbroseValley': {'scale': 900,  'origin_x': -370, 'origin_z': -473},
        'GrandRift':     {'scale': 581,  'origin_x': -290, 'origin_z': -290},
        'Lockdown':      {'scale': 1000, 'origin_x': -500, 'origin_z': -500},
    }

    frames = []
    for day in DAYS:
        folder = os.path.join(DATA_ROOT, day)
        if not os.path.exists(folder):
            continue
        for fname in os.listdir(folder):
            if fname.startswith('.'):
                continue
            try:
                df = pq.read_table(os.path.join(folder, fname)).to_pandas()
                df['event'] = df['event'].apply(
                    lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
                user_id = fname.split('_')[0]
                df['is_bot'] = user_id.isdigit()
                df['day'] = day
                frames.append(df)
            except:
                continue

    df = pd.concat(frames, ignore_index=True)

    def world_to_pixel(row):
        cfg = MAP_CONFIGS.get(row['map_id'])
        if cfg is None:
            return pd.Series([None, None])
        u = (row['x'] - cfg['origin_x']) / cfg['scale']
        v = (row['z'] - cfg['origin_z']) / cfg['scale']
        return pd.Series([u * 1024, (1 - v) * 1024])

    df[['px', 'py']] = df.apply(world_to_pixel, axis=1)
    df['px'] = df['px'].clip(0, 1024)
    df['py'] = df['py'].clip(0, 1024)
    df['ts_seconds'] = (df['ts'].astype('int64') // 1_000_000).astype(int)
    df['ts_seconds'] = df['ts_seconds'] - df['ts_seconds'].min()
    return df

# ── Plotting helpers ──────────────────────────────────────────────────────────
def make_base_fig(map_id):
    img_path = MINIMAP_PATHS[map_id]
    img = Image.open(img_path)
    fig = go.Figure()
    fig.add_layout_image(
        dict(source=img, xref="x", yref="y",
             x=0, y=0, sizex=1024, sizey=1024,
             xanchor="left", yanchor="bottom",
             sizing="stretch", layer="below")
    )
    fig.update_layout(
        xaxis=dict(range=[0, 1024], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[0, 1024], showgrid=False, zeroline=False,
                   visible=False, scaleanchor="x"),
        margin=dict(l=0, r=0, t=0, b=0),
        height=700,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=True,
        legend=dict(bgcolor='rgba(0,0,0,0.6)', font=dict(color='white'))
    )
    return fig

def add_paths(fig, df, show_humans=True, show_bots=True):
    pos = df[df['event'].isin(['Position', 'BotPosition'])].copy()
    for match_id in pos['match_id'].unique()[:30]:  # cap at 30 matches for perf
        m = pos[pos['match_id'] == match_id]
        humans = m[m['is_bot'] == False].sort_values('ts')
        bots   = m[m['is_bot'] == True].sort_values('ts')
        if show_humans and len(humans) > 1:
            for uid in humans['user_id'].unique():
                p = humans[humans['user_id'] == uid]
                fig.add_trace(go.Scatter(
                    x=p['px'], y=p['py'], mode='lines',
                    line=dict(color='#4A90D9', width=1.5),
                    opacity=0.5, name='Human path', showlegend=False,
                    hoverinfo='skip'))
        if show_bots and len(bots) > 1:
            for uid in bots['user_id'].unique():
                p = bots[bots['user_id'] == uid]
                fig.add_trace(go.Scatter(
                    x=p['px'], y=p['py'], mode='lines',
                    line=dict(color='#888888', width=1),
                    opacity=0.3, name='Bot path', showlegend=False,
                    hoverinfo='skip'))
    # Legend proxies
    if show_humans:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
            line=dict(color='#4A90D9', width=2), name='Human path'))
    if show_bots:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
            line=dict(color='#888888', width=2), name='Bot path'))
    return fig

def add_events(fig, df, event_types):
    events = df[df['event'].isin(event_types)]
    for evt in event_types:
        sub = events[events['event'] == evt]
        if len(sub) == 0:
            continue
        fig.add_trace(go.Scatter(
            x=sub['px'], y=sub['py'], mode='markers',
            marker=dict(color=EVENT_COLORS.get(evt, '#fff'),
                        symbol=EVENT_SYMBOLS.get(evt, 'circle'),
                        size=10, line=dict(width=1, color='white')),
            name=evt,
            hovertemplate=f"<b>{evt}</b><br>x: %{{x:.0f}}<br>y: %{{y:.0f}}<extra></extra>"
        ))
    return fig

def add_heatmap(fig, df, event_filter, colorscale, name):
    sub = df[df['event'].isin(event_filter)]
    if len(sub) == 0:
        return fig
    fig.add_trace(go.Histogram2dContour(
        x=sub['px'], y=sub['py'],
        colorscale=colorscale,
        reversescale=False,
        showscale=True,
        opacity=0.6,
        ncontours=20,
        name=name,
        hoverinfo='skip',
        contours=dict(coloring='fill'),
        line=dict(width=0),
    ))
    return fig

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🎮 LILA BLACK — Player Journey Visualizer")
st.caption("Level Design Tool | 5 days of production data")

with st.spinner("Loading data..."):
    df = load_data()

# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")

    map_id = st.selectbox("Map", ['AmbroseValley', 'GrandRift', 'Lockdown'])
    df_map = df[df['map_id'] == map_id]

    day_options = ['All'] + sorted(df_map['day'].unique().tolist())
    day = st.selectbox("Day", day_options)
    if day != 'All':
        df_map = df_map[df_map['day'] == day]

    matches = sorted(df_map['match_id'].unique().tolist())
    match_options = ['All matches'] + matches
    selected_match = st.selectbox("Match", match_options)
    if selected_match != 'All matches':
        df_map = df_map[df_map['match_id'] == selected_match]

    st.divider()
    st.subheader("Display")
    show_humans = st.checkbox("Show human paths", value=True)
    show_bots   = st.checkbox("Show bot paths",   value=False)

    st.subheader("Events to show")
    all_events = ['Kill', 'Killed', 'BotKill', 'BotKilled', 'KilledByStorm', 'Loot']
    selected_events = []
    for evt in all_events:
        if st.checkbox(evt, value=evt in ['Kill', 'Killed', 'KilledByStorm', 'Loot']):
            selected_events.append(evt)

    st.divider()
    st.subheader("Heatmap")
    heatmap_type = st.selectbox("Heatmap overlay", [
        'None', 'Kill zones', 'Death zones', 'High traffic (humans)', 'Storm deaths'])

# ── Stats bar ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total events", f"{len(df_map):,}")
c2.metric("Matches",      df_map['match_id'].nunique())
c3.metric("Human players", df_map[df_map['is_bot']==False]['user_id'].nunique())
c4.metric("Bots",         df_map[df_map['is_bot']==True]['user_id'].nunique())

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🗺️ Map View", "⏱️ Timeline", "📊 Stats"])

with tab1:
    fig = make_base_fig(map_id)

    # Heatmap
    if heatmap_type == 'Kill zones':
        fig = add_heatmap(fig, df_map, ['Kill', 'BotKill'], 'Reds', 'Kill heatmap')
    elif heatmap_type == 'Death zones':
        fig = add_heatmap(fig, df_map, ['Killed', 'BotKilled', 'KilledByStorm'], 'Oranges', 'Death heatmap')
    elif heatmap_type == 'High traffic (humans)':
        fig = add_heatmap(fig, df_map, ['Position'], 'Blues', 'Traffic heatmap')
    elif heatmap_type == 'Storm deaths':
        fig = add_heatmap(fig, df_map, ['KilledByStorm'], 'Purples', 'Storm death heatmap')

    # Paths
    fig = add_paths(fig, df_map, show_humans, show_bots)

    # Events
    if selected_events:
        fig = add_events(fig, df_map, selected_events)

    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Match Timeline Playback")
    if selected_match == 'All matches':
        st.info("Select a specific match from the sidebar to use the timeline.")
    else:
        match_df = df_map.sort_values('ts')
        min_t = int(match_df['ts_seconds'].min())
        max_t = int(match_df['ts_seconds'].max())

        time_val = st.slider("Time (seconds into match)",
                             min_value=min_t, max_value=max_t,
                             value=min_t, step=5)

        window_df = match_df[match_df['ts_seconds'] <= time_val]

        fig2 = make_base_fig(map_id)
        fig2 = add_paths(fig2, window_df, show_humans=True, show_bots=False)
        fig2 = add_events(fig2, window_df, ['Kill', 'Killed', 'BotKill', 'BotKilled', 'KilledByStorm', 'Loot'])

        st.plotly_chart(fig2, use_container_width=True)
        st.caption(f"Showing events from 0s → {time_val}s | "
                   f"{len(window_df):,} events rendered")

with tab3:
    st.subheader("Event Breakdown")
    evt_counts = df_map['event'].value_counts().reset_index()
    evt_counts.columns = ['Event', 'Count']
    st.dataframe(evt_counts, use_container_width=True)

    st.subheader("Humans vs Bots")
    col1, col2 = st.columns(2)
    col1.metric("Human events", len(df_map[df_map['is_bot']==False]))
    col2.metric("Bot events",   len(df_map[df_map['is_bot']==True]))

    st.subheader("Events per Day")
    day_counts = df_map.groupby('day')['event'].count().reset_index()
    day_counts.columns = ['Day', 'Events']
    st.dataframe(day_counts, use_container_width=True)