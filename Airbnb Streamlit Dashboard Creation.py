import os
import pandas as pd
import plotly.express as px
import streamlit as st

#  PAGE CONFIGURATION
st.set_page_config(
    page_title="Airbnb Market Intelligence & Sentiment Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

#  LOAD MASTER DATASET 
@st.cache_data
def load_data():
    file_name = 'master_airbnb_cleaned.csv'
    if not os.path.exists(file_name):
        st.error(f"File '{file_name}' not found. Please ensure it is saved in the same directory.")
        st.stop()
        
    data = pd.read_csv(file_name, low_memory=False)
    
    # Ensure numerical types are properly formatted
    data['price'] = pd.to_numeric(data['price'], errors='coerce')
    data['number_of_reviews'] = pd.to_numeric(data['number_of_reviews'], errors='coerce').fillna(0)
    data['reviews_per_month'] = pd.to_numeric(data['reviews_per_month'], errors='coerce').fillna(0.0)
    
    if 'future_occupancy_rate' in data.columns:
        data['future_occupancy_rate'] = pd.to_numeric(data['future_occupancy_rate'], errors='coerce').fillna(0.0)
    else:
        data['future_occupancy_rate'] = 0.0

    if 'avg_sentiment_score' in data.columns:
        data['avg_sentiment_score'] = pd.to_numeric(data['avg_sentiment_score'], errors='coerce').fillna(0.0)
    else:
        data['avg_sentiment_score'] = 0.0

    if 'sentiment_category' not in data.columns:
        data['sentiment_category'] = 'No Reviews / Neutral'
    else:
        data['sentiment_category'] = data['sentiment_category'].fillna('No Reviews / Neutral')
        
    return data

df = load_data()

#  SIDEBAR FILTERS 
st.sidebar.title("Dashboard Filters")

# Borough Filter
borough_options = ['All'] + sorted(df['neighbourhood_group'].dropna().unique().tolist())
selected_borough = st.sidebar.selectbox("Borough / Neighbourhood Group", borough_options)

# Room Type Filter
room_options = ['All'] + sorted(df['room_type'].dropna().unique().tolist())
selected_room = st.sidebar.selectbox("Room Type", room_options)

# Sentiment Classification Filter
sentiment_options = ['All'] + sorted(df['sentiment_category'].dropna().unique().tolist())
selected_sentiment = st.sidebar.selectbox("Sentiment Category", sentiment_options)

# Nightly Price Range Filter
min_val = int(df['price'].min()) if not df['price'].empty else 0
max_val = int(df['price'].max()) if not df['price'].empty else 500
price_range = st.sidebar.slider(
    "Nightly Price Range ($)",
    min_value=min_val,
    max_value=max_val,
    value=(min_val, max_val)
)

# Apply Filters
filtered = df[(df['price'] >= price_range[0]) & (df['price'] <= price_range[1])].copy()

if selected_borough != 'All':
    filtered = filtered[filtered['neighbourhood_group'] == selected_borough]

if selected_room != 'All':
    filtered = filtered[filtered['room_type'] == selected_room]

if selected_sentiment != 'All':
    filtered = filtered[filtered['sentiment_category'] == selected_sentiment]

#  DASHBOARD HEADER & KPIS 
st.title("Airbnb Performance, Sentiment & Market Dashboard")
st.markdown(
    "Interactive analytics exploring listing density, nightly rates, customer review sentiment, "
    "and future booking occupancy rates."
)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Total Listings", f"{len(filtered):,}")
kpi2.metric(
    "Avg Nightly Price", 
    f"${filtered['price'].mean():.2f}" if len(filtered) else "$0.00"
)
kpi3.metric(
    "Avg Future Occupancy", 
    f"{filtered['future_occupancy_rate'].mean() * 100:.1f}%" if len(filtered) else "0.0%"
)
kpi4.metric(
    "Avg Sentiment Polarity", 
    f"{filtered['avg_sentiment_score'].mean():.2f}" if len(filtered) else "0.00"
)

st.divider()

#  TABBED VISUALIZATIONS 
tab_market, tab_map, tab_sentiment, tab_data = st.tabs([
    "Pricing & Demand",
    "Geospatial Distribution",
    "Review Sentiment",
    "Export Data"
])

# TAB 1: Pricing & Demand
with tab_market:
    col_left, col_right = st.columns(2)
    
    with col_left:
        top_neigh = (
            filtered['neighbourhood']
            .value_counts()
            .head(10)
            .reset_index()
        )
        top_neigh.columns = ['neighbourhood', 'count']
        fig_bar = px.bar(
            top_neigh,
            x='count',
            y='neighbourhood',
            orientation='h',
            title='Top 10 High-Density Neighbourhoods',
            color='count',
            color_continuous_scale='Blues'
        )
        fig_bar.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_right:
        fig_box = px.box(
            filtered,
            x='room_type',
            y='price',
            color='room_type',
            title='Nightly Price Spread by Room Type'
        )
        st.plotly_chart(fig_box, use_container_width=True)

    fig_scatter = px.scatter(
        filtered,
        x='future_occupancy_rate',
        y='price',
        color='room_type',
        hover_data=['name', 'neighbourhood'],
        title='Future Occupancy Rate vs. Nightly Price',
        opacity=0.6
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# TAB 2: Geospatial Map (with Plotly v5 & v6 Compatibility)
with tab_map:
    st.subheader("Geospatial Property Distribution")
    map_df = filtered.dropna(subset=['latitude', 'longitude']).head(2500)
    
    if map_df.empty:
        st.info("No listings with valid coordinates match the current filter selection.")
    else:
        # Check if using Plotly v6+ (px.scatter_map) or Plotly v5- (px.scatter_mapbox)
        if hasattr(px, 'scatter_map'):
            fig_map = px.scatter_map(
                map_df,
                lat='latitude',
                lon='longitude',
                color='price',
                size='price',
                size_max=10,
                hover_name='name',
                hover_data=['neighbourhood', 'price', 'room_type'],
                color_continuous_scale='Turbo',
                zoom=10,
                map_style='carto-positron',
                title='Geospatial Map (Top 2,500 Sampled Listings)'
            )
        else:
            fig_map = px.scatter_mapbox(
                map_df,
                lat='latitude',
                lon='longitude',
                color='price',
                size='price',
                size_max=10,
                hover_name='name',
                hover_data=['neighbourhood', 'price', 'room_type'],
                color_continuous_scale='Turbo',
                zoom=10,
                mapbox_style='carto-positron',
                title='Geospatial Map (Top 2,500 Sampled Listings)'
            )
        st.plotly_chart(fig_map, use_container_width=True)

# TAB 3: Review Sentiment Analysis
with tab_sentiment:
    st.subheader("Customer Review Sentiment Analysis")
    sc1, sc2 = st.columns(2)
    
    with sc1:
        sent_df = (
            filtered['sentiment_category']
            .value_counts()
            .reset_index()
        )
        sent_df.columns = ['sentiment_category', 'listings']
        fig_pie = px.pie(
            sent_df,
            names='sentiment_category',
            values='listings',
            title='Proportion of Sentiment Classifications',
            color='sentiment_category',
            color_discrete_map={
                'Consistently Positive': '#2ca02c',
                'Moderately Positive': '#1f77b4',
                'Critical / Mixed': '#d62728',
                'No Reviews / Neutral': '#7f7f7f'
            }
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with sc2:
        valid_sent = filtered[filtered['sentiment_category'] != 'No Reviews / Neutral']
        if valid_sent.empty:
            st.info("No reviewed listings in the current filter.")
        else:
            fig_box_sent = px.box(
                valid_sent,
                x='neighbourhood_group',
                y='avg_sentiment_score',
                color='neighbourhood_group',
                title='Sentiment Polarity Spread Across Boroughs'
            )
            st.plotly_chart(fig_box_sent, use_container_width=True)

# TAB 4: Data Inspection & CSV Export
with tab_data:
    st.subheader("Filtered Property Dataset")
    st.dataframe(filtered.head(500))

    csv_bytes = filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Filtered Results as CSV",
        data=csv_bytes,
        file_name="filtered_airbnb_data.csv",
        mime="text/csv"
    )