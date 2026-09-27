import os
from pathlib import Path

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Netflix Analytics",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# STYLING
# ============================================================
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        border-bottom: 1px solid #e0e0e0;
    }

    .stTabs [data-baseweb="tab"] {
        font-weight: 500;
        padding-bottom: 10px;
    }

    .insight-box {
        background-color: #f8f9fa;
        border-left: 4px solid #E50914;
        padding: 15px 20px;
        border-radius: 4px;
        margin-top: 15px;
        font-size: 0.95rem;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False


# ============================================================
# DATA DIRECTORY
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


# ============================================================
# SAFE CSV READER
# ============================================================
def safe_read_csv(filename):
    """
    Safely read a CSV from the project's data folder.

    Handles:
    - Missing files
    - Empty files
    - UTF-8 / UTF-8-SIG / CP1252 / Latin-1 encodings
    - Bad CSV rows
    - Column-name cleanup
    """
    filepath = DATA_DIR / filename

    if not filepath.exists():
        return None

    if filepath.stat().st_size == 0:
        return None

    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]

    for encoding in encodings:
        try:
            data = pd.read_csv(
                filepath,
                encoding=encoding,
                on_bad_lines="skip"
            )

            if data is None or data.empty:
                return None

            data.columns = (
                data.columns
                .astype(str)
                .str.strip()
                .str.lower()
                .str.replace(r"\s+", "_", regex=True)
            )

            return data

        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
        except pd.errors.EmptyDataError:
            return None
        except Exception:
            continue

    return None


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def clean_text_column(df, column, default="Unknown"):
    """Create/clean a text column safely."""
    if column not in df.columns:
        df[column] = default

    df[column] = df[column].fillna(default).astype(str).str.strip()
    df.loc[df[column].isin(["", "nan", "None"]), column] = default
    return df


def find_column(df, keywords):
    """
    Find the first column whose name contains one of the keywords.
    """
    for keyword in keywords:
        for column in df.columns:
            if keyword in column:
                return column
    return None


def numeric_column(df, column):
    """Convert a column to numeric safely."""
    if column not in df.columns:
        return pd.Series(index=df.index, dtype="float64")

    return pd.to_numeric(
        df[column].astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False),
        errors="coerce"
    )


# ============================================================
# LOAD DATA
# ============================================================
@st.cache_data(show_spinner=False)
def load_data():
    # --------------------------------------------------------
    # 1. BASE NETFLIX CATALOG
    # --------------------------------------------------------
    df = safe_read_csv("netflix_titles.csv")

    if df is None:
        st.error(
            "❌ Could not load data/netflix_titles.csv. "
            "Make sure the file exists inside the data folder."
        )
        return pd.DataFrame()

    # Required Netflix catalog columns
    required_defaults = {
        "title": "Unknown Title",
        "type": "Unknown",
        "country": "Unknown",
        "director": "Unknown",
        "cast": "Unknown",
        "rating": "Not Rated",
        "listed_in": "Unknown",
        "duration": "Unknown",
        "date_added": "",
        "release_year": np.nan,
        "description": ""
    }

    for column, default in required_defaults.items():
        if column not in df.columns:
            df[column] = default

    # Clean important columns
    for column in ["title", "type", "country", "director", "cast",
                   "rating", "listed_in", "duration", "description"]:
        df = clean_text_column(df, column)

    # Dates
    df["date_added"] = pd.to_datetime(
        df["date_added"].astype(str).str.strip(),
        errors="coerce"
    )
    df["year_added"] = df["date_added"].dt.year

    # Release year
    df["release_year"] = pd.to_numeric(
        df["release_year"],
        errors="coerce"
    )

    # Movie duration
    movies_mask = df["type"].str.lower().eq("movie")

    df["duration_int"] = np.nan
    extracted_duration = (
        df.loc[movies_mask, "duration"]
        .astype(str)
        .str.extract(r"(\d+)", expand=False)
    )
    df.loc[movies_mask, "duration_int"] = pd.to_numeric(
        extracted_duration,
        errors="coerce"
    )

    # Clean title for merging
    df["title_clean"] = (
        df["title"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # netflix_large_dataset_cleaned.csv is NOT merged here.
    #
    # The file shown in the user's project contains user/
    # subscription fields such as:
    # user_id, age_group, gender, region, subscription_type,
    # subscription_start_date, payment_method, etc.
    #
    # It does not contain movie/TV title information, so using
    # it as an IMDb/TMDb title-enrichment dataset is incorrect.
    # --------------------------------------------------------

    # --------------------------------------------------------
    # 2. ROTTEN TOMATOES / METACRITIC / IMDb DATA
    # --------------------------------------------------------
    rt = safe_read_csv(
        "netflix-rotten-tomatoes-metacritic-imdb.csv"
    )

    if rt is not None and not rt.empty:
        title_col = find_column(rt, ["title", "name"])

        if title_col is not None:
            rt["title_clean"] = (
                rt[title_col]
                .astype(str)
                .str.lower()
                .str.strip()
            )

            # Only keep useful critic columns.
            useful_columns = ["title_clean"]

            for column in rt.columns:
                if column == "title_clean":
                    continue

                if any(
                    key in column
                    for key in [
                        "imdb",
                        "rotten",
                        "tomatoes",
                        "metacritic",
                        "score",
                        "rating"
                    ]
                ):
                    if column not in df.columns:
                        useful_columns.append(column)

            if len(useful_columns) > 1:
                rt_small = rt[useful_columns].drop_duplicates(
                    subset=["title_clean"]
                )

                df = pd.merge(
                    df,
                    rt_small,
                    on="title_clean",
                    how="left"
                )

    # --------------------------------------------------------
    # 3. OFFICIAL NETFLIX GLOBAL VIEWERSHIP
    # --------------------------------------------------------
    global_views = safe_read_csv("all-weeks-global.csv")

    if global_views is not None and not global_views.empty:
        title_col = find_column(
            global_views,
            ["title", "show", "name"]
        )

        if title_col is not None:
            global_views["title_clean"] = (
                global_views[title_col]
                .astype(str)
                .str.lower()
                .str.strip()
            )

            # Try to find hours/viewership column.
            hours_col = find_column(
                global_views,
                [
                    "hours_viewed",
                    "hours",
                    "viewed",
                    "view"
                ]
            )

            if hours_col is not None:
                global_views["view_numeric"] = numeric_column(
                    global_views,
                    hours_col
                )

                views_grouped = (
                    global_views
                    .dropna(subset=["view_numeric"])
                    .groupby("title_clean", as_index=False)["view_numeric"]
                    .sum()
                    .rename(
                        columns={
                            "view_numeric": "total_hours_viewed"
                        }
                    )
                )

                if not views_grouped.empty:
                    df = pd.merge(
                        df,
                        views_grouped,
                        on="title_clean",
                        how="left"
                    )

    # --------------------------------------------------------
    # 4. CONTENT INTELLIGENCE
    # --------------------------------------------------------
    intel = safe_read_csv(
        "netflix_content_intelligence_combined.csv"
    )

    if intel is not None and not intel.empty:
        title_col = find_column(intel, ["title", "name"])

        if title_col is not None:
            intel["title_clean"] = (
                intel[title_col]
                .astype(str)
                .str.lower()
                .str.strip()
            )

            useful_columns = ["title_clean"]

            for column in intel.columns:
                if column == "title_clean":
                    continue

                if any(
                    key in column
                    for key in [
                        "sentiment",
                        "popularity",
                        "content_type"
                    ]
                ):
                    if column not in df.columns:
                        useful_columns.append(column)

            if len(useful_columns) > 1:
                intel_small = intel[useful_columns].drop_duplicates(
                    subset=["title_clean"]
                )

                df = pd.merge(
                    df,
                    intel_small,
                    on="title_clean",
                    how="left"
                )

    # Final safety checks
    if "rating" not in df.columns:
        df["rating"] = "Not Rated"

    df["rating"] = (
        df["rating"]
        .fillna("Not Rated")
        .astype(str)
        .str.strip()
    )

    if "country" not in df.columns:
        df["country"] = "Unknown"

    if "listed_in" not in df.columns:
        df["listed_in"] = "Unknown"

    return df


# ============================================================
# LOAD
# ============================================================
with st.spinner("Loading Netflix datasets..."):
    df = load_data()


if df.empty:
    st.error("❌ No Netflix catalog data is available.")
    st.info(
        "Check that data/netflix_titles.csv exists in your GitHub repository."
    )
    st.stop()


# ============================================================
# SIDEBAR FILTERS
# ============================================================
with st.sidebar:
    st.header("🎛️ Filters")

    type_options = sorted(
        df["type"].dropna().astype(str).unique().tolist()
    )

    types = st.multiselect(
        "Content Type",
        options=type_options,
        default=type_options
    )

    valid_years = (
        pd.to_numeric(df["release_year"], errors="coerce")
        .dropna()
        .astype(int)
    )

    if not valid_years.empty:
        year_min = int(valid_years.min())
        year_max = int(valid_years.max())

        default_start = max(2000, year_min)

        if default_start > year_max:
            default_start = year_min

        year_range = st.slider(
            "Release Year",
            min_value=year_min,
            max_value=year_max,
            value=(default_start, year_max)
        )
    else:
        year_range = None
        st.info("Release-year data is unavailable.")

    rating_options = sorted(
        df["rating"].dropna().astype(str).unique().tolist()
    )

    ratings = st.multiselect(
        "Age Rating",
        options=rating_options,
        default=[]
    )

    if st.button("🔄 Reset Filters"):
        st.rerun()


# ============================================================
# APPLY FILTERS
# ============================================================
filtered = df.copy()

if types:
    filtered = filtered[
        filtered["type"].isin(types)
    ]
else:
    filtered = filtered.iloc[0:0]

if year_range is not None:
    filtered = filtered[
        filtered["release_year"].isna()
        |
        (
            (filtered["release_year"] >= year_range[0])
            &
            (filtered["release_year"] <= year_range[1])
        )
    ]

if ratings:
    filtered = filtered[
        filtered["rating"].isin(ratings)
    ]


# ============================================================
# HEADER & METRICS
# ============================================================
st.title("🎬 Netflix Content Analytics")

st.markdown(
    f"Analyzing **{len(df):,}** Netflix titles with catalog, "
    "critic, and viewership data where available."
)

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "Total Titles",
    f"{len(filtered):,}"
)

col2.metric(
    "Movies",
    f"{(filtered['type'] == 'Movie').sum():,}"
)

col3.metric(
    "TV Shows",
    f"{(filtered['type'] == 'TV Show').sum():,}"
)

country_count = (
    filtered["country"]
    .fillna("Unknown")
    .astype(str)
    .str.split(", ")
    .explode()
    .replace("Unknown", np.nan)
    .dropna()
    .nunique()
)

col4.metric(
    "Countries",
    f"{country_count:,}"
)


# Find IMDb column dynamically
imdb_col = None

for column in filtered.columns:
    column_lower = column.lower()

    if "imdb" in column_lower:
        if any(
            word in column_lower
            for word in ["score", "rating", "rate"]
        ):
            imdb_col = column
            break

if imdb_col is not None:
    imdb_values = numeric_column(
        filtered,
        imdb_col
    )

    avg_imdb = imdb_values.mean()

    col5.metric(
        "Avg IMDb Score",
        f"{avg_imdb:.1f}"
        if pd.notna(avg_imdb)
        else "N/A"
    )
else:
    col5.metric(
        "Avg IMDb Score",
        "N/A"
    )

st.divider()


# ============================================================
# EMPTY FILTER RESULT
# ============================================================
if filtered.empty:
    st.warning(
        "No titles match your current filters. "
        "Try changing the filters in the sidebar."
    )
    st.stop()


# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    [
        "📊 Overview",
        "📈 Trends",
        "🌍 Geography",
        "🎭 Genres",
        "⭐ Scores & Ratings",
        "📺 Viewership",
        "🔍 Explore"
    ]
)


# ============================================================
# TAB 1 — OVERVIEW
# ============================================================
with tab1:
    st.subheader("Content Distribution")

    c1, c2 = st.columns(2)

    with c1:
        type_counts = filtered["type"].value_counts()

        fig, ax = plt.subplots(figsize=(6, 6))

        ax.pie(
            type_counts.values,
            labels=type_counts.index,
            autopct="%1.1f%%",
            colors=["#E50914", "#4A4A4A"][:len(type_counts)],
            startangle=90,
            wedgeprops=dict(width=0.4)
        )

        ax.set_title("Movies vs TV Shows")

        st.pyplot(fig)
        plt.close(fig)

    with c2:
        movies_pct = (
            (filtered["type"] == "Movie").sum()
            / len(filtered)
            * 100
        )

        st.markdown(
            f"""
            <div class='insight-box'>
                The current filtered catalog contains
                <b>{movies_pct:.1f}% Movies</b> and
                <b>{100 - movies_pct:.1f}% TV Shows</b>.
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    st.subheader("Top Ratings by Count")

    ratings_data = (
        filtered["rating"]
        .value_counts()
        .head(10)
    )

    if not ratings_data.empty:
        fig, ax = plt.subplots(figsize=(10, 5))

        sns.barplot(
            x=ratings_data.index,
            y=ratings_data.values,
            hue=ratings_data.index,
            palette="Reds_r",
            ax=ax,
            legend=False
        )

        for i, value in enumerate(ratings_data.values):
            ax.text(
                i,
                value + max(ratings_data.values) * 0.01,
                f"{value:,}",
                ha="center",
                fontsize=9
            )

        ax.set_ylabel("Count")
        ax.set_xlabel("Rating")
        plt.xticks(rotation=45)

        st.pyplot(fig)
        plt.close(fig)


# ============================================================
# TAB 2 — TRENDS
# ============================================================
with tab2:
    st.subheader("Content Added Over the Years")

    yearly_data = filtered.dropna(
        subset=["year_added"]
    ).copy()

    yearly_data = yearly_data[
        yearly_data["year_added"] >= 2008
    ]

    yearly = (
        yearly_data["year_added"]
        .value_counts()
        .sort_index()
    )

    if not yearly.empty:
        fig, ax = plt.subplots(figsize=(12, 5))

        ax.plot(
            yearly.index,
            yearly.values,
            color="#E50914",
            linewidth=2.5,
            marker="o"
        )

        ax.fill_between(
            yearly.index,
            yearly.values,
            color="#E50914",
            alpha=0.1
        )

        ax.set_xlabel("Year")
        ax.set_ylabel("Number of Titles")
        ax.set_title("Netflix Titles Added by Year")

        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("No date-added data is available.")

    st.divider()

    st.subheader("Movies vs TV Shows Growth")

    fig, ax = plt.subplots(figsize=(12, 5))

    for content_type, color in [
        ("Movie", "#E50914"),
        ("TV Show", "#4A4A4A")
    ]:
        sub = filtered[
            (filtered["type"] == content_type)
            &
            (filtered["year_added"] >= 2008)
        ]

        trend = (
            sub["year_added"]
            .value_counts()
            .sort_index()
        )

        if not trend.empty:
            ax.plot(
                trend.index,
                trend.values,
                color=color,
                linewidth=2,
                marker="o",
                label=content_type
            )

    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Titles")
    ax.legend()

    st.pyplot(fig)
    plt.close(fig)


# ============================================================
# TAB 3 — GEOGRAPHY
# ============================================================
with tab3:
    st.subheader("Top 10 Content-Producing Countries")

    countries = (
        filtered["country"]
        .fillna("Unknown")
        .astype(str)
        .str.split(", ")
        .explode()
    )

    countries = (
        countries[countries != "Unknown"]
        .value_counts()
        .head(10)
    )

    if not countries.empty:
        fig, ax = plt.subplots(figsize=(10, 6))

        sns.barplot(
            x=countries.values,
            y=countries.index,
            hue=countries.index,
            palette="Reds_r",
            ax=ax,
            legend=False
        )

        for i, value in enumerate(countries.values):
            ax.text(
                value + max(countries.values) * 0.01,
                i,
                f"{value:,}",
                va="center",
                fontsize=10
            )

        ax.set_xlabel("Number of Titles")
        ax.set_ylabel("Country")

        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("No country data is available.")


# ============================================================
# TAB 4 — GENRES
# ============================================================
with tab4:
    st.subheader("Top 10 Genres")

    genres = (
        filtered["listed_in"]
        .fillna("Unknown")
        .astype(str)
        .str.split(", ")
        .explode()
    )

    genres = (
        genres[genres != "Unknown"]
        .value_counts()
        .head(10)
    )

    if not genres.empty:
        fig, ax = plt.subplots(figsize=(10, 6))

        sns.barplot(
            x=genres.values,
            y=genres.index,
            hue=genres.index,
            palette="Reds_r",
            ax=ax,
            legend=False
        )

        for i, value in enumerate(genres.values):
            ax.text(
                value + max(genres.values) * 0.01,
                i,
                f"{value:,}",
                va="center",
                fontsize=10
            )

        ax.set_xlabel("Number of Titles")
        ax.set_ylabel("Genre")

        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("No genre data is available.")

    st.divider()

    st.subheader("Top 10 Longest Movies")

    movies_only = filtered[
        filtered["type"] == "Movie"
    ].dropna(
        subset=["duration_int"]
    )

    if not movies_only.empty:
        longest = (
            movies_only
            .nlargest(10, "duration_int")
            [["title", "duration_int"]]
        )

        fig, ax = plt.subplots(figsize=(10, 5))

        sns.barplot(
            x=longest["duration_int"],
            y=longest["title"],
            hue=longest["title"],
            palette="Reds_r",
            ax=ax,
            legend=False
        )

        for i, value in enumerate(
            longest["duration_int"]
        ):
            ax.text(
                value + 2,
                i,
                f"{int(value)} min",
                va="center",
                fontsize=9
            )

        ax.set_xlabel("Duration (minutes)")
        ax.set_ylabel("Title")

        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("No movie duration data is available.")


# ============================================================
# TAB 5 — SCORES & RATINGS
# ============================================================
with tab5:
    st.subheader("⭐ Critical Reception Analysis")

    if imdb_col is not None:
        c1, c2 = st.columns(2)

        # ----------------------------------------------------
        # IMDb vs Popularity
        # ----------------------------------------------------
        with c1:
            st.markdown("**IMDb Score vs. Popularity**")

            pop_col = None

            for column in filtered.columns:
                column_lower = column.lower()

                if (
                    "popularity" in column_lower
                    or "vote_count" in column_lower
                    or "votes" in column_lower
                ):
                    pop_col = column
                    break

            if pop_col is not None:
                plot_df = filtered[
                    [imdb_col, pop_col, "type"]
                ].copy()

                plot_df[imdb_col] = numeric_column(
                    plot_df,
                    imdb_col
                )

                plot_df[pop_col] = numeric_column(
                    plot_df,
                    pop_col
                )

                plot_df = plot_df.dropna(
                    subset=[imdb_col, pop_col]
                )

                if not plot_df.empty:
                    fig, ax = plt.subplots(figsize=(8, 6))

                    sns.scatterplot(
                        data=plot_df,
                        x=imdb_col,
                        y=pop_col,
                        hue="type",
                        alpha=0.6,
                        ax=ax
                    )

                    ax.set_xlabel("IMDb Score")
                    ax.set_ylabel("Popularity / Votes")

                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.info(
                        "Not enough overlapping IMDb and popularity data."
                    )
            else:
                st.info(
                    "Popularity/vote data is not available."
                )

        # ----------------------------------------------------
        # Top IMDb Rated Movies
        # ----------------------------------------------------
        with c2:
            st.markdown("**Top 10 Highest Rated on IMDb**")

            top_rated = filtered[
                filtered["type"] == "Movie"
            ].copy()

            top_rated[imdb_col] = numeric_column(
                top_rated,
                imdb_col
            )

            top_rated = (
                top_rated
                .dropna(subset=[imdb_col])
                .nlargest(10, imdb_col)
            )

            if not top_rated.empty:
                fig, ax = plt.subplots(figsize=(8, 6))

                sns.barplot(
                    x=top_rated[imdb_col],
                    y=top_rated["title"],
                    hue=top_rated["title"],
                    palette="Reds_r",
                    ax=ax,
                    legend=False
                )

                for i, value in enumerate(
                    top_rated[imdb_col]
                ):
                    ax.text(
                        value + 0.1,
                        i,
                        f"{value:.1f}",
                        va="center",
                        fontsize=9
                    )

                ax.set_xlabel("IMDb Score")

                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info(
                    "No valid IMDb movie scores are available."
                )

        # ----------------------------------------------------
        # IMDb vs Rotten Tomatoes
        # ----------------------------------------------------
        st.divider()

        st.markdown(
            "**Compare Critics: IMDb vs Rotten Tomatoes**"
        )

        rotten_col = None

        for column in filtered.columns:
            column_lower = column.lower()

            if (
                "rotten" in column_lower
                or "tomatoes" in column_lower
            ):
                rotten_col = column
                break

        if rotten_col is not None:
            compare_df = filtered[
                [imdb_col, rotten_col]
            ].copy()

            compare_df[imdb_col] = numeric_column(
                compare_df,
                imdb_col
            )

            compare_df[rotten_col] = numeric_column(
                compare_df,
                rotten_col
            )

            compare_df = compare_df.dropna(
                subset=[imdb_col, rotten_col]
            )

            if not compare_df.empty:
                fig, ax = plt.subplots(figsize=(10, 5))

                ax.scatter(
                    compare_df[rotten_col],
                    compare_df[imdb_col] * 10,
                    alpha=0.5,
                    color="#E50914"
                )

                ax.set_xlabel(
                    "Rotten Tomatoes Score (%)"
                )

                ax.set_ylabel(
                    "IMDb Score (scaled to 100)"
                )

                ax.set_title(
                    "IMDb vs Rotten Tomatoes Score Comparison"
                )

                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info(
                    "Not enough data for critic comparison."
                )
        else:
            st.info(
                "Rotten Tomatoes data is not available."
            )

    else:
        st.warning(
            "⚠️ IMDb score data is not available in the "
            "current datasets."
        )

        with st.expander("🔧 Debug: Loaded columns"):
            st.write(list(df.columns))


# ============================================================
# TAB 6 — VIEWERSHIP
# ============================================================
with tab6:
    st.subheader("📺 Official Viewership Analytics")

    if "total_hours_viewed" in filtered.columns:
        view_data = filtered.dropna(
            subset=["total_hours_viewed"]
        ).copy()

        view_data["total_hours_viewed"] = numeric_column(
            view_data,
            "total_hours_viewed"
        )

        top_viewed = (
            view_data
            .dropna(subset=["total_hours_viewed"])
            .nlargest(15, "total_hours_viewed")
        )

        if not top_viewed.empty:
            fig, ax = plt.subplots(figsize=(10, 6))

            sns.barplot(
                x=top_viewed["total_hours_viewed"],
                y=top_viewed["title"],
                hue=top_viewed["title"],
                palette="Reds_r",
                ax=ax,
                legend=False
            )

            max_value = top_viewed[
                "total_hours_viewed"
            ].max()

            for i, value in enumerate(
                top_viewed["total_hours_viewed"]
            ):
                if value > 1_000_000:
                    label = f"{value / 1_000_000:.1f}M"
                else:
                    label = f"{value / 1_000:.0f}K"

                ax.text(
                    value + max_value * 0.01,
                    i,
                    label,
                    va="center",
                    fontsize=9
                )

            ax.set_xlabel("Total Hours Viewed")
            ax.set_ylabel("Title")
            ax.set_title(
                "Top 15 Most-Watched Titles on Netflix"
            )

            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info(
                "No valid viewership data is available."
            )
    else:
        st.warning(
            "⚠️ Viewership data is not available. "
            "Check all-weeks-global.csv."
        )


# ============================================================
# TAB 7 — EXPLORE
# ============================================================
with tab7:
    st.subheader("🔍 Explore the Catalog")

    search = st.text_input(
        "Search by title",
        placeholder="e.g. Stranger Things, Breaking Bad..."
    )

    view = filtered.copy()

    if search:
        view = view[
            view["title"]
            .astype(str)
            .str.contains(
                search,
                case=False,
                na=False
            )
        ]

    st.markdown(
        f"Showing **{len(view):,}** titles"
    )

    display_cols = [
        "title",
        "type",
        "country",
        "release_year",
        "rating",
        "duration"
    ]

    if imdb_col is not None:
        display_cols.append(imdb_col)

    if "total_hours_viewed" in view.columns:
        display_cols.append(
            "total_hours_viewed"
        )

    display_cols = [
        column
        for column in display_cols
        if column in view.columns
    ]

    rename_map = {
        "title": "Title",
        "type": "Type",
        "country": "Country",
        "release_year": "Year",
        "rating": "Rating",
        "duration": "Duration",
        "total_hours_viewed": "Hours Viewed"
    }

    if imdb_col is not None:
        rename_map[imdb_col] = "IMDb"

    st.dataframe(
        view[display_cols].rename(
            columns=rename_map
        ),
        width="stretch",
        height=500
    )

    csv_data = view[
        display_cols
    ].to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Filtered Data as CSV",
        data=csv_data,
        file_name="netflix_filtered.csv",
        mime="text/csv"
    )


# ============================================================
# FOOTER
# ============================================================
st.divider()

st.markdown(
    """
    <div style='text-align: center; color: #888; font-size: 0.85rem;'>
        Built with Python, Pandas, Matplotlib & Streamlit<br>
        Data: Kaggle Netflix Movies and TV Shows Dataset + Enriched Sources
    </div>
    """,
    unsafe_allow_html=True
)
