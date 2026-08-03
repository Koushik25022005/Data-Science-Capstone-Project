# Data Science Capstone Project — SpaceX Falcon 9 Landing Prediction

This repository contains the notebooks and app for the **IBM Data Science Professional Certificate Capstone**, centered on predicting whether the first stage of a SpaceX Falcon 9 rocket will land successfully. Since SpaceX advertises Falcon 9 launches at a fraction of the cost of competitors — largely because it reuses the first stage — predicting a successful landing helps estimate the cost of a launch and can inform competing bids.

The project follows the full data science lifecycle: data collection, wrangling, exploratory data analysis (SQL + visualization), interactive geospatial analysis, an interactive dashboard, and machine learning classification.

## Project Structure

| File | Description |
|---|---|
| `labs-jupyter-spacex-Data wrangling.ipynb` | Data wrangling lab: cleans the raw SpaceX launch dataset and engineers the landing outcome label used for supervised learning. |
| `jupyter-labs-eda-sql-coursera_sqllite.ipynb` | Exploratory Data Analysis using SQL (SQLite) — querying launch sites, payload mass, mission outcomes, and booster versions. |
| `edadataviz.ipynb` | Exploratory Data Analysis and feature engineering using visualization (Matplotlib/Seaborn) to uncover patterns relevant to landing outcomes. |
| `lab_jupyter_launch_site_location.ipynb` | Interactive visual analytics with **Folium** — mapping launch sites, marker clustering, and proximity analysis (coastline, cities, railways) using Haversine distance. |
| `DV0101EN-Exercise-Generating-Maps-in-Python.ipynb` | Supplementary exercise on generating maps and visualizing geospatial data with Folium. |
| `SpaceX_Machine Learning Prediction_Part_5.ipynb` | Builds the machine learning pipeline: standardizes features, splits train/test data, and tunes/evaluates classifiers (Logistic Regression, SVM, Decision Tree, KNN) via `GridSearchCV`. |
| `spacex-dash-app (1).py` | A **Plotly Dash** web application providing an interactive dashboard for exploring launch records by site and payload mass. |

## Key Objectives

1. Collect and clean SpaceX launch data.
2. Perform EDA using SQL and visualization to identify patterns affecting landing success.
3. Visualize launch site locations and proximities interactively with Folium.
4. Build an interactive dashboard (Dash/Plotly) to explore launch outcomes by site and payload.
5. Train and evaluate classification models to predict first-stage landing success.

## Tech Stack

- **Language:** Python
- **Data handling:** pandas, NumPy, SQLite
- **Visualization:** Matplotlib, Seaborn, Folium, Plotly Express
- **Web app / dashboard:** Dash
- **Machine learning:** scikit-learn (Logistic Regression, SVM, Decision Tree, KNN, `GridSearchCV`, `train_test_split`)

## Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn folium dash plotly scikit-learn prettytable
```

Jupyter Notebook or JupyterLab is required to run the `.ipynb` files.

### Running the Notebooks

Open any notebook in Jupyter and run the cells sequentially:

```bash
jupyter notebook
```

Recommended order: data wrangling → SQL EDA → visualization EDA → launch site location analysis → machine learning prediction.

### Running the Dashboard

```bash
python "spacex-dash-app (1).py"
```

The app requires `spacex_launch_dash.csv` (the cleaned launch dataset) to be present in the same directory. Once running, open the local URL shown in the terminal (typically `http://127.0.0.1:8050`) to view the interactive dashboard, which lets you filter launch records by site and payload mass range.

## Results

The machine learning pipeline evaluates multiple classifiers to identify the model that best predicts Falcon 9 first-stage landing outcomes, with model selection driven by cross-validated accuracy on the test set.

## Acknowledgements

This project is based on labs from the **IBM Data Science Professional Certificate** (Applied Data Science Capstone) on Coursera / Skills Network.

## Author

**Koushik Sripathi**
GitHub: [Koushik25022005](https://github.com/Koushik25022005)
