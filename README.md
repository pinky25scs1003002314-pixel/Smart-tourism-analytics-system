# Smart Tourism Analytics System

A beginner-friendly academic project that demonstrates tourism data cleaning, preprocessing, exploratory analysis, machine-learning regression, interactive charts and an India destination map.

## Features
- Upload a tourism CSV or use the included sample dataset.
- Clean duplicate rows, missing values, numeric columns and text fields.
- Calculate total visitors.
- Explore visitor totals by state, month and destination.
- Train a Random Forest regression model to estimate visitor counts.
- Evaluate the model with MAE, RMSE and R².
- View destinations on an interactive map.
- Download the cleaned CSV.

## Setup
1. Install Python 3.10 or newer.
2. Open a terminal in this project folder.
3. Create and activate a virtual environment (recommended):

   Windows:
   ```bash
   python -m venv .venv
   .venv\\Scripts\\activate
   ```

   macOS/Linux:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

4. Install libraries:
   ```bash
   pip install -r requirements.txt
   ```

5. Run:
   ```bash
   streamlit run app.py
   ```

## Dataset columns
The CSV must contain:
`year, month, state, city, destination, domestic_visitors, international_visitors, latitude, longitude`

The included CSV contains illustrative sample records for demonstrating the workflow. Replace it with a reliable, properly cited tourism dataset before making real-world claims or submitting final research results.

## Important limitations
- The included dataset is a small sample for software demonstration, not official government statistics.
- The ML model is a baseline Random Forest model. Its metrics are only meaningful when trained and tested on a sufficiently large, representative dataset.
- The model estimates visitor counts from year, month, state and destination. It does not establish causal relationships.
- The map displays the coordinates supplied in the dataset.
