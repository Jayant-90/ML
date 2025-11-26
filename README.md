## Rose LIBS Classification – Streamlit + Render

This project wraps your notebook model in a deployable Streamlit web app and exposes it via Render.

### Project structure

- `Ml_Project.ipynb` – original exploratory notebook and model.
- `train_model.py` – script that trains an SVM model and saves `model.pkl`.
- `streamlit_app.py` – Streamlit frontend for single and batch predictions.
- `requirements.txt` – Python dependencies for local use and Render.
- `render.yaml` – Render service definition for deployment.

### 1. Prepare the data and train the model locally

1. Place your CSV file `roses_LIBS_50000.csv` in the same folder as these files.
2. (Optional but recommended) Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate  # on Windows
pip install -r requirements.txt
```

3. Train the model and create `model.pkl`:

```bash
python train_model.py
```

This will:

- Load `roses_LIBS_50000.csv`
- Train a StandardScaler + SVC (RBF) pipeline
- Print accuracy and classification metrics
- Save the trained pipeline to `model.pkl`

### 2. Run the Streamlit app locally

After `model.pkl` has been created:

```bash
streamlit run streamlit_app.py
```

Then open the URL shown in the terminal (typically `http://localhost:8501`).

The app supports:

- **Single Prediction**: manually enter LIBS feature values and get a predicted class.
- **Batch Prediction**: upload a CSV with the same LIBS feature columns (excluding `Class`) and download predictions.

### 3. Deploy to Render

1. Push this folder to a Git repository (GitHub, GitLab, etc.).
2. In Render:
   - Create a **New Web Service** from your repo.
   - Render will detect `render.yaml` and configure the service.
3. The build will run:
   - `pip install -r requirements.txt`
4. The service will start with:
   - `streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0`

> **Note:** You must **include `model.pkl` in the repo** (after training locally) so Render can load it. Alternatively, mount persistent storage and train on Render, but the simplest is to commit `model.pkl`.

### 4. Regenerating or improving the model

If you change the notebook or want to re-train:

1. Update `train_model.py` or your data as needed.
2. Re-run:

```bash
python train_model.py
```

3. Commit the new `model.pkl` and deploy again to Render.


