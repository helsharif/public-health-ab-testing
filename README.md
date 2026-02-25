# Public Health A/B Testing & Predictive Analytics (XGBoost)

<p align="center">
  <img src="images/outreach_campaign_infographic.png" alt="CDC-Style Outreach Campaign A/B Test" width="900">
</p>

**End-to-end experimentation + predictive modeling on simulated CDC-style public health outreach data.**
This project demonstrates how to design and analyze randomized A/B tests using **permutation testing** and how to extend experimental insights with **predictive analytics (XGBoost)** to identify which populations are most likely to engage with public health interventions.

> **Core focus:** Experimental design, causal thinking, and ML-driven decision support in a public health / healthcare context.

---

## 🚀 Project Highlights

* **Task:** A/B testing + predictive analytics
* **Domain:** Public health outreach (CDC-style messaging campaigns)
* **Experiment:** Permutation-based A/B testing
* **ML (planned):** XGBoost classification for engagement prediction
* **Tech Stack:** Python, Pandas, NumPy, Plotly, Scikit-learn, XGBoost
* **Outputs:** Statistical testing, uplift visualization, reproducible notebooks

---

## 🧠 Motivation & Problem Statement

Public health agencies often deploy large-scale outreach campaigns (SMS, email, IVR) to encourage preventive care such as vaccination scheduling. However, choosing which message strategy to deploy at scale is non-trivial.

**Key questions:**

* Does a personalized message outperform a standard reminder?
* How large is the uplift?
* Which population segments are most likely to respond?
* Can we predict engagement to optimize future campaigns?

This project simulates a randomized CDC-style outreach experiment and applies:

* **Permutation testing** to quantify causal impact
* **Predictive modeling (XGBoost)** to forecast scheduling behavior

---

## 📊 Dataset Description

The dataset is **synthetically generated** to resemble a realistic CDC outreach scenario and includes:

* Demographics: `age`, `sex`, `region`
* Risk & barriers: `risk_score`, `barriers_index`
* Engagement history: `prior_cdc_interactions_90d`, `prior_appointments_1y`, `missed_appointments_1y`
* Experiment design: `message_variant` (A vs B)
* Engagement signals: `opened`, `clicked`
* Outcomes:

  * `scheduled_7d` (primary A/B test outcome)
  * `completed_30d` (secondary downstream outcome)

> **Important:** The treatment assignment (`message_variant`) is randomized, enabling valid causal inference via permutation testing.

Dataset location:

```
data/cdc_outreach_ab_synthetic.csv
```

---

## 🔬 A/B Testing Methodology (Permutation Test)

Instead of relying on parametric assumptions, this project uses a **permutation test** to construct the null distribution of lift under random assignment:

**Steps:**

1. Compute observed lift:
   [
   \text{Lift} = \frac{p_B - p_A}{p_A}
   ]
2. Shuffle treatment labels thousands of times
3. Recompute lift for each shuffle
4. Estimate p-value as the probability of observing a lift as extreme as the real one
5. Visualize the null distribution with Plotly

This approach:

* Requires minimal assumptions
* Is robust and interpretable
* Mirrors real-world experimentation workflows

---

## 🤖 Predictive Analytics (XGBoost – Planned Extension)

A follow-up notebook will extend the analysis by training an **XGBoost classifier** to predict:

* Probability of scheduling within 7 days (`scheduled_7d`)
* (Optional) Probability of completion within 30 days (`completed_30d`)

**Use cases:**

* Target high-likelihood responders
* Identify high-barrier populations
* Support resource allocation and campaign design

**Planned techniques:**

* Feature encoding
* Train/validation splits
* ROC-AUC, precision/recall
* Feature importance / SHAP explanations

---

## 📁 Repository Structure

```
.
├── data/
│   └── cdc_outreach_ab_synthetic.csv
├── images/
│   └── outreach_campaign_infographic.png
├── src/
│   └── ab_test_data_generator.py
├── 01_data_generation_and_AB_testing.ipynb
├── datasci_xgb_skl_env001.yml
└── README.md
```


---

## ▶️ How to Run This Project

### 1️⃣ Create the Conda Environment

```bash
conda env create -f datasci_xgb_skl_env001.yml
conda activate datasci_xgb_skl_env001
```

### 2️⃣ Launch Jupyter

```bash
jupyter notebook
```

### 3️⃣ Run the Notebook

Open:

```
01_data_generation_and_AB_testing.ipynb
```

This notebook:

* Generates the synthetic dataset
* Computes observed uplift
* Runs the permutation-based A/B test
* Visualizes the null distribution
* Interprets results in a public health context

---

## 📈 Example Outputs

* Null distribution of relative lift
* Observed uplift indicator
* Statistical significance via permutation p-value
* Outreach campaign infographic (Message A vs Message B)

---

## ⚙️ Tech Stack

* **Language:** Python
* **Data:** Pandas, NumPy
* **Visualization:** Plotly
* **Statistics:** A/B testing
* **ML (planned):** XGBoost
* **Reproducibility:** Conda environment YAML

---

## ⚠️ Disclaimer

This project is for **portfolio and educational purposes only**.
The dataset is **synthetic** and does **not** represent real CDC data or real individuals.

---

## 👤 Author

**Husayn El Sharif**
Senior Data Scientist / Machine Learning Engineer

---

## 📌 Portfolio Relevance

This project highlights:

* A/B testing with causal reasoning
* Permutation-based statistical inference
* Visualization of experimental results
* Healthcare/public health analytics framing
* Foundations for ML-driven targeting (XGBoost)

---
