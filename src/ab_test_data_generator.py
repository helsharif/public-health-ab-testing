import numpy as np
import pandas as pd
import os

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def generate_cdc_ab_dataset(
    n=20000,
    seed=42,
    treatment_rate=0.5
):
    """
    Generates a synthetic CDC-style outreach A/B test dataset.

    Key properties:
    - message_variant is randomized (A/B), so A/B comparisons are unbiased.
    - Variant B has a positive average treatment effect on scheduling.
    - Treatment effect is heterogeneous (larger for some groups).
    - Includes realistic covariates for XGBoost modeling.
    """
    rng = np.random.default_rng(seed)

    # --- Core demographics / baseline attributes ---
    age = rng.integers(18, 86, size=n)
    sex = rng.choice(["F", "M"], size=n, p=[0.52, 0.48])

    # Simple region / site marker (e.g., Atlanta metro vs nearby counties)
    region = rng.choice(["ATL-Core", "ATL-Metro", "North-GA", "South-GA"], size=n, p=[0.35, 0.35, 0.15, 0.15])

    # Risk score (0-1) capturing comorbidities / vulnerability
    # Correlated with age and some randomness
    risk_score = np.clip(
        0.15 + 0.007 * (age - 18) + rng.normal(0, 0.12, size=n),
        0, 1
    )

    # Prior engagement features
    prior_cdc_interactions_90d = rng.poisson(lam=np.clip(0.6 + 2.0 * risk_score, 0.2, 4.0), size=n)
    prior_appointments_1y = rng.poisson(lam=np.clip(0.3 + 1.3 * risk_score, 0.1, 3.0), size=n)
    missed_appointments_1y = rng.binomial(n=prior_appointments_1y + 1, p=np.clip(0.08 + 0.18*(1-risk_score), 0.05, 0.35))

    # Socio-behavioral proxy: "barriers" (transport/time/tech friction)
    # Higher barriers reduce scheduling
    barriers_index = np.clip(
        rng.normal(loc=0.0, scale=1.0, size=n)
        + 0.8 * (region == "South-GA").astype(int)
        + 0.25 * (region == "North-GA").astype(int)
        + 0.15 * (age > 70).astype(int),
        -3, 3
    )

    # Delivery channel
    channel = rng.choice(["SMS", "Email", "IVR"], size=n, p=[0.65, 0.25, 0.10])

    # Time sent (hour) and weekday (0=Mon..6=Sun)
    send_hour = rng.integers(8, 21, size=n)  # 8am-8pm
    weekday = rng.integers(0, 7, size=n)

    # Randomized assignment
    message_variant = rng.choice(["A", "B"], size=n, p=[1 - treatment_rate, treatment_rate])

    # --- Simulate intermediate engagement (opens/clicks) ---
    # Engagement depends on channel, barriers, prior interactions, and variant (B slightly better)
    base_open = (
        -0.4
        + 0.25 * (channel == "SMS").astype(int)
        + 0.10 * (channel == "Email").astype(int)
        - 0.22 * barriers_index
        + 0.08 * np.log1p(prior_cdc_interactions_90d)
        + 0.10 * (send_hour >= 17).astype(int)  # evenings slightly better
        + 0.08 * (weekday >= 5).astype(int)      # weekends slightly better
    )
    open_prob = sigmoid(base_open + 0.10 * (message_variant == "B").astype(int))
    opened = rng.binomial(1, open_prob)

    # Click-through conditional on open
    base_click = (
        -1.2
        + 0.45 * opened
        + 0.18 * (channel == "SMS").astype(int)
        - 0.20 * barriers_index
        + 0.10 * np.log1p(prior_appointments_1y)
    )
    click_prob = sigmoid(base_click + 0.15 * (message_variant == "B").astype(int))
    clicked = rng.binomial(1, click_prob)

    # --- Primary outcome: scheduled within 7 days ---
    # Baseline log-odds influenced by risk, barriers, history, engagement.
    baseline = (
        -2.0
        + 0.95 * risk_score
        - 0.55 * barriers_index
        + 0.25 * np.log1p(prior_appointments_1y)
        - 0.35 * (missed_appointments_1y > 0).astype(int)
        + 0.55 * opened
        + 0.75 * clicked
        + 0.12 * (channel == "SMS").astype(int)
    )

    # Treatment effect (B) with heterogeneity:
    # - B helps more for high barriers (friction-reducing link)
    # - B helps more for older + higher risk (personalization)
    treat = (message_variant == "B").astype(int)
    hetero_effect = (
        0.18 * treat
        + 0.10 * treat * (barriers_index > 1.0).astype(int)
        + 0.08 * treat * (age >= 60).astype(int)
        + 0.10 * treat * (risk_score >= 0.6).astype(int)
    )

    schedule_prob = sigmoid(baseline + hetero_effect)
    scheduled_7d = rng.binomial(1, schedule_prob)

    # --- Secondary outcome: completed within 30 days ---
    # Completion depends strongly on scheduling + barriers + risk + missed history
    comp_base = (
        -1.7
        + 2.2 * scheduled_7d
        + 0.35 * risk_score
        - 0.45 * barriers_index
        - 0.25 * (missed_appointments_1y > 0).astype(int)
        + 0.10 * np.log1p(prior_appointments_1y)
    )
    completed_30d = rng.binomial(1, sigmoid(comp_base))

    df = pd.DataFrame({
        "person_id": np.arange(1, n + 1),
        "age": age,
        "sex": sex,
        "region": region,
        "risk_score": np.round(risk_score, 3),
        "barriers_index": np.round(barriers_index, 3),
        "channel": channel,
        "weekday": weekday,
        "send_hour": send_hour,
        "prior_cdc_interactions_90d": prior_cdc_interactions_90d,
        "prior_appointments_1y": prior_appointments_1y,
        "missed_appointments_1y": missed_appointments_1y,
        "message_variant": message_variant,
        "opened": opened,
        "clicked": clicked,
        "scheduled_7d": scheduled_7d,
        "completed_30d": completed_30d,
    })

    return df

if __name__ == "__main__":
    df = generate_cdc_ab_dataset(n=20000, seed=42)
    print(df.head())
    print("\nScheduling rate by variant:")
    print(df.groupby("message_variant")["scheduled_7d"].mean())

    # Ensure data directory exists
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    output_path = os.path.join(data_dir, "cdc_outreach_ab_synthetic.csv")
    df.to_csv(output_path, index=False)

    print(f"\nSaved: {output_path}")