# Sanitary Flow Decomposition: Client-Friendly Explanation

## What this model is trying to do
The plant sees **total flow** every hour.  
This project separates that total into two parts:

1. **Sanitary flow**: normal wastewater from homes/businesses.
2. **Extraneous flow**: unwanted extra water entering the system (for example groundwater infiltration and tide-related effects).

In simple form:

`Total Plant Flow = Sanitary Flow + Extraneous Flow`

The model does this decomposition hour-by-hour, while checking that daily totals still make sense.

---

## What data are used
- Hourly plant flow
- Hourly tide level
- Hourly rainfall
- Daily water usage

These are aligned on a common hourly timeline.

---

## How travel-time lag to the plant is handled
Water used by customers does not all arrive at the plant instantly.  
The code models this by blending:

- same-day usage, and
- previous-day usage

using a configurable split (default is about two-thirds same-day, one-third previous-day).  
This creates an adjusted usage signal that better matches when sanitary flow actually reaches the plant.

Why this matters:
- If lag is ignored, sanitary flow can look too early/late.
- With lag adjustment, the sanitary estimate is better aligned with observed plant behavior.

---

## How daily usage is converted to hourly sanitary flow
This is one of the most important parts of the method.

The challenge:
- Water usage data are daily totals.
- Plant flow data are hourly.
- So the model must convert one daily sanitary value into a realistic 24-hour sanitary pattern.

The process is:

1. Build an adjusted daily usage signal  
The model first applies travel-time lag to daily usage (same-day + previous-day blend) so usage timing better matches plant timing.

2. Convert adjusted usage into daily sanitary flow  
For each season (DJF/MAM/JJA/SON), the model estimates a seasonal RF (return factor).  
Then:
- `sanitary_daily = RF × adjusted_daily_usage`

This gives one sanitary flow level per day.

3. Learn an hourly shape factor from dry periods  
On dry days, the model learns an hourly “shape” (for example low overnight, higher in daytime peaks).  
This shape is called `s_shape`.

4. Force hourly shape to conserve daily total  
The learned hourly shape is normalized so its daily average is 1.0.  
That means when hourly values are summed/averaged over the day, they still match `sanitary_daily`.

5. Allocate daily sanitary total into 24 hours  
The final hourly sanitary estimate is:
- `sanitary_hourly_pred = sanitary_daily × sanitary_shape`

Because shape is normalized daily, this allocation preserves the daily sanitary volume while giving realistic hourly variation.

6. Compute extraneous flow from remaining share  
After sanitary is allocated hourly, the remaining flow needed to match plant observations is assigned to extraneous flow.

Why this matters for clients:
- It prevents “double counting” or drifting totals.
- It allows daily customer-use information to drive hourly decomposition without inventing extra volume.
- It keeps hourly sanitary patterns realistic while staying consistent with daily mass balance.

---

## How rain response lag is handled (dry vs wet classification)
Rain does not always affect plant flow immediately.  
The code checks for rain events and then looks ahead in a lag window (default roughly 12 to 24 hours) to see if plant flow rises enough.

If rain is followed by a meaningful plant response in that lag window, those hours are treated as rain-affected (“wet”).  
Hours not affected are treated as “dry.”

Why this matters:
- Dry periods are used to learn baseline sanitary and extraneous behavior.
- Wet periods are handled separately so storm effects do not distort dry-weather learning.

---

## How tidal lag is handled
The effect of tide at the plant is delayed.  
The code automatically estimates a **tidal lag** by testing candidate lags (up to 24 hours) and selecting the lag with strongest relationship between tide and dry-weather plant behavior.

Then it builds tide features at that selected lag (and several fixed lags like 1, 2, 3, 6, 12 hours).

Why this matters:
- The model can capture “tide now, plant response later,” which is typical in coastal/influenced systems.

---

## How the groundwater (GW) proxy is created
Groundwater influence changes slowly, so the model creates a smooth **GW proxy**:

1. Start from dry-day extraneous flow estimates.
2. Smooth them using a rolling median over multiple days.
3. Interpolate to hourly values.
4. Also compute GW change rate (hour-to-hour slope).

This gives the model a slow-moving background signal for infiltration pressure.

Why this matters:
- Extraneous flow is not only short-term (like tide); some of it is persistent and slow-varying.

---

## How training and testing work
The model uses dry days only for learning decomposition behavior.

### Training set vs test set
- Dry days are split by time into:
  - **Training** (earlier portion)
  - **Holdout test** (later portion)

“Holdout” means the model does **not** train on those days; they are used to check performance honestly.

### What is trained
Two linked sub-models are trained iteratively:

1. **Extraneous model** (predicts hourly extraneous shape/level from tide, GW proxy, calendar patterns, etc.)
2. **Sanitary-shape model** (predicts how sanitary flow is distributed across hours of the day)

The process alternates between these two so each improves the other.

### What is measured on test data
On holdout dry days, the code compares reconstructed plant flow vs observed plant flow using:

- **MAE** (mean absolute error): average size of errors.
- **RMSE** (root mean squared error): like MAE but penalizes large misses more.
- **R²**: how much variation is explained (closer to 1 is better).

It also reports night-vs-day metrics so we can see where performance improves or worsens.

---

## What “is being tested” in this project
The model is testing whether known physical behaviors are present and usable:

1. **Lagged response behavior**
   - rain-to-plant lag
   - tide-to-plant lag
   - usage travel-time lag

2. **Slow groundwater influence**
   - represented by the GW proxy and its trend

3. **Hour-of-day sanitary pattern**
   - especially lower sanitary activity during wee hours

4. **Seasonal behavior**
   - return factor (RF) is estimated by season and can vary through the year

The final objective is not only good fit metrics, but also physically plausible decomposition.

---

## Plain-language glossary
- **Lag**: a delay between cause and observed effect.
- **Proxy**: a substitute variable used to represent something not measured directly.
- **Feature**: an input signal the model uses (for example tide lag or GW proxy).
- **Holdout test**: unseen data used only for evaluation, not training.
- **RF (Return Factor)**: scaling factor linking usage to expected sanitary flow.
- **Decomposition**: splitting total flow into sanitary and extraneous components.

---

## Key takeaway for non-technical stakeholders
This workflow is designed to be both:

1. **Data-driven** (machine learning learns patterns from history), and  
2. **Physics-aware** (it explicitly handles travel-time lag, rain lag, tide lag, and slow groundwater effects).

So the model is not a “black box only”; it combines engineering logic with statistical learning.
