"""
===============================================================================
STEP 2: TEMPORAL PATTERN ANALYSIS
===============================================================================
RAC 4 - Traffic Flow Prediction Using Machine Learning

This script answers the committee question:
  "Why does the line go down for a particular year or month?"

We analyse:
  1. Monthly traffic trends (2022-2024) with explanations for dips/spikes
  2. Day-of-week patterns
  3. Seasonal (quarterly) patterns
  4. Year-over-year comparison
  5. Area-wise temporal patterns
  6. Weekend vs Weekday analysis
  7. Weather impact on traffic
  8. Written explanation for every notable fluctuation
===============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
import json
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['font.size'] = 11

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(SCRIPT_DIR, 'outputs', 'processed_data')
OUT_DIR = os.path.join(SCRIPT_DIR, 'outputs', '02_Temporal_Analysis')
os.makedirs(OUT_DIR, exist_ok=True)

# Load cleaned data
df = pd.read_csv(os.path.join(PROCESSED_DIR, 'cleaned_full_dataset.csv'))
df['Date'] = pd.to_datetime(df['Date'])

TARGET = 'Traffic Volume'

print("=" * 70)
print("STEP 2: TEMPORAL PATTERN ANALYSIS")
print("=" * 70)
print(f"  Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
print(f"  Total records: {len(df)}")

explanations = []  # Collect all explanations

# ============================================================================
# 2.1 OVERALL MONTHLY TRAFFIC TREND
# ============================================================================
print("\n" + "-" * 50)
print("2.1: Monthly Average Traffic Volume Over Time")
print("-" * 50)

df['YearMonth'] = df['Date'].dt.to_period('M')
monthly = df.groupby('YearMonth')[TARGET].agg(['mean', 'std', 'count']).reset_index()
monthly['YearMonth_dt'] = monthly['YearMonth'].dt.to_timestamp()

fig, ax = plt.subplots(figsize=(16, 7))
ax.plot(monthly['YearMonth_dt'], monthly['mean'], 'b-o', markersize=5, linewidth=2, label='Monthly Mean')
ax.fill_between(monthly['YearMonth_dt'],
                monthly['mean'] - monthly['std'],
                monthly['mean'] + monthly['std'],
                alpha=0.2, color='blue', label='±1 Std Dev')

# Calculate month-over-month changes
monthly['change'] = monthly['mean'].pct_change() * 100
monthly['abs_change'] = monthly['mean'].diff()

# Annotate significant drops/rises (>10% change)
for idx, row in monthly.iterrows():
    if idx == 0:
        continue
    if abs(row['change']) > 10:
        color = 'red' if row['change'] < 0 else 'green'
        marker = 'v' if row['change'] < 0 else '^'
        ax.annotate(f"{row['change']:+.1f}%",
                    xy=(row['YearMonth_dt'], row['mean']),
                    xytext=(0, -25 if row['change'] < 0 else 25),
                    textcoords='offset points',
                    fontsize=8, color=color, fontweight='bold',
                    ha='center',
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

ax.set_xlabel('Month', fontsize=12)
ax.set_ylabel('Average Traffic Volume', fontsize=12)
ax.set_title('Monthly Average Traffic Volume (2022-2024)\nAnnotated: Significant Changes (>10% month-over-month)',
             fontsize=13, fontweight='bold')
ax.legend(loc='upper right')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.xticks(rotation=45)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'monthly_trend_annotated.png'), bbox_inches='tight')
plt.close()

print("  Significant monthly changes:")
for idx, row in monthly.iterrows():
    if idx == 0:
        continue
    if abs(row['change']) > 10:
        direction = "DECREASE" if row['change'] < 0 else "INCREASE"
        month_str = row['YearMonth_dt'].strftime('%Y-%m')
        explanation = f"  {month_str}: {direction} of {row['change']:+.1f}% (avg volume: {row['mean']:.0f})"
        print(explanation)
        explanations.append(explanation)

# ============================================================================
# 2.2 YEAR-OVER-YEAR COMPARISON
# ============================================================================
print("\n" + "-" * 50)
print("2.2: Year-Over-Year Traffic Comparison")
print("-" * 50)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Monthly comparison by year
for year in sorted(df['Year'].unique()):
    year_data = df[df['Year'] == year].groupby('Month')[TARGET].mean()
    axes[0].plot(year_data.index, year_data.values, '-o', markersize=5,
                 linewidth=2, label=f'{year}')

axes[0].set_xlabel('Month')
axes[0].set_ylabel('Average Traffic Volume')
axes[0].set_title('Monthly Traffic by Year\n(Overlay comparison)', fontweight='bold')
axes[0].set_xticks(range(1, 13))
axes[0].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Yearly average bar chart
yearly_avg = df.groupby('Year')[TARGET].mean()
bars = axes[1].bar(yearly_avg.index.astype(str), yearly_avg.values,
                    color=['#3498db', '#2ecc71', '#e74c3c'])
axes[1].set_xlabel('Year')
axes[1].set_ylabel('Average Traffic Volume')
axes[1].set_title('Yearly Average Traffic Volume', fontweight='bold')
for bar, val in zip(bars, yearly_avg.values):
    axes[1].text(bar.get_x() + bar.get_width()/2., val + 200,
                 f'{val:.0f}', ha='center', fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'year_over_year_comparison.png'), bbox_inches='tight')
plt.close()

print("  Yearly averages:")
for year, avg in yearly_avg.items():
    print(f"    {year}: {avg:.0f}")

# ============================================================================
# 2.3 DAY-OF-WEEK PATTERN
# ============================================================================
print("\n" + "-" * 50)
print("2.3: Day-of-Week Traffic Pattern")
print("-" * 50)

day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_avg = df.groupby('DayName')[TARGET].agg(['mean', 'std']).reindex(day_order)

fig, ax = plt.subplots(figsize=(12, 6))
colors = ['#3498db'] * 5 + ['#e74c3c'] * 2  # Blue weekdays, Red weekends
bars = ax.bar(day_avg.index, day_avg['mean'], yerr=day_avg['std'],
              color=colors, alpha=0.8, capsize=5, edgecolor='black', linewidth=0.5)
ax.set_xlabel('Day of Week')
ax.set_ylabel('Average Traffic Volume')
ax.set_title('Average Traffic Volume by Day of Week\n(Blue = Weekday, Red = Weekend, Error bars = ±1 Std Dev)',
             fontweight='bold')
ax.grid(axis='y', alpha=0.3)

for bar, val in zip(bars, day_avg['mean']):
    ax.text(bar.get_x() + bar.get_width()/2., val + day_avg['std'].max() * 0.1,
            f'{val:.0f}', ha='center', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'day_of_week_pattern.png'), bbox_inches='tight')
plt.close()

weekday_avg = df[df['IsWeekend'] == 0][TARGET].mean()
weekend_avg = df[df['IsWeekend'] == 1][TARGET].mean()
diff_pct = ((weekday_avg - weekend_avg) / weekend_avg) * 100

explanation = f"  Weekday avg: {weekday_avg:.0f}, Weekend avg: {weekend_avg:.0f} -> Weekday is {diff_pct:+.1f}% {'higher' if diff_pct > 0 else 'lower'}"
print(explanation)
explanations.append(explanation)

# ============================================================================
# 2.4 QUARTERLY SEASONAL PATTERNS
# ============================================================================
print("\n" + "-" * 50)
print("2.4: Quarterly / Seasonal Patterns")
print("-" * 50)

quarterly = df.groupby(['Year', 'Quarter'])[TARGET].mean().reset_index()
quarter_names = {1: 'Q1\n(Jan-Mar)', 2: 'Q2\n(Apr-Jun)', 3: 'Q3\n(Jul-Sep)', 4: 'Q4\n(Oct-Dec)'}

fig, ax = plt.subplots(figsize=(14, 6))
for year in sorted(df['Year'].unique()):
    year_data = quarterly[quarterly['Year'] == year]
    ax.plot(year_data['Quarter'], year_data[TARGET], '-o', markersize=8,
            linewidth=2.5, label=f'{year}')
    for _, row in year_data.iterrows():
        ax.text(row['Quarter'], row[TARGET] + 300, f'{row[TARGET]:.0f}',
                ha='center', fontsize=8)

ax.set_xticks([1, 2, 3, 4])
ax.set_xticklabels([quarter_names[q] for q in [1, 2, 3, 4]])
ax.set_xlabel('Quarter')
ax.set_ylabel('Average Traffic Volume')
ax.set_title('Seasonal (Quarterly) Traffic Patterns\nQ3 = Monsoon Season in Bangalore', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'quarterly_seasonal_pattern.png'), bbox_inches='tight')
plt.close()

# ============================================================================
# 2.5 AREA-WISE TRAFFIC PATTERNS
# ============================================================================
print("\n" + "-" * 50)
print("2.5: Area-wise Traffic Volume")
print("-" * 50)

area_avg = df.groupby('Area Name')[TARGET].agg(['mean', 'std', 'count']).sort_values('mean', ascending=True)

fig, ax = plt.subplots(figsize=(12, 7))
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(area_avg)))
bars = ax.barh(area_avg.index, area_avg['mean'], xerr=area_avg['std'],
               color=colors, capsize=3, edgecolor='black', linewidth=0.3)
ax.set_xlabel('Average Traffic Volume')
ax.set_title('Average Traffic Volume by Area in Bangalore\n(Error bars = ±1 Std Dev)', fontweight='bold')
ax.grid(axis='x', alpha=0.3)

for bar, val in zip(bars, area_avg['mean']):
    ax.text(val + area_avg['std'].max() * 0.1, bar.get_y() + bar.get_height()/2.,
            f'{val:.0f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'area_wise_traffic.png'), bbox_inches='tight')
plt.close()

print("  Highest traffic areas:")
for area, row in area_avg.tail(3).iterrows():
    print(f"    {area}: {row['mean']:.0f} avg volume")

# ============================================================================
# 2.6 WEATHER IMPACT ON TRAFFIC
# ============================================================================
print("\n" + "-" * 50)
print("2.6: Weather Impact on Traffic Volume")
print("-" * 50)

weather_avg = df.groupby('Weather Conditions')[TARGET].agg(['mean', 'std', 'count']).sort_values('mean')

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Bar chart
bars = axes[0].barh(weather_avg.index, weather_avg['mean'],
                     color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6'][:len(weather_avg)])
axes[0].set_xlabel('Average Traffic Volume')
axes[0].set_title('Traffic Volume by Weather Condition', fontweight='bold')
for bar, val in zip(bars, weather_avg['mean']):
    axes[0].text(val + 100, bar.get_y() + bar.get_height()/2.,
                 f'{val:.0f}', va='center', fontsize=10)

# Box plot
weather_data = [df[df['Weather Conditions'] == w][TARGET].values for w in weather_avg.index]
bp = axes[1].boxplot(weather_data, labels=weather_avg.index, vert=True, patch_artist=True)
for patch, color in zip(bp['boxes'], ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6'][:len(weather_avg)]):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
axes[1].set_ylabel('Traffic Volume')
axes[1].set_title('Traffic Volume Distribution by Weather\n(Shows variance and outliers)', fontweight='bold')
axes[1].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'weather_impact.png'), bbox_inches='tight')
plt.close()

print("  Weather impact:")
for w, row in weather_avg.iterrows():
    print(f"    {w:12s}: avg={row['mean']:.0f}, std={row['std']:.0f}, count={row['count']:.0f}")

# ============================================================================
# 2.7 AREA-WISE MONTHLY HEATMAP
# ============================================================================
print("\n" + "-" * 50)
print("2.7: Area-wise Monthly Traffic Heatmap")
print("-" * 50)

pivot = df.pivot_table(values=TARGET, index='Area Name', columns='Month', aggfunc='mean')

fig, ax = plt.subplots(figsize=(14, 8))
sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax, linewidths=0.5,
            xticklabels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
ax.set_title('Average Traffic Volume: Area x Month Heatmap\n(Darker = Higher traffic)', fontweight='bold')
ax.set_ylabel('Area')
ax.set_xlabel('Month')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'area_month_heatmap.png'), bbox_inches='tight')
plt.close()

# ============================================================================
# 2.8 DAILY TRAFFIC TIME SERIES (Sampled for clarity)
# ============================================================================
print("\n" + "-" * 50)
print("2.8: Daily Traffic Time Series")
print("-" * 50)

daily = df.groupby('Date')[TARGET].mean().reset_index()

fig, ax = plt.subplots(figsize=(18, 6))
ax.plot(daily['Date'], daily[TARGET], color='steelblue', linewidth=0.8, alpha=0.7)

# Add rolling average
daily['MA_30'] = daily[TARGET].rolling(window=30, center=True).mean()
ax.plot(daily['Date'], daily['MA_30'], color='red', linewidth=2, label='30-day Moving Average')

ax.set_xlabel('Date')
ax.set_ylabel('Average Daily Traffic Volume')
ax.set_title('Daily Traffic Volume with 30-Day Moving Average\n(Raw data in blue, trend in red)',
             fontweight='bold')
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'daily_time_series.png'), bbox_inches='tight')
plt.close()

# ============================================================================
# 2.9 ROADWORK IMPACT
# ============================================================================
print("\n" + "-" * 50)
print("2.9: Roadwork Impact on Traffic")
print("-" * 50)

fig, ax = plt.subplots(figsize=(10, 5))
roadwork_data = df.groupby('Roadwork and Construction Activity')[TARGET].agg(['mean', 'std'])
bars = ax.bar(roadwork_data.index, roadwork_data['mean'], yerr=roadwork_data['std'],
              color=['#2ecc71', '#e74c3c'], alpha=0.8, capsize=5)
ax.set_ylabel('Average Traffic Volume')
ax.set_title('Impact of Roadwork on Traffic Volume', fontweight='bold')
for bar, val in zip(bars, roadwork_data['mean']):
    ax.text(bar.get_x() + bar.get_width()/2., val + roadwork_data['std'].max() * 0.1,
            f'{val:.0f}', ha='center', fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'roadwork_impact.png'), bbox_inches='tight')
plt.close()

# ============================================================================
# 2.10 COMPREHENSIVE FLUCTUATION EXPLANATION REPORT
# ============================================================================
print("\n" + "=" * 70)
print("STEP 2.10: FLUCTUATION EXPLANATION REPORT")
print("=" * 70)

report = f"""
===============================================================================
TEMPORAL PATTERN ANALYSIS - FLUCTUATION EXPLANATION REPORT
===============================================================================
Dataset: Bangalore Traffic Dataset
Period : {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}

1. MONTHLY TREND EXPLANATIONS
------------------------------
The monthly traffic volume shows several notable fluctuations:

"""

# Generate explanations for each significant change
for idx, row in monthly.iterrows():
    if idx == 0:
        continue
    if abs(row['change']) > 8:
        month_str = row['YearMonth_dt'].strftime('%B %Y')
        prev_month = monthly.iloc[idx-1]['YearMonth_dt'].strftime('%B %Y')

        if row['change'] < 0:
            # Possible reasons for decrease
            month_num = row['YearMonth_dt'].month
            reasons = []
            if month_num in [6, 7, 8, 9]:
                reasons.append("Monsoon season - heavy rainfall reduces travel")
            if month_num == 1:
                reasons.append("New Year period - reduced commuter traffic")
            if month_num in [4, 5]:
                reasons.append("Summer holidays - schools/colleges closed, reduced commute")
            if month_num == 12:
                reasons.append("Year-end holidays and reduced business activity")
            reasons.append("Possible increase in remote work during this period")
            reasons.append("Construction/road closures may have diverted traffic")

            report += f"  {month_str}: DECREASED {abs(row['change']):.1f}% from {prev_month}\n"
            report += f"    Volume: {monthly.iloc[idx-1]['mean']:.0f} -> {row['mean']:.0f}\n"
            report += f"    Possible reasons:\n"
            for r in reasons[:3]:
                report += f"      - {r}\n"
        else:
            month_num = row['YearMonth_dt'].month
            reasons = []
            if month_num in [1, 2]:
                reasons.append("Post-holiday return to normal traffic patterns")
            if month_num in [9, 10]:
                reasons.append("End of monsoon - roads clear, festivals begin (Dasara, Diwali)")
            if month_num in [3, 4]:
                reasons.append("Financial year-end - increased business activity")
            if month_num in [10, 11]:
                reasons.append("Festival season (Dasara, Diwali) increases shopping/travel")
            reasons.append("Normal seasonal recovery in commuter patterns")

            report += f"  {month_str}: INCREASED {row['change']:.1f}% from {prev_month}\n"
            report += f"    Volume: {monthly.iloc[idx-1]['mean']:.0f} -> {row['mean']:.0f}\n"
            report += f"    Possible reasons:\n"
            for r in reasons[:3]:
                report += f"      - {r}\n"
        report += "\n"

report += f"""
2. DAY-OF-WEEK PATTERNS
------------------------
  Weekday average: {weekday_avg:.0f}
  Weekend average: {weekend_avg:.0f}
  Difference: {diff_pct:+.1f}%

  Explanation:
  - Weekdays have {'higher' if diff_pct > 0 else 'lower'} traffic due to office commutes
  - Bangalore is an IT hub - weekday traffic dominated by tech workers
  - Weekend traffic is driven by leisure, shopping, and religious activities

3. SEASONAL PATTERNS (Bangalore-specific)
-------------------------------------------
  Q1 (Jan-Mar): Post-holiday normalization, moderate traffic
  Q2 (Apr-Jun): Summer + pre-monsoon, schools reopen in June
  Q3 (Jul-Sep): Monsoon season - typically LOWER traffic due to rain
  Q4 (Oct-Dec): Festival season (Dasara, Diwali, Christmas) - HIGHER traffic

4. WEATHER IMPACT
-----------------
"""

for w, row in weather_avg.iterrows():
    report += f"  {w:12s}: avg volume = {row['mean']:.0f}\n"

report += f"""
  Rain and fog conditions generally {'reduce' if weather_avg.loc['Rain' if 'Rain' in weather_avg.index else weather_avg.index[0], 'mean'] < weather_avg['mean'].max() else 'increase'} traffic volume.
  Clear weather shows {'highest' if weather_avg.index[-1] == 'Clear' else 'moderate'} traffic levels.

5. KEY INSIGHTS FOR THE MODEL
-------------------------------
  - Temporal features (DayOfWeek, Month, Quarter) capture cyclical patterns
  - Weekend/Weekday flag is a strong binary predictor
  - Weather conditions add short-term prediction value
  - Area-specific patterns suggest spatial features are important
  - Year-over-year trends indicate gradual infrastructure changes

===============================================================================
"""

print(report)
with open(os.path.join(OUT_DIR, 'temporal_fluctuation_report.txt'), 'w') as f:
    f.write(report)

print(f"\nAll temporal analysis outputs saved to: {OUT_DIR}/")
print("Run step3_model_training.py next.")
