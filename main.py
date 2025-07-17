import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

pd.options.display.float_format = '{:,.2f}'.format

df_hh_income = pd.read_csv('data/Median_Household_Income_2015.csv', encoding="windows-1252")
df_pct_poverty = pd.read_csv('data/Pct_People_Below_Poverty_Level.csv', encoding="windows-1252")
df_pct_completed_hs = pd.read_csv('data/Pct_Over_25_Completed_High_School.csv', encoding="windows-1252")
df_share_race_city = pd.read_csv('data/Share_of_Race_By_City.csv', encoding="windows-1252")
df_fatalities = pd.read_csv('data/Deaths_by_Police_US.csv', encoding="windows-1252")

def run_challenge(description, func):
    print(f"\nChallenge: {description}")
    input("Press ENTER to see the result...\n")
    func()

df_hh_income = df_hh_income.fillna(0)
df_pct_completed_hs = df_pct_completed_hs.fillna(0)
df_fatalities = df_fatalities.fillna(0)

## Chart the Poverty Rate in each US State------------------------------------------------------------------------------
def poverty_rate_us_states():
    df_pct_poverty['poverty_rate'] = df_pct_poverty['poverty_rate'].replace('', np.nan)
    df_pct_poverty['poverty_rate'] = df_pct_poverty['poverty_rate'].str.replace('%', '', regex=False)
    df_pct_poverty['poverty_rate'] = pd.to_numeric(df_pct_poverty['poverty_rate'], errors='coerce')

    state_poverty = df_pct_poverty.groupby('Geographic Area')['poverty_rate'].mean()
    state_poverty_sorted = state_poverty.sort_values(ascending=False)

    plt.figure(figsize=(12, 8))

    plt.bar(state_poverty_sorted.index, state_poverty_sorted.values, color='skyblue', edgecolor='black')
    plt.title('Average Poverty Rate by US State (Highest to Lowest)')
    plt.ylabel('Poverty Rate (%)')
    plt.xlabel('State')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

## Chart the High School Graduation Rate by US State--------------------------------------------------------------------
def high_school_grad_rate_us_states():
    global df_pct_completed_hs
    df_pct_completed_hs['percent_completed_hs'] = df_pct_completed_hs['percent_completed_hs'].replace('', np.nan)
    df_pct_completed_hs['percent_completed_hs'] = df_pct_completed_hs['percent_completed_hs'].str.replace('%', '', regex=False)
    df_pct_completed_hs['percent_completed_hs'] = pd.to_numeric(df_pct_completed_hs['percent_completed_hs'], errors='coerce')

    df_pct_completed_hs = df_pct_completed_hs.fillna(0)

    state_rates = df_pct_completed_hs.groupby('Geographic Area')['percent_completed_hs'].mean()
    sorted_state_rates = state_rates.sort_values(ascending=True)

    plt.figure(figsize=(12, 8))
    plt.bar(sorted_state_rates.index, sorted_state_rates.values, edgecolor='black')
    plt.title('High School Graduation Rate by US State')
    plt.xlabel('States')
    plt.ylabel('Rates')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

## Visualise the Relationship between Poverty Rates and High School Graduation Rates------------------------------------
def relation_between_poverty_and_high_grad():
    state_poverty = df_pct_poverty.groupby('Geographic Area')['poverty_rate'].mean()
    state_rates = df_pct_completed_hs.groupby('Geographic Area')['percent_completed_hs'].mean()
    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax2 = ax1.twinx()

    ax1.set_ylim(min(state_poverty), max(state_poverty))
    ax2.set_ylim(min(state_rates), max(state_rates))

    ax1.plot(state_poverty, label='Poverty Rate', color='red')
    ax2.plot(state_rates, '--', label='Diploma Rate', color='black')

    ax1.set_xticks(range(len(state_poverty)))
    ax1.set_xticklabels(state_poverty.index, rotation=90)

    plt.title('Relationship between Poverty Rates and High School Graduation Rates')
    plt.tight_layout()

    ## Seaborn
    print("Creating now Seaborn .jointplot() with a Kernel Density Estimate (KDE) for the same relationship")
    df_states = pd.DataFrame({
        'Poverty Rate': state_poverty,
        'Diploma Rate': state_rates
    })
    sns.jointplot(data=df_states, x='Poverty Rate', y='Diploma Rate', kind='kde', fill=True)
    plt.tight_layout()
    plt.show()

    print("Seaborn's .lmplot() or .regplot() to show a linear regression between the poverty ratio and the high school graduation ratio.")
    sns.lmplot(data=df_states, x='Poverty Rate', y='Diploma Rate')
    plt.title('Linear Regression using lmplot')
    plt.tight_layout()
    plt.show()


## Create a Bar Chart with Subsections Showing the Racial Makeup of Each US State--------------------------------------------------------------------
def racial_makeup_us_states():
    df_share_race_city.columns.to_list()
    df_share_race_city[['share_white', 'share_black', 'share_hispanic', 'share_asian', 'share_native_american']] = \
    df_share_race_city[
        ['share_white', 'share_black', 'share_hispanic', 'share_asian', 'share_native_american']].replace('', np.nan)

    df_share_race_city[['share_white', 'share_black', 'share_hispanic', 'share_asian', 'share_native_american']] = \
    df_share_race_city[
        ['share_white', 'share_black', 'share_hispanic', 'share_asian', 'share_native_american']].replace('%', '',
                                                                                                          regex=False)

    cols = ['share_white', 'share_black', 'share_hispanic', 'share_asian', 'share_native_american']
    df_share_race_city[cols] = df_share_race_city[cols].apply(pd.to_numeric, errors='coerce')

    races_state = df_share_race_city.groupby('Geographic area')[
        ['share_white', 'share_black', 'share_hispanic', 'share_asian', 'share_native_american']].mean()

    races_state.plot(kind='bar', stacked=False, figsize=(15, 5))
    plt.title('Racial Makeup of Each US State')
    plt.xlabel('States')
    plt.ylabel('Population')
    plt.tight_layout()
    plt.show()

## Create Donut Chart by of People Killed by Race-----------------------------------------------------------------------
def chart_people_killed_race():
    killed_by_race = df_fatalities['race'].value_counts()

    plt.pie(killed_by_race.values, labels=killed_by_race.index,
            autopct='%1.1f%%', pctdistance=0.85)

    centre_circle = plt.Circle((0, 0), 0.70, fc='white')

    plt.title('Fatalities by Race')
    plt.show()

## Create a Chart Comparing the Total Number of Deaths of Men and Women-------------------------------------------------
def men_women_numbers():
    gender_differnces = df_fatalities['gender'].value_counts()

    plt.pie(gender_differnces.values, labels=gender_differnces.index,
            autopct='%1.1f%%', pctdistance=0.85)

    centre_circle = plt.Circle((0, 0), 0.70, fc='white')

    plt.title('Deaths by Gender')
    plt.show()

## Create a Box Plot Showing the Age and Manner of Death----------------------------------------------------------------
def age_manner_death():
    df_fatalities.columns.to_list()
    df = df_fatalities.dropna(subset=["age", "manner_of_death", "gender"])
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=df,
        x="manner_of_death",  # category on the x-axis
        y="age",  # numerical variable for box plot
        hue="gender"  # separates by gender
    )
    plt.title("Distribution of Age by Manner of Death and Gender")
    plt.xlabel("Manner of Death")
    plt.ylabel("Age")
    plt.legend(title="Gender")
    plt.show()

## Were People Armed? And with what type of weapon?---------------------------------------------------------------------------------------------------
def people_armed_or_not():
    weapon_victims = df_fatalities['armed'].value_counts()
    sorted_weapon_victims = weapon_victims.sort_values(ascending=False)
    top_weapons = sorted_weapon_victims.head(10)

    plt.figure(figsize=(12, 8))
    plt.bar(top_weapons.index.astype(str), top_weapons.values, edgecolor='black')
    plt.title('Most found weapons on victims')
    plt.xlabel('Victim Weapon')
    plt.ylabel('Numbers')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    print("How many of the people killed by police were armed with guns versus unarmed?")
    df = df_fatalities.copy()
    df['armed'] = df['armed'].fillna('Unknown')

    df['armed_status'] = df['armed'].apply(lambda x: 'Unarmed' if 'unarmed' in str(x).lower() else 'Armed')

    armed_or_unarmed = df['armed_status'].value_counts()

    plt.pie(armed_or_unarmed.values, labels=armed_or_unarmed.index,
            autopct='%1.1f%%', pctdistance=0.85)

    centre_circle = plt.Circle((0, 0), 0.70, fc='white')

    plt.title('Victims Unarmed or Armed')
    plt.show()

## How Old Were the People Killed?--------------------------------------------------------------------------------------
def age_people_killed():
    print("People under 25")
    df_fatalities.drop(df_fatalities[df_fatalities['age'] == 0].index, inplace = True)

    df_fatalities['under_25'] = df_fatalities['age'].apply(lambda x: 'Under 25' if x < 25 else 'Above 25')

    under_or_above_25 = df_fatalities['under_25'].value_counts()

    plt.pie(under_or_above_25.values, labels=under_or_above_25.index,
           autopct='%1.1f%%', pctdistance=0.85)

    centre_circle = plt.Circle((0, 0), 0.70, fc='white')

    plt.title('Victims Under or Above 25')
    plt.show()

    print("Histogram and KDE plot that shows the distribution of ages of the people killed by police.")
    ages = df_fatalities['age'].dropna()

    plt.figure(figsize=(10, 6))
    sns.histplot(ages, bins=30, color='lightseagreen', edgecolor='black', kde=False)
    plt.title('Histogram of Ages of People Killed by Police')
    plt.xlabel('Age')
    plt.ylabel('Number of People')
    plt.grid(True)
    plt.show()

    print("Seaborn KDE Plot")
    plt.figure(figsize=(10, 6))
    sns.histplot(ages, kde=True, bins=30, color='skyblue', edgecolor='black')
    plt.title('Distribution of Ages of People Killed by Police')
    plt.xlabel('Age')
    plt.ylabel('Count / Density')
    plt.grid(True)
    plt.show()

    print("Create a seperate KDE plot for each race. Is there a difference between the distributions?")
    df_fatalities.drop(df_fatalities[df_fatalities['race'] == 0].index, inplace=True)
    df_clean = df_fatalities[['age', 'race']].dropna()

    plt.figure(figsize=(12, 6))
    sns.kdeplot(data=df_clean, x='age', hue='race', common_norm=False, fill=True)
    plt.show()

## Race of People Killed -----------------------------------------------------------------------------------------------
def people_killed_by_race():
    races_count = df_fatalities['race'].value_counts()

    plt.figure(figsize=(12, 8))
    bars = plt.bar(races_count.index.astype(str), races_count.values, edgecolor='black')

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 5,
                 f'{int(height)}',
                 ha='center', va='bottom', fontsize=12)

    plt.title('Specific Race Victims Numbers')
    plt.xlabel('Race')
    plt.ylabel('Count/Density')
    plt.tight_layout()
    plt.show()

## Mental Illness and Police Killings ----------------------------------------------------------------------------------
def rate_mental_illness_killed():
    signs_of_mental_illness = df_fatalities['signs_of_mental_illness'].value_counts()

    plt.pie(signs_of_mental_illness.values, labels=signs_of_mental_illness.index,
            autopct='%1.1f%%', pctdistance=0.85)

    centre_circle = plt.Circle((0, 0), 0.70, fc='white')

    plt.title('Victims with Sign of Mental Illness')
    plt.show()

## In Which Cities Do the Most Police Killings Take Place?--------------------------------------------------------------
def cities_most_killings():
    victim_in_city = df_fatalities['city'].value_counts()
    sorted_victim_in_city = victim_in_city.sort_values(ascending=False)
    top_cities = sorted_victim_in_city.head(10)

    plt.figure(figsize=(12, 8))
    plt.bar(top_cities.index.astype(str), top_cities.values, edgecolor='black')
    plt.title('Top Cities with more victims')
    plt.xlabel('City')
    plt.ylabel('Numbers')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

## Rate of Death by Race -----------------------------------------------------------------------------------------------
def rate_death_race():
    df_city_race = pd.crosstab(df_fatalities['city'], df_fatalities['race'])
    desired_race_order = ['A', 'W', 'H', 'B', 'O', 'N']
    df_city_race = df_city_race.reindex(columns=desired_race_order, fill_value=0).reset_index()

    cols = ['A', 'W', 'H', 'B', 'O', 'N']
    df_city_race[cols] = df_city_race[cols].apply(pd.to_numeric, errors='coerce')

    df_city_race['total'] = df_city_race[cols].sum(axis=1)
    top_cities = df_city_race.sort_values(by='total', ascending=False).head(10)
    top_cities.set_index('city')[cols].plot(kind='bar', stacked=False, figsize=(15, 5))

    plt.title('Racial Makeup of Each US State')
    plt.xlabel('States')
    plt.ylabel('Population')
    plt.tight_layout()
    plt.show()

## Create a Choropleth Map of Police Killings by US State --------------------------------------------------------------
def choropleth_map_us_killings_police():
    state_counts = df_fatalities['state'].value_counts().reset_index()
    state_counts.columns = ['state', 'deaths']

    fig = px.choropleth(
        state_counts,
        locations='state',
        locationmode='USA-states',
        color='deaths',
        scope='usa',
        color_continuous_scale='Reds',
        labels={'deaths': 'Deaths'},
        title='Police Killings by US State'
    )

    fig.show()

## Number of Police Killings Over Time ---------------------------------------------------------------------------------
def police_killings_over_time():
    df_fatalities['date'] = pd.to_datetime(df_fatalities['date'], format='mixed', dayfirst=True, errors='coerce')
    df_fatalities['year_month'] = df_fatalities['date'].dt.to_period('M')

    monthly_counts = df_fatalities.groupby('year_month').size().reset_index(name='count')
    monthly_counts['year_month'] = monthly_counts['year_month'].dt.to_timestamp()


    month_range = pd.date_range(
        start=monthly_counts['year_month'].min(),
        end=monthly_counts['year_month'].max(),
        freq='MS'
    )

    plt.figure(figsize=(15, 6))
    plt.plot(monthly_counts['year_month'], monthly_counts['count'], linestyle='-')


    plt.xticks(month_range, rotation=45)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    plt.title('Monthly Police Killings in the US')
    plt.xlabel('Month')
    plt.ylabel('Number of Fatalities')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ------------------ Run Challenges ------------------

run_challenge("Chart the Poverty Rate in each US State", poverty_rate_us_states)

run_challenge("Chart the High School Graduation Rate by US State", high_school_grad_rate_us_states)

run_challenge("Visualise the Relationship between Poverty Rates and High School Graduation Rates", relation_between_poverty_and_high_grad)

run_challenge("Create a Bar Chart with Subsections Showing the Racial Makeup of Each US State", racial_makeup_us_states)

run_challenge("Create Donut Chart by of People Killed by Race", chart_people_killed_race)

run_challenge("Create a Chart Comparing the Total Number of Deaths of Men and Women", men_women_numbers)

run_challenge("Create a Box Plot Showing the Age and Manner of Death", age_manner_death)

run_challenge("Were People Armed? And with what type of weapon?", people_armed_or_not)

run_challenge("How Old Were the People Killed?", age_people_killed)

run_challenge("Race of People Killed", people_killed_by_race)

run_challenge("Mental Illness and Police Killings", rate_mental_illness_killed)

run_challenge("In Which Cities Do the Most Police Killings Take Place?", cities_most_killings)

run_challenge("Rate of Death by Race", rate_death_race)

run_challenge("Create a Choropleth Map of Police Killings by US State", choropleth_map_us_killings_police)

run_challenge("Number of Police Killings Over Time", police_killings_over_time)
