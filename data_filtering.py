import numpy as np
import pandas as pd
import seaborn as sns
import string
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load datasets
sleep_df = pd.read_csv(r'C:\Users\abhis\OneDrive\Desktop\Smart Health\Smart-Health-Monitor\datasets\Sleep_health_and_lifestyle_dataset.csv')
obesity_df = pd.read_csv(r'C:\Users\abhis\OneDrive\Desktop\Smart Health\Smart-Health-Monitor\datasets\obesity_data.csv')
mental_health_df = pd.read_csv(r'Smart-Health-Monitor/datasets/Combined Data.csv')
physical_level_df = pd.read_csv(r'C:\Users\abhis\OneDrive\Desktop\Smart Health\Smart-Health-Monitor\datasets\extended_physical_activity_dataset.csv')

# Function to process sleep data
def load_and_process_sleep_data(dataframe, show_matrix=False):
    """
    Cleans and processes the sleep dataset.
    """
    # Drop unnecessary columns
    dataframe.drop(['Person ID', 'Blood Pressure', 'Heart Rate'], axis=1, inplace=True)
    dataframe.dropna(inplace=True)
    dataframe.dropna(axis=1, inplace=True)
    dataframe.drop_duplicates(inplace=True)
    dataframe = dataframe.rename(columns={'Sleep Duration': 'Sleep_Duration', 'Quality of Sleep': 'Sleep_Quality', 'Physical Activity Level': 'Physical_activity_duration', 'Stress Level': 'Stress_Level', 'Daily Steps': 'Daily_Steps'})

    # Merge certain outputs into one
    dataframe['Sleep Disorder'] = dataframe['Sleep Disorder'].replace('Sleep Apnea', 'Insomnia')
    dataframe['BMI Category'] = dataframe['BMI Category'].replace('Normal', 'Normal Weight')

    if show_matrix:
        num_vars = dataframe.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if 'Gender' in num_vars:
            num_vars.remove('Gender')
        corr = dataframe[num_vars].corr()

        plt.figure(figsize=(8, 6))  # Adjust the figure size if necessary
        sns.heatmap(corr, cmap='coolwarm', cbar=True, annot=True, fmt=".2f", annot_kws={"size": 10}, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    
        plt.xticks(rotation=45, ha='right', fontsize=10)  # Rotate x-axis labels to fit
        plt.yticks(fontsize=10)  # Adjust y-axis label font size
        
        plt.tight_layout()  # Ensure everything fits within the figure
        plt.show()

# Process the sleep dataset
load_and_process_sleep_data(sleep_df, show_matrix=True)


# Function to process obesity data
def load_and_process_obesity_data(dataframe):
    """
    Cleans and processes the obesity dataset.
    """
    dataframe.dropna(inplace=True)
    dataframe.dropna(axis=1, inplace=True)
    
    # Encode categorical variables
    le = LabelEncoder()
    dataframe['Gender'] = le.fit_transform(dataframe['Gender'])

# Process the obesity dataset
load_and_process_obesity_data(obesity_df)

# Clean the mental health dataset
mental_health_df.drop(mental_health_df.columns[0], axis=1, inplace=True)
mental_health_df.dropna(inplace=True)
mental_health_df.dropna(axis=1, inplace=True)

# Clean and load the physical activity dataset
def clean_text(text):
    for p in string.punctuation:
        text = text.replace(p, '')
    text = text.lower()
    return text

mental_health_df['statement'] = mental_health_df['statement'].apply(clean_text)
print(mental_health_df.head(10))



# Data Visualization
def plot_sleep_data(dataframe):
    """
    Plots scatter plots for sleep data.
    """
    # Create a figure with 1 row and 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

    # First scatter plot: Sleep Duration vs Quality of Sleep
    sns.scatterplot(x='Sleep Duration', y='Quality of Sleep', hue='Sleep Disorder', data=dataframe, ax=axes[0])
    axes[0].set_xlabel('Sleep Duration (hours)')
    axes[0].set_ylabel('Quality of Sleep')
    axes[0].set_title('Sleep Duration vs Quality of Sleep')
    axes[0].legend(loc='upper left', bbox_to_anchor=(1, 1))  # Move legend outside the plot

    # Second scatter plot: Age vs Stress Level
    sns.scatterplot(x='Age', y='Stress Level', hue='Sleep Disorder', data=dataframe, ax=axes[1])
    axes[1].set_xlabel('Age')
    axes[1].set_ylabel('Stress Level')
    axes[1].set_title('Age vs Stress Level')
    axes[1].legend(loc='upper left', bbox_to_anchor=(1, 1))  # Move legend outside the plot

    # Adjust layout to increase space between plots and ensure they fit properly
    plt.subplots_adjust(wspace=0.4)
    plt.tight_layout()

    # Show the figure
    plt.show()



def plot_obesity_data(dataframe):
    """
    Plots scatter plots for obesity data.
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='Age', y='Weight', hue='ObesityCategory', data=dataframe)
    plt.xlabel('Age')
    plt.ylabel('Weight')
    plt.title('Age vs Weight by Obesity Category')
    plt.show()

# Call the plotting functions
plot_sleep_data(sleep_df)
plot_obesity_data(obesity_df)