import matplotlib.pyplot as plt # For plotting
import numpy as np  # linear algebra
import pandas as pd
import re #regex

# Put data in dataframe
data = pd.read_csv("training_data.csv")
print(data.shape) # Check shape is correct

# Clean data
# Remove incomplete rows (with NaN's / Na's)
cleaned_df = data.dropna(how='any')
print(cleaned_df.shape) # Check shape is correct

# Separate data into three dataframes sorted by the three paintings
memory_df = cleaned_df.loc[cleaned_df['Painting'] == 'The Persistence of Memory']
lily_df = cleaned_df.loc[cleaned_df['Painting'] == 'The Water Lily Pond']
night_df = cleaned_df.loc[cleaned_df['Painting'] == 'The Starry Night']

# # Check std and mean
# memory_df.describe()
# lily_df.describe()
# night_df.describe()
# memory_df["On a scale of 1–10, how intense is the emotion conveyed by the artwork?"].dtype # Check type is int/float

# Organize column names, etc. for plotting box plots
columns = data.columns.tolist() # all columns' names
numerical_cols = [2, 8, 9] # indices of the completely numerical columns like "intensity of emotion"
likert_cols = [4, 5, 6, 7] # indices of the columns with scales like "This art piece makes me feel calm: 4 - Agree"
shortened_names = ['ID', 'Painting', 'Emotional Intensity', 'Description', 'Sombre', 'Content', 'Calm', 'Uneasy', 'Noticed #Colours', 'Noticed #Objects', 'Price', 'Room', 'Who with', 'Season', 'Food', 'Soundtrack']

# Make box plot function
def make_box_plots(list_of_indices):
    """
    This function takes a list of indices (of the columns you want) 
    and make and display corresponding box plots sequentially
    """
    for i in list_of_indices:
        # Column name used for filtering
        col = columns[i]

        # Filter by column name or/and convert to numeric values by stripping away non-numeric values
        memory_vals = extract_numeric(memory_df, col)
        lily_vals = extract_numeric(lily_df, col)
        night_vals = extract_numeric(night_df, col)
           
        # Plot boxplots together
        plt.figure(figsize=(8,5))
        plt.boxplot(
            [memory_vals, lily_vals, night_vals],
            labels=['Persistence of Memory', 'Water Lily Pond', 'Starry Night']
        )
        plt.ylabel(shortened_names[i])
        plt.title(shortened_names[i] + " Distribution by Painting")

        # Show box plot
        plt.show()

def extract_numeric(df, col):
    """
    This extracts numeric values from a dataframe column
    - For Likert-style '4 - Agree', takes the number
    - For prices or text with numbers, extracts the first number
    - If no number, replaces with 0
    Returns a list of floats for boxplots
    """
    numeric_list = []
    for val in df[col]:
        if pd.isna(val):
            numeric_list.append(0)
        else:
            # Convert to string and find first number (integer or decimal)
            match = re.search(r'\d+(?:,\d{3})*\.?\d*', str(val))
            if match:
                # Remove commas in numbers (e.g., "1,000" → "1000")
                num = match.group().replace(',', '')
                numeric_list.append(float(num))
            else:
                numeric_list.append(0)
    return numeric_list

# Make box plots
make_box_plots(numerical_cols)
make_box_plots(likert_cols)
make_box_plots([10]) # this is the column with price, current box plot is really bad, 
#should fix the outliers and then do this again