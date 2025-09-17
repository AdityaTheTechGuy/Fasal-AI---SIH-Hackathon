import pandas as pd

try:
    # 1. Define filenames
    file1 = 'crop_recommendation_dataset.csv'
    file2 = 'Crop_recommendation.csv'
    output_filename = 'crop_recommendation_combined.csv'

    # 2. Load both datasets
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    print(f"Loaded '{file1}' (Shape: {df1.shape}) and '{file2}' (Shape: {df2.shape})")

    # 3. Standardize Columns (if necessary)
    # This script assumes columns are the same as per our previous exploration.
    # If not, add your df2.rename() calls here.
    if not df1.columns.equals(df2.columns):
        print("Columns are different. Ensure they are standardized before merging.")
    
    # 4. Concatenate the datasets
    combined_df = pd.concat([df1, df2], ignore_index=True)
    print(f"Combined shape before cleaning: {combined_df.shape}")

    # 5. Check for and remove duplicates
    duplicates = combined_df.duplicated().sum()
    print(f"Found and removing {duplicates} duplicate rows.")
    combined_df.drop_duplicates(inplace=True)
    
    # 6. Save the new, combined dataset
    combined_df.to_csv(output_filename, index=False)
    
    print("\n✅ Success!")
    print(f"Final combined dataset shape: {combined_df.shape}")
    print(f"Saved to '{output_filename}'")

except FileNotFoundError as e:
    print(f"❌ Error: Could not find a file. Make sure the script is in the same folder as your CSV files.")
    print(e)
except Exception as e:
    print(f"An error occurred: {e}")