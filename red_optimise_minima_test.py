import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def diffusion_model(t, D):
    m_oh = 1*0.001    # Molarity converted to per m^3
    v_0 = 10 * 1e-9     # Drop volume
    V = 50 * 1e-6      # Agar volume
    rad = 75 * 1e-3     # Petri dish radius
    H = V/(np.pi*rad**2)    # Agar height
    M = m_oh * v_0 / H    # Initial total number of moles per thickness of agar

    # Theoretical formula for radius based on diffusion, ensure t, D, and M are all positive
    with np.errstate(divide='ignore', invalid='ignore'):
        # Prevent log10(0) by adding a very small number inside the log
        term_inside_log = np.maximum(4 * np.pi * 60 * t * D / M, 1e-10)
        # Prevent sqrt of negative numbers by ensuring the term is non-negative
        term = np.maximum(4 * D * 60 * t * (6 - np.log10(term_inside_log)), 0)
        r = np.sqrt(term) * 100 + 0.43 # Convert to cm
        # Ensure the result is finite
        return np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)

data_file_path = os.path.join('image_data', 'extracted_red_data.csv')
df = pd.read_csv(data_file_path)
df['Average_Radius'] = df['Average_Radius'].astype(float)

# Extract minute values from the image filenames and convert to integers
minute_str_df = df['Image_Path'].str.extract(r'image_data/red_spot/redten(\d+)\.jpg')
minute_str_df = minute_str_df.dropna()
df['Minute'] = minute_str_df[0].astype(int)

# Ensure no missing values in 'Average_Radius' column
df = df.dropna(subset=['Average_Radius'])

# Reset starting time value to 0
df['Minute'] = df['Minute'] - df['Minute'].min()
# Exclude any non-positive minute values
df = df[df['Minute'] > 0]

# Sort the DataFrame by 'Minute' to ensure a proper time series plot
df = df.sort_values(by='Minute')

# Define the range of D values to test
D_values = np.logspace(-12, -8, 100)
squared_errors = []
lowest_D_value = 10e10

# Calculate the sum of squared errors for each D value
for D in D_values:
    predicted_r = diffusion_model(df['Minute'].values, D)
    squared_error = np.sum((predicted_r - df['Average_Radius'].values) ** 2)
    if squared_error < lowest_D_value:
        lowest_D_value = squared_error
        best_D = D
    squared_errors.append(squared_error)

# Print minima
print(f'Lowest D value: {best_D:.2e} m^2/s')
print(f'Sum of Squared Errors: {lowest_D_value:.2e}')       

# Plot the sum of squared errors against D values
plt.figure(figsize=(10, 5))
plt.semilogx(D_values, squared_errors, marker='o')
plt.xlabel('D value (cm^2/s)')
plt.ylabel('Sum of Squared Errors')
plt.title('Sum of Squared Errors for Different D values')
plt.grid(True)

plt.savefig('Optimisation Curve of D.pdf')

plt.show()


