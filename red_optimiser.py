import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import numpy as np
import os

def diffusion_model(t, D):
    # Set equation coefficients
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
        r = np.sqrt(term) * 100  # Convert to cm
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

# Initial parameter estimates (D)
initial_guess = [3e-10]

# Fitting the diffusion_model to the data
# The bounds are set to positive values since D should be positive
optimal_parameter, covariance = curve_fit(
    f=diffusion_model,
    xdata=df['Minute'].values,
    ydata=df['Average_Radius'].values,
    p0=initial_guess,
    bounds=([0, np.inf]),  # Bounds for D
)

# Extract the optimized values of D and M
optimized_D = optimal_parameter[0]
print(f"Optimized D: {optimized_D:.2e} cm^2/s")

# Plotting the experimental data
plt.figure(figsize=(10, 5))
plt.plot(df['Minute'], df['Average_Radius'], marker='o', linestyle='-', label='Experimental Data')

# Plotting the model prediction with the optimized D and M
optimized_r = diffusion_model(df['Minute'].values, optimized_D)
plt.plot(df['Minute'], optimized_r, marker='None', linestyle='--', label='Model Prediction')

plt.title('Red Drop Radius Over Time')
plt.xlabel('Minutes')
plt.ylabel('Average Radius (cm)')
plt.legend()
plt.grid(True)
plt.ylim(bottom=0)

plt.show()
