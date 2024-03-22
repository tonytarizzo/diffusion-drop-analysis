import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import numpy as np
import os

data_file_path = os.path.join('image_data', 'extracted_red_data.csv')
df = pd.read_csv(data_file_path)
df['Average_Radius'] = df['Average_Radius'].astype(float)

# Extract minute values from the image filenames and convert to integers
minute_str_df = df['Image_Path'].str.extract(r'image_data/red_spot/redten(\d+)\.jpg')
minute_str_df = minute_str_df.dropna()
df['Minute'] = minute_str_df[0].astype(int)

# Ensure no missing values in 'Average_Radius' column
df = df.dropna(subset=['Average_Radius'])

# Reset starting time value to 0 for when drop appears
df['Minute'] = df['Minute'] - df['Minute'].min()

# Sort the DataFrame by 'Minute' to ensure a proper time series plot
df = df.sort_values(by='Minute')

# Set equation coefficients
m_oh = 1*0.001    # Molarity converted to per m^3
v_0 = 10 * 1e-9     # Drop volume
V = 50 * 1e-6      # Agar volume
rad = 75 * 1e-3     # Petri dish radius
H = V/(np.pi*rad**2)    # Agar height
M = m_oh * v_0 / H    # Initial total number of moles per thickness of agar

D = 5.09e-10    # Diffusion Coefficient
t = df['Minute'][1:] # Time removing 0 
r = np.sqrt(4 * D * 60 * t * (6 - np.log10(4 * np.pi * 60 * t * D / M))) * 100 + 0.43 # Converted to cm and seconds

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(df['Minute'], df['Average_Radius'], marker=',', linestyle='solid')
plt.plot(t, r, marker=',', linestyle='--')

plt.title('Red Drop Radius Over Time')
plt.xlabel('Minutes')
plt.ylabel('Average Radius (cm)')
plt.grid(True)
plt.ylim(bottom=0)

plt.savefig('Red Drop Experimental Curve.pdf')

plt.show()

