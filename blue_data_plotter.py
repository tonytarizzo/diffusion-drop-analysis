import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt
import numpy as np
import os

data_file_path = os.path.join('image_data', 'extracted_blue_data.csv')
df = pd.read_csv(data_file_path)
df['Average_Radius'] = df['Average_Radius'].astype(float)

# Extract minute values from the image filenames and convert to integers
minute_str_df = df['Image_Path'].str.extract(r'image_data/blue_spot/blue3ten(\d+)\.jpg')
minute_str_df = minute_str_df.dropna()
df['Minute'] = minute_str_df[0].astype(int)

# Ensure no missing values in 'Radius' column
df = df.dropna(subset=['Average_Radius'])

# Reset starting time value to 0
df['Minute'] = df['Minute'] - df['Minute'].min()

# Sort the DataFrame by 'Minute' to ensure a proper time series plot
df = df.sort_values(by='Minute')

# # Set equation coefficients
mf = 1.232
mpb = 0.002
v_0 = 10 * 1e-9     # Drop volume
V = 50 * 1e-6      # Agar volume
rad = 75 * 1e-3     # Petri dish radius
H = V/(np.pi*rad**2)    # Agar height
print("H:",H)
r_inf = sqrt((4 * mf * v_0) / (3 * mpb * H)) * 100 # Convert to cm
print(f'r_inf = {r_inf:.2f} cm')

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(df['Minute'], df['Average_Radius'], marker=',', linestyle='solid')
plt.axhline(y=r_inf, color='r', linestyle='--', label=f'r_inf = {r_inf:.2f} cm')

# plt.title('Blue Drop Radius Over Time')
plt.xlabel('Minutes')
plt.ylabel('Average Radius (cm)')
plt.legend(['Experimental', 'Theoretical'])
plt.grid(True)
plt.ylim(bottom=0)

plt.savefig('Blue Drop Theory vs Experimental.pdf')

plt.show()