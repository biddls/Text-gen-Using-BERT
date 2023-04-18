import numpy as np
import matplotlib.pyplot as plt
import csv

from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde
from tqdm import tqdm
import warnings
from sklearn import linear_model
from matplotlib.colors import LinearSegmentedColormap

# Ignore warnings
warnings.filterwarnings("error")

# Load the data
data = []
with open('metrics.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    [data.append([np.array([int(col) for col in row]), len(row)]) for row in csv_reader if len(row) != 0]

accuracy = []
# median of none zero values
medianNoneZero = []
# max none zero value
maxNoneZero = []
for arr, _len in tqdm(data, desc='Calculating'):
    a = (arr == 0).sum()
    # skips if all values are zero
    if arr.sum() == 0 or a == 0:
        continue
    accuracy.append(a/_len)
    medianNoneZero.append(np.median(arr[arr != 0]))
    maxNoneZero.append(np.max(arr[arr != 0]))

ransac = linear_model.RANSACRegressor(min_samples=200, max_trials=1000)
accuracy = np.array(accuracy).reshape(-1, 1)
medianNoneZero = np.array(medianNoneZero).reshape(-1, 1)
maxNoneZero = np.array(maxNoneZero).reshape(-1, 1)

# "Viridis-like" colormap with white background
white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
    (0, '#ffffff'),
    (1e-20, '#440053'),
    (0.2, '#404388'),
    (0.4, '#2a788e'),
    (0.6, '#21a784'),
    (0.8, '#78d151'),
    (1, '#fde624'),
], N=256)

# Plot the data
# fig = plt.figure(figsize=(10, 5))

fig = plt.figure(figsize=(10, 5), constrained_layout=True)
gs = GridSpec(3, 3, figure=fig)

# Histogram plotting
print('Plotting histograms')
plt.subplot(gs[0, :])
binwidth = 500
bins = list(range(0, int(np.max(maxNoneZero)) + binwidth, binwidth))
plt.hist(maxNoneZero,
         bins=bins,
         alpha=0.5,
         label='Max',
         density = True
         )
plt.plot(bins, gaussian_kde(maxNoneZero.reshape(-1))(bins), label='Max KDE')
bins = range(0, int(np.max(medianNoneZero)) + binwidth, binwidth)
plt.hist(medianNoneZero,
         bins=bins,
         alpha=0.5,
         label='Median',
         density = True
         )
plt.grid()
plt.title('None zero values')
plt.yscale('log')
plt.legend(loc='upper right')
# plt.show()
# exit()
# Scatter plots
print("Plotting scatter plots 1")
ax = plt.subplot(2, 2, 3)
# Fit linear regression via the least squares with numpy.polyfit
# It returns a slope (b) and intercept (a)
# deg=1 means linear fit (i.e. polynomial of degree 1)
b, a = np.polyfit(accuracy.reshape(-1), medianNoneZero.reshape(-1), deg=1)
ransac.fit(accuracy, medianNoneZero)
# Create sequence of 100 numbers from 0 to 100
xseq = np.linspace(0, 1, num=1000)
line_y_ransac = ransac.predict(xseq.reshape(-1, 1))
# Plot regression line
plt.plot(xseq, a + b * xseq,
         color="b",
         lw=1
         ,label='Least squares')
plt.plot(xseq, line_y_ransac,
            color="r",
            lw=1
            ,label='RANSAC'
        )

xy = np.vstack([accuracy.reshape(-1), medianNoneZero.reshape(-1)])
z = gaussian_kde(xy)(xy)

plt.scatter(accuracy, medianNoneZero, s=1, c=z)
ymin = 0.9
ax.set_ylim(ymin=ymin)
plt.title('Accuracy vs Median of none zero values')
plt.yscale('log')
plt.legend(loc='lower left')

print("Plotting scatter plots 2")
ax = plt.subplot(2, 2, 4)
# Fit linear regression via the least squares with numpy.polyfit
# It returns a slope (b) and intercept (a)
# deg=1 means linear fit (i.e. polynomial of degree 1)
b, a = np.polyfit(accuracy.reshape(-1), maxNoneZero.reshape(-1), deg=1)
ransac.fit(accuracy, maxNoneZero)
# Create sequence of 100 numbers from 0 to 100
xseq = np.linspace(0, 1, num=1000)
line_y_ransac = ransac.predict(xseq.reshape(-1, 1))
# Plot regression line
plt.plot(xseq, a + b * xseq,
            color="b",
            lw=1
            ,label='Least squares'
        )
plt.plot(xseq, line_y_ransac,
            color="r",
            lw=1
            ,label='RANSAC'
        )
xy = np.vstack([accuracy.reshape(-1), maxNoneZero.reshape(-1)])
z = gaussian_kde(xy)(xy)

plt.scatter(accuracy, maxNoneZero, s=1, c=z, cmap=white_viridis)
plt.title('Accuracy vs Max of none zero values')
plt.yscale('log')
plt.legend(loc='lower left')
ax.set_ylim(ymin=ymin)
plt.show()
