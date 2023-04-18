import numpy as np
import matplotlib.pyplot as plt
import csv

# Load the data
data = []
with open('metrics.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    [data.append([int(col) for col in row]) for row in csv_reader]

temp_ = np.array([el for arr in data for el in arr])
cm = plt.get_cmap('gist_rainbow')

# Plot the data in a bar chart
temp = temp_[temp_ != 0]
count = temp.shape[0]

bins=[i for i in range(1, 11)]
bins_=[str(i) for i in range(1, 10)] + ['10+']
binsData = [int(np.count_nonzero(temp == bin if bin != 10 else temp >= bin)) for bin in bins]

binsData_ = [i / count for i in binsData]

plt.bar(bins_, binsData_, color=[cm(1.*i/len(bins)) for i in range(len(bins))])
plt.xlabel('Number of indices away from the correct answer')
plt.ylabel('Proportion')
plt.title('The distribution of the models accuracy,\nexcluding the correct answer')
plt.show()

# pie chart
temp = temp_
count = temp.shape[0]
bins_=[str(i) for i in range(0, 10)] + ['10+']
binsData = [int(np.count_nonzero(temp == bin if bin != 10 else temp >= bin)) for bin in range(11)]

unique, counts = np.unique(temp_, return_counts=True)
temp_ = np.asarray((unique, counts)).T

cm = plt.get_cmap('gist_rainbow')
patches, texts = plt.pie(binsData,
        labels=['0'] + ['' for i in range(1, 10)] + ['10+'],
        colors=[cm(1.*i/len(binsData)) for i in range(len(binsData))],
        startangle=90,
        radius=1.2,
        explode=[0.1 if (i == 0 or i == 10) else 0 for i in range(len(binsData))],
        labeldistance=0.8,
        )
labels = [f'{i} | {j} | {100*j/count:1.1f}%' for i,j in zip(bins_, binsData)]


leg = plt.legend(patches, labels,
            loc='center left',
            bbox_to_anchor=(-.15, .5),
            # font="monospace",
            fontsize=10)
leg._legend_box.align = "left"
leg._legend_box.font = "monospace"

plt.title('Pie chart of the models accuracy,\nincluding the correct answer')
plt.axis('equal')
plt.show()
