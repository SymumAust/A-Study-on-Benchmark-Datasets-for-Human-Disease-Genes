import matplotlib.pyplot as plt

models = ["SVM", "RF", "DT", "NN"]
accuracy = [0.6041, 0.8571, 0.7291, 0.8065]

colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd']  # Bold colors for each model

plt.figure(figsize=(10, 6))
bars = plt.bar(models, accuracy, color=colors)
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')

# Adding data values on top of each bar
for bar, acc in zip(bars, accuracy):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval, round(acc, 4), ha='center', va='bottom', fontweight='bold')

plt.ylim(0, 1)  # Set y-axis limits to ensure accuracy values are displayed appropriately
plt.show()
