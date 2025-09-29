import csv

# Предположим, что у тебя есть уже 4 результата
components_list_done = [2, 12, 22, 32]
accuracies_done = [0.600, 0.553, 0.579, 0.584]
times_done = [214.983, 5894.675, 1997.135, 2122.804]

with open("pca_results_partial_manual.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["n_components", "accuracy", "time_seconds"])
    
    for n, acc, t in zip(components_list_done, accuracies_done, times_done):
        writer.writerow([n, acc, t])

print("Промежуточные результаты вручную сохранены!")