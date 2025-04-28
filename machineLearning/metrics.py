def accuracy(TP, TN, FP, FN):
    return (TP + TN) / (TP + TN + FP + FN);

#how many between the all positive classified are actually positive?
def precision(TP, TN, FP, FN):
    return (TP) / (TP + FP );

#corretly positivve classified to all the actual positive
def recall(TP, TN, FP, FN):
    return (TP) / (TP + FN);


def F1Score(TP, TN, FP, FN):
    return (2) / (1/recall(42,32,8,18) + 1/precision(42,32,8,18));

print(accuracy(42,32,8,18));
print(precision(42,32,8,18));
print(recall(42,32,8,18));
print(F1Score(42,32,8,18));