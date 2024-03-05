import numpy as np
from sklearn import metrics

# K (number of shots)
x = np.array([1., 10., 30., 50., 100., 300., 500., 1000.])
x_log = np.log(x) / np.log(1000)
# Average Recall scores
y = np.array([0.0, 18.2, 27.0, 30.6, 34.7, 40.2, 42.8, 46.1]) # 双卡，oln模型， AUC = 25.5188

y *= 0.01
auc = metrics.auc(x_log, y)
print('AUC score:', auc)
