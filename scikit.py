import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt

# 设置matplotlib支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 避免坐标轴负号显示问题

# 读取CSV文件
data = pd.read_csv('category.csv')

# 数据预处理，将汉字描述转换为特征数据
X = data.iloc[:, :-1]  # 前6列是特征
y = data.iloc[:, -1]   # 最后一列是标签

# 使用OneHotEncoder进行独热编码
ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [i for i in range(X.shape[1])])],
                       remainder='passthrough')

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练决策树模型
dt_clf = DecisionTreeClassifier(random_state=42)
dt_clf.fit(ct.fit_transform(X_train), y_train)
dt_pred = dt_clf.predict(ct.transform(X_test))
dt_accuracy = accuracy_score(y_test, dt_pred)

# 训练KNN模型
knn_clf = KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(ct.fit_transform(X_train), y_train)
knn_pred = knn_clf.predict(ct.transform(X_test))
knn_accuracy = accuracy_score(y_test, knn_pred)

# 训练朴素贝叶斯模型
gnb_clf = GaussianNB()
gnb_clf.fit(ct.fit_transform(X_train), y_train)
gnb_pred = gnb_clf.predict(ct.transform(X_test))
gnb_accuracy = accuracy_score(y_test, gnb_pred)

# 训练SVM模型
svm_clf = SVC(kernel='linear')
svm_clf.fit(ct.fit_transform(X_train), y_train)
svm_pred = svm_clf.predict(ct.transform(X_test))
svm_accuracy = accuracy_score(y_test, svm_pred)

# 训练随机森林模型
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(ct.fit_transform(X_train), y_train)
rf_pred = rf_clf.predict(ct.transform(X_test))
rf_accuracy = accuracy_score(y_test, rf_pred)

# 训练梯度提升树模型
gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gb_clf.fit(ct.fit_transform(X_train), y_train)
gb_pred = gb_clf.predict(ct.transform(X_test))
gb_accuracy = accuracy_score(y_test, gb_pred)

# 模型名称和对应的准确率
models = ['决策树', 'KNN', '朴素贝叶斯', 'SVM', '随机森林', '梯度提升树']
accuracies = [dt_accuracy, knn_accuracy, gnb_accuracy, svm_accuracy, rf_accuracy, gb_accuracy]

# 生成柱状图
plt.figure(figsize=(10, 6))
plt.bar(models, accuracies, color='skyblue')
plt.xlabel('模型')
plt.ylabel('准确率')
plt.title('不同模型的准确率')
plt.ylim(0, 1)  # 假设准确率的范围在0到1之间
plt.xticks(rotation=45)  # 旋转x轴标签以便更好地显示
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 显示图形
plt.show()


# 2220667吴旺阳