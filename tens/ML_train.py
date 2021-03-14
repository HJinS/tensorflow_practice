from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.metrics import mean_squared_error
# 우선 데이터세트 로드
X, y = load_data()

# 훈련, 테스트 데이터세트 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 파이프라인을 통한 전처리
pipeline = Pipeline([
                        ('imputer', SimpleImputer(strategy="median")),
                        ('attribs_adder', CombinedAttributesAdder()),
                        ('std_scaler', StandardScaler())
                    ])
X_train = pipeline.fit_transform(X_train)

# 모델 선언 및 학습: Model은 존재하지 않는 클래스로, 알맞은 모델 클래스를 선언하면 됩니다.
model = Model() model.fit(X_train, y_train)

# 모델 테스트세트 예측
X_test = pipeline.transform(X_test)
y_pred = model.predict(X_test)

# 모델 평가: 예를 들어 MSE, RMSE 평가 기준 값입니다.
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

