from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def evaluate_and_plot(
    model,
    X_train,
    X_test,
    y_train_scaled,
    y_test_scaled,
    scaler_y,
    feature_selection=True,
    cv_splits=5,
    scoring='neg_mean_squared_error',
    plot_top_n=30,
    random_state=None
):

    # Özellik seçimi
    if feature_selection:
        rfecv = RFECV(
            estimator=model,
            step=1,
            cv=KFold(n_splits=cv_splits, shuffle=True, random_state=random_state),
            scoring=scoring,
            n_jobs=-1
        )
        rfecv.fit(X_train, y_train_scaled.ravel())
        support = rfecv.support_
        print(f"Seçilen özellik sayısı: {rfecv.n_features_}")
        print("En iyi özellikler:", X_train.columns[support].tolist())
        X_train = X_train.loc[:, support]
        X_test = X_test.loc[:, support]
        selected = X_train.columns.tolist()
    else:
        selected = X_train.columns.tolist()

    # Model eğitimi
    model.fit(X_train, y_train_scaled.ravel())
    train_score = model.score(X_train, y_train_scaled)
    test_score = model.score(X_test, y_test_scaled)

    # Tahmin ve ters ölçekleme
    y_pred_scaled = model.predict(X_test)
    y_pred_log = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
    y_pred_price = np.expm1(y_pred_log)
    y_test_log = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1))
    y_test_price = np.expm1(y_test_log)

    # Değerlendirme metrikleri
    mse = mean_squared_error(y_test_price, y_pred_price)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test_price, y_pred_price)
    r2 = r2_score(y_test_price, y_pred_price)

    print(f"MAPE: {mape * 100:.2f}%")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Skoru: {r2:.4f}")

    # Özellik önemleri / katsayılar
    importance_df = None
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_.flatten())
    else:
        importances = None

    if importances is not None:
        importance_df = pd.DataFrame({'feature': selected, 'importance': importances})
        importance_df = importance_df.sort_values('importance', ascending=False)
        print(importance_df)

        # Grafik
        top_feats = importance_df.head(plot_top_n)
        plt.figure(figsize=(10, 6))
        plt.barh(top_feats['feature'][::-1], top_feats['importance'][::-1])
        plt.xlabel("Önemi / |Katsayı|")
        plt.title("En Önemli Özellikler / En Yüksek Mutlak Katsayılar")
        plt.tight_layout()
        plt.show()

    # Tahmin ve gerçek değerlerin karşılaştırılması
    plt.figure(figsize=(10,6))
    plt.scatter(y_test_price, y_pred_price, alpha=0.5)
    plt.plot([y_test_price.min(), y_test_price.max()], [y_test_price.min(), y_test_price.max()], 'r--', lw=2)
    plt.xlabel('Gerçek Değerler')
    plt.ylabel('Tahmin Değerleri')
    plt.title('Gerçek vs Tahmin')
    plt.show()

    # Kalıntı grafiği
    residuals = y_test_price - y_pred_price
    plt.figure(figsize=(10,6))
    plt.scatter(y_pred_price, residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Tahmin Değerleri')
    plt.ylabel('Residuals')
    plt.title("Residual Plot")
    plt.show()

    # Sonuç sözlüğü
    results = {
        'metrics': {
            'train_score': train_score,
            'test_score': test_score,
            'mape': mape,
            'rmse': rmse,
            'r2': r2
        },
        'selected_features': selected if feature_selection else None,
        'importance_df': importance_df,
        'y_pred_price': y_pred_price,
        'y_test_price': y_test_price
    }
    return results