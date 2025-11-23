import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import os

output_dir = 'notebooks/XGBoost'

# Функция для стандартизации названий регионов
def standardize_region_names(df):
    """Стандартизация названий регионов"""
    df_clean = df.copy()
    
    # Создаем словарь замен
    replacements = {
        'Ненецкий авт.округ': 'Ненецкий автономный округ',
        'Hенецкий авт.округ': 'Ненецкий автономный округ',
        '  Ненецкий автономный округ': 'Ненецкий автономный округ',
        
        'Ямало-Ненецкий авт.округ': 'Ямало-Ненецкий автономный округ',
        'Ямало-Hенецкий авт.округ': 'Ямало-Ненецкий автономный округ',
        '  Ямало-Ненецкий автономный округ': 'Ямало-Ненецкий автономный округ',
        
        'Ханты-Мансийский авт.округ-Югра': 'Ханты-Мансийский автономный округ - Югра',
        '  Ханты-Мансийский автономный округ - Югра': 'Ханты-Мансийский автономный округ - Югра',
        
        'Республика Татарстан(Татарстан)': 'Республика Татарстан',
        'Чувашская Республика(Чувашия)': 'Чувашская Республика',
        'Республика Северная Осетия- Алания': 'Республика Северная Осетия-Алания',
        
        'Oмская область': 'Омская область',
        'Hижегородская область': 'Нижегородская область',
        
        'г. Севастополь': 'г.Севастополь',
        'г.Москва': 'г.Москва',
        'г.Санкт-Петербург': 'г.Санкт-Петербург',
        
        'Чукотский авт.округ': 'Чукотский автономный округ',
        'Чукотский автономный округ': 'Чукотский автономный округ'
    }
    
    # Применяем замены
    df_clean['Регион'] = df_clean['Регион'].replace(replacements)
    df_clean['Регион'] = df_clean['Регион'].str.strip()
    
    return df_clean

class SKRXGBoostForecaster:
    def __init__(self, random_state=42):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_fitted = False
        self.last_known_data = None
        self.random_state = random_state
        self.first_year = None
        
    def clean_numeric_columns(self, df):
        """
        Очистка числовых колонок
        """
        df = df.copy()
        
        numeric_columns = [col for col in df.columns if col not in ['Регион', 'Год', 'СКР']]
        
        for col in numeric_columns:
            if df[col].dtype == 'object':
                df[col] = (df[col]
                          .astype(str)
                          .str.replace('\xa0', '')
                          .str.replace(' ', '')
                          .str.replace(',', '.')
                          .str.replace('−', '-')
                          .str.replace('–', '-')
                          )
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if 'СКР' in df.columns:
            df['СКР'] = pd.to_numeric(df['СКР'], errors='coerce')
        
        return df
        
    def prepare_features(self, df, is_training=True):
        """
        Подготовка признаков для СКР с сохранением данных с 2014 года
        """
        df = df.copy().sort_values(['Регион', 'Год'])
        
        # Определяем первый год для расчета тренда
        if self.first_year is None:
            self.first_year = df['Год'].min()
        
        print(f"Исходные данные: {df['Год'].min()}-{df['Год'].max()}, {len(df)} строк")
        
        # Создание лаговых признаков
        if is_training or df['Год'].min() < 2024:
            df['lag1_СКР'] = df.groupby('Регион')['СКР'].shift(1)
            df['lag2_СКР'] = df.groupby('Регион')['СКР'].shift(2)
            df['lag1_Браков'] = df.groupby('Регион')['Браков'].shift(1)
            df['lag1_Разводов'] = df.groupby('Регион')['Разводов'].shift(1)
            df['lag1_Число родившихся'] = df.groupby('Регион')['Число родившихся'].shift(1)
            
            # Скользящие средние с min_periods=1 для сохранения данных
            df['СКР_MA2'] = df.groupby('Регион')['СКР'].transform(lambda x: x.rolling(2, min_periods=1).mean())
            df['СКР_MA3'] = df.groupby('Регион')['СКР'].transform(lambda x: x.rolling(3, min_periods=1).mean())
        
        # Создание тренда времени
        df['year_trend'] = df['Год'] - self.first_year
        df['год_от_начала'] = df['Год'] - self.first_year
        
        # Создание производных показателей
        df['Браков_на_1000'] = df['Браков'] / df['Численность населения'] * 1000
        df['Разводов_на_1000'] = df['Разводов'] / df['Численность населения'] * 1000
        df['Родившихся_на_1000'] = df['Число родившихся'] / df['Численность населения'] * 1000
        df['Преступлений_на_1000'] = df['Кол-во преступлений'] / df['Численность населения'] * 1000
        
        # Дополнительные фичи
        df['соотношение_браков_разводов'] = df['Браков'] / (df['Разводов'] + 1)
        df['Социально_экономический_индекс'] = (
            df['Средняя ЗП'] / df['Величина прожиточного минимума'] - 
            (df['Уровень бедности'] / 100)
        )
        
        # Относительные изменения
        df['изменение_населения'] = df.groupby('Регион')['Численность населения'].pct_change()
        df['изменение_ВРП'] = df.groupby('Регион')['Валовой региональный продукт на душу населения (ОКВЭД 2)'].pct_change()
        
        if is_training:
            print(f"Данные до обработки NaN: {len(df)} строк")
            
            # Заполняем числовые колонки медианами по регионам
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col not in ['Год']:
                    df[col] = df.groupby('Регион')[col].transform(
                        lambda x: x.fillna(x.median()) if not x.isnull().all() else x
                    )
            
            # Если все еще есть NaN, заполняем общими медианами
            df = df.fillna(df.median(numeric_only=True))
            
            print(f"Данные после обработки NaN: {len(df)} строк")
            print(f"Годы после обработки: {df['Год'].min()}-{df['Год'].max()}")
        
        return df
    
    def train_test_split_temporal(self, df, test_years=[2021, 2022, 2023]):
        """
        Разделение на train/test по времени
        """
        train_mask = ~df['Год'].isin(test_years)
        test_mask = df['Год'].isin(test_years)
        
        X_train = df[train_mask][self.feature_names]
        X_test = df[test_mask][self.feature_names]
        y_train = df[train_mask]['СКР']
        y_test = df[test_mask]['СКР']
        
        return X_train, X_test, y_train, y_test, train_mask, test_mask
    
    def fit(self, df):
        """
        Обучение XGBoost модели для СКР на расширенных данных
        """
        # Очистка и подготовка данных
        df_clean = self.clean_numeric_columns(df)
        df_processed = self.prepare_features(df_clean, is_training=True)
        
        print(f"Данные за период: {df_processed['Год'].min()}-{df_processed['Год'].max()}")
        
        # Сохраняем последние известные данные для прогноза
        self.last_known_data = df_processed[df_processed['Год'] == 2023].copy()
        
        # Определение признаков для СКР
        self.feature_names = [
            # Демографические
            'Численность населения', 
            'Число родившихся',
            'Браков', 
            'Разводов',
            
            # Социально-экономические
            'Введено в действие общей площади жилых домов на 1000 человек населения',
            'Кол-во преступлений', 
            'Уровень безработицы', 
            'Уровень бедности', 
            'Величина прожиточного минимума',
            'Валовой региональный продукт на душу населения (ОКВЭД 2)',
            'Средняя ЗП',
            
            # Лаговые признаки
            'lag1_СКР', 
            'lag2_СКР',
            'lag1_Браков', 
            'lag1_Разводов',
            'lag1_Число родившихся',
            
            # Скользящие средние
            'СКР_MA2',
            'СКР_MA3',
            
            # Тренд и время
            'year_trend',
            'год_от_начала',
            
            # Производные показатели
            'Браков_на_1000', 
            'Разводов_на_1000',
            'Родившихся_на_1000',
            'Преступлений_на_1000',
            'соотношение_браков_разводов',
            'Социально_экономический_индекс',
            'изменение_населения',
            'изменение_ВРП'
        ]
        
        # Убираем признаки, которых нет в данных
        available_features = [f for f in self.feature_names if f in df_processed.columns]
        self.feature_names = available_features
        
        print(f"Используется {len(self.feature_names)} признаков")
        
        # Разделение на train/test
        X_train, X_test, y_train, y_test, train_mask, test_mask = self.train_test_split_temporal(df_processed)
        
        print(f"Размер train: {X_train.shape}, test: {X_test.shape}")
        print(f"Период обучения: {df_processed[train_mask]['Год'].min()}-{df_processed[train_mask]['Год'].max()}")
        print(f"Период тестирования: {df_processed[test_mask]['Год'].min()}-{df_processed[test_mask]['Год'].max()}")
        
        # Масштабирование признаков
        scale_features = [f for f in self.feature_names if not f.startswith(('lag', 'СКР_MA', 'изменение_'))]
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled[scale_features] = self.scaler.fit_transform(X_train[scale_features])
        X_test_scaled[scale_features] = self.scaler.transform(X_test[scale_features])
        
        # Создание и обучение XGBoost модели
        self.model = xgb.XGBRegressor(
            max_depth=8,
            learning_rate=0.05,
            n_estimators=200,
            random_state=self.random_state,
            n_jobs=-1,
            subsample=0.8,
            colsample_bytree=0.8
        )
        
        print("Обучение XGBoost модели на расширенных данных...")
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True
        
        # Предсказание на тесте
        y_pred = self.model.predict(X_test_scaled)
        
        # Метрики качества
        self.calculate_metrics(y_test, y_pred)
        
        # Дополнительная валидация
        self.cross_validation(X_train_scaled, y_train)
        
        # Сохранение результатов
        self.results = {
            'X_train': X_train_scaled, 'X_test': X_test_scaled,
            'y_train': y_train, 'y_test': y_test, 'y_pred': y_pred,
            'df_processed': df_processed
        }
        
        return self
    
    def calculate_metrics(self, y_true, y_pred):
        """
        Расчет метрик качества для СКР
        """
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        print("\n" + "="*50)
        print("МЕТРИКИ КАЧЕСТВА XGBoost ДЛЯ СКР (2014-2023)")
        print("="*50)
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R²: {r2:.4f}")
        print(f"Средний СКР в тесте: {y_true.mean():.4f}")
        print(f"Относительная ошибка: {rmse/y_true.mean()*100:.2f}%")
        
        # Дополнительные метрики
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        max_error = np.max(np.abs(y_true - y_pred))
        print(f"MAPE: {mape:.2f}%")
        print(f"Максимальная ошибка: {max_error:.4f}")
        
        self.metrics = {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape, 'MaxError': max_error}
    
    def cross_validation(self, X_train, y_train, cv=5):
        """
        Кросс-валидация для оценки устойчивости модели
        """
        print("\nКросс-валидация XGBoost (RMSE):")
        
        temp_model = xgb.XGBRegressor(
            max_depth=8,
            learning_rate=0.05,
            n_estimators=200,
            random_state=self.random_state,
            n_jobs=-1,
            subsample=0.8,
            colsample_bytree=0.8
        )
        
        try:
            scores = cross_val_score(temp_model, X_train, y_train, 
                                   scoring='neg_mean_squared_error', cv=cv)
            rmse_scores = np.sqrt(-scores)
            print(f"Среднее: {rmse_scores.mean():.4f} (+/- {rmse_scores.std() * 2:.4f})")
        except Exception as e:
            print(f"Ошибка при кросс-валидации: {e}")
    
    def save_model(self, filepath=None):
        """
        Сохранение обученной модели и всех компонентов
        """
        if not self.is_fitted:
            print("Модель не обучена!")
            return False
        
        if filepath is None:
            filepath = os.path.join(output_dir, 'skr_xgboost_model.pkl')
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'first_year': self.first_year,
            'last_known_data': self.last_known_data,
            'metrics': self.metrics
        }
        
        joblib.dump(model_data, filepath)
        print(f"XGBoost модель СКР сохранена в {filepath}")
        return True
    
    def load_model(self, filepath=None):
        """
        Загрузка обученной модели
        """
        if filepath is None:
            filepath = os.path.join(output_dir, 'skr_xgboost_model.pkl')
            
        if not os.path.exists(filepath):
            print(f"Файл модели {filepath} не найден!")
            return False
        
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.first_year = model_data['first_year']
        self.last_known_data = model_data['last_known_data']
        self.metrics = model_data['metrics']
        self.is_fitted = True
        
        print(f"XGBoost модель СКР загружена из {filepath}")
        return True

    def prepare_final_output(self, predictions_df, target_year=2024):
        """
        Подготовка финального файла для GitHub пайплайна
        """
        final_output = pd.DataFrame({
            'Регион': predictions_df['Регион'],
            'Год': predictions_df['Год'],
            'СКР': predictions_df['СКР_предыдущий'],
            'predictions': predictions_df['СКР_прогноз']
        })
        
        return final_output

    def predict_future(self, df, future_years=[2024]):
        """
        Прогноз СКР на будущие периоды
        """
        if not self.is_fitted:
            print("Сначала обучите модель!")
            return None
        
        if self.last_known_data is None:
            print("Нет данных для прогноза!")
            return None
        
        print(f"\nСоздание прогноза СКР на {future_years} год (XGBoost)...")
        
        all_predictions = []
        
        for year in future_years:
            year_predictions = []
            
            for _, last_row in self.last_known_data.iterrows():
                # Создаем строку для прогноза
                future_row = last_row.copy()
                future_row['Год'] = year
                future_row['year_trend'] = year - self.first_year
                future_row['год_от_начала'] = year - self.first_year
                
                # Прогноз основных показателей
                future_row['Численность населения'] = last_row['Численность населения'] * 1.003
                future_row['Число родившихся'] = last_row['Число родившихся'] * 0.995
                future_row['Браков'] = last_row['Браков'] * 1.01
                future_row['Разводов'] = last_row['Разводов'] * 1.005
                future_row['Валовой региональный продукт на душу населения (ОКВЭД 2)'] = last_row['Валовой региональный продукт на душу населения (ОКВЭД 2)'] * 1.02
                future_row['Средняя ЗП'] = last_row['Средняя ЗП'] * 1.04
                future_row['Величина прожиточного минимума'] = last_row['Величина прожиточного минимума'] * 1.03
                future_row['Уровень бедности'] = last_row['Уровень бедности'] * 0.98
                
                # Лаги берем из последнего известного года
                future_row['lag1_СКР'] = last_row['СКР']
                future_row['lag2_СКР'] = last_row['lag1_СКР']
                future_row['lag1_Браков'] = last_row['Браков']
                future_row['lag1_Разводов'] = last_row['Разводов']
                future_row['lag1_Число родившихся'] = last_row['Число родившихся']
                
                # Пересчитываем скользящие средние
                future_row['СКР_MA2'] = (future_row['lag1_СКР'] + future_row['СКР']) / 2
                future_row['СКР_MA3'] = (future_row['lag2_СКР'] + future_row['lag1_СКР'] + future_row['СКР']) / 3
                
                # Пересчитываем производные показатели
                future_row['Браков_на_1000'] = future_row['Браков'] / future_row['Численность населения'] * 1000
                future_row['Разводов_на_1000'] = future_row['Разводов'] / future_row['Численность населения'] * 1000
                future_row['Родившихся_на_1000'] = future_row['Число родившихся'] / future_row['Численность населения'] * 1000
                future_row['Преступлений_на_1000'] = future_row['Кол-во преступлений'] / future_row['Численность населения'] * 1000
                future_row['соотношение_браков_разводов'] = future_row['Браков'] / (future_row['Разводов'] + 1)
                future_row['Социально_экономический_индекс'] = (
                    future_row['Средняя ЗП'] / future_row['Величина прожиточного минимума'] - 
                    (future_row['Уровень бедности'] / 100)
                )
                future_row['изменение_населения'] = (future_row['Численность населения'] - last_row['Численность населения']) / last_row['Численность населения']
                future_row['изменение_ВРП'] = (future_row['Валовой региональный продукт на душу населения (ОКВЭД 2)'] - last_row['Валовой региональный продукт на душу населения (ОКВЭД 2)']) / last_row['Валовой региональный продукт на душу населения (ОКВЭД 2)']
                
                year_predictions.append(future_row)
            
            # Прогноз на конкретный год
            future_df = pd.DataFrame(year_predictions)
            X_future = future_df[self.feature_names]
            
            # Масштабирование
            scale_features = [f for f in self.feature_names if not f.startswith(('lag', 'СКР_MA', 'изменение_'))]
            X_future_scaled = X_future.copy()
            X_future_scaled[scale_features] = self.scaler.transform(X_future[scale_features])
            
            # Прогноз СКР
            predictions = self.model.predict(X_future_scaled)
            
            # Сохраняем результаты
            for i, (_, last_row) in enumerate(self.last_known_data.iterrows()):
                all_predictions.append({
                    'Регион': last_row['Регион'],
                    'Год': year,
                    'СКР_прогноз': predictions[i],
                    'СКР_предыдущий': last_row['СКР'],
                    'Изменение_СКР': predictions[i] - last_row['СКР']
                })
            
            # Обновляем последние данные
            self.last_known_data = future_df.copy()
            self.last_known_data['СКР'] = predictions
        
        results_df = pd.DataFrame(all_predictions)
        
        print("Прогноз XGBoost успешно создан")
        return results_df

if __name__ == "__main__":
    df_skr = pd.read_excel('Финальный вариант/общая_СКР (2).xlsx')
    
    # Стандартизация названий регионов
    print("Стандартизация названий регионов...")
    df_skr = standardize_region_names(df_skr)
    print(f"Уникальных регионов после очистки: {df_skr['Регион'].nunique()}")
    
    print("="*70)
    print("XGBOOST МОДЕЛЬ ДЛЯ ПРОГНОЗА СКР (2014-2023)")
    print("="*70)
    
    # Диагностика данных
    print(f"Размер данных: {df_skr.shape}")
    print(f"Период данных: {df_skr['Год'].min()}-{df_skr['Год'].max()}")
    print(f"Количество регионов: {df_skr['Регион'].nunique()}")
    
    # Обучение модели
    skr_forecaster = SKRXGBoostForecaster()
    skr_forecaster.fit(df_skr)
    
    # Сохранение модели
    skr_forecaster.save_model()
    
    # Прогноз на 2024 год
    future_predictions = skr_forecaster.predict_future(df_skr, [2024])
    
    if future_predictions is not None:
        # Подготовка финального вывода для GitHub пайплайна
        final_output = skr_forecaster.prepare_final_output(future_predictions, 2024)
        
        # Сохранение в формате для пайплайна
        predictions_filepath = os.path.join(output_dir, 'predictions_afr.xlsx')
        final_output.to_excel(predictions_filepath, index=False)
        print(f"✓ Финальный файл для пайплайна сохранен как '{predictions_filepath}'")
        
        # Вывод статистики
        print(f"\nСТАТИСТИКА ПРОГНОЗА XGBoost:")
        print(f"Средний СКР в 2024: {future_predictions['СКР_прогноз'].mean():.3f}")
        print(f"Регионов с ростом СКР: {(future_predictions['Изменение_СКР'] > 0).sum()}")
        print(f"Регионов со снижением СКР: {(future_predictions['Изменение_СКР'] < 0).sum()}")
        
        # Пример первых строк финального файла
        print(f"\nПервые 5 строк финального файла:")
        print(final_output.head())
        
        # Дополнительно: сохранение полных результатов для анализа
        full_results_filepath = os.path.join(output_dir, 'full_skr_predictions_results_xgboost.csv')
        future_predictions.to_csv(full_results_filepath, index=False)
        print(f"Полные результаты XGBoost сохранены в '{full_results_filepath}'")