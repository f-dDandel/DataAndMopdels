# Random Forest для ОПЖ с данными за 2014-2023 и прогнозом на 2019
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Создаем папку для сохранения результатов
output_dir = 'notebooks/Random Forest'

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
    
    return df_clean

class OPJForecaster:
    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            max_depth=10,
            min_samples_split=5,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_fitted = False
        self.last_known_data = None
        self.first_year = None
        
    def clean_numeric_columns(self, df):
        """
        Очистка числовых колонок
        """
        df = df.copy()
        
        numeric_columns = [col for col in df.columns if col not in ['Регион', 'Год', 'ОПЖ']]
        
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
        
        if 'ОПЖ' in df.columns:
            df['ОПЖ'] = pd.to_numeric(df['ОПЖ'], errors='coerce')
        
        return df
        
    def prepare_features(self, df, is_training=True):
        """
        Подготовка признаков для ОПЖ с сохранением данных с 2014 года
        """
        df = df.copy().sort_values(['Регион', 'Год'])
        
        # Определяем первый год для расчета тренда
        if self.first_year is None:
            self.first_year = df['Год'].min()
        
        print(f"Исходные данные: {df['Год'].min()}-{df['Год'].max()}, {len(df)} строк")
        
        # Создание лаговых признаков
        if is_training or df['Год'].min() < 2024:
            df['lag1_ОПЖ'] = df.groupby('Регион')['ОПЖ'].shift(1)
            df['lag2_ОПЖ'] = df.groupby('Регион')['ОПЖ'].shift(2)
            df['lag1_Число умерших'] = df.groupby('Регион')['Число умерших'].shift(1)
            df['lag1_Младенческая смертность коэф'] = df.groupby('Регион')['Младенческая смертность коэф'].shift(1)
            df['lag1_Общая численность инвалидов'] = df.groupby('Регион')['Общая численность инвалидов'].shift(1)
            
            # Скользящие средние с min_periods=1 для сохранения данных
            df['ОПЖ_MA2'] = df.groupby('Регион')['ОПЖ'].transform(lambda x: x.rolling(2, min_periods=1).mean())
            df['ОПЖ_MA3'] = df.groupby('Регион')['ОПЖ'].transform(lambda x: x.rolling(3, min_periods=1).mean())
        
        # Создание тренда времени
        df['year_trend'] = df['Год'] - self.first_year
        df['год_от_начала'] = df['Год'] - self.first_year
        
        # Создание производных медицинских и социальных показателей
        df['Врачей_на_10k'] = df['Численность врачей всех специальностей'] / df['Численность населения'] * 10000
        df['Умерших_на_1000'] = df['Число умерших'] / df['Численность населения'] * 1000
        df['Инвалидов_на_1000'] = df['Общая численность инвалидов'] / df['Численность населения'] * 1000
        df['Преступлений_на_1000'] = df['Кол-во преступлений'] / df['Численность населения'] * 1000
        df['Браков_на_1000'] = df['Браков'] / df['Численность населения'] * 1000
        df['Разводов_на_1000'] = df['Разводов'] / df['Численность населения'] * 1000
        
        # Индексы развития
        df['Индекс_здравоохранения'] = (
            df['Врачей_на_10k'] + 
            df['Число больничных организаций на конец отчетного года'] + 
            df['Число санаторно-курортных организаций']
        ) / 3
        
        df['Социальный_индекс'] = (
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
                if col not in ['Год']:  # не заполняем год
                    # Заполняем медианами по регионам
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
        Разделение на train/test по времени с учетом большего количества данных
        """
        train_mask = ~df['Год'].isin(test_years)
        test_mask = df['Год'].isin(test_years)
        
        X_train = df[train_mask][self.feature_names]
        X_test = df[test_mask][self.feature_names]
        y_train = df[train_mask]['ОПЖ']
        y_test = df[test_mask]['ОПЖ']
        
        return X_train, X_test, y_train, y_test, train_mask, test_mask
    
    def fit(self, df, test_years=[2019, 2020, 2021]):
        """
        Обучение модели для ОПЖ на расширенных данных
        """
        # Очистка и подготовка данных
        df_clean = self.clean_numeric_columns(df)
        df_processed = self.prepare_features(df_clean, is_training=True)
        
        print(f"Данные за период: {df_processed['Год'].min()}-{df_processed['Год'].max()}")
        
        # Сохраняем последние известные данные для прогноза
        self.last_known_data = df_processed[df_processed['Год'] == 2018].copy()
        
        # Определение признаков для ОПЖ с учетом расширенного набора данных
        self.feature_names = [
            # Базовые демографические
            'Численность населения', 
            'Число умерших', 
            'Общая численность инвалидов',
            'Браков',
            'Разводов',
            
            # Медицинские факторы
            'Младенческая смертность коэф', 
            'Численность врачей всех специальностей', 
            'Число больничных организаций на конец отчетного года', 
            'Число санаторно-курортных организаций',
            
            # Социально-экономические
            'Валовой региональный продукт на душу населения (ОКВЭД 2)',
            'Величина прожиточного минимума', 
            'Уровень бедности', 
            'Средняя ЗП',
            'Кол-во преступлений',
            
            # Лаговые признаки
            'lag1_ОПЖ', 
            'lag2_ОПЖ',
            'lag1_Число умерших', 
            'lag1_Младенческая смертность коэф', 
            'lag1_Общая численность инвалидов',
            
            # Скользящие средние
            'ОПЖ_MA2',
            'ОПЖ_MA3',
            
            # Тренд и время
            'year_trend',
            'год_от_начала',
            
            # Производные показатели
            'Врачей_на_10k', 
            'Умерших_на_1000', 
            'Инвалидов_на_1000', 
            'Преступлений_на_1000',
            'Браков_на_1000',
            'Разводов_на_1000',
            'Индекс_здравоохранения', 
            'Социальный_индекс',
            'изменение_населения',
            'изменение_ВРП'
        ]
        
        # Убираем признаки, которых нет в данных
        available_features = [f for f in self.feature_names if f in df_processed.columns]
        self.feature_names = available_features
        
        print(f"Используется {len(self.feature_names)} признаков")
        
        # Разделение на train/test
        X_train, X_test, y_train, y_test, train_mask, test_mask = self.train_test_split_temporal(df_processed, test_years)
        
        print(f"Размер train: {X_train.shape}, test: {X_test.shape}")
        print(f"Период обучения: {df_processed[train_mask]['Год'].min()}-{df_processed[train_mask]['Год'].max()}")
        print(f"Период тестирования: {df_processed[test_mask]['Год'].min()}-{df_processed[test_mask]['Год'].max()}")
        
        # Масштабирование признаков
        scale_features = [f for f in self.feature_names if not f.startswith(('lag', 'ОПЖ_MA', 'изменение_'))]
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled[scale_features] = self.scaler.fit_transform(X_train[scale_features])
        X_test_scaled[scale_features] = self.scaler.transform(X_test[scale_features])
        
        # Обучение модели
        print("Обучение модели на расширенных данных...")
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
            'df_processed': df_processed,
            'test_years': test_years
        }
        
        return self
    
    def calculate_metrics(self, y_true, y_pred):
        """
        Расчет метрик качества для ОПЖ
        """
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        print("\n" + "="*50)
        print("МЕТРИКИ КАЧЕСТВА Random Forest ДЛЯ ОПЖ")
        print("="*50)
        print(f"RMSE: {rmse:.4f} лет")
        print(f"MAE: {mae:.4f} лет")
        print(f"R²: {r2:.4f}")
        print(f"Средняя ОПЖ в тесте: {y_true.mean():.2f} лет")
        print(f"Относительная ошибка: {rmse/y_true.mean()*100:.2f}%")
        
        # Дополнительные метрики
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        max_error = np.max(np.abs(y_true - y_pred))
        median_error = np.median(np.abs(y_true - y_pred))
        print(f"MAPE: {mape:.2f}%")
        print(f"Максимальная ошибка: {max_error:.2f} лет")
        print(f"Медианная ошибка: {median_error:.2f} лет")
        
        self.metrics = {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape, 'MaxError': max_error, 'MedianError': median_error}
    
    def cross_validation(self, X_train, y_train, cv=5):
        """
        Кросс-валидация для оценки устойчивости модели
        """
        print("\nКросс-валидация (RMSE):")
        scores = cross_val_score(self.model, X_train, y_train, 
                               scoring='neg_mean_squared_error', cv=cv)
        rmse_scores = np.sqrt(-scores)
        print(f"Среднее: {rmse_scores.mean():.4f} (+/- {rmse_scores.std() * 2:.4f})")
    
    def save_model(self, filepath=None):
        """
        Сохранение обученной модели и всех компонентов
        """
        if not self.is_fitted:
            print("Модель не обучена!")
            return False
        
        if filepath is None:
            filepath = os.path.join(output_dir, 'opj_random_forest_model.pkl')
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'first_year': self.first_year,
            'last_known_data': self.last_known_data,
            'metrics': self.metrics
        }
        
        joblib.dump(model_data, filepath)
        print(f"✓ Модель сохранена в {filepath}")
        return True
    
    def load_model(self, filepath=None):
        """
        Загрузка обученной модели
        """
        if filepath is None:
            filepath = os.path.join(output_dir, 'opj_random_forest_model.pkl')
            
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
        
        print(f"✓ Модель загружена из {filepath}")
        return True

    def prepare_final_output(self, predictions_df, target_year=2019):
        """
        Подготовка финального файла с прогнозами и реальными значениями
        """
        # Создаем DataFrame с прогнозами и реальными значениями
        final_output = pd.DataFrame({
            'Регион': predictions_df['Регион'],
            'Год': predictions_df['Год'],
            'ОПЖ_прогноз': predictions_df['ОПЖ_прогноз'],
            'ОПЖ_реальный': predictions_df['ОПЖ_реальный'],
            'Ошибка_абсолютная': predictions_df['Ошибка_абсолютная'],
            'Ошибка_относительная_%': predictions_df['Ошибка_относительная_%']
        })
        
        return final_output

    def predict_for_year(self, df, target_year=2019):
        """
        Прогноз ОПЖ на 2019 год и сравнение с реальными значениями
        """
        if not self.is_fitted:
            print("Сначала обучите модель!")
            return None
        
        print(f"\nСоздание прогноза ОПЖ на {target_year} год...")
        
        # Получаем реальные значения за 2019 год
        real_2019 = df[df['Год'] == target_year][['Регион', 'ОПЖ']].copy()
        real_2019 = real_2019.rename(columns={'ОПЖ': 'ОПЖ_реальный'})
        
        # Получаем данные за 2018 год для прогноза
        data_2018 = self.last_known_data.copy()
        
        all_predictions = []
        
        for _, last_row in data_2018.iterrows():
            # Создаем строку для прогноза
            future_row = last_row.copy()
            future_row['Год'] = target_year
            future_row['year_trend'] = target_year - self.first_year
            future_row['год_от_начала'] = target_year - self.first_year
            
            # Прогноз основных показателей
            future_row['Численность населения'] = last_row['Численность населения'] * 1.003
            future_row['Число умерших'] = last_row['Число умерших'] * 0.995
            future_row['Общая численность инвалидов'] = last_row['Общая численность инвалидов']
            future_row['Младенческая смертность коэф'] = last_row['Младенческая смертность коэф'] * 0.98
            future_row['Валовой региональный продукт на душу населения (ОКВЭД 2)'] = last_row['Валовой региональный продукт на душу населения (ОКВЭД 2)'] * 1.02
            future_row['Средняя ЗП'] = last_row['Средняя ЗП'] * 1.04
            future_row['Численность врачей всех специальностей'] = last_row['Численность врачей всех специальностей'] * 1.02
            future_row['Величина прожиточного минимума'] = last_row['Величина прожиточного минимума'] * 1.03
            future_row['Уровень бедности'] = last_row['Уровень бедности'] * 0.98
            future_row['Браков'] = last_row['Браков'] * 1.01
            future_row['Разводов'] = last_row['Разводов'] * 1.005
            
            # Лаги берем из 2018 года
            future_row['lag1_ОПЖ'] = last_row['ОПЖ']
            future_row['lag2_ОПЖ'] = last_row['lag1_ОПЖ']
            future_row['lag1_Число умерших'] = last_row['Число умерших']
            future_row['lag1_Младенческая смертность коэф'] = last_row['Младенческая смертность коэф']
            future_row['lag1_Общая численность инвалидов'] = last_row['Общая численность инвалидов']
            
            # Пересчитываем скользящие средние
            future_row['ОПЖ_MA2'] = (future_row['lag1_ОПЖ'] + future_row['ОПЖ']) / 2
            future_row['ОПЖ_MA3'] = (future_row['lag2_ОПЖ'] + future_row['lag1_ОПЖ'] + future_row['ОПЖ']) / 3
            
            # Пересчитываем производные показатели
            future_row['Врачей_на_10k'] = future_row['Численность врачей всех специальностей'] / future_row['Численность населения'] * 10000
            future_row['Умерших_на_1000'] = future_row['Число умерших'] / future_row['Численность населения'] * 1000
            future_row['Инвалидов_на_1000'] = future_row['Общая численность инвалидов'] / future_row['Численность населения'] * 1000
            future_row['Преступлений_на_1000'] = future_row['Кол-во преступлений'] / future_row['Численность населения'] * 1000
            future_row['Браков_на_1000'] = future_row['Браков'] / future_row['Численность населения'] * 1000
            future_row['Разводов_на_1000'] = future_row['Разводов'] / future_row['Численность населения'] * 1000
            future_row['Индекс_здравоохранения'] = (
                future_row['Врачей_на_10k'] + 
                future_row['Число больничных организаций на конец отчетного года'] + 
                future_row['Число санаторно-курортных организаций']
            ) / 3
            future_row['Социальный_индекс'] = (
                future_row['Средняя ЗП'] / future_row['Величина прожиточного минимума'] - 
                (future_row['Уровень бедности'] / 100)
            )
            future_row['изменение_населения'] = (future_row['Численность населения'] - last_row['Численность населения']) / last_row['Численность населения']
            future_row['изменение_ВРП'] = (future_row['Валовой региональный продукт на душу населения (ОКВЭД 2)'] - last_row['Валовой региональный продукт на душу населения (ОКВЭД 2)']) / last_row['Валовой региональный продукт на душу населения (ОКВЭД 2)']
            
            all_predictions.append(future_row)
        
        # Создаем DataFrame для прогноза
        future_df = pd.DataFrame(all_predictions)
        X_future = future_df[self.feature_names]
        
        # Масштабирование
        scale_features = [f for f in self.feature_names if not f.startswith(('lag', 'ОПЖ_MA', 'изменение_'))]
        X_future_scaled = X_future.copy()
        X_future_scaled[scale_features] = self.scaler.transform(X_future[scale_features])
        
        # Прогноз ОПЖ
        predictions = self.model.predict(X_future_scaled)
        
        # Собираем результаты
        results = []
        for i, (_, last_row) in enumerate(data_2018.iterrows()):
            results.append({
                'Регион': last_row['Регион'],
                'Год': target_year,
                'ОПЖ_прогноз': predictions[i],
                'ОПЖ_2018': last_row['ОПЖ']
            })
        
        results_df = pd.DataFrame(results)
        
        # Объединяем с реальными значениями
        comparison_df = pd.merge(results_df, real_2019, on='Регион', how='inner')
        
        # Рассчитываем ошибки
        comparison_df['Ошибка_абсолютная'] = comparison_df['ОПЖ_реальный'] - comparison_df['ОПЖ_прогноз']
        comparison_df['Ошибка_абсолютная_abs'] = np.abs(comparison_df['Ошибка_абсолютная'])
        comparison_df['Ошибка_относительная_%'] = (comparison_df['Ошибка_абсолютная'] / comparison_df['ОПЖ_реальный']) * 100
        
        print(f"\nПрогноз на {target_year} год успешно создан")
        print(f"Сравнение выполнено для {len(comparison_df)} регионов")
        
        # Рассчитываем метрики для прогноза
        self.calculate_prediction_metrics(comparison_df['ОПЖ_реальный'], comparison_df['ОПЖ_прогноз'])
        
        return comparison_df
    
    def calculate_prediction_metrics(self, y_true, y_pred):
        """
        Расчет метрик качества для прогноза
        """
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        print("\n" + "="*50)
        print(f"МЕТРИКИ КАЧЕСТВА ПРОГНОЗА НА 2019 ГОД")
        print("="*50)
        print(f"RMSE: {rmse:.4f} лет")
        print(f"MAE: {mae:.4f} лет")
        print(f"R²: {r2:.4f}")
        print(f"MAPE: {mape:.2f}%")
        print(f"Средняя реальная ОПЖ: {y_true.mean():.2f} лет")
        print(f"Средняя прогнозная ОПЖ: {y_pred.mean():.2f} лет")
        print(f"Разница: {y_true.mean() - y_pred.mean():.4f} лет")
        
        # Сохраняем метрики прогноза
        self.prediction_metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape,
            'Mean_Actual': y_true.mean(),
            'Mean_Predicted': y_pred.mean()
        }

# ИСПОЛЬЗОВАНИЕ:
if __name__ == "__main__":
    # Загрузка данных для ОПЖ 
    df_opj = pd.read_excel('Финальный вариант/общая_ОПЖ (2).xlsx')
    
    # Стандартизация названий регионов
    print("Стандартизация названий регионов...")
    df_opj = standardize_region_names(df_opj)
    print(f"Уникальных регионов после очистки: {df_opj['Регион'].nunique()}")
    
    # Проверка наличия данных за 2019 год
    years_in_data = sorted(df_opj['Год'].unique())
    print(f"Годы в данных: {years_in_data}")
    
    if 2019 not in df_opj['Год'].unique():
        print("ВНИМАНИЕ: В данных нет 2019 года!")
        print("Используем последний доступный год для тестирования...")
        last_year = max(df_opj['Год'].unique())
        print(f"Последний доступный год: {last_year}")
    else:
        print("Данные за 2019 год присутствуют")
    
    print("="*70)
    print("МОДЕЛЬ СЛУЧАЙНОГО ЛЕСА ДЛЯ ПРОГНОЗА ОПЖ")
    print("Тестирование на 2019 году")
    print("="*70)
    
    # Диагностика данных
    print(f"Размер данных: {df_opj.shape}")
    print(f"Период данных: {df_opj['Год'].min()}-{df_opj['Год'].max()}")
    print(f"Количество регионов: {df_opj['Регион'].nunique()}")
    
    # Обучение модели на данных до 2018 года
    print("\nОбучение модели на данных до 2018 года...")
    opj_forecaster = OPJForecaster(n_estimators=200)
    
    # Обучаем на данных до 2018 года, тестируем на 2019
    opj_forecaster.fit(df_opj, test_years=[2019, 2020, 2021])
    
    # Сохранение модели
    opj_forecaster.save_model()
    
    # Прогноз на 2019 год
    comparison_results = opj_forecaster.predict_for_year(df_opj, target_year=2019)
    
    if comparison_results is not None:
        # Подготовка финального вывода
        final_output = opj_forecaster.prepare_final_output(comparison_results, target_year=2019)
        
        # Сохранение результатов
        predictions_filepath = os.path.join(output_dir, 'opj_predictions_2019_comparison.xlsx')
        final_output.to_excel(predictions_filepath, index=False)
        print(f"\nФинальный файл с прогнозами и сравнением сохранен как '{predictions_filepath}'")
        
        # Вывод детальной статистики
        print(f"\nДЕТАЛЬНАЯ СТАТИСТИКА ПРОГНОЗА НА 2019 ГОД:")
        print("-" * 50)
        print(f"Всего регионов: {len(comparison_results)}")
        print(f"Средняя реальная ОПЖ: {comparison_results['ОПЖ_реальный'].mean():.2f} лет")
        print(f"Средняя прогнозная ОПЖ: {comparison_results['ОПЖ_прогноз'].mean():.2f} лет")
        print(f"Средняя абсолютная ошибка: {comparison_results['Ошибка_абсолютная_abs'].mean():.3f} лет")
        print(f"Медианная абсолютная ошибка: {comparison_results['Ошибка_абсолютная_abs'].median():.3f} лет")
        print(f"Максимальная ошибка: {comparison_results['Ошибка_абсолютная_abs'].max():.3f} лет")
        print(f"Минимальная ошибка: {comparison_results['Ошибка_абсолютная_abs'].min():.3f} лет")
        
        # Топ регионов по точности прогноза
        print(f"\nТОП-10 РЕГИОНОВ С НАИБОЛЕЕ ТОЧНЫМ ПРОГНОЗОМ:")
        top_accurate = comparison_results.nsmallest(10, 'Ошибка_абсолютная_abs')
        for i, row in top_accurate.iterrows():
            print(f"  {row['Регион'][:30]:<30} | Ошибка: {row['Ошибка_абсолютная_abs']:.3f} лет | "
                  f"Реальный: {row['ОПЖ_реальный']:.2f} | Прогноз: {row['ОПЖ_прогноз']:.2f}")
        
        # Топ регионов с наибольшей ошибкой
        print(f"\nТОП-10 РЕГИОНОВ С НАИБОЛЬШЕЙ ОШИБКОЙ:")
        top_error = comparison_results.nlargest(10, 'Ошибка_абсолютная_abs')
        for i, row in top_error.iterrows():
            print(f"  {row['Регион'][:30]:<30} | Ошибка: {row['Ошибка_абсолютная_abs']:.3f} лет | "
                  f"Реальный: {row['ОПЖ_реальный']:.2f} | Прогноз: {row['ОПЖ_прогноз']:.2f}")
        
        # Гистограмма ошибок
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.hist(comparison_results['Ошибка_абсолютная'], bins=30, edgecolor='black', alpha=0.7)
        plt.xlabel('Ошибка прогноза (лет)')
        plt.ylabel('Количество регионов')
        plt.title('Распределение ошибок прогноза ОПЖ на 2019 год')
        plt.grid(True, alpha=0.3)
        
        # Сохранение графика
        plot_filepath = os.path.join(output_dir, 'opj_prediction_errors_2019.png')
        plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
        print(f"График распределения ошибок сохранен как '{plot_filepath}'")
        
        print("\n" + "="*70)
        print("ПРОГНОЗ ДЛЯ СКР:")
        print("="*70)
        
        # Загрузка данных для СКР
        df_skr = pd.read_excel('Финальный вариант/общая_СКР (2).xlsx')
        
        # Стандартизация названий регионов
        print("Стандартизация названий регионов для СКР...")
        df_skr = standardize_region_names(df_skr)
        
        print("\n" + "="*70)
        print("ВСЕ РАСЧЕТЫ ЗАВЕРШЕНЫ!")
        print(f"Результаты сохранены в папке: {output_dir}")
        print("="*70)