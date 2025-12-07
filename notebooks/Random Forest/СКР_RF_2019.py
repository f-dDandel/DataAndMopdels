# Random Forest для СКР с данными за 2014-2023
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

class SKRForecaster:
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
        Подготовка признаков с данными за 2014-2023
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
            
            # Скользящие средние с min_periods=1 чтобы сохранить больше данных
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
        
        # Экономические индексы
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
    
    def train_test_split_temporal(self, df, test_years=[2019, 2020, 2021]):
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
    
    def fit(self, df, test_years=[2019, 2020, 2021]):
        """
        Обучение модели на расширенных данных
        """
        # Очистка и подготовка данных
        df_clean = self.clean_numeric_columns(df)
        df_processed = self.prepare_features(df_clean, is_training=True)
        
        print(f"Данные за период: {df_processed['Год'].min()}-{df_processed['Год'].max()}")
        
        # Сохраняем последние известные данные для прогноза (2018 год для предсказания 2019)
        self.last_known_data = df_processed[df_processed['Год'] == 2018].copy()
        
        # Определение признаков
        self.feature_names = [
            # Демографические
            'Численность населения', 
            'Число родившихся',
            'Браков', 
            'Разводов',
            
            # Социально-экономические
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
            'Социально_экономический_индекс',
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
        scale_features = [f for f in self.feature_names if not f.startswith(('lag', 'СКР_MA', 'изменение_'))]
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
        Расчет метрик качества
        """
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        print("\n" + "="*50)
        print("МЕТРИКИ КАЧЕСТВА Random Forest ДЛЯ СКР")
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
            filepath = os.path.join(output_dir, 'skr_random_forest_model.pkl')
        
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
            filepath = os.path.join(output_dir, 'skr_random_forest_model.pkl')
            
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
            'СКР_прогноз': predictions_df['СКР_прогноз'],
            'СКР_реальный': predictions_df['СКР_реальный'],
            'СКР_предыдущий_год': predictions_df['СКР_предыдущий_год'],
            'Ошибка_абсолютная': predictions_df['Ошибка_абсолютная'],
            'Ошибка_относительная_%': predictions_df['Ошибка_относительная_%']
        })
        
        return final_output

    def predict_for_year(self, df, target_year=2019):
        """
        Прогноз СКР на 2019 год и сравнение с реальными значениями
        """
        if not self.is_fitted:
            print("Сначала обучите модель!")
            return None
        
        print(f"\nСоздание прогноза СКР на {target_year} год...")
        
        # Получаем реальные значения за 2019 год
        real_2019 = df[df['Год'] == target_year][['Регион', 'СКР']].copy()
        real_2019 = real_2019.rename(columns={'СКР': 'СКР_реальный'})
        
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
            # Используем умеренные коэффициенты роста, характерные для 2018-2019 годов
            population_factor = 1.002  # Небольшой прирост населения
            births_factor = 0.998      # Небольшое снижение рождаемости
            marriages_factor = 1.005   # Небольшой рост браков
            divorces_factor = 1.003    # Небольшой рост разводов
            grp_factor = 1.015         # Умеренный рост ВРП
            salary_factor = 1.03       # Умеренный рост зарплат
            living_wage_factor = 1.025 # Рост прожиточного минимума
            poverty_factor = 0.99      # Небольшое снижение бедности
            
            # Применяем коэффициенты
            future_row['Численность населения'] = last_row['Численность населения'] * population_factor
            future_row['Число родившихся'] = last_row['Число родившихся'] * births_factor
            future_row['Браков'] = last_row['Браков'] * marriages_factor
            future_row['Разводов'] = last_row['Разводов'] * divorces_factor
            future_row['Валовой региональный продукт на душу населения (ОКВЭД 2)'] = last_row['Валовой региональный продукт на душу населения (ОКВЭД 2)'] * grp_factor
            future_row['Средняя ЗП'] = last_row['Средняя ЗП'] * salary_factor
            future_row['Величина прожиточного минимума'] = last_row['Величина прожиточного минимума'] * living_wage_factor
            future_row['Уровень бедности'] = last_row['Уровень бедности'] * poverty_factor
            
            # Лаги берем из 2018 года
            future_row['lag1_СКР'] = last_row['СКР']
            future_row['lag2_СКР'] = last_row.get('lag1_СКР', last_row['СКР'])
            future_row['lag1_Браков'] = last_row['Браков']
            future_row['lag1_Разводов'] = last_row['Разводов']
            future_row['lag1_Число родившихся'] = last_row['Число родившихся']
            
            # Пересчитываем скользящие средние
            future_row['СКР_MA2'] = (future_row['lag1_СКР'] + last_row['СКР']) / 2
            future_row['СКР_MA3'] = (last_row.get('lag2_СКР', last_row['СКР']) + last_row['СКР'] + future_row['lag1_СКР']) / 3
            
            # Пересчитываем производные показатели
            future_row['Браков_на_1000'] = future_row['Браков'] / future_row['Численность населения'] * 1000
            future_row['Разводов_на_1000'] = future_row['Разводов'] / future_row['Численность населения'] * 1000
            future_row['Родившихся_на_1000'] = future_row['Число родившихся'] / future_row['Численность населения'] * 1000
            future_row['Преступлений_на_1000'] = future_row['Кол-во преступлений'] / future_row['Численность населения'] * 1000
            future_row['Социально_экономический_индекс'] = (
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
        scale_features = [f for f in self.feature_names if not f.startswith(('lag', 'СКР_MA', 'изменение_'))]
        X_future_scaled = X_future.copy()
        X_future_scaled[scale_features] = self.scaler.transform(X_future[scale_features])
        
        # Прогноз СКР
        predictions = self.model.predict(X_future_scaled)
        
        # Собираем результаты
        results = []
        for i, (_, last_row) in enumerate(data_2018.iterrows()):
            results.append({
                'Регион': last_row['Регион'],
                'Год': target_year,
                'СКР_прогноз': predictions[i],
                'СКР_предыдущий_год': last_row['СКР']
            })
        
        results_df = pd.DataFrame(results)
        
        # Объединяем с реальными значениями
        comparison_df = pd.merge(results_df, real_2019, on='Регион', how='inner')
        
        # Рассчитываем ошибки
        comparison_df['Ошибка_абсолютная'] = comparison_df['СКР_реальный'] - comparison_df['СКР_прогноз']
        comparison_df['Ошибка_абсолютная_abs'] = np.abs(comparison_df['Ошибка_абсолютная'])
        comparison_df['Ошибка_относительная_%'] = (comparison_df['Ошибка_абсолютная'] / comparison_df['СКР_реальный']) * 100
        
        print(f"\nПрогноз на {target_year} год успешно создан")
        print(f"Сравнение выполнено для {len(comparison_df)} регионов")
        
        # Рассчитываем метрики для прогноза
        self.calculate_prediction_metrics(comparison_df['СКР_реальный'], comparison_df['СКР_прогноз'])
        
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
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R²: {r2:.4f}")
        print(f"MAPE: {mape:.2f}%")
        print(f"Средний реальный СКР: {y_true.mean():.4f}")
        print(f"Средний прогнозный СКР: {y_pred.mean():.4f}")
        print(f"Разница: {y_true.mean() - y_pred.mean():.4f}")
        
        # Сохраняем метрики прогноза
        self.prediction_metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape,
            'Mean_Actual': y_true.mean(),
            'Mean_Predicted': y_pred.mean()
        }

if __name__ == "__main__":
    # Загрузка данных
    df = pd.read_excel('Финальный вариант/общая_СКР (2).xlsx')
    
    # Стандартизация названий регионов
    print("Стандартизация названий регионов...")
    df = standardize_region_names(df)
    print(f"Уникальных регионов после очистки: {df['Регион'].nunique()}")
    
    # Проверка наличия данных за 2019 год
    years_in_data = sorted(df['Год'].unique())
    print(f"Годы в данных: {years_in_data}")
    
    if 2019 not in df['Год'].unique():
        print("ВНИМАНИЕ: В данных нет 2019 года!")
        print("Используем последний доступный год для тестирования...")
        last_year = max(df['Год'].unique())
        print(f"Последний доступный год: {last_year}")
    else:
        print("✓ Данные за 2019 год присутствуют")
    
    print("="*70)
    print("МОДЕЛЬ СЛУЧАЙНОГО ЛЕСА ДЛЯ ПРОГНОЗА СКР")
    print("Тестирование на 2019 году")
    print("="*70)
    
    # Диагностика данных
    print(f"Размер данных: {df.shape}")
    print(f"Период данных: {df['Год'].min()}-{df['Год'].max()}")
    print(f"Количество регионов: {df['Регион'].nunique()}")
    
    # Обучение модели на данных до 2018 года
    print("\nОбучение модели на данных до 2018 года...")
    forecaster = SKRForecaster(n_estimators=200)
    
    # Обучаем на данных до 2018 года, тестируем на 2019
    forecaster.fit(df, test_years=[2019, 2020, 2021])
    
    # Сохранение модели
    forecaster.save_model()
    
    # Прогноз на 2019 год
    comparison_results = forecaster.predict_for_year(df, target_year=2019)
    
    if comparison_results is not None:
        # Подготовка финального вывода
        final_output = forecaster.prepare_final_output(comparison_results, target_year=2019)
        
        # Сохранение результатов
        predictions_filepath = os.path.join(output_dir, 'skr_predictions_2019_comparison.xlsx')
        final_output.to_excel(predictions_filepath, index=False)
        print(f"\n✓ Финальный файл с прогнозами и сравнением сохранен как '{predictions_filepath}'")
        
        # Вывод детальной статистики
        print(f"\nДЕТАЛЬНАЯ СТАТИСТИКА ПРОГНОЗА НА 2019 ГОД:")
        print("-" * 50)
        print(f"Всего регионов: {len(comparison_results)}")
        print(f"Средний реальный СКР: {comparison_results['СКР_реальный'].mean():.4f}")
        print(f"Средний прогнозный СКР: {comparison_results['СКР_прогноз'].mean():.4f}")
        print(f"Средняя абсолютная ошибка: {comparison_results['Ошибка_абсолютная_abs'].mean():.4f}")
        print(f"Медианная абсолютная ошибка: {comparison_results['Ошибка_абсолютная_abs'].median():.4f}")
        print(f"Максимальная ошибка: {comparison_results['Ошибка_абсолютная_abs'].max():.4f}")
        print(f"Минимальная ошибка: {comparison_results['Ошибка_абсолютная_abs'].min():.4f}")
        
        # Статистика по ошибкам
        print(f"\nСТАТИСТИКА ПО ОШИБКАМ:")
        error_positive = (comparison_results['Ошибка_абсолютная'] > 0).sum()
        error_negative = (comparison_results['Ошибка_абсолютная'] < 0).sum()
        print(f"Прогноз завышен (реальный > прогноз): {error_positive} регионов")
        print(f"Прогноз занижен (реальный < прогноз): {error_negative} регионов")
        
        # Топ регионов по точности прогноза
        print(f"\nТОП-10 РЕГИОНОВ С НАИБОЛЕЕ ТОЧНЫМ ПРОГНОЗОМ:")
        top_accurate = comparison_results.nsmallest(10, 'Ошибка_абсолютная_abs')
        for i, row in top_accurate.iterrows():
            print(f"  {row['Регион'][:30]:<30} | Ошибка: {row['Ошибка_абсолютная_abs']:.4f} | "
                  f"Реальный: {row['СКР_реальный']:.4f} | Прогноз: {row['СКР_прогноз']:.4f}")
        
        # Топ регионов с наибольшей ошибкой
        print(f"\nТОП-10 РЕГИОНОВ С НАИБОЛЬШЕЙ ОШИБКОЙ:")
        top_error = comparison_results.nlargest(10, 'Ошибка_абсолютная_abs')
        for i, row in top_error.iterrows():
            print(f"  {row['Регион'][:30]:<30} | Ошибка: {row['Ошибка_абсолютная_abs']:.4f} | "
                  f"Реальный: {row['СКР_реальный']:.4f} | Прогноз: {row['СКР_прогноз']:.4f}")
        
        # Анализ распределения ошибок
        error_quartiles = np.percentile(comparison_results['Ошибка_абсолютная_abs'], [25, 50, 75])
        print(f"\nКВАРТИЛИ АБСОЛЮТНОЙ ОШИБКИ:")
        print(f"  25-й перцентиль: {error_quartiles[0]:.4f}")
        print(f"  50-й перцентиль (медиана): {error_quartiles[1]:.4f}")
        print(f"  75-й перцентиль: {error_quartiles[2]:.4f}")
        
        # Гистограмма ошибок
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 8))
        
        # График распределения ошибок
        plt.subplot(2, 2, 1)
        plt.hist(comparison_results['Ошибка_абсолютная'], bins=30, edgecolor='black', alpha=0.7)
        plt.xlabel('Ошибка прогноза СКР')
        plt.ylabel('Количество регионов')
        plt.title('Распределение ошибок прогноза СКР на 2019 год')
        plt.grid(True, alpha=0.3)
        
        # График реальных vs прогнозных значений
        plt.subplot(2, 2, 2)
        plt.scatter(comparison_results['СКР_реальный'], comparison_results['СКР_прогноз'], alpha=0.6)
        plt.plot([comparison_results['СКР_реальный'].min(), comparison_results['СКР_реальный'].max()],
                [comparison_results['СКР_реальный'].min(), comparison_results['СКР_реальный'].max()], 
                'r--', label='Идеальный прогноз')
        plt.xlabel('Реальный СКР')
        plt.ylabel('Прогнозный СКР')
        plt.title('Реальные vs Прогнозные значения СКР')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # График распределения относительных ошибок
        plt.subplot(2, 2, 3)
        plt.hist(comparison_results['Ошибка_относительная_%'], bins=30, edgecolor='black', alpha=0.7)
        plt.xlabel('Относительная ошибка (%)')
        plt.ylabel('Количество регионов')
        plt.title('Распределение относительных ошибок')
        plt.grid(True, alpha=0.3)
        
        # Боксплот ошибок
        plt.subplot(2, 2, 4)
        plt.boxplot(comparison_results['Ошибка_абсолютная_abs'])
        plt.ylabel('Абсолютная ошибка')
        plt.title('Боксплот абсолютных ошибок')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Сохранение графика
        plot_filepath = os.path.join(output_dir, 'skr_prediction_analysis_2019.png')
        plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
        print(f"✓ График анализа сохранен как '{plot_filepath}'")
        
        # Сохранение сводной таблицы
        summary_filepath = os.path.join(output_dir, 'skr_prediction_summary_2019.csv')
        
        summary_stats = {
            'Метрика': ['RMSE', 'MAE', 'R2', 'MAPE (%)', 'Средний реальный', 'Средний прогноз', 'Разница средних'],
            'Значение': [
                forecaster.prediction_metrics['RMSE'],
                forecaster.prediction_metrics['MAE'],
                forecaster.prediction_metrics['R2'],
                forecaster.prediction_metrics['MAPE'],
                forecaster.prediction_metrics['Mean_Actual'],
                forecaster.prediction_metrics['Mean_Predicted'],
                forecaster.prediction_metrics['Mean_Actual'] - forecaster.prediction_metrics['Mean_Predicted']
            ]
        }
        
        summary_df = pd.DataFrame(summary_stats)
        summary_df.to_csv(summary_filepath, index=False)
        print(f"✓ Сводная статистика сохранена как '{summary_filepath}'")
        
        print("\n" + "="*70)
        print("ВСЕ РАСЧЕТЫ ЗАВЕРШЕНЫ!")
        print(f"Результаты сохранены в папке: {output_dir}")
        print("="*70)