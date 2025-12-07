import pandas as pd
import re
from scipy.stats import pearsonr
import numpy as np
separator = "=" * 100

# Функция для стандартизации названий регионов
def standardize_region_names(df):
    """Стандартизация названий регионов"""
    df_clean = df.copy()
    
    # Создаем словарь замен
    replacements = {
        'Ненецкий авт.округ': 'Ненецкий автономный округ',
        'Hенецкий авт.округ': 'Ненецкий автономный округ',  # латинская H
        '  Ненецкий автономный округ': 'Ненецкий автономный округ',
        
        'Ямало-Ненецкий авт.округ': 'Ямало-Ненецкий автономный округ',
        'Ямало-Hенецкий авт.округ': 'Ямало-Ненецкий автономный округ',  # латинская H
        '  Ямало-Ненецкий автономный округ': 'Ямало-Ненецкий автономный округ',
        
        'Ханты-Мансийский авт.округ-Югра': 'Ханты-Мансийский автономный округ - Югра',
        '  Ханты-Мансийский автономный округ - Югра': 'Ханты-Мансийский автономный округ - Югра',
        
        'Республика Татарстан(Татарстан)': 'Республика Татарстан',
        'Чувашская Республика(Чувашия)': 'Чувашская Республика',
        'Республика Северная Осетия- Алания': 'Республика Северная Осетия-Алания',
        
        'Oмская область': 'Омская область',  # латинская O
        'Hижегородская область': 'Нижегородская область',  # латинская H
        
        'г. Севастополь': 'г.Севастополь',
        'г.Москва': 'г.Москва',
        'г.Санкт-Петербург': 'г.Санкт-Петербург',
        
        'Чукотский авт.округ': 'Чукотский автономный округ',
        'Чукотский автономный округ': 'Чукотский автономный округ'
    }
    
    # Применяем замены
    df_clean['Регион'] = df_clean['Регион'].replace(replacements)
    
    # Удаляем лишние пробелы
    df_clean['Регион'] = df_clean['Регион'].str.strip()
    
    return df_clean

def clean_numeric_columns(df, target):
    """
    Очистка числовых колонок от пробелов, нестандартных разделителей и символов
    """
    df = df.copy()
    df = df.drop(['Регион', 'Год'], axis='columns')
    numeric_columns = [col for col in df.columns if col not in [target]]
    
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
    
    if target in df.columns:
        df[target] = pd.to_numeric(df[target], errors='coerce')
    
    return df


def corr_pvalues(df):
    """
    Вычисляет матрицу p-value для попарных корреляций Пирсона между числовыми столбцами DataFrame.
    Возвращает DataFrame тех же индексов/столбцов с p-value.
    """
    # Оставляем только числовые столбцы
    num_df = df.select_dtypes(include=[np.number]).copy()
    cols = num_df.columns
    pvals = pd.DataFrame(np.ones((len(cols), len(cols))), index=cols, columns=cols)

    for i, col_i in enumerate(cols):
        for j, col_j in enumerate(cols):
            if j <= i:
                # диагональ и нижний треугольник — можно оставить 1 или зеркально заполнить
                continue
            x = num_df[col_i]
            y = num_df[col_j]
            mask = x.notna() & y.notna()
            if mask.sum() < 2:
                p = np.nan
            else:
                try:
                    _, p = pearsonr(x[mask], y[mask])
                except Exception:
                    p = np.nan
            pvals.at[col_i, col_j] = p
            pvals.at[col_j, col_i] = p

    return pvals

def parse_block(text):
    """
    Парсит текстовый блок в Markdown
    """
    blocks = [b.strip() for b in text.strip().split("\n\n")]
    md_parts = []

    # numeric pattern: integers, floats, optional scientific notation
    num_re = re.compile(r"^-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$")

    for block in blocks:
        lines = [ln for ln in block.split("\n") if ln.strip()]
        rows = []

        for line in lines:
            line_str = line.strip()
            # сначала пытаемся поймать две числовые величины в конце строки
            m2 = re.match(r"^(.+?)\s+(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$", line_str)
            if m2:
                key = m2.group(1).strip()
                nums = [m2.group(2), m2.group(3)]
            else:
                # пробуем одну числовую величину в конце строки
                m1 = re.match(r"^(.+?)\s+(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$", line_str)
                if m1:
                    key = m1.group(1).strip()
                    nums = [m1.group(2)]
                else:
                    continue

            rows.append([key] + nums)

        if not rows:
            continue

        # determine number of numeric columns
        num_cols = max(len(r) - 1 for r in rows)
        if num_cols == 1:
            cols = ["Показатель", "Корреляция"]
        else:
            # build column names: Корреляция, pvalue, and additional if present
            cols = ["Показатель"]
            if num_cols >= 1:
                cols.append("Корреляция")
            if num_cols >= 2:
                cols.append("pvalue")
            for k in range(3, num_cols + 1):
                cols.append(f"val{k}")

        # normalize rows to have same length
        norm_rows = []
        for r in rows:
            key = r[0]
            nums = r[1:]
            # pad with empty strings if needed
            nums = nums + [""] * (num_cols - len(nums))
            norm_rows.append([key] + nums)

        df = pd.DataFrame(norm_rows, columns=cols)
        md_parts.append(df.to_markdown(index=False))

    return md_parts


def parse_multiple_texts(text_list):
    """
    Парсит блоки parse_block в один Markdown
    """
    all_md = [text_list[0]] # первый блок - описание

    for text in text_list[1:]:
        md_tables = parse_block(text)
        for table in md_tables:
            all_md.append(table)
            all_md.append(separator)

    return "\n\n".join(all_md)

if __name__ == "__main__":
    # Run check correlation pipeline
    # Проверка для ОПЖ
    result = []
    df = pd.read_excel('общая_ОПЖ (2).xlsx')
    df = standardize_region_names(df)
    df = clean_numeric_columns(df, target='ОПЖ')
    # Корреляции и p-value для ОПЖ
    corr_matrix = df.corr(method='pearson')
    pval_matrix = corr_pvalues(df)
    print("Pearson correlations (ОПЖ):")
    print(corr_matrix['ОПЖ'])
    print("P-values (ОПЖ):")
    print(pval_matrix['ОПЖ'])
    
    # Собираем единый текстовый блок: каждая строка - показатель, r, p
    lines = []
    for idx in corr_matrix.index:
        r = corr_matrix.at[idx, 'ОПЖ'] if 'ОПЖ' in corr_matrix.columns else ''
        p = pval_matrix.at[idx, 'ОПЖ'] if 'ОПЖ' in pval_matrix.columns else ''
        r_str = f"{r:.6f}" if isinstance(r, (float, np.floating)) else str(r)
        p_str = f"{p:.6e}" if isinstance(p, (float, np.floating)) else str(p)
        lines.append(f"{idx}    {r_str}    {p_str}")

    raw_text_opj = "Target: ОПЖ\n\n" + "\n".join(lines)
    result.append(raw_text_opj)
    # Проверка для СКР
    df = pd.read_excel('общая_СКР (2).xlsx')
    df = standardize_region_names(df)
    df = clean_numeric_columns(df, target='СКР')

    corr_matrix = df.corr(method='pearson')
    pval_matrix = corr_pvalues(df)
    print("Pearson correlations (СКР):")
    print(corr_matrix['СКР'])
    print("P-values (СКР):")
    print(pval_matrix['СКР'])
    print("p-value между СКР и Численность населения:", pval_matrix.at['СКР', 'Численность населения'])
    lines = []
    for idx in corr_matrix.index:
        r = corr_matrix.at[idx, 'СКР'] if 'СКР' in corr_matrix.columns else ''
        p = pval_matrix.at[idx, 'СКР'] if 'СКР' in pval_matrix.columns else ''
        r_str = f"{r:.6f}" if isinstance(r, (float, np.floating)) else str(r)
        p_str = f"{p:.6e}" if isinstance(p, (float, np.floating)) else str(p)
        lines.append(f"{idx}    {r_str}    {p_str}")

    raw_text_skr = "Target: СКР\n\n" + "\n".join(lines)
    result.append(raw_text_skr)

    description_text = (
        "# Корреляционный анализ показателей Ожидаемой продолжительности жизни и Среднего коэффициента рождаемости по регионам России\n\n"
        "В данном файле приведены результаты корреляционного анализа "
        "по регионам России. "
        "Каждая таблица содержит коэффициенты корреляции Пирсона "
        "между целевым показателем и другими переменными в наборе данных."
        "В таблицу включены только те показатели, которые показали наибольшее влияние на таргет."
    )
    result.insert(0, description_text)
    final_md = parse_multiple_texts(result)
    with open("Readme.md", "w", encoding="utf-8") as f:
        f.write(final_md)