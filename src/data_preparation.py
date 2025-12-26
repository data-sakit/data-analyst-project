import pandas as pd
import numpy as np
import sqlite3
import gzip

# ============================================================
# STEP 1: LOAD DATA FROM DATABASE
# ============================================================

path = r'C:\Users\lenovo\Desktop\skillbox\d_a\result_data\shop_database.db'
conn = sqlite3.connect(path)

# Check tables
query_tables = "SELECT NAME FROM sqlite_master WHERE type='table'"
tables = pd.read_sql(query_tables, conn)
print("Tables in database")
print(tables)

# Load first 5 rows
df_purchases_5r = pd.read_sql("select * from purchases limit 5", conn)
df_coeffs_5r = pd.read_sql("select * from personal_data_coeffs limit 5", conn)
df_personal_5r = pd.read_sql("select * from personal_data limit 5", conn)

# Load full tables
df_purchases = pd.read_sql("select * from purchases", conn)
df_coeffs = pd.read_sql("select * from personal_data_coeffs", conn)
df_personal = pd.read_sql("select * from personal_data", conn)

print("Basic information about tables:")
print("\npurchases shape:", df_purchases.shape)
print("personal_data_coeffs shape:", df_coeffs.shape)
print("personal_data shape:", df_personal.shape)

conn.close()

# ============================================================
# STEP 2: FILTER BY COUNTRY
# ============================================================

print("\nInitial counts:")
print(f"Unique customers in personal_data: {df_personal['id'].nunique()}")
print(f"Unique customers in personal_data_coeffs: {df_coeffs['id'].nunique()}")
print(f"Unique customers in purchases: {df_purchases['id'].nunique()}")

print("\nFiltering personal_data: keeping only country = 32")
before_filter = df_personal.shape[0]
df_personal_filtered = df_personal[df_personal['country'] == 32]
after_filter = df_personal_filtered.shape[0]

print(f"Before filter: {before_filter} rows")
print(f"After filter: {after_filter} rows")
print(f"Removed: {before_filter - after_filter} rows")

filtered_customer_ids = df_personal_filtered['id'].tolist()
df_coeffs_filtered = df_coeffs[df_coeffs["id"].isin(filtered_customer_ids)]
df_purchases_filtered = df_purchases[df_purchases['id'].isin(filtered_customer_ids)]

# ============================================================
# STEP 3: CLEANING AND FIXING MISSING VALUES
# ============================================================

df_personal_clean = df_personal_filtered.copy()
df_coeffs_clean = df_coeffs_filtered.copy()
df_purchases_clean = df_purchases_filtered.copy()

# Fill missing colours
df_purchases_clean['colour'] = df_purchases_clean['colour'].fillna('unknown')

# Simplify colours
df_purchases_clean['colour_simple'] = df_purchases_clean['colour'].apply(
    lambda x: x.split('/')[0] if '/' in str(x) else x
)

# Fill missing product_sex
product_sex_mode = df_purchases_clean['product_sex'].mode()[0]
df_purchases_clean['product_sex'] = df_purchases_clean['product_sex'].fillna(product_sex_mode)

# Fix negative costs
df_purchases_clean.loc[df_purchases_clean['cost'] < 0, 'cost'] = abs(
    df_purchases_clean.loc[df_purchases_clean['cost'] < 0, 'cost']
)

# Fix -inf in ac_coef
df_coeffs_clean['ac_coef'] = df_coeffs_clean['ac_coef'].replace(float('-inf'), np.nan)
ac_coef_median = df_coeffs_clean['ac_coef'].dropna().median()
df_coeffs_clean['ac_coef'] = df_coeffs_clean['ac_coef'].fillna(ac_coef_median)

# ============================================================
# STEP 4: MERGE TABLES AND CREATE FEATURES
# ============================================================

df_merged = pd.merge(df_personal_clean, df_coeffs_clean, on='id', how='inner')

customer_purchases = df_purchases_clean.groupby('id').agg({
    'product': 'count',
    'cost': 'sum',
    'base_sale': 'mean',
    'product_sex': 'mean',
    'dt': ['min', 'max']
})

customer_purchases.columns = [
    'total_purchases',
    'total_spent',
    'discount_rate',
    'avg_product_sex',
    'first_purchase_day',
    'last_purchase_day'
]

customer_purchases = customer_purchases.reset_index()

df_final = pd.merge(df_merged, customer_purchases, on='id', how='left')

df_final['total_purchases'] = df_final['total_purchases'].fillna(0)
df_final['total_spent'] = df_final['total_spent'].fillna(0)
df_final['discount_rate'] = df_final['discount_rate'].fillna(0)
df_final['avg_product_sex'] = df_final['avg_product_sex'].fillna(0.5)
df_final['first_purchase_day'] = df_final['first_purchase_day'].fillna(-1)
df_final['last_purchase_day'] = df_final['last_purchase_day'].fillna(-1)

# ============================================================
# STEP 5: SAVE customers_final.csv
# ============================================================

df_final.to_csv(
    r'C:\Users\lenovo\Desktop\skillbox\d_a\result_data\customers_final.csv',
    index=False
)

# ============================================================
# STEP 6: LOAD ADDITIONAL personal_data.csv.gz
# ============================================================

df_additional = pd.read_csv(
    r'C:\Users\lenovo\Desktop\skillbox\d_a\personal_data.csv.gz',
    compression='gzip'
)

df_additional.to_csv(
    r'C:\Users\lenovo\Desktop\skillbox\d_a\result_data\df_additional.csv',
    index=False
)

# ============================================================
# STEP 7: PREPARE DATA FOR BINARY CLASSIFICATION
# ============================================================

print("=" * 60)
print("STEP 7: PREPARING DATA FOR BINARY CLASSIFICATION")
print("=" * 60)

gender_dist = df_final['gender'].value_counts()

common_features = ['age', 'education', 'city', 'country']

x_train_common = df_final[common_features].copy()
y_train = df_final['gender'].copy()
x_test = df_additional[common_features].copy()

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False, drop='first')

education_encoded_train = encoder.fit_transform(x_train_common[['education']])
education_encoded_test = encoder.transform(x_test[['education']])

x_train_encoded = x_train_common.drop('education', axis=1)
x_test_encoded = x_test.drop('education', axis=1)

col_name = encoder.get_feature_names_out()[0]

x_train_encoded[col_name] = education_encoded_train[:, 0]
x_test_encoded[col_name] = education_encoded_test[:, 0]

x_train_encoded.to_csv('X_train_prepared.csv', index=False)
x_test_encoded.to_csv('X_test_prepared.csv', index=False)
y_train.to_csv('y_train.csv', index=False)

print("Prepared datasets saved.")
