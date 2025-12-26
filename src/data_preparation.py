import pandas as pd
import numpy as np
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# -----------------------------
# STEP 1 — Load raw data
# -----------------------------

# Load SQLite database
conn = sqlite3.connect(r'C:\Users\lenovo\Desktop\skillbox\d_a\data_raw\shop_database.db')
df_shop = pd.read_sql_query("SELECT * FROM shop", conn)
conn.close()

# Load personal data
df_personal = pd.read_csv(r'C:\Users\lenovo\Desktop\skillbox\d_a\data_raw\personal_data.csv.gz')

# Load additional ID lists
test_file = r'C:\Users\lenovo\Desktop\skillbox\d_a\data_raw\ids_first_company_positive.txt'
control_file = r'C:\Users\lenovo\Desktop\skillbox\d_a\data_raw\ids_first_company_negative.txt'

ids_positive = pd.read_csv(test_file, header=None)[0].tolist()
ids_negative = pd.read_csv(control_file, header=None)[0].tolist()

# -----------------------------
# STEP 2 — Filter and clean data
# -----------------------------

# Filter by country
df_personal = df_personal[df_personal['country'] == 'first_company']

# Remove invalid rows
df_personal = df_personal[df_personal['id'].notna()]

# Fix categories
df_personal['color'] = df_personal['color'].replace({
    'blak': 'black',
    'whtie': 'white',
    'gren': 'green'
})

# Replace invalid coefficients
df_personal['coefficient'] = df_personal['coefficient'].replace([-np.inf, np.inf], np.nan)
df_personal['coefficient'] = df_personal['coefficient'].fillna(df_personal['coefficient'].median())

# -----------------------------
# STEP 3 — Aggregate purchase data
# -----------------------------

df_shop['price'] = df_shop['price'].replace([-np.inf, np.inf], np.nan)
df_shop['price'] = df_shop['price'].fillna(df_shop['price'].median())

df_additional = df_shop.groupby('id').agg({
    'price': ['mean', 'sum', 'count']
})

df_additional.columns = ['avg_price', 'total_spent', 'purchase_count']
df_additional = df_additional.reset_index()

# -----------------------------
# STEP 4 — Merge personal + aggregated data
# -----------------------------

df_final = df_personal.merge(df_additional, on='id', how='left')

# Fill missing aggregated values
df_final[['avg_price', 'total_spent', 'purchase_count']] = \
    df_final[['avg_price', 'total_spent', 'purchase_count']].fillna(0)

# -----------------------------
# STEP 5 — Prepare train/test
# -----------------------------

# Target
y = df_final['gender']
X = df_final.drop(columns=['gender'])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# One-hot encoding
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

X_train_encoded = pd.DataFrame(
    encoder.fit_transform(X_train.select_dtypes(include=['object'])),
    index=X_train.index
)

X_test_encoded = pd.DataFrame(
    encoder.transform(X_test.select_dtypes(include=['object'])),
    index=X_test.index
)

# Add numeric columns
X_train_encoded = pd.concat([X_train_encoded, X_train.select_dtypes(exclude=['object'])], axis=1)
X_test_encoded = pd.concat([X_test_encoded, X_test.select_dtypes(exclude=['object'])], axis=1)

# -----------------------------
# STEP 6 — Save processed data
# -----------------------------

output_path = r'C:\Users\lenovo\Desktop\skillbox\d_a\data_processed'

df_final.to_csv(output_path + r'\customers_final.csv', index=False)
df_additional.to_csv(output_path + r'\df_additional.csv', index=False)
X_train_encoded.to_csv(output_path + r'\X_train_prepared.csv', index=False)
X_test_encoded.to_csv(output_path + r'\X_test_prepared.csv', index=False)
y_train.to_csv(output_path + r'\y_train.csv', index=False)

print("Data preparation completed successfully.")
