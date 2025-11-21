import pandas as pd
import numpy as np

# Paths
payments_path = 'olist_order_payments_dataset.csv'
reviews_path = 'olist_order_reviews_dataset.csv'

# 1. Load
payments = pd.read_csv(payments_path)
reviews = pd.read_csv(reviews_path)

# 2. Basic info
print("Payments shape:", payments.shape)
print("Reviews shape:", reviews.shape)

print("\nPayments dtypes:\n", payments.dtypes)
print("\nReviews dtypes:\n", reviews.dtypes)

# 3. Missing values summary
print("\nMissing values (payments):\n", payments.isna().sum())
print("\nMissing values (reviews):\n", reviews.isna().sum())

# 4. Basic stats
print("\nPayment value stats:\n", payments['payment_value'].describe())
print("\nTop payment types:\n", payments['payment_type'].value_counts())
print("\nReview score distribution:\n", reviews['review_score'].value_counts())

# 5. Convert date columns in reviews to datetime
for col in ['review_creation_date', 'review_answer_timestamp']:
    reviews[col] = pd.to_datetime(reviews[col], errors='coerce')

# compute response time in hours
reviews['review_response_time_hours'] = (
    (reviews['review_answer_timestamp'] - reviews['review_creation_date'])
    .dt.total_seconds() / 3600
)

# flag missing comment
reviews['is_comment_missing'] = reviews['review_comment_message'].isna()

# 6. Aggregate payments by order_id
payments_agg = payments.groupby('order_id').agg(
    total_payment=('payment_value','sum'),
    num_payments=('payment_value','count'),
    avg_payment=('payment_value','mean'),
    max_payment=('payment_value','max'),
    payment_types=('payment_type', lambda x: ','.join(sorted(x.unique())))
).reset_index()

# 7. Merge reviews and payments
reviews_first = reviews.sort_values('review_creation_date') \
                       .drop_duplicates('order_id', keep='first')

merged = payments_agg.merge(
    reviews_first[['order_id','review_id','review_score','review_comment_message',
                   'is_comment_missing','review_response_time_hours',
                   'review_creation_date']],
    on='order_id', how='left'
)

# 8. Type corrections & duplicates
merged['num_payments'] = merged['num_payments'].astype(int)
merged = merged.drop_duplicates(subset=['order_id'])

# 9. Outlier flagging
merged['is_payment_outlier'] = merged['total_payment'] > 10000

# 10. Clean text fields
merged['review_comment_message'] = merged['review_comment_message'] \
    .fillna("No comment provided").astype(str)

# 11. Summary
print("\nMerged shape:", merged.shape)
print("\nMissing per column:\n", merged.isna().sum())

# 12. Save output
merged.to_csv('olist_milestone1_cleaned.csv', index=False)
print("Saved cleaned file as olist_milestone1_cleaned.csv")
