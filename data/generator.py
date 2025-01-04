import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Sample data lists
products = [
    ("Bread", "Bakery"), ("Milk", "Dairy"), ("Eggs", "Dairy"),
    ("Butter", "Dairy"), ("Apple", "Fruits"), ("Banana", "Fruits"),
    ("Chicken", "Meat"), ("Beef", "Meat"), ("Tomato", "Vegetables"),
    ("Potatoes", "Vegetables"), ("Chocolate", "Sweets"), ("Cookies", "Sweets"),
    ("Pasta", "Dry Products"), ("Rice", "Dry Products"), ("Coffee", "Beverages"),
    ("Tea", "Beverages"), ("Orange Juice", "Beverages"), ("Water", "Beverages"),
    ("Cheese", "Dairy"), ("Ham", "Meat")
]

# Function to generate random transactions
def generate_transaction_data(n_transactions=3000):
    data = []
    start_date = datetime(2024, 1, 1)
    for _ in range(n_transactions):
        # Random number of products in one transaction
        n_products = random.randint(1, 5)
        chosen_products = random.sample(products, n_products)

        # Generate data for each product
        transaction_id = random.randint(1000, 9999)
        date = start_date + timedelta(days=random.randint(0, 180))
        date_str = date.strftime("%Y-%m-%d")

        for product, category in chosen_products:
            quantity = random.randint(1, 10)
            price_per_unit = round(random.uniform(2.0, 50.0), 2)  # Price from 2 to 50 PLN
            total_price = round(quantity * price_per_unit, 2)
            data.append([
                transaction_id, date_str, product, category, quantity, price_per_unit, total_price
            ])

    return pd.DataFrame(data, columns=[
        "Transaction_ID", "Date", "Product", "Category", "Quantity", "Price_Per_Unit", "Total_Price"
    ])

# Generate data
n_rows = 3000  # Maximum number of rows
data = generate_transaction_data(n_transactions=n_rows // 5)

# Save to CSV file
output_file = "sales_data.csv"
data.to_csv(output_file, index=False, encoding='utf-8')

print(f"CSV file has been generated: {output_file}")
print(data.head())
