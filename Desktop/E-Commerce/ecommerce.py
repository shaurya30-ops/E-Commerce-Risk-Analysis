import pandas as pd
df = pd.read_csv(r"C:\Users\SHAURYA\Downloads\data.csv\data.csv", encoding="ISO-8859-1")
print(df.head())
print(df.isnull().sum())
print(df.describe())
print(df.shape)
print(df.isnull().sum().sum())
print(df.describe())
print(df.head())
fraud_df = df[(df["Quantity"] < 0) | (df["UnitPrice"] < 0)]
print(fraud_df.head())
high_value_transactions = df[df["Quantity"] > 1000]
print(high_value_transactions.head())
print(f"Potential fraud transactions: {len(fraud_df)}")
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
sns.boxplot(x=df["Quantity"])
plt.title("Transaction Quantity Distribution")
plt.show()
plt.figure(figsize=(10, 6))
sns.boxplot(x=df["UnitPrice"])
plt.title("Unit Price Distribution")
plt.show()
df["FraudFlag"] = df.apply(lambda row: 1 if row["Quantity"] < 0 or row["UnitPrice"] < 0 else 0, axis=1)
print(df["FraudFlag"].value_counts())
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])  
df["Month"] = df["InvoiceDate"].dt.month 
fraud_by_month = df[df["FraudFlag"] == 1].groupby("Month")["InvoiceNo"].count()
plt.figure(figsize=(10,5))
fraud_by_month.plot(kind="bar", color="red")
plt.title("Fraud Cases by Month")
plt.xlabel("Month")
plt.ylabel("Number of Fraud Cases")
plt.show()
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"]) 
last_purchase = df.groupby("CustomerID")["InvoiceDate"].max().reset_index()
last_purchase.columns = ["CustomerID", "LastPurchaseDate"]
import datetime
cutoff_date = df["InvoiceDate"].max() - datetime.timedelta(days=180)
churn_customers = last_purchase[last_purchase["LastPurchaseDate"] < cutoff_date]
print(f"Total churn-risk customers: {len(churn_customers)}")
print(churn_customers.head())
import matplotlib.pyplot as plt
churn_customers["LastPurchaseMonth"] = churn_customers["LastPurchaseDate"].dt.month
churn_trends = churn_customers.groupby("LastPurchaseMonth")["CustomerID"].count()
plt.figure(figsize=(10,5))
churn_trends.plot(kind="bar", color="blue")
plt.title("Customer Churn Trends by Month")
plt.xlabel("Month")
plt.ylabel("Number of Churn Customers")
plt.show()
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["Month"] = df["InvoiceDate"].dt.to_period("M") 
monthly_sales = df.groupby("Month")["Quantity"].sum()
import matplotlib.pyplot as plt
plt.figure(figsize=(12,6))
monthly_sales.plot(marker='o', color="green")
plt.title("Monthly Sales Trends")
plt.xlabel("Month")
plt.ylabel("Total Quantity Sold")
plt.xticks(rotation=45)
plt.show()
from sklearn.linear_model import LinearRegression
import numpy as np
df["MonthNumeric"] = np.arange(len(df["Month"])) 
X = df["MonthNumeric"].values.reshape(-1,1)
y = monthly_sales.values
model = LinearRegression()
model.fit(X, y)
future_months = np.arange(len(df["Month"]), len(df["Month"]) + 6).reshape(-1,1)
future_sales = model.predict(future_months)
plt.figure(figsize=(12,6))
plt.plot(df["MonthNumeric"], y, marker='o', label="Actual Sales", color="blue")
plt.plot(future_months, future_sales, marker='o', label="Predicted Sales", color="red")
plt.title("Sales Forecast for Next 6 Months")
plt.xlabel("Month")
plt.ylabel("Total Quantity Sold")
plt.legend()
plt.show()








