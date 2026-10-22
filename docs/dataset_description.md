# Dataset Description

The project uses the UCI Online Retail II dataset.

Source: https://archive.ics.uci.edu/dataset/502/online+retail

The dataset contains transactions for a UK-based online retail store between 01 December 2009 and 09 December 2011. Important fields are:

- Invoice or InvoiceNo: invoice identifier
- StockCode: product identifier
- Description: product name
- Quantity: units purchased
- InvoiceDate: transaction date and time
- Price or UnitPrice: product unit price
- Customer ID or CustomerID: customer identifier
- Country: customer country

Cleaning rules used in the project:

- Preserve the raw dataset.
- Create processed CSV files separately.
- Treat invoice numbers starting with C or rows with non-positive quantity as cancellations or returns.
- Use only positive quantity and positive price rows for churn features, recommendations, and forecasting.
- Exclude missing customer IDs from customer-level churn modeling.

