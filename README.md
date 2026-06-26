# Customer Churn Prediction
>- This project aims to minimise financial losses associated with customer search. Instead of plainly focusing on maximizing the accuracy of our model, we focus on other metrics such as precision and recall.
>- Our dataset consists of over 7000 rows and 33 columns of data. It is based in the U.S. state of California.
>- Kaggle link: https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset

## Initial Data Preprocessing 
- We dropped ['Country', 'State', 'Count', 'CustomerID', 'Zip Code', 'City', 'Latitude', 'Longitude', 'Lat Long', 'Churn Label', 'Churn Reason', 'Churn Score', 'CLTV', 'Total Charges'] columns as they were either redundant(Churn Label, , Lat Long) or useless (Country, State, Count, Customer_ID) correlated (Total Charges (being the product of Tenure Months and Monthly Charges)) or leaking data (CLTV, Total Charges, Churn Score). Latitude and Longitude were plotted on a graph with the churn value being the colour differentiating between them, but we could not find a suitable or differntiating factor.
  
  ![Latitude_Longitude_vs_Churn](plots/lat_long_churn.png)

## Data Insights
### Contract vs Churn
- When we plot contract vs churn, we get this heatmap,
 ![Contract vs Churn](plots/contract_vs_churn.png)
- This heatmap strongly suggests that people with month-to-month contract are most likely to churn, around 42.71% people churn, while for a One Year Contract, around 11.27% churn, while for a Two Year Contract, only around 2.83% people churn. The risk of churn is significantly higher when there is a smaller term contract.
### Contract Terms
- When we plot a boxplot having the values to Tenure Months for all the contract types: Month-to-month, One Year, Two Years, we observe certain outliers.
#### Month-to-Month
 ![Boxplot for Month-to-month](plots/Month-to-Month_box.png)
#### One Year
 ![Boxplot for One Year](plots/OneYear_box.png)
#### Two Years
 ![Boxplot for Two Years](plots/TwoYear_box.png)

 -There are 17 outliers for Month-to-month and 79 Outliers for Two Years and none for One year. Out of the 17 outliers for Month-to-month, 12 do not churn while 5 churn, which tells that about 70.59% of the outliers are willing to stay. We can also infer that people who have stayed for longer tenure (in terms of months) are more likely to be loyal and not churn. Bigger the term of the contract, greater is the retention rate.

 ## Model Training and Insights
 ### Model 1
 - We applied One-Hot Encoding on all the categorical columns and 
