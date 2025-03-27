import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import seaborn as sns

def generate_Nigerian_retail_data():
    # More comprehensive product categories
    product_categories = [
        ['Rice', 'Palm Oil', 'Tomato', 'Onions'],
        ['Bread', 'Eggs', 'Butter', 'Milk'],
        ['Chicken', 'Spices', 'Plantain', 'Tomato'],
        ['Rice', 'Chicken', 'Spices'],
        ['Bread', 'Eggs', 'Milk'],
        ['Palm Oil', 'Tomato', 'Onions', 'Chicken'],
        ['Rice', 'Beans', 'Pepper'],
        ['Bread', 'Butter', 'Eggs'],
        ['Plantain', 'Chicken', 'Spices'],
        ['Rice', 'Palm Oil', 'Chicken']
    ]
    return product_categories

# Generate and Prepare Data
transactions = generate_Nigerian_retail_data()
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Apriori and Association Rules
frequent_itemsets = apriori(df, min_support=0.3, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

# Multiple Visualizations
plt.figure(figsize=(20,15))

# 1. Scatter Plot of Association Rules
plt.subplot(2,2,1)
sns.scatterplot(data=rules, x='support', y='confidence', size='lift', hue='lift', palette='viridis')
plt.title('Association Rules: Support vs Confidence')
plt.xlabel('Support')
plt.ylabel('Confidence')

# 2. Top Rules Bar Plot
plt.subplot(2,2,2)
top_rules = rules.sort_values('lift', ascending=False).head(10)
plt.bar(
    [f"{list(x)[0]} → {list(y)[0]}" for x, y in zip(top_rules['antecedents'], top_rules['consequents'])], 
    top_rules['lift']
)
plt.title('Top 10 Association Rules by Lift')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Lift')

# 3. Frequent Itemsets Heatmap
plt.subplot(2,2,3)
frequent_pivot = frequent_itemsets.pivot_table(
    index='itemsets', 
    columns='support', 
    aggfunc='size'
).fillna(0)
sns.heatmap(frequent_pivot, cmap='YlGnBu')
plt.title('Frequent Itemsets Heatmap')

# 4. Rule Metrics Distribution
plt.subplot(2,2,4)
sns.boxplot(data=rules[['support', 'confidence', 'lift']])
plt.title('Distribution of Rule Metrics')

plt.tight_layout()
plt.savefig('comprehensive_market_basket_analysis.png')
plt.close()

# Print and Export Insights
print("🇳🇬 Nigerian Retail Market Basket Analysis Insights 🛒")
print("\n1. Top 5 Frequent Itemsets:")
print(frequent_itemsets.head())

print("\n2. Top Association Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values('lift', ascending=False).head())

# Export Results
frequent_itemsets.to_csv('frequent_itemsets.csv', index=False)
rules.to_csv('association_rules.csv', index=False)

# Actionable Recommendations
def generate_recommendations(rules):
    return [
        f"When customers buy {list(rule['antecedents'])}, recommend {list(rule['consequents'])} (Lift: {rule['lift']:.2f})"
        for _, rule in rules.sort_values('lift', ascending=False).head(10).iterrows()
    ]

retailer_recommendations = generate_recommendations(rules)
print("\n3. Top Retailer Recommendations:")
for rec in retailer_recommendations:
    print(rec)