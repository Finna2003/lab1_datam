from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
import networkx as nx


# Завантаження даних
df = pd.read_csv('Groceries_dataset.csv')

# Перетворення стовбця дати з правильним параметром для дати
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

# Групування за датою та ідентифікатором покупця та підрахунок унікальних продуктів
transactions = df.groupby(['Date', 'Member_number'])['itemDescription'].unique().reset_index(name='items')

# Preparation for Apriori Algorithm
records_final = transactions['items'].tolist()

# Running Apriori Algorithm
results = list(apriori(records_final, min_support=0.005, use_colnames=True, max_len=3))

# Convert results to DataFrame
frequent_itemsets = pd.DataFrame(
    [(tuple(itemset.items), itemset.support) for itemset in results],
    columns=['itemsets', 'support']
)

# Sorting the DataFrame
frequent_itemsets = frequent_itemsets.sort_values(by='support', ascending=False)

# Filtering itemsets with more than one item
filtered_itemsets = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: len(x) > 1)]
top_filtered_itemsets = filtered_itemsets.head(25)
top_filtered_itemsets.loc[:, 'itemsets'] = top_filtered_itemsets['itemsets'].apply(lambda x: ','.join(map(str, x)))


# Visualization of Apriori Results
plt.figure(figsize=(12, 8))
plt.barh(top_filtered_itemsets['itemsets'].astype(str), top_filtered_itemsets['support'], color='grey')
plt.xlabel('Support')
plt.ylabel('Itemsets')
plt.title('Top 25 Most Frequent Item Combinations (Sets > 1)')
plt.show()

encoder = TransactionEncoder()
onehot = encoder.fit(records_final).transform(records_final)
onehot_df = pd.DataFrame(onehot, columns=encoder.columns_)

# Now, run Apriori again using mlxtend to get the correct format
from mlxtend.frequent_patterns import apriori

frequent_itemsets = apriori(onehot_df, min_support=0.005, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.01)
rules = rules.sort_values('lift', ascending=False)
rules = rules[['antecedents','consequents','support', 'confidence', 'lift' ]]
print(rules.head(25))

rules_to_display = rules.head(25)
G = nx.DiGraph()
for idx, rule in rules_to_display.iterrows():
    antecedents = ','.join(rule['antecedents'])
    consequents = ','.join(rule['consequents'])
    lift_value = f'{rule["lift"]:.4f}'
    G.add_edge(antecedents, consequents, weight=lift_value)

plt.figure(figsize=(12, 12))
pos = nx.spring_layout (G, seed=42)
nx.draw(G, pos, with_labels=True, font_size=8, font_color='black', node_size=2000, node_color='lightgreen', font_weight='bold', edge_color='gray', linewidths=1, alpha=0.7, arrowsize=20)

plt.title('Association Rules Network Graph') 
plt.show()

