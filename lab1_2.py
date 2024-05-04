from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori

# Завантаження даних
df = pd.read_csv('Groceries_dataset.csv')

# Перетворення стовбця дати
df['Date'] = pd.to_datetime(df['Date'])

# Групування за датою та ідентифікатором покупця та підрахунок унікальних продуктів
transactions = df.groupby(['Date', 'Member_number'])['itemDescription'].unique().reset_index(name='items')

# Підрахунок кількості транзакцій для кожного унікального набору продуктів
sublist_counts = Counter(tuple(items) for items in transactions['items'])

# Отримання 25 найбільш часто вживаних наборів продуктів
top_sublists = sublist_counts.most_common(25)

# Розпакування кортежів і конвертація наборів у рядки
sublists, counts = zip(*top_sublists)
sublists = [', '.join(sublist) for sublist in sublists]

# Створення горизонтального графіка
plt.figure(figsize=(10, 6))
plt.barh(sublists, counts, color='grey')
plt.xlabel('Count')
plt.ylabel('Transaction Items')
plt.title('25 Most Frequent Transaction Items')
plt.tight_layout()
plt.show()
