import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar o CSV
arquivo = pd.read_csv('C:/Users/Joao/PycharmProjects/trabalhoEstatistica/animes.csv')
df = arquivo.copy()
pd.set_option('display.max_columns', None)  # Mostrar todas as colunas

print(df.head())
print("***********************************\n\n")
print(df.info())
print("***********************************\n\n")

print("Verificando se existem dados nulos:")
print(df.isnull().sum())
print("\nVerificando se existem dados duplicados:")
print(df.duplicated().sum())

print("***********************************\n\n")

# Remover duplicatas
df = df.drop_duplicates()

# Verificar se as duplicatas foram removidas
print(f"Total de duplicatas após a remoção: {df.duplicated().sum()}")

# Preencher valores nulos com 0 nas colunas 'episodes', 'ranked', 'score'
df['episodes'] = df['episodes'].fillna(0)
df['ranked'] = df['ranked'].fillna(0)
df['score'] = df['score'].fillna(0)
df['synopsis'] = df['synopsis'].fillna("Sinopse não disponível")
df['img_url'] = df['img_url'].fillna("qualquer")
print("***********************************\n\n")

# Estatísticas descritivas
print(df.describe(include='all'))
print("\n\n")

# Criar histogramas para as variáveis quantitativas
df[['episodes', 'members', 'popularity', 'ranked', 'score']].hist(bins=30, figsize=(10, 8))
plt.suptitle('Histogramas das Variáveis Quantitativas', fontsize=16)
plt.show()

# Criar box-plots para as variáveis quantitativas
plt.figure(figsize=(12, 8))

for i, coluna in enumerate(['episodes', 'members', 'popularity', 'ranked', 'score'], 1):
    plt.subplot(2, 3, i)
    sns.boxplot(y=df[coluna])
    plt.title(f'Box-Plot de {coluna.capitalize()}')

plt.tight_layout()
plt.suptitle('Box-Plots das Variáveis Quantitativas', fontsize=16)
plt.subplots_adjust(top=0.88)
plt.show()

# Identificar animes com número de episódios atípico
episodios_outliers = df[df['episodes'] > df['episodes'].quantile(0.95)]  # Exemplo para identificar os 5% maiores valores
print("Animes com número de episódios atípico:")
print(episodios_outliers[['title', 'episodes']])

print("***********************************\n\n")

# Selecionar apenas colunas numéricas para calcular a matriz de correlação
df_numerico = df[['episodes', 'members', 'popularity', 'ranked', 'score']]

# Calcular a matriz de correlação usando Pearson
correlation_matrix = df_numerico.corr(method='pearson')

# Exibir a matriz de correlação
print("Matriz de correlação:")
print(correlation_matrix)

print("***********************************\n\n")

print("Segmentação por gênero")
# Verificar a lista de gêneros
df['genre'] = df['genre'].fillna("Unknown")
df['first_genre'] = df['genre'].apply(lambda x: x.split(',')[0])

# Agrupar por primeiro gênero e calcular as médias das variáveis de interesse
genre_grouped = df.groupby('first_genre').agg({
    'popularity': 'mean',
    'score': 'mean',
    'episodes': 'mean',
    'members': 'mean'
}).sort_values(by='popularity', ascending=True)  # Ordenar por popularidade crescente

print(genre_grouped)
print("***********************************\n\n")

print("Segmentação por número de episódios")
# Criar categorias para o número de episódios
df['episode_length'] = pd.cut(df['episodes'], bins=[-1, 12, 26, float('inf')], labels=['Curto', 'Médio', 'Longo'])

# Agrupar por categoria de número de episódios e calcular as médias das variáveis de interesse
episode_grouped = df.groupby('episode_length').agg({
    'popularity': 'mean',
    'score': 'mean',
    'members': 'mean'
})

print(episode_grouped)

print("***********************************\n\n")

print("Segmentação por lançamento")
# Preencher valores nulos na coluna de lançamento
df['aired'] = df['aired'].fillna("0000")

# Extrair o ano do lançamento
df['year'] = df['aired'].str.extract(r'(\d{4})').astype(float)

# Criar a coluna de décadas
df['decade'] = (df['year'] // 10) * 10

# Agrupar por década e calcular as médias das variáveis de interesse
decade_grouped = df.groupby('decade').agg({
    'popularity': 'mean',
    'score': 'mean',
    'members': 'mean'
}).sort_values(by='decade')

print(decade_grouped)

