import streamlit as st
import pandas as pd
import plotly.express as px
from data_processing import load_data, load_rename_mapping

# Carregar os dados processados
df = load_data()

# Carregar o mapeamento de nomes dos procedimentos
rename_mapping = load_rename_mapping()

# Título da página
st.title("Análises Comparativas de Internações")
st.markdown("""
Nesta página, realizamos comparações de internações hospitalares considerando diferentes variáveis:
- Frequência por região
- Frequência por tipo de procedimento
- Frequência por faixa populacional
- Mapa de calor baseado na localização geográfica das internações
""")

# ---------------------------------------------
# Seção 1: Comparação entre Regiões
# ---------------------------------------------
st.header("1. Comparação entre Regiões")

# Agrupar dados por região
regiao_data = df.groupby('regiao_nome').size().reset_index(name='Frequência')
regiao_data = regiao_data.sort_values(by='Frequência', ascending=False)

st.write("Frequência de Internações por Região:")
st.dataframe(regiao_data)

# Gráfico de barras
fig_regiao = px.bar(
    regiao_data,
    x='regiao_nome',
    y='Frequência',
    title="Frequência de Internações por Região",
    labels={'regiao_nome': 'Região', 'Frequência': 'Número de Internações'},
)
st.plotly_chart(fig_regiao)
# ---------------------------------------------
# Seção 2: Frequência por Tipo de Procedimento
# ---------------------------------------------
st.header("2. Frequência por Tipo de Procedimento")

# Selecionar apenas as colunas de procedimentos (baseado nos nomes renomeados)
procedimento_cols = [col for col in df.columns if col in rename_mapping.values()]

# Verificar os dados das colunas de procedimentos
st.write("Verificando os valores nas colunas de procedimentos:")
st.write(df[procedimento_cols].head())

# Somar os valores das colunas de procedimentos
procedimento_data = pd.DataFrame(df[procedimento_cols].sum(), columns=['Frequência']).reset_index()
procedimento_data.columns = ['Nome do Procedimento', 'Frequência']

# Garantir que a coluna 'Frequência' seja numérica
procedimento_data['Frequência'] = pd.to_numeric(procedimento_data['Frequência'], errors='coerce')

# Remover linhas com valores inválidos em 'Frequência'
procedimento_data = procedimento_data.dropna(subset=['Frequência'])

# Converter 'Frequência' para inteiro (se necessário)
procedimento_data['Frequência'] = procedimento_data['Frequência'].astype(int)

# Filtrar para remover o "Valor total dos procedimentos"
procedimento_data = procedimento_data[procedimento_data['Nome do Procedimento'] != 'Valor total dos procedimentos']

# Filtrar os 10 procedimentos mais frequentes
procedimento_data = procedimento_data.sort_values(by='Frequência', ascending=False).head(10)

# Exibir o DataFrame processado
st.write("Top 10 Procedimentos com Mais Internações:")
st.dataframe(procedimento_data)

# Gráfico de barras
fig_procedimento = px.bar(
    procedimento_data,
    x='Nome do Procedimento',
    y='Frequência',
    title="Frequência por Tipo de Procedimento (Top 10)",
    labels={'Nome do Procedimento': 'Procedimento', 'Frequência': 'Número de Internações'},
)
st.plotly_chart(fig_procedimento)

# Gráfico de pizza
fig_pizza = px.pie(
    procedimento_data,
    values='Frequência',
    names='Nome do Procedimento',
    title="Distribuição de Frequências por Procedimento (Top 10)",
)
st.plotly_chart(fig_pizza)

# ---------------------------------------------
# Seção 3: Frequência por Faixa Populacional
# ---------------------------------------------
st.header("3. Frequência por Faixa Populacional")

# Agrupar dados por faixa populacional
populacao_data = df.groupby('faixa_populacao').size().reset_index(name='Frequência')
populacao_data = populacao_data.sort_values(by='Frequência', ascending=False)

st.write("Frequência de Internações por Faixa Populacional:")
st.dataframe(populacao_data)

# Gráfico de barras
fig_populacao = px.bar(
    populacao_data,
    x='faixa_populacao',
    y='Frequência',
    title="Frequência de Internações por Faixa Populacional",
    labels={'faixa_populacao': 'Faixa Populacional', 'Frequência': 'Número de Internações'},
)
st.plotly_chart(fig_populacao)