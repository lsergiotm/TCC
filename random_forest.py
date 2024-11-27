import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from data_processing import load_data

# Carregar os dados usando o script data_processing.py
df = load_data()

# Exibir colunas e prévia dos dados carregados
st.title("Modelo Random Forest - Previsão de Custos")
st.subheader("Visualização dos Dados Carregados")
st.write("Colunas disponíveis no DataFrame:")
st.write(df.columns.tolist())
st.write("Prévia dos dados carregados:")
st.dataframe(df.head())

# Verificar valores nulos e únicos antes de prosseguir
st.write("Valores únicos por coluna:")
st.write(df.nunique())
st.write("Contagem de valores nulos:")
st.write(df.isnull().sum())

# Verificar se o dataframe está vazio
if df.empty:
    st.error("O dataframe está vazio após o carregamento. Verifique os dados de entrada.")
else:
    # Selecionar e converter colunas relevantes
    st.write("Processando colunas relevantes...")
    try:
        df['faixa_populacao'] = pd.to_numeric(df['faixa_populacao'], errors='coerce')
        df['Quantidade total de procedimentos'] = pd.to_numeric(df['Quantidade total de procedimentos'], errors='coerce')
        df['Valor total dos procedimentos'] = pd.to_numeric(df['Valor total dos procedimentos'], errors='coerce')
    except KeyError:
        st.error("Algumas colunas necessárias não estão disponíveis no DataFrame.")

    # Tratar valores nulos
    st.write("Substituindo valores nulos por 0...")
    df.fillna(0, inplace=True)

    # Remover linhas com valores não numéricos restantes
    df = df.dropna()

    # Exibir informações após a limpeza
    st.write("Dados após limpeza e conversão:")
    st.dataframe(df[['faixa_populacao', 'Quantidade total de procedimentos', 'Valor total dos procedimentos']].head())
    st.write("Quantidade de valores nulos após a limpeza:")
    st.write(df[['faixa_populacao', 'Quantidade total de procedimentos', 'Valor total dos procedimentos']].isnull().sum())

    # Verificar se existem dados suficientes para treinamento
    if df.shape[0] < 10:
        st.error("Dados insuficientes para treinamento após a limpeza. Verifique os dados de entrada.")
    else:
        # Separar os dados em features (X) e target (y)
        X = df[['faixa_populacao', 'Quantidade total de procedimentos']]
        y = df['Valor total dos procedimentos']

        # Interface do Streamlit para ajustar parâmetros do modelo
        st.sidebar.header("Configurações do Modelo")
        n_estimators = st.sidebar.slider("Número de Árvores (n_estimators):", min_value=10, max_value=500, value=100, step=10)

        # Dividir os dados em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Treinar o modelo Random Forest
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)

        # Fazer previsões
        y_pred = model.predict(X_test)

        # Calcular métricas
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        # Importância das variáveis
        feature_importances = pd.DataFrame({
            'Variável': X.columns,
            'Importância': model.feature_importances_
        }).sort_values(by='Importância', ascending=False)

        # Exibição das métricas
        st.subheader("Métricas do Modelo")
        st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
        st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
        st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")
        st.write(f"**R² Score:** {r2:.2f}")

        # Interpretação das métricas
        st.subheader("Interpretação das Métricas")
        if r2 > 0.9:
            st.success("O modelo apresenta uma excelente capacidade de explicar a variabilidade nos dados.")
        elif r2 > 0.75:
            st.info("O modelo apresenta uma boa capacidade de explicação, mas ainda há espaço para melhorias.")
        else:
            st.warning("O modelo apresenta baixa capacidade de explicação. Considere ajustar os parâmetros ou utilizar mais dados.")

        st.markdown("""
        **MAE:** Indica o erro médio absoluto entre os valores previstos e reais. Valores menores são melhores.  
        **MSE e RMSE:** Medem o erro quadrático médio. RMSE é mais interpretável, sendo na mesma escala dos valores reais.  
        **R²:** Mede o percentual de variabilidade explicada pelo modelo. Quanto mais próximo de 1, melhor.
        """)

        # Visualização da importância das variáveis
        st.subheader("Importância das Variáveis")
        feature_importances_sorted = feature_importances.sort_values(by="Importância", ascending=True)
        fig_importance = px.bar(
            feature_importances_sorted,
            x='Importância',
            y='Variável',
            orientation='h',
            text='Importância',
            title="Importância das Variáveis",
            labels={'Importância': 'Importância', 'Variável': 'Variáveis'},
            template="plotly_white"
        )
        fig_importance.update_traces(
            marker_color='skyblue',
            texttemplate='%{text:.2f}',
            textposition='outside'
        )
        st.plotly_chart(fig_importance)

        # Gráfico de comparação entre valores reais e previstos
        st.subheader("Comparação de Valores Reais vs Previstos")
        fig_comparison = go.Figure()
        fig_comparison.add_trace(go.Scatter(
            x=y_test,
            y=y_pred,
            mode='markers',
            name='Valores',
            marker=dict(color='blue')
        ))
        fig_comparison.add_trace(go.Scatter(
            x=[y_test.min(), y_test.max()],
            y=[y_test.min(), y_test.max()],
            mode='lines',
            name='Linha Ideal',
            line=dict(color='red', dash='dot')
        ))
        fig_comparison.update_layout(
            title="Comparação de Valores Reais vs Previstos",
            xaxis_title="Valor Real (R$)",
            yaxis_title="Valor Previsto (R$)",
            showlegend=True
        )
        st.plotly_chart(fig_comparison)

        # Exibir dados reais e previstos em tabela
        st.subheader("Tabela Comparativa de Valores")
        comparison_df = pd.DataFrame({'Valor Real (R$)': y_test, 'Valor Previsto (R$)': y_pred})
        st.dataframe(comparison_df)
