# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Dashboard Gestão", layout="wide")

# -----------------------
# 1. Gerar dados simulados
# -----------------------
@st.cache_data
def gerar_dados(seed=42):
    np.random.seed(seed)
    meses = pd.date_range(end=pd.Timestamp.today(), periods=12, freq='M')
    produtos = ["Produto A", "Produto B", "Produto C"]
    dados = []

    for i, m in enumerate(meses):
        faturamento_total = 15000 + i * 700 + np.random.normal(0,1500)
        despesas_total = 9000 + i * 200 + np.random.normal(0,1000)
        clientes = 100 + i*3 + int(np.random.normal(0,5))

        shares = np.random.dirichlet(np.ones(len(produtos)))
        for p, s in zip(produtos, shares):
            qtd = int(faturamento_total * s / 50)
            dados.append({
                "mes": m.strftime("%Y-%m"),
                "produto": p,
                "quantidade": qtd,
                "faturamento": round(faturamento_total * s + np.random.normal(0,200)),
                "despesa": round(despesas_total/len(produtos) + np.random.normal(0,50)),
                "clientes": clientes
            })
    return pd.DataFrame(dados)

df = gerar_dados()

# -----------------------
# 2. Agregações e KPIs
# -----------------------
st.title("Dashboard de Gestão - Teste")
st.markdown("""
Este dashboard é um **exemplo de aplicação prática de IA** em gestão:  
- **Previsão de faturamento** para os próximos meses (IA básica com regressão linear).  
- **Análises automáticas** de tendências e alertas sobre faturamento, lucro e produtos.  
""")

mensal = df.groupby("mes").agg(
    faturamento=("faturamento","sum"),
    despesa=("despesa","sum"),
    clientes=("clientes","max")
).reset_index()
mensal["lucro"] = mensal["faturamento"] - mensal["despesa"]
mensal["margem_pct"] = (mensal["lucro"] / mensal["faturamento"]).fillna(0)

k1, k2, k3 = st.columns(3)
k1.metric("Faturamento (últ. mês)", f"R$ {mensal.iloc[-1].faturamento:,.0f}")
k2.metric("Despesa (últ. mês)", f"R$ {mensal.iloc[-1].despesa:,.0f}")
k3.metric("Lucro (últ. mês)", f"R$ {mensal.iloc[-1].lucro:,.0f}", delta=f"{mensal.iloc[-1].margem_pct*100:.1f}%")

# -----------------------
# 3. Gráficos principais
# -----------------------
col1, col2 = st.columns((2,1))
with col1:
    fig_fat = px.line(mensal, x="mes", y="faturamento", title="Faturamento Mensal", markers=True)
    st.plotly_chart(fig_fat, use_container_width=True)

    fig_lucro = px.bar(mensal, x="mes", y="lucro", title="Lucro Mensal")
    st.plotly_chart(fig_lucro, use_container_width=True)

with col2:
    ult6 = df[df['mes'].isin(mensal['mes'].iloc[-6:])]
    top_prod = ult6.groupby("produto").agg(faturamento=("faturamento","sum")).reset_index()
    st.subheader("Top produtos (últ. 6 meses)")
    st.dataframe(top_prod.sort_values("faturamento", ascending=False))

# -----------------------
# 4. Previsão simples (IA)
# -----------------------
st.subheader("Previsão Faturamento (próx. 3 meses)")
st.markdown("**IA em ação:** usamos regressão linear para identificar tendência histórica e prever faturamento futuro automaticamente.")

mensal_model = mensal.copy()
mensal_model['indice'] = np.arange(len(mensal_model))
mensal_model['mes_dt'] = pd.to_datetime(mensal_model['mes'] + "-01")  # datetime

X = mensal_model[['indice']]
y = mensal_model['faturamento']
model = LinearRegression()
model.fit(X, y)

futuro_idx = np.arange(X.values[-1][0]+1, X.values[-1][0]+4).reshape(-1,1)
preds = model.predict(futuro_idx)

# Datas futuras com pd.DateOffset
ultima_data = mensal_model['mes_dt'].iloc[-1]
datas_fut = [ultima_data + pd.DateOffset(months=i) for i in range(1,4)]
pred_df = pd.DataFrame({"mes_dt": datas_fut, "faturamento": preds})

# Concatenar histórico + previsão
df_plot = pd.concat([mensal_model[['mes_dt','faturamento']], pred_df])
df_plot['mes_dt'] = pd.to_datetime(df_plot['mes_dt'])  # garantir datetime

fig_prev = px.line(df_plot, x="mes_dt", y="faturamento", title="Histórico + Previsão")

# Linha "agora" com add_shape
fig_prev.add_shape(
    type="line",
    x0=ultima_data,
    x1=ultima_data,
    y0=0,
    y1=1,
    xref='x',
    yref='paper',
    line=dict(color="red", width=2, dash="dash")
)

# Adicionar anotação "agora"
fig_prev.add_annotation(
    x=ultima_data,
    y=max(df_plot['faturamento'])*1.02,
    text="agora",
    showarrow=True,
    arrowhead=3,
    ax=0,
    ay=-30
)

st.plotly_chart(fig_prev, use_container_width=True)

# -----------------------
# 5. Insights automáticos
# -----------------------
st.subheader("Insights automáticos (IA)")
st.markdown("O dashboard gera automaticamente alertas e recomendações com base nos dados:")

insights = []

if mensal['faturamento'].iloc[-1] < mensal['faturamento'].mean():
    insights.append("Faturamento abaixo da média — atenção a vendas.")
else:
    insights.append("Faturamento dentro do esperado.")

if mensal['lucro'].iloc[-1] < mensal['lucro'].mean():
    insights.append("Lucro abaixo da média — reveja despesas.")

top_nome = ult6.groupby("produto").faturamento.sum().idxmax()
insights.append(f"Produto com melhor desempenho (últ. 6 meses): {top_nome}")

for i, item in enumerate(insights,1):
    st.info(f"{i}. {item}")

# -----------------------
# 6. Dados e export
# -----------------------
st.subheader("Dados resumidos")
st.dataframe(mensal)
csv = mensal.to_csv(index=False).encode('utf-8')
st.download_button("Baixar CSV", csv, "resumo.csv", "text/csv")
