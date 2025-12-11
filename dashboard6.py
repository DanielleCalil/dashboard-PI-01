import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

# Configuração da Página (Deve ser o primeiro comando)
st.set_page_config(page_title="Dashboard de Saúde - Análise de Correlações", layout="wide")

# --- FUNÇÃO DE CARREGAMENTO E TRATAMENTO DE DADOS ---
@st.cache_data
def carregar_dados():
    # Ler o arquivo CSV
    # Tenta ler com a codificação padrão, se falhar tenta latin-1
    try:
        df = pd.read_csv("saude_processada.csv")
    except UnicodeDecodeError:
        df = pd.read_csv("saude_processada.csv", encoding="latin-1")

    # 1. Tratamento RIGOROSO de Valores "NÃO DEFINIDO"
    # Converte para string primeiro para evitar erros
    # Remove espaços em branco no início e fim
    df['diagnostico'] = df['diagnostico'].astype(str).str.strip()
    
    # Regex poderosa: substitui .NÃO DEFINIDO. (com ou sem acento, com ou sem pontos) por NaN
    # Explicação do regex: ^ (inicio) \.? (ponto opcional) N[ÃA]O (NÃO ou NAO) \s (espaço) DEFINIDO \.? (ponto opcional) $ (fim)
    df['diagnostico'] = df['diagnostico'].replace(r'^\.?N[ÃA]O\s+DEFINIDO\.?$', np.nan, regex=True)
    
    # Remove strings que viraram "nan" texto por acidente
    df['diagnostico'] = df['diagnostico'].replace('nan', np.nan)

    # 2. Conversão de Datas
    df['dataNascimento'] = pd.to_datetime(df['dataNascimento'], errors='coerce')
    df['dataEntrada'] = pd.to_datetime(df['dataEntrada'], errors='coerce')

    # 3. Cálculo da Idade (Baseada na Data de Entrada)
    df['idade_no_atendimento'] = (df['dataEntrada'] - df['dataNascimento']).dt.days // 365
    
    # Remover idades negativas ou irreais
    df = df[(df['idade_no_atendimento'] >= 0) & (df['idade_no_atendimento'] < 120)]

    # 4. Criação de Faixas Etárias
    bins = [0, 2, 12, 19, 59, 120]
    labels = ['Bebê (0-2)', 'Criança (3-12)', 'Adolescente (13-19)', 'Adulto (20-59)', 'Idoso (60+)']
    df['faixa_etaria'] = pd.cut(df['idade_no_atendimento'], bins=bins, labels=labels, right=False)

    return df

# Carrega os dados
try:
    df = carregar_dados()
except FileNotFoundError:
    st.error("Erro: O arquivo 'saude_processada.csv' não foi encontrado na mesma pasta do script.")
    st.stop()

# Definir labels globalmente
labels_faixa = ['Bebê (0-2)', 'Criança (3-12)', 'Adolescente (13-19)', 'Adulto (20-59)', 'Idoso (60+)']

# --- BARRA LATERAL (FILTROS GERAIS) ---
st.sidebar.header("Filtros Globais")

# Filtro de Cidade
if 'cidade' in df.columns:
    cidades = st.sidebar.multiselect(
        "Selecione a Cidade:",
        options=df['cidade'].unique(),
        default=df['cidade'].unique()
    )
else:
    cidades = []

# Filtro de Data
min_date = df['dataEntrada'].min().date()
max_date = df['dataEntrada'].max().date()

try:
    data_inicio, data_fim = st.sidebar.date_input(
        "Período de Análise:",
        [min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )
except ValueError:
    st.sidebar.warning("Selecione um período válido.")
    data_inicio, data_fim = min_date, max_date

# Aplicar Filtros
df_filtered = df[
    (df['cidade'].isin(cidades)) &
    (df['dataEntrada'].dt.date >= data_inicio) &
    (df['dataEntrada'].dt.date <= data_fim)
]

# --- LAYOUT DO DASHBOARD ---

st.title("🏥 Dashboard de Análise Clínica e Epidemiológica")
st.markdown("Este painel visa identificar correlações entre perfis demográficos e diagnósticos clínicos.")

# KPIs
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Total de Atendimentos", f"{len(df_filtered):,}")
kpi2.metric("Pacientes Únicos", f"{df_filtered['_id'].nunique():,}")
kpi3.metric("Média de Idade", f"{df_filtered['idade_no_atendimento'].mean():.1f} anos")

# KPI de Diagnóstico mais comum (ignora Nulos)
try:
    top_diag = df_filtered['diagnostico'].dropna().value_counts().idxmax()
except:
    top_diag = "N/A"
kpi4.metric("Diagnóstico + Comum", top_diag)

st.markdown("---")

# --- SEÇÃO 1: VISÃO GERAL ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Evolução dos Atendimentos")
    vendas_tempo = df_filtered.set_index('dataEntrada').resample('M').size().reset_index(name='Atendimentos')
    fig_line = px.line(vendas_tempo, x='dataEntrada', y='Atendimentos', markers=True, 
                       title="Tendência Temporal")
    st.plotly_chart(fig_line, use_container_width=True)

with col2:
    st.subheader("Distribuição por Gênero")
    fig_pie = px.pie(df_filtered, names='sexo', title="Gênero", hole=0.4,
                     color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig_pie, use_container_width=True)

# --- SEÇÃO 2: CORRELAÇÕES ---
st.markdown("## 🔍 Análise de Correlações")
st.info("Utilize esta seção para validar hipóteses (Ex: 'Mulheres jovens têm mais enxaqueca?').")

tab1, tab2 = st.tabs(["Por Diagnóstico", "Por Palavra-Chave na Queixa"])

with tab1:
    # Remove NaNs da lista de seleção para não aparecer opção em branco
    lista_diagnosticos = df_filtered['diagnostico'].dropna().unique()
    
    if len(lista_diagnosticos) > 0:
        diag_selecionado = st.selectbox("Selecione um Diagnóstico:", options=np.sort(lista_diagnosticos))

        if diag_selecionado:
            df_diag = df_filtered[df_filtered['diagnostico'] == diag_selecionado]
            
            st.markdown(f"### Perfil: **{diag_selecionado}** ({len(df_diag)} casos)")
            
            c1, c2 = st.columns(2)
            with c1:
                fig_bar_age = px.histogram(df_diag, x="faixa_etaria", color="sexo", 
                                           title="Faixa Etária e Sexo",
                                           category_orders={"faixa_etaria": labels_faixa},
                                           text_auto=True, barmode='group')
                st.plotly_chart(fig_bar_age, use_container_width=True)
                
            with c2:
                fig_hist = px.histogram(df_diag, x="idade_no_atendimento", nbins=20, 
                                        title="Idade Detalhada",
                                        color_discrete_sequence=['#636EFA'])
                fig_hist.update_layout(bargap=0.1)
                st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.warning("Não há diagnósticos definidos no período selecionado.")

with tab2:
    termo_busca = st.text_input("Buscar termo na QUEIXA (Ex: dor, febre):", "")
    
    if termo_busca:
        df_queixa = df_filtered[df_filtered['queixa'].astype(str).str.contains(termo_busca, case=False, na=False)]
        
        st.markdown(f"### Perfil para termo: **'{termo_busca}'** ({len(df_queixa)} casos)")
        
        if not df_queixa.empty:
            c3, c4 = st.columns(2)
            with c3:
                heatmap_data = pd.crosstab(df_queixa['faixa_etaria'], df_queixa['sexo'])
                fig_heat = px.imshow(heatmap_data, text_auto=True, aspect="auto",
                                     title=f"Mapa de Calor: Intensidade",
                                     color_continuous_scale='Viridis')
                st.plotly_chart(fig_heat, use_container_width=True)
            with c4:
                fig_box = px.box(df_queixa, x="sexo", y="idade_no_atendimento", points="all",
                                 title="Boxplot de Idade", color="sexo")
                st.plotly_chart(fig_box, use_container_width=True)

# --- SEÇÃO 3: RANKING ---
st.markdown("---")
st.subheader("Top 10 Diagnósticos (Excluindo 'Não Definido')")

# Filtra explicitamente nulos antes de contar
df_clean_diag = df_filtered.dropna(subset=['diagnostico'])
top_diagnosticos = df_clean_diag['diagnostico'].value_counts().head(10).reset_index()
top_diagnosticos.columns = ['Diagnóstico', 'Contagem']

if not top_diagnosticos.empty:
    fig_bar_horiz = px.bar(top_diagnosticos, x='Contagem', y='Diagnóstico', orientation='h',
                           title="Ranking de Ocorrências", text='Contagem',
                           color='Contagem', color_continuous_scale='Blues')
    fig_bar_horiz.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_bar_horiz, use_container_width=True)
else:
    st.info("Sem dados suficientes para o ranking.")