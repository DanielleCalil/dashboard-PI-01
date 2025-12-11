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
    # Tenta ler com a codificação padrão, se falhar tenta latin-1 (comum em arquivos pt-br)
    try:
        df = pd.read_csv("saude_processada.csv")
    except UnicodeDecodeError:
        df = pd.read_csv("saude_processada.csv", encoding="latin-1")

    # 1. Tratamento do campo DIAGNÓSTICO (Remover .NÃO DEFINIDO.)
    # Substitui '.NÃO DEFINIDO.' por NaN (nulo) e remove espaços extras
    df['diagnostico'] = df['diagnostico'].replace('.NÃO DEFINIDO.', np.nan)
    df['diagnostico'] = df['diagnostico'].str.strip()

    # 2. Conversão de Datas
    df['dataNascimento'] = pd.to_datetime(df['dataNascimento'], errors='coerce')
    df['dataEntrada'] = pd.to_datetime(df['dataEntrada'], errors='coerce')

    # 3. Cálculo da Idade (Baseada na Data de Entrada, não hoje)
    # Isso é importante para saber a idade que o paciente tinha NA HORA do atendimento
    df['idade_no_atendimento'] = (df['dataEntrada'] - df['dataNascimento']).dt.days // 365
    
    # Remover idades negativas ou irreais (erro de cadastro)
    df = df[(df['idade_no_atendimento'] >= 0) & (df['idade_no_atendimento'] < 120)]

    # 4. Criação de Faixas Etárias (Para facilitar correlações como "Crianças entre 3 e 6")
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

# Definir labels globalmente para uso em gráficos
labels = ['Bebê (0-2)', 'Criança (3-12)', 'Adolescente (13-19)', 'Adulto (20-59)', 'Idoso (60+)']

# --- BARRA LATERAL (FILTROS GERAIS) ---
st.sidebar.header("Filtros Globais")

# Filtro de Cidade
cidades = st.sidebar.multiselect(
    "Selecione a Cidade:",
    options=df['cidade'].unique(),
    default=df['cidade'].unique()
)

# Filtro de Data (Período)
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

# Aplicar Filtros ao DataFrame Principal
df_filtered = df[
    (df['cidade'].isin(cidades)) &
    (df['dataEntrada'].dt.date >= data_inicio) &
    (df['dataEntrada'].dt.date <= data_fim)
]

# --- LAYOUT DO DASHBOARD ---

st.title("🏥 Dashboard de Análise Clínica e Epidemiológica")
st.markdown("Este painel visa identificar correlações entre perfis demográficos e diagnósticos clínicos.")

# KPIs (Indicadores Chave)
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Total de Atendimentos", f"{len(df_filtered):,}")
kpi2.metric("Pacientes Únicos", f"{df_filtered['_id'].nunique():,}")
kpi3.metric("Média de Idade", f"{df_filtered['idade_no_atendimento'].mean():.1f} anos")
try:
    top_diag = df_filtered['diagnostico'].value_counts().idxmax()
except:
    top_diag = "N/A"
kpi4.metric("Diagnóstico + Comum", top_diag)

st.markdown("---")

# --- SEÇÃO 1: VISÃO GERAL E TEMPORAL ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Evolução dos Atendimentos (Sazonalidade)")
    # Agrupar por mês/ano para ver tendências
    vendas_tempo = df_filtered.set_index('dataEntrada').resample('M').size().reset_index(name='Atendimentos')
    fig_line = px.line(vendas_tempo, x='dataEntrada', y='Atendimentos', markers=True, 
                       title="Tendência de Atendimentos ao Longo do Tempo")
    fig_line.update_layout(xaxis_title="Data", yaxis_title="Qtd. Atendimentos")
    st.plotly_chart(fig_line, use_container_width=True)

with col2:
    st.subheader("Distribuição por Gênero")
    fig_pie = px.pie(df_filtered, names='sexo', title="Proporção Masculino vs Feminino", hole=0.4,
                     color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig_pie, use_container_width=True)

# --- SEÇÃO 2: MOTOR DE CORRELAÇÃO (O que você pediu) ---
st.markdown("## 🔍 Análise de Correlações: Quem tem o quê?")
st.info("Utilize esta seção para validar hipóteses (Ex: 'Mulheres jovens têm mais enxaqueca?').")

# Dividir a lógica de correlação
tab1, tab2 = st.tabs(["Por Diagnóstico", "Por Palavra-Chave na Queixa"])

with tab1:
    # Escolher os Top 20 diagnósticos para não poluir, mas permitir busca
    lista_diagnosticos = df_filtered['diagnostico'].dropna().unique()
    diag_selecionado = st.selectbox("Selecione um Diagnóstico para investigar:", options=np.sort(lista_diagnosticos))

    if diag_selecionado:
        # Filtrar apenas dados desse diagnóstico
        df_diag = df_filtered[df_filtered['diagnostico'] == diag_selecionado]
        
        st.markdown(f"### Perfil dos Pacientes com: **{diag_selecionado}** ({len(df_diag)} casos)")
        
        c1, c2 = st.columns(2)
        
        # Correlação 1: Faixa Etária
        with c1:
            fig_bar_age = px.histogram(df_diag, x="faixa_etaria", color="sexo", 
                                       title="Distribuição por Faixa Etária e Sexo",
                                       category_orders={"faixa_etaria": labels},
                                       text_auto=True, barmode='group')
            st.plotly_chart(fig_bar_age, use_container_width=True)
            
        # Correlação 2: Histograma de Idade Detalhada
        with c2:
            fig_hist = px.histogram(df_diag, x="idade_no_atendimento", nbins=20, 
                                    title="Histograma Detalhado de Idade",
                                    color_discrete_sequence=['#636EFA'])
            fig_hist.update_layout(bargap=0.1)
            st.plotly_chart(fig_hist, use_container_width=True)

with tab2:
    st.markdown("Procure por termos específicos na queixa do paciente (Ex: 'dor de cabeça', 'febre', 'vomito').")
    termo_busca = st.text_input("Digite um termo para buscar na QUEIXA:", "")
    
    if termo_busca:
        # Filtra onde a queixa contém o termo (case insensitive)
        df_queixa = df_filtered[df_filtered['queixa'].astype(str).str.contains(termo_busca, case=False, na=False)]
        
        st.markdown(f"### Perfil de quem reclama de: **'{termo_busca}'** ({len(df_queixa)} casos)")
        
        if not df_queixa.empty:
            c3, c4 = st.columns(2)
            
            # Mapa de Calor: Idade x Sexo (Visualmente impactante para correlações)
            with c3:
                # Criar tabela de contingência
                heatmap_data = pd.crosstab(df_queixa['faixa_etaria'], df_queixa['sexo'])
                fig_heat = px.imshow(heatmap_data, text_auto=True, aspect="auto",
                                     title=f"Mapa de Calor: Intensidade de '{termo_busca}' por Grupo",
                                     color_continuous_scale='Viridis')
                st.plotly_chart(fig_heat, use_container_width=True)
                
            with c4:
                # Boxplot para ver a distribuição exata da idade
                fig_box = px.box(df_queixa, x="sexo", y="idade_no_atendimento", points="all",
                                 title="Distribuição de Idade (Boxplot)",
                                 color="sexo")
                st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.warning("Nenhum registro encontrado com esse termo.")

# --- SEÇÃO 3: RANKINGS GERAIS ---
st.markdown("---")
st.subheader("Top Diagnósticos Gerais (excluindo nulos)")

# Calcular Top 10 diagnósticos
top_diagnosticos = df_filtered['diagnostico'].value_counts().head(10).reset_index()
top_diagnosticos.columns = ['Diagnóstico', 'Contagem']

fig_bar_horiz = px.bar(top_diagnosticos, x='Contagem', y='Diagnóstico', orientation='h',
                       title="10 Diagnósticos Mais Frequentes", text='Contagem',
                       color='Contagem', color_continuous_scale='Blues')
fig_bar_horiz.update_layout(yaxis={'categoryorder':'total ascending'})
st.plotly_chart(fig_bar_horiz, use_container_width=True)