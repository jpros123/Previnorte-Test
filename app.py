import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from datetime import datetime, date
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.grid import grid

# Configuração da página
st.set_page_config(layout="wide")

# Função para construir o sidebar
def build_sidebar():
    st.image("images/273731962_442203134141407_7743180946701135051_n.png")
    
    # Carregar lista de tickers
    try:
        ticker_list = pd.read_csv("tickers_ibra.csv", index_col=0)
    except Exception as e:
        st.error(f"Erro ao carregar lista de tickers: {e}")
        return None, None
    
    # Seleção de empresas
    tickers = st.multiselect(
        label="Selecione as Empresas", 
        options=ticker_list, 
        placeholder='Códigos'
    )
    
    # Adicionar sufixo ".SA" aos tickers
    tickers = [t + ".SA" for t in tickers]
    
    # Seleção de datas
    start_date = st.date_input("De", value=date(2020, 1, 2))
    end_date = st.date_input("Até", value=date.today())

    if tickers:
        try:
            # Baixar dados dos tickers
            prices = yf.download(tickers, start=start_date, end=end_date)["Close"]
            
            # Ajustar estrutura dos dados para um ticker
            if len(tickers) == 1:
                prices = prices.to_frame()
                prices.columns = [tickers[0].rstrip(".SA")]
            
            # Remover sufixo ".SA" dos nomes das colunas
            prices.columns = prices.columns.str.rstrip(".SA")
            
            # Adicionar dados do IBOV
            prices['IBOV'] = yf.download("^BVSP", start=start_date, end=end_date)["Close"]
            
            return tickers, prices
        except Exception as e:
            st.error(f"Erro ao baixar dados: {e}")
            return None, None
    return None, None

# Função para calcular métricas
def calcular_métricas(prices):
    # Calcular pesos iguais para cada ativo
    weights = np.ones(len([c for c in prices.columns if c != 'IBOV'])) / len([c for c in prices.columns if c != 'IBOV'])
    
    # Calcular preço do portfólio
    prices['portfolio'] = prices.drop("IBOV", axis=1) @ weights
    
    # Normalizar preços
    norm_prices = 100 * prices / prices.iloc[0]
    
    # Calcular retornos
    returns = prices.pct_change()[1:]
    
    # Calcular volatilidades anualizadas
    vols = returns.std() * np.sqrt(252)
    
    # Calcular retornos totais
    rets = (norm_prices.iloc[-1] - 100) / 100
    
    # Calcular Sharpe Ratio
    sharpe_ratios = rets / vols
    
    # Calcular Beta
    betas = returns.cov().loc[:, 'IBOV'] / returns['IBOV'].var()
    
    return norm_prices, returns, vols, rets, sharpe_ratios, betas

# Função para construir a página principal
def build_main(tickers, prices):
    norm_prices, returns, vols, rets, sharpe_ratios, betas = calcular_métricas(prices)

    # Criação de grid para exibir métricas
    mygrid = grid(2, 2, 2, 2, 2, 2, vertical_align="top")
    
    for t in prices.columns:
        c = mygrid.container(border=True)
        c.subheader(t, divider="red")
        
        # Dividir container em colunas
        colA, colB, colC, colD, colE = c.columns(5)
        
        # Exibir ícone
        if t == "portfolio":
            colA.image("images/pie-chart-dollar-svgrepo-com.svg")
        elif t == "IBOV":
            colA.image("images/bovespa-1.svg")
        else:
            try:
                colA.image(f'https://raw.githubusercontent.com/thefintz/icones-b3/main/icones/{t}.png', width=85)
            except Exception as e:
                st.warning(f"Ícone não encontrado para {t}.")
        
        # Exibir métricas
        colB.metric(label="Retorno", value=f"{rets[t]:.0%}")
        colC.metric(label="Volatilidade", value=f"{vols[t]:.0%}")
        colD.metric(label="Sharpe Ratio", value=f"{sharpe_ratios[t]:.2f}")
        colE.metric(label="Beta", value=f"{betas[t]:.2f}")
        
        # Estilizar cartões de métricas
        style_metric_cards(background_color='rgba(255,255,255,0)')

    # Dividir a página em duas colunas
    col1, col2 = st.columns(2, gap='large')
    
    with col1:
        st.subheader("Desempenho Relativo")
        st.line_chart(norm_prices, height=600)

    with col2:
        st.subheader("Risco-Retorno")
        fig = px.scatter(
            x=vols,
            y=rets,
            text=vols.index,
            color=rets / vols,
            color_continuous_scale=px.colors.sequential.Bluered_r
        )
        
        # Configuração do gráfico
        fig.update_traces(
            textfont_color='white', 
            marker=dict(size=45),
            textfont_size=10,                  
        )
        fig.layout.yaxis.title = 'Retorno Total'
        fig.layout.xaxis.title = 'Volatilidade (anualizada)'
        fig.layout.height = 600
        fig.layout.xaxis.tickformat = ".0%"
        fig.layout.yaxis.tickformat = ".0%"        
        fig.layout.coloraxis.colorbar.title = 'Sharpe'
        
        st.plotly_chart(fig, use_container_width=True)

# Construir sidebar
with st.sidebar:
    tickers, prices = build_sidebar()

# Título da página
st.title('Fundos Previnorte')

# Construir página principal se houver tickers selecionados
if tickers:
    build_main(tickers, prices)
