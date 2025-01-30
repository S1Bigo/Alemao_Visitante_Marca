import pandas as pd
import streamlit as st
import pickle
import xgboost as xgb


dados_consulta = pd.read_csv('https://www.football-data.co.uk/mmz4281/2425/D1.csv')
dados_consulta['Temporada'] = '2025'
dados_consulta['Temporada'] = dados_consulta['Temporada'].astype(int)

column_mapping = {
    'Div': 'League Division',
    'Date': 'Data',
    'Time': 'Horario',
    'HomeTeam': 'Time_Mandante',
    'AwayTeam': 'Time_Visitante',
    'FTHG': 'Gols_Mandante',
    'FTAG': 'Gols_Visitante',
    'FTR': 'Full Time Result (H=Home Win, D=Draw, A=Away Win)',
    'Res': 'Full Time Result (H=Home Win, D=Draw, A=Away Win)',
    'HTHG': 'Gols_1_Tempo_Mandante',
    'HTAG': 'Gols_1_Tempo_Visitante',
    'HTR': 'Half Time Result (H=Home Win, D=Draw, A=Away Win)',
    'Attendance': 'Publico_Presente',
    'Referee': 'Arbitro',
    'HS': 'Chutes_Mandante',
    'AS': 'Chutes_Visitante',
    'HST': 'Chute_a_Gol_Mandante',
    'AST': 'Chute_a_Gol_Visitante',
    'HHW': 'Bola_Trave_Mandante',
    'AHW': 'Bola_Trave_Visitante',
    'HC': 'Escanteio_Mandante',
    'AC': 'Escanteio_Visitante',
    'HF': 'Faltas_Mandante',
    'AF': 'Faltas_Visitante',
    'HFKC': 'Bolas_Paradas_Mandante',
    'AFKC': 'Bolas_Paradas_Visitante',
    'HO': 'Impedimentos_Mandante',
    'AO': 'Impedimentos_Visitante',
    'HY': 'Cartao_Amarelo_Mandante',
    'AY': 'Cartao_Amarelo_Visitante',
    'HR': 'Cartao_Vermelho_Mandante',
    'AR': 'Cartao_Vermelho_Visitante',
    'HBP': 'Amarelo+Vermelho_Mandante',  # 10 = amarelo, 25 = vermelho
    'ABP': 'Amarelo+Vermelho_Visitante'  # 10 = amarelo, 25 = vermelho
}

dados_consulta.rename(columns=column_mapping, inplace=True)

colunas_para_remover = [
    'B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'PSH', 'PSD', 'PSA', 'WHH', 'WHD', 'WHA',
    'MaxH', 'MaxD', 'MaxA', 'AvgH', 'AvgD', 'AvgA', 'B365>2.5', 'B365<2.5', 'P>2.5', 'P<2.5',
    'Max>2.5', 'Max<2.5', 'Avg>2.5', 'Avg<2.5', 'AHh', 'B365AHH', 'B365AHA', 'PAHH', 'PAHA', 'MaxAHH', 'MaxAHA',
    'AvgAHH', 'AvgAHA', 'B365CH', 'B365CD', 'B365CA', 'BWCH', 'BWD', 'BWCA', 'PSCH', 'PSCD',
    'PSCA', 'WHCH', 'WHCD', 'WHCA', 'MaxCH', 'MaxCD', 'MaxCA', 'AvgCH', 'AvgCD', 'AvgCA',
    'B365C>2.5', 'B365C<2.5', 'PC>2.5', 'PC<2.5', 'MaxC>2.5', 'MaxC<2.5', 'AvgC>2.5', 'AvgC<2.5', 'AHCh', 'B365CAHH',
    'B365CAHA', 'PCAHH', 'PCAHA', 'MaxCAHH', 'MaxCAHA', 'AvgCAHH', 'AvgCAHA', 'BFH', 'BFD', 'BFA', '1XBH', '1XBD',
    '1XBA', 'BFEH', 'BFED', 'BFEA', 'BFE>2.5', 'BFE<2.5', 'BFEAHH', 'BFEAHA', 'BFCH', 'BFCD', 'BFCA', '1XBCH',
    '1XBCD', '1XBCA', 'BFECH', 'BFECD', 'BFECA', 'BFEC>2.5', 'BFEC<2.5', 'BFECAHH', 'BFECAHA', 'BWCD', 'League Division', 'Horario'
]
dados_consulta = dados_consulta.drop(columns=colunas_para_remover)

dados_consulta['marcou_no_jogo_passado_mandante(0=Nao)'] = dados_consulta.groupby(['Time_Mandante', 'Temporada'])['Gols_Mandante'].shift(1).apply(lambda x: 0 if x == 0 else 1)
dados_consulta['marcou_no_jogo_passado_visitante(0=Nao)'] = dados_consulta.groupby(['Time_Visitante', 'Temporada'])['Gols_Visitante'].shift(1).apply(lambda x: 0 if x == 0 else 1)
dados_consulta['sofreu_no_jogo_passado_mandante(0=Nao)'] = dados_consulta.groupby(['Time_Mandante', 'Temporada'])['Gols_Visitante'].shift(1).apply(lambda x: 0 if x == 0 else 1)
dados_consulta['sofreu_no_jogo_passado_visitante(0=Nao)'] = dados_consulta.groupby(['Time_Visitante', 'Temporada'])['Gols_Mandante'].shift(1).apply(lambda x: 0 if x == 0 else 1)


with open('modelo_AlemaoVisitanteMarca.pkl', 'rb') as f:
       modelo = pickle.load(f)


# Adicionando textos ao layout do Streamlit
st.title('Previsão de Restultado de Futebol do Campeonato Alemão(Bundesliga)')

st.caption('''Este projeto utiliza a biblioteca XGBoost para prever se em uma partida de futebol da Bundesliga haverá mais de 2(dois) gols. O modelo
           criado foi treinado com dados da temporada 2018/19 até a temporada 2024/25 e possui uma acurácia de geral de 70% nos dados de teste.
           O usuário pode inserir os times mandante e visitante os quais deseja a previsão, e o modelo dirá o resultado previsto.''')

st.subheader('Insira os times que jogarão:')



Time_Mandante = st.selectbox(
    "Time Mandante:",
    (dados_consulta['Time_Mandante'].unique()), key="mandante")

Time_Visitante = st.selectbox(
    "Time Visitante:",
    (dados_consulta['Time_Visitante'].unique()), key="visitante")

Temporada = int(2025)

if Time_Mandante and Time_Visitante and Temporada:
    nova_previsao = {'Gols_Mandante_Media': [dados[(dados['Time_Mandante'] == Time_Mandante) & (dados['Temporada'] == Temporada)]['Gols_Mandante'].mean()],
                  'Gols_Visitante_Media': [dados[(dados['Time_Visitante'] == Time_Visitante) & (dados['Temporada'] == Temporada)]['Gols_Visitante'].mean()],
           'Gols_Sofridos_Mandante_Media': [dados[(dados['Time_Mandante'] == Time_Mandante) & (dados['Temporada'] == Temporada)]['Gols_Visitante'].mean()],
           'Gols_Sofridos_Visitante_Media': [dados[(dados['Time_Visitante'] == Time_Visitante) & (dados['Temporada'] == Temporada)]['Gols_Mandante'].mean()],
           'media_partidas_marcando_mandante': [dados[(dados['Time_Mandante'] == Time_Mandante) & (dados['Temporada'] == Temporada)]['Mandante_Marcou'].mean()],
           'media_partidas_marcando_visitante': [dados[(dados['Time_Visitante'] == Time_Visitante) & (dados['Temporada'] == Temporada)]['Visitante_Marcou'].mean()],
           'media_partidas_sofrendo_mandante': [dados[(dados['Time_Mandante'] == Time_Mandante) & (dados['Temporada'] == Temporada)]['Visitante_Marcou'].mean()],
                  'media_partidas_sofrendo_visitante': [dados[(dados['Time_Visitante'] == Time_Visitante) & (dados['Temporada'] == Temporada)]['Mandante_Marcou'].mean()],
           'marcou_no_jogo_passado_mandante(0=Nao)': [dados.loc[dados[dados['Time_Mandante'] == Time_Mandante].index[-1], 'Mandante_Marcou']],
                  'sofreu_no_jogo_passado_mandante(0=Nao)': [dados.loc[dados[dados['Time_Mandante'] == Time_Mandante].index[-1], 'Visitante_Marcou']],
                  'sofreu_no_jogo_passado_visitante(0=Nao)': [dados.loc[dados[dados['Time_Visitante'] == Time_Visitante].index[-1], 'Mandante_Marcou']],
                 'Escanteio_Mandante_Media': [dados[(dados['Time_Mandante'] == Time_Mandante) & (dados['Temporada'] == Temporada)]['Escanteio_Mandante'].mean()],
                  'Escanteio_Visitante_Media': [dados[(dados['Time_Visitante'] == Time_Visitante) & (dados['Temporada'] == Temporada)]['Escanteio_Visitante'].mean()],
           'Escanteio_Sofridos_Mandante_Media': [dados[(dados['Time_Mandante'] == Time_Mandante) & (dados['Temporada'] == Temporada)]['Escanteio_Visitante'].mean()],
                  'Escanteio_Sofridos_Visitante_Media': [dados[(dados['Time_Visitante'] == Time_Visitante) & (dados['Temporada'] == Temporada)]['Escanteio_Mandante'].mean()],
                 'Partidas_mais_de_2gols_mandante': [dados[(dados['Time_Mandante'] == Time_Mandante) & (dados['Temporada'] == Temporada)]['Partidas_mais_de_2gols'].mean()],
                 'Partidas_mais_de_2gols_visitante': [dados[(dados['Time_Visitante'] == Time_Visitante) & (dados['Temporada'] == Temporada)]['Partidas_mais_de_2gols'].mean()],
                 'Faltas_Mandante_Media': [dados[(dados['Time_Mandante'] == Time_Mandante) & (dados['Temporada'] == Temporada)]['Faltas_Mandante'].mean()],
                 'Faltas_Visitante_Media': [dados[(dados['Time_Visitante'] == Time_Visitante) & (dados['Temporada'] == Temporada)]['Faltas_Visitante'].mean()],
                 'Chute_a_Gol_Mandante_Media': [dados[(dados['Time_Mandante'] == Time_Mandante) & (dados['Temporada'] == Temporada)]['Chute_a_Gol_Mandante'].mean()],
                 'Chute_a_Gol_Visitante_Media': [dados[(dados['Time_Visitante'] == Time_Visitante) & (dados['Temporada'] == Temporada)]['Chute_a_Gol_Visitante'].mean()]
                   }

nova_previsao_df = pd.DataFrame(nova_previsao)

if 'previsao_feita' not in st.session_state:
    st.session_state['previsao_feita'] = False
    st.session_state['dados_previsao'] = None

if st.button('Prever'):
    st.session_state.previsao_feita = True
    previsao = modelo.predict(nova_previsao_df)
    st.session_state['dados_previsao'] = previsao
    st.write(f"Previsão: {'Sim' if previsao[0] == 1 else 'Não'}")

    if st.button("Resetar"):
        st.session_state.previsao_feita = None  # Reseta a previsão na session_state
        st.session_state['dados_previsao'] = None  # Reseta os dados da previsão na session_state
        st.experimental_rerun()  # Recarrega a página

