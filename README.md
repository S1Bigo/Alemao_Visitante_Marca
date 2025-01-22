Previsão de Gols no Futebol com Regressão Logística e XGBoost
Este projeto utiliza técnicas de Machine Learning, como Regressão Logística e XGBoost, para prever se o time visitante marcará um gol durante uma partida de futebol. O modelo foi treinado com dados históricos de partidas e utiliza variáveis relevantes (como estatísticas de jogo, desempenho anterior e outros fatores) para fazer previsões sobre o resultado de futuras partidas.

Tecnologias Utilizadas
Python: Linguagem principal para desenvolvimento.
XGBoost: Biblioteca para otimização e treinamento de modelos de aprendizado de máquina baseados em gradient boosting.
Optuna: Biblioteca para otimização de hiperparâmetros do modelo.
Streamlit: Framework utilizado para criar uma interface interativa que permite aos usuários inserir dados e obter previsões em tempo real.
Pandas e Numpy: Bibliotecas para manipulação e análise de dados.
Scikit-learn: Biblioteca para pré-processamento de dados e métricas de avaliação de modelos.
Objetivo
O objetivo principal deste projeto é criar um modelo preditivo que informe a probabilidade de um time visitante marcar um gol durante uma partida de futebol. Isso pode ser útil para análise de desempenho de times e para insights em apostas esportivas, entre outras aplicações.

Descrição do Modelo
O modelo utiliza Regressão Logística como base, para modelar a probabilidade de um gol ser marcado. XGBoost foi implementado para melhorar a precisão do modelo, aplicando uma técnica de gradient boosting que geralmente fornece resultados mais robustos em tarefas de classificação. Para otimizar os parâmetros do modelo, foi usada a biblioteca Optuna, que realiza uma busca eficiente pelos melhores hiperparâmetros.

Etapas do Projeto
Coleta de Dados: Dados históricos das partidas de futebol, incluindo informações como número de gols, desempenho dos times e outras variáveis relevantes.
Pré-processamento de Dados: Limpeza e transformação dos dados, incluindo normalização e codificação de variáveis categóricas.
Treinamento do Modelo: Construção do modelo de Regressão Logística e otimização utilizando XGBoost e Optuna.
Avaliação do Modelo: Testes do modelo utilizando métricas como Acurácia, Precisão e Recall.
Desenvolvimento da Interface no Streamlit: Criação de uma aplicação interativa onde o usuário pode inserir dados e obter previsões.
