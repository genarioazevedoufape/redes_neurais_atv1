import pandas as pd
import streamlit as st
from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguegamefinder, playergamelog

SEASON = '2024-25'

@st.cache_data
def get_nba_teams():
    """Retorna uma lista de todos os times da NBA."""
    return teams.get_teams()

def get_team_id(team_name):
    """Retorna o ID de um time pelo nome."""
    nba_teams = get_nba_teams()
    team = next((t for t in nba_teams if t['full_name'] == team_name), None)
    return team['id'] if team else None

def get_available_teams():
    """Retorna uma lista de nomes de times disponíveis."""
    teams_list = get_nba_teams()
    return sorted([team['full_name'] for team in teams_list])

@st.cache_data
def load_team_game_log(team_id):
    """
    Carrega o log de jogos de um time para a temporada 2024-2025.
    Retorna um DataFrame com as estatísticas do jogo.
    """
    try:
        game_finder = leaguegamefinder.LeagueGameFinder(
            team_id_nullable=team_id,
            season_nullable=SEASON
        )
        games = game_finder.get_data_frames()[0]
        
        # Filtrar apenas os jogos da temporada 2024-2025 (embora já tenhamos passado a season_nullable)
        games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
        
        # Selecionar e renomear colunas relevantes para a análise
        data = games[[
            'GAME_DATE', 'MATCHUP', 'WL', 'PTS', 'REB', 'AST', 'FG_PCT', 'FG3_PCT', 'FT_PCT',
            'OREB', 'DREB', 'STL', 'BLK', 'TOV', 'PF', 'PLUS_MINUS'
        ]].copy()
        
        # Converter 'WL' (Win/Loss) para variável binária (1 para Vitória, 0 para Derrota)
        data['WIN'] = data['WL'].apply(lambda x: 1 if x == 'W' else 0)
        
        return data.sort_values(by='GAME_DATE', ascending=True)
        
    except Exception as e:
        print(f"Erro ao carregar dados do time (ID: {team_id}) para a temporada {SEASON}: {e}")
        return pd.DataFrame()

@st.cache_data
def load_player_game_log(player_id):
    """
    Carrega o log de jogos de um jogador para a temporada 2024-2025.
    Retorna um DataFrame com as estatísticas do jogo.
    """
    try:
        log = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=SEASON
        )
        df = log.get_data_frames()[0]
        
        # Selecionar e renomear colunas relevantes
        data = df[[
            'GAME_DATE', 'MATCHUP', 'WL', 'PTS', 'REB', 'AST', 'FG_PCT', 'FG3_PCT', 'FT_PCT',
            'OREB', 'DREB', 'STL', 'BLK', 'TOV', 'PF', 'PLUS_MINUS'
        ]].copy()
        
        data['GAME_DATE'] = pd.to_datetime(data['GAME_DATE'])
        data['WIN'] = data['WL'].apply(lambda x: 1 if x == 'W' else 0)
        
        return data.sort_values(by='GAME_DATE', ascending=True)
        
    except Exception as e:
        print(f"Erro ao carregar dados do jogador (ID: {player_id}) para a temporada {SEASON}: {e}")
        return pd.DataFrame()

# Funções de suporte para o Streamlit
def get_available_stats_columns(df):
    """Retorna as colunas de estatísticas disponíveis para seleção de X e Y."""
    if df.empty:
        return []
    
    # Colunas que representam estatísticas numéricas
    stats_cols = [
        'PTS', 'REB', 'AST', 'FG_PCT', 'FG3_PCT', 'FT_PCT',
        'OREB', 'DREB', 'STL', 'BLK', 'TOV', 'PF', 'PLUS_MINUS', 'WIN'
    ]
    
    # Filtra as colunas que realmente existem no DataFrame
    return [col for col in stats_cols if col in df.columns]

if __name__ == '__main__':
    team_name = "Boston Celtics"
    team_id = get_team_id(team_name)
    
    if team_id:
        print(f"ID do {team_name}: {team_id}")
        df = load_team_game_log(team_id)
        if not df.empty:
            print(f"Dados carregados para {team_name} ({len(df)} jogos):")
            print(df.head())
            print("\nColunas de estatísticas disponíveis:")
            print(get_available_stats_columns(df))
        else:
            print(f"Nenhum dado encontrado para {team_name} na temporada {SEASON}.")
    else:
        print(f"Time '{team_name}' não encontrado.")

