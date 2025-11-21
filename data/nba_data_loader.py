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

@st.cache_data
def load_player_game_log(team_id, season=SEASON):
    """
    Carrega o player game log para TODOS os jogadores que atuaram pelo time na temporada.
    Versão robusta que lida com diferentes estruturas de dados da NBA API.
    """
    try:
        st.info("🔄 Conectando à NBA API para buscar dados dos jogadores...")
        
        # Primeiro, vamos obter a lista de jogadores do time
        from nba_api.stats.endpoints import commonteamroster
        
        # Obter roster do time com tratamento de erro
        try:
            roster = commonteamroster.CommonTeamRoster(team_id=team_id, season=season)
            roster_df = roster.get_data_frames()[0]
            
            if roster_df.empty:
                st.warning(f"Nenhum jogador encontrado no roster do time {team_id} para a temporada {season}.")
                return create_sample_player_data(), create_sample_player_agg_data()
                
            # Mapear nomes de colunas possíveis
            player_id_col = None
            player_name_col = None
            
            for col in roster_df.columns:
                if 'PLAYER' in col and 'ID' in col:
                    player_id_col = col
                elif 'PLAYER' in col and 'NAME' in col:
                    player_name_col = col
            
            # Fallback para nomes padrão
            if not player_id_col and 'PLAYER_ID' in roster_df.columns:
                player_id_col = 'PLAYER_ID'
            if not player_name_col and 'PLAYER_NAME' in roster_df.columns:
                player_name_col = 'PLAYER_NAME'
            
            if not player_id_col or not player_name_col:
                st.warning("Não foi possível identificar as colunas de jogador no roster.")
                return create_sample_player_data(), create_sample_player_agg_data()
                
            player_ids = roster_df[player_id_col].tolist()
            player_names = roster_df[player_name_col].tolist()
            
            st.info(f"📊 Encontrados {len(player_ids)} jogadores no roster. Buscando estatísticas...")
            
        except Exception as roster_error:
            st.warning(f"Erro ao buscar roster do time: {roster_error}")
            return create_sample_player_data(), create_sample_player_agg_data()
        
        all_player_games = []
        successful_players = 0
        
        # Buscar dados para cada jogador
        for i, (player_id, player_name) in enumerate(zip(player_ids, player_names)):
            try:
                if i > 0:
                    import time
                    time.sleep(0.3)  # Pequena pausa para não sobrecarregar a API
                
                # PlayerGameLog para o jogador específico
                pl = playergamelog.PlayerGameLog(
                    player_id=player_id, 
                    season=season,
                    timeout=45
                )
                player_df = pl.get_data_frames()[0]
                
                if not player_df.empty:
                    # Adicionar informações do jogador manualmente
                    player_df['PLAYER_ID'] = player_id
                    player_df['PLAYER_NAME'] = player_name
                    all_player_games.append(player_df)
                    successful_players += 1
                    
            except Exception as e:
                st.warning(f"⚠️ Erro ao buscar dados do jogador {player_name} (ID: {player_id}): {str(e)[:100]}...")
                continue
        
        st.info(f"✅ Dados obtidos para {successful_players} de {len(player_ids)} jogadores")
        
        if not all_player_games:
            st.warning("Nenhum dado de jogador foi retornado pela NBA API. Usando dados de exemplo.")
            return create_sample_player_data(), create_sample_player_agg_data()
        
        # Combinar todos os dados
        df_players_games = pd.concat(all_player_games, ignore_index=True)
        
        # Verificar se temos as colunas necessárias
        required_cols = ['PLAYER_ID', 'PLAYER_NAME']
        missing_cols = [col for col in required_cols if col not in df_players_games.columns]
        
        if missing_cols:
            st.warning(f"Colunas faltantes nos dados: {missing_cols}. Corrigindo...")
            # Se PLAYER_NAME está faltando, criar a partir do roster
            if 'PLAYER_NAME' not in df_players_games.columns and 'PLAYER_ID' in df_players_games.columns:
                player_name_map = dict(zip(player_ids, player_names))
                df_players_games['PLAYER_NAME'] = df_players_games['PLAYER_ID'].map(player_name_map)
        
        # Processar e limpar dados
        date_col = None
        for col in df_players_games.columns:
            if 'DATE' in col:
                date_col = col
                break
        
        if date_col:
            df_players_games['GAME_DATE'] = pd.to_datetime(df_players_games[date_col])
        else:
            df_players_games['GAME_DATE'] = pd.to_datetime('2024-10-01')  # Data padrão
        
        # Identificar colunas numéricas disponíveis
        possible_numeric_cols = [
            'PTS', 'REB', 'AST', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 
            'OREB', 'DREB', 'STL', 'BLK', 'TOV', 'PF', 'PLUS_MINUS',
            'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'MIN'
        ]
        
        available_numeric_cols = []
        for col in possible_numeric_cols:
            if col in df_players_games.columns:
                available_numeric_cols.append(col)
                # Converter para numérico
                df_players_games[col] = pd.to_numeric(df_players_games[col], errors='coerce')
        
        st.info(f"📈 Colunas numéricas encontradas: {len(available_numeric_cols)}")
        
        # Agregação por jogador
        if available_numeric_cols:
            agg_funcs = {}
            for col in available_numeric_cols:
                agg_funcs[col] = ['mean', 'sum', 'count']
            
            # Garantir que temos as colunas de agrupamento
            groupby_cols = []
            if 'PLAYER_ID' in df_players_games.columns:
                groupby_cols.append('PLAYER_ID')
            if 'PLAYER_NAME' in df_players_games.columns:
                groupby_cols.append('PLAYER_NAME')
            
            if groupby_cols:
                df_agg = df_players_games.groupby(groupby_cols).agg(agg_funcs)
                
                # Flatten MultiIndex columns
                df_agg.columns = ['_'.join(col).strip() for col in df_agg.columns.values]
                df_agg = df_agg.reset_index()
                
                # Calcular jogos jogados
                game_count = df_players_games.groupby(groupby_cols).size()
                if len(groupby_cols) == 2:
                    df_agg['GAMES_PLAYED'] = df_agg.apply(
                        lambda row: game_count.get((row['PLAYER_ID'], row['PLAYER_NAME']), 0), 
                        axis=1
                    )
                else:
                    df_agg['GAMES_PLAYED'] = df_agg[groupby_cols[0]].map(game_count)
                
                # Ordenar por jogos
                df_agg = df_agg.sort_values(by='GAMES_PLAYED', ascending=False)
                
                st.success(f"✅ Dados de {len(df_agg)} jogadores processados com sucesso!")
                return df_players_games.sort_values('GAME_DATE'), df_agg
        
        # Fallback se a agregação falhar
        st.warning("Agregação falhou, retornando dados básicos...")
        return df_players_games, create_sample_player_agg_data()
            
    except Exception as e:
        st.error(f"❌ Erro crítico ao carregar player game log: {e}")
        st.info("Usando dados de exemplo para demonstração...")
        return create_sample_player_data(), create_sample_player_agg_data()

def create_sample_player_data():
    """Cria dados de exemplo para desenvolvimento quando a API falha"""
    import numpy as np
    sample_data = {
        'PLAYER_ID': [1, 1, 2, 2, 3, 3],
        'PLAYER_NAME': ['LeBron James', 'LeBron James', 'Stephen Curry', 'Stephen Curry', 'Giannis Antetokounmpo', 'Giannis Antetokounmpo'],
        'GAME_DATE': pd.to_datetime(['2024-10-25', '2024-10-27', '2024-10-25', '2024-10-27', '2024-10-26', '2024-10-28']),
        'PTS': [25, 30, 32, 28, 35, 29],
        'REB': [8, 10, 5, 6, 12, 11],
        'AST': [6, 8, 7, 9, 8, 6],
        'FG_PCT': [0.48, 0.52, 0.51, 0.47, 0.55, 0.53],
        'STL': [2, 1, 3, 2, 1, 2],
        'BLK': [1, 2, 0, 1, 3, 2],
        'FG3_PCT': [0.35, 0.38, 0.45, 0.42, 0.28, 0.30]
    }
    return pd.DataFrame(sample_data)

def create_sample_player_agg_data():
    """Cria dados agregados de exemplo para desenvolvimento"""
    sample_agg = {
        'PLAYER_ID': [1, 2, 3],
        'PLAYER_NAME': ['LeBron James', 'Stephen Curry', 'Giannis Antetokounmpo'],
        'PTS_mean': [27.5, 30.0, 32.0],
        'PTS_sum': [55, 60, 64],
        'REB_mean': [9.0, 5.5, 11.5],
        'REB_sum': [18, 11, 23],
        'AST_mean': [7.0, 8.0, 7.0],
        'AST_sum': [14, 16, 14],
        'FG_PCT_mean': [0.50, 0.49, 0.54],
        'FG3_PCT_mean': [0.365, 0.435, 0.29],
        'STL_mean': [1.5, 2.5, 1.5],
        'BLK_mean': [1.5, 0.5, 2.5],
        'GAMES_PLAYED': [2, 2, 2]
    }
    return pd.DataFrame(sample_agg)

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

