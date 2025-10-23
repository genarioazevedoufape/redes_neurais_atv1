## `data/nba_data_loader.py`
"""Baixa dados de estatísticas por jogador/time usando nba_api.
Funções retornam DataFrames prontos para análise.
"""
from nba_api.stats.endpoints import leaguedashplayerstats, leaguedashteamstats
import pandas as pd

def get_player_stats_season(season: str = '2024-25', measure_type: str = 'Base') -> pd.DataFrame:
    """Retorna estatísticas por jogador para a temporada indicada.
    season: formato '2024-25'
    measure_type: 'Base' ou 'Advanced'
    """
    # a nba_api pode demorar; considere aplicar cache no Streamlit
    df = leaguedashplayerstats.LeagueDashPlayerStats(season=season, measure_type_simple=measure_type).get_data_frames()[0]
    # rename columns mais amigáveis (opcional)
    return df


def get_team_stats_season(season: str = '2024-25', measure_type: str = 'Base') -> pd.DataFrame:
    df = leaguedashteamstats.LeagueDashTeamStats(season=season, measure_type_simple=measure_type).get_data_frames()[0]
    return df


def sample_player_game_logs(player_id: int, season: str = '2024-25') -> pd.DataFrame:
    """Placeholder: Dependendo do endpoint que você preferir, implemente a extração de game logs por jogador.
    Por simplicidade aqui focamos em season-aggregate stats (leaguedashplayerstats).
    """
    # Para logs por jogo existe `playergamelog.PlayerGameLog` — pode ser adicionado conforme necessidade
    raise NotImplementedError("Implemente game logs se desejar previsões por partida (PlayerGameLog)")
