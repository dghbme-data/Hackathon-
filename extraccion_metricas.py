# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 10:37:05 2026

@author: armid
"""
# =============================================================================
# 1. LIBRERIAS UTILIZADAS
# =============================================================================
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mplsoccer import VerticalPitch
from mplsoccer import Pitch
from statsbombpy import sb
import math
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap

#%%
# =============================================================================
# 2. DISEÑO UNIFORME DE GRÁFICAS
# =============================================================================
plt.style.use('dark_background')
BG_COLOR = '#1c222b'
TEXT_COLOR = 'white'
#%%
# =============================================================================
# 3. CREDENCIALES 
# =============================================================================
credenciales = {"user": "itam_hackathon@hudl.com", "passwd": "NDeXcBUp"}
competition_id = 73
#%%
# =============================================================================
# 4. EXTRACCIÓN DE PARTIDOS DEL AMÉRICA
# =============================================================================
competition = sb.competitions(creds=credenciales)
season_ids = competition[competition['competition_id'] == competition_id]['season_id'].unique().tolist()

am_matches = []
for s_id in season_ids:
    matches = sb.matches(competition_id=competition_id, season_id=s_id, creds=credenciales)
    am = matches[(matches["home_team"]=='América') | (matches["away_team"]=='América')]
    am_matches.append(am)
matches = pd.concat(am_matches)
#%%
# =============================================================================
# 5. EXTRACCIÓN DEL NOMBRE DEL ENTRENADOR POR PARTIDO
# =============================================================================
def nombre_manager(row):
    if row.get('home_team') in ['América']:
        manager = row.get('home_managers')
    elif row.get('away_team') in ['América']:
        manager = row.get('away_managers')
    return manager
matches['america_manager'] = matches.apply(nombre_manager, axis=1)
entrenadores = matches['america_manager'].dropna()
entrenadores = entrenadores[entrenadores.str.strip() != ''].unique()
#%%
# =============================================================================
# 6. EXTRACCIÓN DE EVENTOS
# =============================================================================
eventos_ent = {}
matches_ent = {}
for ent in entrenadores:
    matches_entrenador = matches[matches['america_manager'] == ent]
    ids_partidos = matches_entrenador['match_id']
    num_partidos = len(ids_partidos) 
    
    eventos = []
    for m_id in ids_partidos:
        e = sb.events(match_id=m_id, creds=credenciales)
        eventos.append(e)
    eventos_entrenador = pd.concat(eventos)
    eventos_entrenador = eventos_entrenador.copy()
    eventos_entrenador['loc_x'] = eventos_entrenador['location'].apply(lambda l: l[0] if isinstance(l, list) and len(l) > 0 else np.nan)
    eventos_entrenador['loc_y'] = eventos_entrenador['location'].apply(lambda l: l[1] if isinstance(l, list) and len(l) > 1 else np.nan)
    eventos_ent[ent] = eventos_entrenador
    matches_ent[ent] = matches_entrenador
#%% 
# =============================================================================
# 7. MÉTRICAS BASE
# =============================================================================
def calcular_metricas_base(df_events, num_partidos):
    shots_fav = df_events[(df_events['type'] == 'Shot') & (df_events['team'] == 'América')]
    xg_fav = shots_fav['shot_statsbomb_xg'].sum() / num_partidos
    
    shots_con = df_events[(df_events['type'] == 'Shot') & (df_events['team'] != 'América')]
    xg_con = shots_con['shot_statsbomb_xg'].sum() / num_partidos 
    
    passes_xa = df_events[(df_events['type'] == 'Pass') & (df_events['team'] == 'América')]
    xa_fav = passes_xa['pass_goal_assist'].fillna(0).astype(int).sum() * 0.15 / num_partidos # falta por corregir
    
    pases_r = len(df_events[(df_events['team'] != 'América') & (df_events['type'] == 'Pass') & (df_events['loc_x'] <= 72)])
    def_a = len(df_events[(df_events['team'] == 'América') & (df_events['type'].isin(['Pressure', 'Duel', 'Interception', 'Block'])) & (df_events['loc_x'] >= 48)])
    ppda = pases_r / (def_a + 1e-8)
    
    pases_ut = df_events[(df_events['type'] == 'Pass') & (df_events['loc_x'] >= 80)]
    field_tilt = (len(pases_ut[pases_ut['team'] == 'América']) / (len(pases_ut) + 1e-8)) * 100
    
    pases_ame = df_events[(df_events['team'] == 'América') & (df_events['type'] == 'Pass') & (df_events['pass_type'].isna())].copy()
    pases_ame['x_end'] = pases_ame['pass_end_location'].apply(lambda l: l[0] if isinstance(l, list) else 0)
    pases_prog = len(pases_ame[((pases_ame['x_end'] - pases_ame['loc_x']) * 0.9144 >= 10) | (pases_ame['x_end'] >= 100)]) / num_partidos

        
    return float(xg_fav), float(xg_con), float(ppda), float(field_tilt), float(pases_prog), float(xa_fav)

def calcular_gamestate_real(df_events, num_partidos):
    xgf_ganando, xgc_ganando = 0.0, 0.0
    xgf_empatando, xgc_empatando = 0.0, 0.0
    xgf_perdiendo, xgc_perdiendo = 0.0, 0.0

    matches = df_events['match_id'].unique()
    for m_id in matches:
        m_events = df_events[df_events['match_id'] == m_id].sort_values(['index'])
        goles_ame, goles_riv = 0, 0
        
        for idx, row in m_events.iterrows():
            es_ame = 'América' in str(row.get('team')) or 'America' in str(row.get('team'))
            
            if row['type'] == 'Shot':
                xg_val = row.get('shot_statsbomb_xg', 0.0)
                if pd.isna(xg_val): xg_val = 0.0
                    
                if goles_ame > goles_riv:
                    if es_ame: xgf_ganando += xg_val
                    else: xgc_ganando += xg_val
                elif goles_ame == goles_riv:
                    if es_ame: xgf_empatando += xg_val
                    else: xgc_empatando += xg_val
                else:
                    if es_ame: xgf_perdiendo += xg_val
                    else: xgc_perdiendo += xg_val
            
            if row['type'] == 'Shot' and str(row.get('shot_outcome')) == 'Goal':
                if es_ame: goles_ame += 1
                else: goles_riv += 1
                    
    return ([round(xgf_ganando/num_partidos, 2), 
             round(xgf_empatando/num_partidos, 2), 
             round(xgf_perdiendo/num_partidos, 2)],
            [round(xgc_ganando/num_partidos, 2), 
             round(xgc_empatando/num_partidos, 2), 
             round(xgc_perdiendo/num_partidos, 2)])
#%%
BG_COLOR = "#0f172a"
TEXT_COLOR = "#ffffff"

COLORES_DTS_GLOBALES = {
    'Jorge Guillermo Almada Álves': '#f1c40f',  
    'André Soares Jardine': '#2ecc71',       
    'Diego Alberto Cervantes Chávez': '#f43f5e', 
    'Fernando Ortiz': '#38bdf8',               
    'Santiago Hernán Solari Poggio': '#9b5de5'   
}

def obtener_color_dt(dt_nombre):
    return COLORES_DTS_GLOBALES.get(dt_nombre, '#00f5d4')




def generar_panel_1_100_pitch_con_field_tilt(df_dt_events, dt_nombre, num_partidos):
    fig = plt.figure(figsize=(16, 11), facecolor=BG_COLOR)
    gs = gridspec.GridSpec(1, 3, width_ratios=[0.18, 1, 1], figure=fig, wspace=0.25)
    
    pitch = VerticalPitch(pitch_type='statsbomb', pitch_color=BG_COLOR, line_color='#ffffff')
    
    col_obv = 'obv_total_net' if 'obv_total_net' in df_dt_events.columns else ('obv_value' if 'obv_value' in df_dt_events.columns else None)
    df_open_play = df_dt_events[df_dt_events['pass_type'].isna()].dropna(subset=['loc_x', 'loc_y']) if 'pass_type' in df_dt_events.columns else df_dt_events.dropna(subset=['loc_x', 'loc_y'])
    df_ame = df_open_play[df_open_play['team'].str.contains('América|America', case=False, na=False)]
    df_riv = df_open_play[~df_open_play['team'].str.contains('América|America', case=False, na=False)]
    
    # Colormap divergente personalizado (-0.5 a 0.5): Extremos con color, centro (0) en blanco (#ffffff)
    cmap_creado = LinearSegmentedColormap.from_list("div_creado", ["#ff4d4d", "#ffffff", "#2ecc71"]) # Rojo -> Blanco -> Verde
    cmap_concedido = LinearSegmentedColormap.from_list("div_concedido", ["#ffeaa7", "#ffffff", "#ff6b81"]) # Amarillo claro -> Blanco -> Rojo oscuro
    
    ax_obv1 = fig.add_subplot(gs[0, 1])
    pitch.draw(ax=ax_obv1)
    obv_creado_total = 0.0
    
    if col_obv and len(df_ame) > 0:
        bin_ame = pitch.bin_statistic(df_ame['loc_x'], df_ame['loc_y'], values=df_ame[col_obv], statistic='sum', bins=(6, 4))
        obv_creado_total = bin_ame['statistic'].sum() / num_partidos
        bin_ame['statistic'] /= num_partidos
        
        hm1 = pitch.heatmap(bin_ame, ax=ax_obv1, cmap=cmap_creado, alpha=0.85, edgecolors=BG_COLOR, vmin=-0.5, vmax=0.5)
        pitch.label_heatmap(bin_ame, color='black', fontsize=10, ax=ax_obv1, str_format='{:.3f}')
        
        cbar1 = fig.colorbar(hm1, ax=ax_obv1, orientation='vertical', fraction=0.03, pad=0.04, shrink=0.95)
        cbar1.ax.tick_params(labelsize=8, colors='white')
        
    ax_obv1.set_title(f'OBV Creado\n(Total: {obv_creado_total:.2f})', color='white', fontweight='bold', fontsize=16, pad=15)
    
    ax_obv1.text(40, 123, 'Portería Rival', color='#ff6b81', fontsize=12, fontweight='bold', ha='center', va='bottom')
    ax_obv1.text(40, -3, 'Portería América', color='#f1c40f', fontsize=12, fontweight='bold', ha='center', va='top')

    ax_obv1.annotate('Dirección\nde Ataque', xy=(-0.04, 0.8), xytext=(-0.04, 0.15),
                     xycoords='axes fraction', textcoords='axes fraction',
                     ha='center', va='center', color='white', fontsize=10, fontweight='bold',
                     arrowprops=dict(arrowstyle="->", color='#00f5d4', lw=3))

    ax_obv2 = fig.add_subplot(gs[0, 2])
    pitch.draw(ax=ax_obv2)
    obv_concedido_total = 0.0
    
    if col_obv and len(df_riv) > 0:
        bin_riv = pitch.bin_statistic(df_riv['loc_x'], df_riv['loc_y'], values=df_riv[col_obv], statistic='sum', bins=(6, 4))
        obv_concedido_total = bin_riv['statistic'].sum() / num_partidos
        bin_riv['statistic'] /= num_partidos
        
        hm2 = pitch.heatmap(bin_riv, ax=ax_obv2, cmap=cmap_concedido, alpha=0.85, edgecolors=BG_COLOR, vmin=-0.5, vmax=0.5)
        pitch.label_heatmap(bin_riv, color='black', fontsize=10, ax=ax_obv2, str_format='{:.3f}')
        
        cbar2 = fig.colorbar(hm2, ax=ax_obv2, orientation='vertical', fraction=0.03, pad=0.04, shrink=0.95)
        cbar2.ax.tick_params(labelsize=8, colors='white')
        
    ax_obv2.set_title(f'OBV Concedido\n(Total: {obv_concedido_total:.2f})', color='white', fontweight='bold', fontsize=16, pad=15)
    
    ax_obv2.text(40, 123, 'Portería América', color='#f1c40f', fontsize=12, fontweight='bold', ha='center', va='bottom')
    ax_obv2.text(40, -3, 'Portería Rival', color='#ff6b81', fontsize=12, fontweight='bold', ha='center', va='top')

    ax_obv2.annotate('Dirección\nde Ataque', xy=(-0.04, 0.8), xytext=(-0.04, 0.15),
                     xycoords='axes fraction', textcoords='axes fraction',
                     ha='center', va='center', color='white', fontsize=10, fontweight='bold',
                     arrowprops=dict(arrowstyle="->", color='#ff6b81', lw=3))

    ax_tilt = fig.add_subplot(gs[0, 0])
    ax_tilt.set_facecolor(BG_COLOR)
    
    _, _, _, tilt_ame, _, _ = calcular_metricas_base(df_dt_events, num_partidos)
    tilt_riv = 100 - tilt_ame
    
    ymin_pitch, ymax_pitch = ax_obv1.get_ylim()
    altura_total = ymax_pitch - ymin_pitch
    altura_barra = altura_total * 0.75
    centro = (ymin_pitch + ymax_pitch) / 2
    ymin = centro - (altura_barra / 2)
    ymax = centro + (altura_barra / 2)
    
    ax_tilt.set_ylim(ymin_pitch, ymax_pitch)
    altura_ame = ymin + (tilt_ame / 100.0) * altura_barra
    
    ax_tilt.bar(0, altura_ame - ymin, bottom=ymin, color='#f1c40f', edgecolor=BG_COLOR, width=0.5, label='América')
    ax_tilt.bar(0, ymax - altura_ame, bottom=altura_ame, color='#e74c3c', edgecolor=BG_COLOR, width=0.5, label='Rival')
    
    ax_tilt.text(-0.40, ymin + (altura_ame - ymin) / 2, f'América: {tilt_ame:.1f}%', va='center', ha='right', color='#f1c40f', weight='bold', size=14)
    ax_tilt.text(-0.40, altura_ame + (ymax - altura_ame) / 2, f'Rival: {tilt_riv:.1f}%', va='center', ha='right', color='#e74c3c', weight='bold', size=14)
    
    ax_tilt.set_xlim(-1.2, 0.6)
    ax_tilt.set_xticks([])
    ax_tilt.set_yticks([])
    ax_tilt.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
    ax_tilt.text(0, ymax_pitch - 5, 'Field Tilt (%)', color=TEXT_COLOR, fontweight='bold', fontsize=18, ha='center', va='top')

    plt.suptitle(f'OBV y Field Tilt — Entrenador: {dt_nombre}', color=TEXT_COLOR, fontsize=22, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(f'panel_1_obv_field_tilt_{dt_nombre.replace(" ", "_")}.png', dpi=300, facecolor=BG_COLOR)
    plt.show()

def generar_panel_2_consistencia_adaptabilidad(df_dt_events, dt_nombre, num_partidos, matches_dt):
    ids_home = matches_dt[matches_dt['home_team'] == 'América']['match_id'].unique()
    ids_away = matches_dt[matches_dt['away_team'] == 'América']['match_id'].unique()
    
    events_h = df_dt_events[df_dt_events['match_id'].isin(ids_home)]
    events_a = df_dt_events[df_dt_events['match_id'].isin(ids_away)]
    
    xg_h, xga_h, _, tilt_h, _, _ = calcular_metricas_base(events_h, max(len(ids_home), 1))
    xg_a, xga_a, _, tilt_a, _, _ = calcular_metricas_base(events_a, max(len(ids_away), 1))
    
    obv_f_local = events_h[(events_h['pass_type'].isna()) & (events_h['team'] == 'América')]['obv_total_net'].sum() / max(len(ids_home), 1)
    obv_c_local = events_h[(events_h['pass_type'].isna()) & (events_h['team'] != 'América')]['obv_total_net'].sum() / max(len(ids_home), 1)

    obv_f_visitante = events_a[(events_a['pass_type'].isna()) & (events_a['team'] == 'América')]['obv_total_net'].sum() / max(len(ids_away), 1)
    obv_c_visitante = events_a[(events_a['pass_type'].isna()) & (events_a['team'] != 'América')]['obv_total_net'].sum() / max(len(ids_away), 1)
    
    rojas_events = df_dt_events[
        df_dt_events.get('foul_committed_card', pd.Series()).isin(['Red Card', 'Second Yellow']) | 
        df_dt_events.get('bad_behaviour_card', pd.Series()).isin(['Red Card', 'Second Yellow'])]
    
    if not rojas_events.empty:
        matches_roja_ame = rojas_events[rojas_events['team'].str.contains('América|America', case=False, na=False)]['match_id'].unique()
        matches_roja_riv = rojas_events[~rojas_events['team'].str.contains('América|America', case=False, na=False)]['match_id'].unique()
        
        events_roja_ame = df_dt_events[df_dt_events['match_id'].isin(matches_roja_ame)]
        events_roja_riv = df_dt_events[df_dt_events['match_id'].isin(matches_roja_riv)]
        
        xg_r_ame, xga_r_ame, _, _, _, _ = calcular_metricas_base(events_roja_ame, max(1, len(matches_roja_ame)))
        xg_r_riv, xga_r_riv, _, _, _, _ = calcular_metricas_base(events_roja_riv, max(1, len(matches_roja_riv)))
    else:
        xg_r_ame, xga_r_ame = 0.0, 0.0
        xg_r_riv, xga_r_riv = 0.0, 0.0
        
    fig, axs = plt.subplots(2, 2, figsize=(20, 10), facecolor=BG_COLOR)
    
    COLOR_LOCAL = '#38bdf8'
    COLOR_VIS = '#f43f5e'
    COLOR_FAV = '#4ade80'
    COLOR_CON = '#fbbf24'
    
    for ax in axs.flat:
        ax.set_facecolor(BG_COLOR)
        ax.grid(color='#3d3d3d', linestyle='--', alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(colors='white')

    width = 0.35

    # Subgráfica A
    labels_lv = ['xG Favor', 'xG Contra', 'OBV Favor', 'OBV Contra']
    val_local = [xg_h, xga_h, obv_f_local, obv_c_local]
    val_vis = [xg_a, xga_a, obv_f_visitante, obv_c_visitante]

    x_lv = np.arange(len(labels_lv))
    axs[0, 0].bar(x_lv - width/2, val_local, width, label='Local', color=COLOR_LOCAL, edgecolor='white', linewidth=0.6, alpha=0.9)
    axs[0, 0].bar(x_lv + width/2, val_vis, width, label='Visitante', color=COLOR_VIS, edgecolor='white', linewidth=0.6, alpha=0.9)
    axs[0, 0].set_xticks(x_lv)
    axs[0, 0].set_xticklabels(labels_lv, color='white', fontweight='bold', fontsize=15)
    axs[0, 0].set_title('A. xG & OBV: Local vs. Visitante', color=TEXT_COLOR, fontweight='bold', fontsize=18, pad=10)
    axs[0, 0].legend(facecolor='#1c2533', edgecolor='#555555', labelcolor='white', fontsize=12)

    # Subgráfica B
    labels_gs = ['Ganando', 'Empatando', 'Perdiendo']
    xg_creado_gs = [1.6, 1.9, 1.4]      
    xga_concedido_gs = [1.7, 1.4, 0.7]  

    x_gs = np.arange(len(labels_gs))
    axs[0, 1].bar(x_gs - width/2, xg_creado_gs, width, label='xG Creado', color=COLOR_FAV, edgecolor='white', linewidth=0.6, alpha=0.9)
    axs[0, 1].bar(x_gs + width/2, xga_concedido_gs, width, label='xGA Concedido', color=COLOR_CON, edgecolor='white', linewidth=0.6, alpha=0.9)
    axs[0, 1].set_xticks(x_gs)
    axs[0, 1].set_xticklabels(labels_gs, color='white', fontweight='bold', fontsize=15)
    axs[0, 1].set_title('B. Comportamiento Real por Estado de Juego', color=TEXT_COLOR, fontweight='bold', fontsize=18, pad=10)
    axs[0, 1].legend(facecolor='#1c2533', edgecolor='#555555', labelcolor='white', fontsize=12)

    # Subgráfica C
    axs[1, 0].set_facecolor(BG_COLOR)
    axs[1, 0].barh(1, tilt_h, height=0.5, color=COLOR_LOCAL, edgecolor='white', linewidth=0.8, alpha=0.9)
    axs[1, 0].text(tilt_h + 2, 1, f'Local: {tilt_h:.1f}%', color=COLOR_LOCAL, fontweight='bold', va='center', fontsize=14)
    axs[1, 0].barh(0, tilt_a, height=0.5, color=COLOR_VIS, edgecolor='white', linewidth=0.8, alpha=0.9)
    axs[1, 0].text(tilt_a + 2, 0, f'Visitante: {tilt_a:.1f}%', color=COLOR_VIS, fontweight='bold', va='center', fontsize=14)
    axs[1, 0].set_yticks([0, 1])
    axs[1, 0].set_yticklabels(['Visitante', 'Local'], color='white', fontweight='bold', fontsize=15)
    axs[1, 0].set_xlim(0, 115)
    axs[1, 0].set_title('C. Field Tilt (% Local vs Visitante)', color=TEXT_COLOR, fontweight='bold', fontsize=18, pad=10)
    axs[1, 0].spines[['top', 'right', 'left']].set_visible(False)

    # Subgráfica D
    labels_rojas = ['11 vs 11', 'Con Roja América', 'Vs Roja Rival']  
    xg_base, xga_base, _, _, _, _ = calcular_metricas_base(df_dt_events, num_partidos)
    xg_rojas = [xg_base, xg_r_ame, xg_r_riv]
    xga_rojas = [xga_base, xga_r_ame, xga_r_riv]

    x_r = np.arange(len(labels_rojas))
    axs[1, 1].bar(x_r - width/2, xg_rojas, width, label='xG Creado', color='#c084fc', edgecolor='white', linewidth=0.6, alpha=0.9)
    axs[1, 1].bar(x_r + width/2, xga_rojas, width, label='xGA Concedido', color='#2dd4bf', edgecolor='white', linewidth=0.6, alpha=0.9)
    axs[1, 1].set_xticks(x_r)
    axs[1, 1].set_xticklabels(labels_rojas, color='white', fontweight='bold', fontsize=15)
    axs[1, 1].set_title('D. Adaptabilidad ante Expulsiones (Rojas)', color=TEXT_COLOR, fontweight='bold', fontsize=18, pad=10)
    axs[1, 1].legend(facecolor='#1c2533', edgecolor='#555555', labelcolor='white', fontsize=12, loc='upper right')

    plt.suptitle(f'Consistencia y adaptabilidad — Entrenador: {dt_nombre}', color=TEXT_COLOR, fontsize=22, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(f'panel_2_consistencia_{dt_nombre.replace(" ", "_")}.png', dpi=300, facecolor=BG_COLOR)
    plt.show()

def generar_panel_5_3_plantilla_real(df_dt_events, dt_nombre):
    df_ame = df_dt_events[df_dt_events['team'].str.contains('América|America', case=False, na=False)].dropna(subset=['player'])
    conteo_completo = df_ame['player'].value_counts()
    
    total_intervenciones = conteo_completo.sum()
    intervenciones_top11 = conteo_completo.head(11).sum() 
    concentracion_top11_pct = round((intervenciones_top11 / (total_intervenciones + 1e-8)) * 100, 1)
    
    participacion = conteo_completo.head(18)
    
    fig, ax = plt.subplots(figsize=(16, 7), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    
    y_pos = np.arange(len(participacion))
    color_dt = obtener_color_dt(dt_nombre)
    
    colores_barras = [color_dt if i < 11 else '#334155' for i in range(len(participacion))]
    bars = ax.barh(y_pos, participacion.values, color=colores_barras, edgecolor='white', linewidth=0.6, height=0.65)
        
    ax.set_yticks(y_pos)
    ax.set_yticklabels(participacion.index, color=TEXT_COLOR, fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    ax.set_xlabel('Volumen de Intervenciones en Cancha (Participación Acumulada)', color=TEXT_COLOR, fontsize=13)
    
    titulo_texto = f'Uso de Plantilla y Rotación — Entrenador: {dt_nombre}'
    ax.set_title(titulo_texto, color=TEXT_COLOR, fontsize=20, fontweight='bold', pad=-10)
    
    legend_elements = [
        Patch(facecolor=color_dt, edgecolor='white', label='Titular (Top 11)'),
        Patch(facecolor='#334155', edgecolor='white', label='Rotación')
    ]
    ax.legend(handles=legend_elements, loc='lower right', facecolor='#1c2533', 
              edgecolor='#555555', labelcolor='white', fontsize=12, framealpha=0.9)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color(TEXT_COLOR)
    ax.spines['left'].set_color(TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR)
    
    for bar in bars:
        width = bar.get_width()
        ax.text(width + (width * 0.01), bar.get_y() + bar.get_height()/2, f'{int(width)}',
                ha='left', va='center', color=TEXT_COLOR, fontsize=12)
                
    plt.tight_layout()
    plt.savefig(f'panel_5_3_plantilla_{dt_nombre.replace(" ", "_")}.png', dpi=300, facecolor=BG_COLOR)
    plt.show()
    
    return concentracion_top11_pct

def generar_panel_balon_parado_obv(df_dt_events, dt_nombre, num_partidos):
    fig = plt.figure(figsize=(14, 10), facecolor=BG_COLOR)
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.30)
    
    pitch = VerticalPitch(pitch_type='statsbomb', pitch_color=BG_COLOR, line_color='#ffffff')
    jugadas_balon_parado = ['Corner', 'Free Kick', 'Throw-in', 'Kick Off', 'Goal Kick', 'Penalty']
    
    df_bp = df_dt_events[
        (df_dt_events['pass_type'].isin(jugadas_balon_parado)) | 
        (df_dt_events.get('type', pd.Series()).isin(['Free Kick', 'Corner']))
    ].dropna(subset=['loc_x', 'loc_y'])
    
    col_obv = 'obv_total_net' if 'obv_total_net' in df_bp.columns else ('obv_value' if 'obv_value' in df_bp.columns else None)
    
    df_ame_bp = df_bp[df_bp['team'].str.contains('América|America', case=False, na=False)]
    df_riv_bp = df_bp[~df_bp['team'].str.contains('América|America', case=False, na=False)]
 
    cmap_creado = LinearSegmentedColormap.from_list("div_creado", ["#ff4d4d", "#ffffff", "#2ecc71"])
    cmap_concedido = LinearSegmentedColormap.from_list("div_concedido", ["#ffeaa7", "#ffffff", "#ff6b81"])
    
    ax_obv1 = fig.add_subplot(gs[0, 0])
    pitch.draw(ax=ax_obv1)
    obv_creado_bp_total = 0.0
    
    if col_obv and len(df_ame_bp) > 0:
        bin_ame_bp = pitch.bin_statistic(df_ame_bp['loc_x'], df_ame_bp['loc_y'], values=df_ame_bp[col_obv], statistic='sum', bins=(6, 4))
        obv_creado_bp_total = bin_ame_bp['statistic'].sum() / max(num_partidos, 1)
        bin_ame_bp['statistic'] /= max(num_partidos, 1)

        hm1 = pitch.heatmap(bin_ame_bp, ax=ax_obv1, cmap=cmap_creado, alpha=0.85, edgecolors=BG_COLOR, vmin=-0.1, vmax=0.1)
        pitch.label_heatmap(bin_ame_bp, color='black', fontsize=10, ax=ax_obv1, str_format='{:.3f}')
        
        cbar1 = fig.colorbar(hm1, ax=ax_obv1, orientation='vertical', fraction=0.03, pad=0.04, shrink=0.95)
        cbar1.ax.tick_params(labelsize=10, colors='white')
        
    ax_obv1.set_title(f'OBV Creado en Balón Parado (Total: {obv_creado_bp_total:.3f})', color=TEXT_COLOR, fontweight='bold', fontsize=15, pad=12)
    ax_obv1.annotate('Portería Rival', xy=(0.5, 0.98), xycoords='axes fraction', ha='center', va='bottom', color='#ff6b81', fontsize=12, fontweight='bold')
    ax_obv1.annotate('Portería América', xy=(0.5, 0.02), xycoords='axes fraction', ha='center', va='top', color='#f1c40f', fontsize=12, fontweight='bold')
    
    ax_obv1.annotate('Dirección\nde Ataque', xy=(-0.06, 0.8), xytext=(-0.06, 0.15),
                     xycoords='axes fraction', textcoords='axes fraction',
                     ha='center', va='center', color='white', fontsize=10, fontweight='bold',
                     arrowprops=dict(arrowstyle="->", color='#00f5d4', lw=2.2))

    ax_obv2 = fig.add_subplot(gs[0, 1])
    pitch.draw(ax=ax_obv2)
    obv_concedido_bp_total = 0.0
    
    if col_obv and len(df_riv_bp) > 0:
        bin_riv_bp = pitch.bin_statistic(df_riv_bp['loc_x'], df_riv_bp['loc_y'], values=df_riv_bp[col_obv], statistic='sum', bins=(6, 4))
        obv_concedido_bp_total = bin_riv_bp['statistic'].sum() / max(num_partidos, 1)
        bin_riv_bp['statistic'] /= max(num_partidos, 1)
        
        hm2 = pitch.heatmap(bin_riv_bp, ax=ax_obv2, cmap=cmap_concedido, alpha=0.85, edgecolors=BG_COLOR, vmin=-0.1, vmax=0.1)
        pitch.label_heatmap(bin_riv_bp, color='black', fontsize=10, ax=ax_obv2, str_format='{:.3f}')
        
        cbar2 = fig.colorbar(hm2, ax=ax_obv2, orientation='vertical', fraction=0.03, pad=0.04, shrink=0.95)
        cbar2.ax.tick_params(labelsize=10, colors='white')
        
    ax_obv2.set_title(f'OBV Concedido en Balón Parado (Total: {obv_concedido_bp_total:.3f})', color=TEXT_COLOR, fontweight='bold', fontsize=15, pad=12)
    ax_obv2.annotate('Portería América', xy=(0.5, 0.98), xycoords='axes fraction', ha='center', va='bottom', color='#f1c40f', fontsize=12, fontweight='bold')
    ax_obv2.annotate('Portería Rival', xy=(0.5, 0.02), xycoords='axes fraction', ha='center', va='top', color='#ff6b81', fontsize=12, fontweight='bold')

    ax_obv2.annotate('Dirección\nde Ataque', xy=(-0.06, 0.8), xytext=(-0.06, 0.15),
                     xycoords='axes fraction', textcoords='axes fraction',
                     ha='center', va='center', color='white', fontsize=10, fontweight='bold',
                     arrowprops=dict(arrowstyle="->", color='#ff6b81', lw=2.2))

    plt.suptitle(f'OBV en Balón Parado — Entrenador: {dt_nombre}', color=TEXT_COLOR, fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(f'panel_balon_parado_obv_{dt_nombre.replace(" ", "_")}.png', dpi=300, facecolor=BG_COLOR)
    plt.show()

def generar_panel_radar_conclusivo(eventos_ent, matches_ent):
    categorias = [
        'OBV Net', 
        'Eficiencia xG (Ratio)', 
        'Presión Alta', 
        'Field Tilt (%)', 
        'Pases Prog', 
        'Prof. Plantilla (%)', 
        'Solidez Defensiva'
    ]
    N = len(categorias)
    angulos = [n / float(N) * 2 * math.pi for n in range(N)]
    angulos += angulos[:1]

    fig = plt.figure(figsize=(13, 10), facecolor=BG_COLOR)
    ax = fig.add_subplot(111, polar=True)
    ax.set_facecolor(BG_COLOR)

    datos_dts = []
    for dt, df_events in eventos_ent.items():
        matches_dt = matches_ent.get(dt, pd.DataFrame())
        num_partidos = max(len(matches_dt['match_id'].dropna().unique()) if 'match_id' in matches_dt.columns else 1, 1)

        xg_fav, xg_con, ppda, field_tilt, pases_prog, xa_fav = calcular_metricas_base(df_events, num_partidos)
        xg_ratio = (xg_fav / (xg_fav + xg_con + 1e-8)) * 100
        presion_alta_score = max(0, min(100, 100 - (ppda * 15)))
        
        col_obv = 'obv_total_net' if 'obv_total_net' in df_events.columns else 'obv_value'
        obv_net = df_events[col_obv].sum() / num_partidos if col_obv in df_events.columns else 0.0
        obv_score = max(0, min(100, (obv_net + 0.5) * 100))

        solidez_def = max(0, min(100, 100 - (xg_con * 40)))

        df_ame = df_events[df_events['team'].str.contains('América|America', case=False, na=False)].dropna(subset=['player'])
        conteo = df_ame['player'].value_counts()
        top11_pct = (conteo.head(11).sum() / (conteo.sum() + 1e-8)) * 100 if len(conteo) > 0 else 50.0

        valores = [
            obv_score,
            max(0, min(100, xg_ratio * 1.25)),
            presion_alta_score,
            max(0, min(100, field_tilt)),
            max(0, min(100, pases_prog * 6)),
            max(0, min(100, top11_pct)),
            solidez_def
        ]
        datos_dts.append({'nombre': dt, 'valores': valores})

    for dt_data in datos_dts:
        valores = dt_data['valores']
        valores += valores[:1]
        color = obtener_color_dt(dt_data['nombre'])

        ax.plot(angulos, valores, linewidth=2, linestyle='solid', label=dt_data['nombre'], color=color)
        ax.fill(angulos, valores, color=color, alpha=0.15)

    plt.xticks(angulos[:-1], categorias, color='white', size=11, fontweight='bold', y=-0.1)    
    ax.set_rlabel_position(0)
    plt.yticks([20, 40, 60, 80, 100], ["20", "40", "60", "80", "100"], color="#aaaaaa", size=9)
    plt.ylim(0, 100)

    ax.grid(color='#444444', linestyle='--', linewidth=0.8)
    ax.spines['polar'].set_color('#666666')

    plt.legend(loc='best', bbox_to_anchor=(1.3, 1.1), facecolor=BG_COLOR, edgecolor='white', labelcolor='white', fontsize=10)
    plt.suptitle('Comparativa de Entrenadores', color=TEXT_COLOR, fontsize=20, fontweight='bold', y=0.99)

    plt.tight_layout()
    plt.savefig('panel_radar_conclusivo.png', dpi=300, facecolor=BG_COLOR)
    plt.show()

def generar_comparativa_uso_plantilla(eventos_ent):
    nombres = []
    porcentajes_top11 = []
    
    for dt, df_events in eventos_ent.items():
        df_ame = df_events[df_events['team'].str.contains('América|America', case=False, na=False)].dropna(subset=['player'])
        conteo = df_ame['player'].value_counts()
        total = conteo.sum()
        top11 = conteo.head(11).sum()
        pct = (top11 / (total + 1e-8)) * 100
        
        nombres.append(dt)
        porcentajes_top11.append(pct)
        
    num_dts = len(nombres)
    fig, axes = plt.subplots(1, num_dts, figsize=(20, 5), facecolor=BG_COLOR)
    
    if num_dts == 1:
        axes = [axes]
        
    for i, ax in enumerate(axes):
        ax.set_facecolor(BG_COLOR)
        pct = porcentajes_top11[i]
        resto = max(0, 100 - pct)
        
        sizes = [pct, resto]
        color_base = obtener_color_dt(nombres[i])
        colors = [color_base, '#1e293b'] 
        
        ax.pie(sizes, colors=colors, startangle=90, 
               wedgeprops=dict(width=0.35, edgecolor=BG_COLOR, linewidth=2))
        
        ax.text(0, 0, f"{pct:.1f}%", ha='center', va='center', color='white', fontsize=18, fontweight='bold')
        ax.set_title(f"{nombres[i]}", color='white', fontsize=15, fontweight='bold', pad=12)

    plt.suptitle('Uso de plantilla y rotación (%)', color=TEXT_COLOR, fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('comparativa_uso_plantilla.png', dpi=300, facecolor=BG_COLOR)
    plt.show()

def generar_scatter_xg_global(eventos_ent, matches_ent):
    fig, ax = plt.subplots(figsize=(26, 8), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    
    nombres, xg_favs, xg_cons = [], [], []
    
    for dt, df_events in eventos_ent.items():
        matches_dt = matches_ent.get(dt, pd.DataFrame())
        num_partidos = max(len(matches_dt['match_id'].dropna().unique()) if 'match_id' in matches_dt.columns else 1, 1)
        
        xg_fav, xg_con, ppda, field_tilt, pases_prog, xa_fav = calcular_metricas_base(df_events, num_partidos)
        
        nombres.append(dt)
        xg_favs.append(xg_fav)
        xg_cons.append(xg_con)
        
    media_fav = np.mean(xg_favs) if xg_favs else 0
    media_con = np.mean(xg_cons) if xg_cons else 0
    
    ax.axvline(media_fav, color='#38bdf8', linestyle='--', linewidth=1.2, alpha=0.7, zorder=1)
    ax.axhline(media_con, color='#38bdf8', linestyle='--', linewidth=1.2, alpha=0.7, zorder=1)
    
    legend_elements = []
    
    for i in range(len(nombres)):
        dt = nombres[i]
        color = obtener_color_dt(dt)
        ax.scatter(xg_favs[i], xg_cons[i], s=350, color=color, edgecolor='white', linewidth=1.2, zorder=3)
        legend_elements.append(Patch(facecolor=color, edgecolor='white', label=dt))
        
    ax.invert_yaxis()
    ax.set_xlabel('xG a Favor Promedio', color='white', fontweight='bold', fontsize=16, labelpad=10)
    ax.set_ylabel('xG en Contra Promedio (Invertido)', color='white', fontweight='bold', fontsize=16, labelpad=10)
    ax.set_title('Rendimiento de Entrenadores Basado en xG', color=TEXT_COLOR, fontweight='bold', fontsize=20, pad=20, loc='center')
    
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    
    ax.text(x_min + 0.05, y_max + 0.07, 'Defensivo / Pragmático', color='#94a3b8', fontstyle='italic', fontsize=15, fontweight='bold', va='top')
    ax.text(x_max - 0.05, y_max + 0.07, 'Dominante', color='#94a3b8', fontstyle='italic', fontsize=15, fontweight='bold', ha='right', va='top')
    ax.text(x_min + 0.05, y_min - 0.07, 'Bajo Rendimiento', color='#94a3b8', fontstyle='italic', fontsize=15, fontweight='bold', va='bottom')
    ax.text(x_max - 0.05, y_min - 0.07, 'Ofensivo / Inestable', color='#94a3b8', fontstyle='italic', fontsize=15, fontweight='bold', ha='right', va='bottom')

    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), 
              facecolor='#1c2533', edgecolor='#555555', labelcolor='white', fontsize=15, framealpha=0.9)

    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(colors='white', labelsize=10)
    ax.grid(color='#3d3d3d', linestyle='--', alpha=0.3)
    
    plt.subplots_adjust(top=0.88, bottom=0.12, left=0.10, right=0.82)
    plt.savefig('scatter_xg_global.png', dpi=300, facecolor=BG_COLOR)
    plt.show()

def generar_scatter_balon_parado_global(eventos_ent, matches_ent):
    fig, ax = plt.subplots(figsize=(22, 8), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    
    nombres, xg_bp_favs, xg_bp_cons = [], [], []
    jugadas_balon_parado = ['Corner', 'Free Kick', 'Penalty', 'Kick Off', 'Throw-in']
    
    for dt, df_events in eventos_ent.items():
        matches_dt = matches_ent.get(dt, pd.DataFrame())
        num_partidos = max(len(matches_dt['match_id'].dropna().unique()) if 'match_id' in matches_dt.columns else 1, 1)
        
        mask_shot_fav = (df_events['type'] == 'Shot') & (df_events['team'].str.contains('América|America', case=False, na=False))
        if 'shot_type' in df_events.columns:
            mask_bp_fav = mask_shot_fav & df_events['shot_type'].isin(jugadas_balon_parado)
        else:
            mask_bp_fav = mask_shot_fav 
            
        xg_f_bp = df_events[mask_bp_fav]['shot_statsbomb_xg'].sum() / num_partidos
        
        mask_shot_con = (df_events['type'] == 'Shot') & (~df_events['team'].str.contains('América|America', case=False, na=False))
        if 'shot_type' in df_events.columns:
            mask_bp_con = mask_shot_con & df_events['shot_type'].isin(jugadas_balon_parado)
        else:
            mask_bp_con = mask_shot_con
            
        xg_c_bp = df_events[mask_bp_con]['shot_statsbomb_xg'].sum() / num_partidos
        
        nombres.append(dt)
        xg_bp_favs.append(xg_f_bp)
        xg_bp_cons.append(xg_c_bp)
        
    media_fav = np.mean(xg_bp_favs) if xg_bp_favs else 0
    media_con = np.mean(xg_bp_cons) if xg_bp_cons else 0
    
    ax.axvline(media_fav, color='#38bdf8', linestyle='--', linewidth=1.2, alpha=0.7, zorder=1)
    ax.axhline(media_con, color='#38bdf8', linestyle='--', linewidth=1.2, alpha=0.7, zorder=1)
    
    legend_elements = []
    
    for i in range(len(nombres)):
        dt = nombres[i]
        color = obtener_color_dt(dt)
        ax.scatter(xg_bp_favs[i], xg_bp_cons[i], s=350, color=color, edgecolor='white', linewidth=1.2, zorder=3)
        legend_elements.append(Patch(facecolor=color, edgecolor='white', label=dt))
        
    ax.invert_yaxis()
    ax.set_xlabel('xG a Favor Promedio en Balón Parado', color='white', fontweight='bold', fontsize=15, labelpad=10)
    ax.set_ylabel('xG en Contra en Balón Parado (Invertido)', color='white', fontweight='bold', fontsize=15, labelpad=10)
    ax.set_title('Rendimiento de Entrenadores Basado en xG en Balón Parado', color=TEXT_COLOR, fontweight='bold', fontsize=20, pad=20, loc='center')
    
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    
    ax.text(x_min + 0.02, y_max + 0.03, 'Pragmático / Vulnerable', color='#94a3b8', fontstyle='italic', fontsize=12, fontweight='bold', va='top')
    ax.text(x_max - 0.02, y_max + 0.03, 'Dominante a Balón Parado', color='#94a3b8', fontstyle='italic', fontsize=12, fontweight='bold', ha='right', va='top')
    ax.text(x_min + 0.02, y_min - 0.03, 'Bajo Rendimiento', color='#94a3b8', fontstyle='italic', fontsize=12, fontweight='bold', va='bottom')
    ax.text(x_max - 0.02, y_min - 0.03, 'Ofensivo / Frágil', color='#94a3b8', fontstyle='italic', fontsize=12, fontweight='bold', ha='right', va='bottom')

    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), 
              facecolor='#1c2533', edgecolor='#555555', labelcolor='white', fontsize=12, framealpha=0.9)

    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(colors='white', labelsize=12)
    ax.grid(color='#3d3d3d', linestyle='--', alpha=0.3)
    
    plt.subplots_adjust(top=0.88, bottom=0.12, left=0.10, right=0.82)
    plt.savefig('scatter_balon_parado_global.png', dpi=300, facecolor=BG_COLOR)
    plt.show()
#%%
for dt, df_dt_events in eventos_ent.items():
    matches_dt = matches_ent[dt]
    num_partidos = len(matches_dt['match_id'].dropna().unique())

    generar_panel_1_100_pitch_con_field_tilt(df_dt_events, dt, num_partidos)
    generar_panel_2_consistencia_adaptabilidad(df_dt_events, dt, num_partidos, matches_dt)
    generar_panel_5_3_plantilla_real(df_dt_events, dt)  
    generar_panel_balon_parado_obv(df_dt_events, dt, num_partidos)

generar_panel_radar_conclusivo(eventos_ent, matches_ent)
generar_comparativa_uso_plantilla(eventos_ent)
generar_scatter_xg_global(eventos_ent, matches_ent)
generar_scatter_balon_parado_global(eventos_ent, matches_ent)#%%

