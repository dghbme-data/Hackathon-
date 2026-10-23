#%% Observaciones
import ast
import numpy as np
import pandas as pd
from statsbombpy import sb
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import concurrent.futures
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import seaborn as sns

import matplotlib.patches as patches
from transformers import pipeline
import tensorflow as tf
import keras
keras.mixed_precision.set_global_policy('mixed_float16')
import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ["HF_TOKEN"] = "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"
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
import torch
import random
import keras

#%%

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

#%%
# =============================================================================
# Seccion 1: OBTENER PARTIDOS A LOS QUE TENEMOS ACCESO 
# =============================================================================
credenciales = {"user": "itam_hackathon@hudl.com", "passwd": "NDeXcBUp"} # Deigo cambia este por tu correo y contraseña, R: Nataly, es la misma.
competition_id = 73 # partidos solo liga MX
competition = sb.competitions(creds=credenciales) # Obtener los partidos a los que tenemos acceso
season_id = competition[competition['competition_id'] == competition_id]['season_id'].to_list() # Id de temporadas solo para liga MX

all_matches = []
for s_id in season_id:
    matches = sb.matches(competition_id=competition_id, season_id=s_id, creds=credenciales)
    all_matches.append(matches)

matches = pd.concat(all_matches) #Todos los partidos de todas las temporadas de la liga MX

#%%
# =============================================================================
# Seccion 2: SEPARAR ID's EN CONJUNTOS (entrenamiento, validación y prueba)
# =============================================================================
train_s = ['2021/2022', '2022/2023', '2023/2024'] # ['2021/2022', '2022/2023', '2023/2024', '2024/2025]
val_s = ['2024/2025']                                      
test_s = ['2025/2026']   # ['2026/2027'] 

train_id = matches[matches['season'].isin(train_s)]['match_id'].unique().tolist()
val_id = matches[matches['season'].isin(val_s)]['match_id'].unique().tolist()

# Partidos solo del América
america_matches = (matches['home_team'] == 'América') | (matches['away_team'] == 'América')

america_train_ids = matches[america_matches & matches['season'].isin(train_s)]['match_id'].unique().tolist()
america_val_ids   = matches[america_matches & matches['season'].isin(val_s)]['match_id'].unique().tolist()
america_test_ids  = matches[america_matches & matches['season'].isin(test_s)]['match_id'].unique().tolist()

# Pequeño candado de seguridad para confirmar que sí encontró datos
print(f"Partidos encontrados -> Entrenamiento: {len(america_train_ids)} | Validación: {len(america_val_ids)} | Prueba: {len(america_test_ids)}")

#%%
# =============================================================================
# Seccion 3: FUNCIÓN PARA EXTRAER TODAS LAS VARIABLES NECESARIAS
# =============================================================================

def extraer_variables(match_id, credenciales, matches_df):
    events = sb.events(match_id=match_id, creds=credenciales, include_360_metrics=True).sort_values(by=['index']).reset_index(drop=True)
    
    # Mapeos
    jugadas_balon_parado = {'Corner', 'Free Kick', 'Throw-in', 'Kick Off', 'Goal Kick', 'Penalty'}
    
    play_pattern_id = {
        'Regular Play': 0, 'From Corner': 1, 'From Free Kick': 2, 'From Throw In': 3, 
        'Other': 4, 'From Counter': 5, 'From Goal Kick': 6, 'From Keeper': 7, 'From Kick Off': 8 
    } 
    
    event_type_id = {
        'Pass': 0, 'Carry': 1, 'Shot': 2, 'Pressure': 3,
        'Ball Recovery': 4, 'Interception': 4, 'Clearance': 4, 'Block': 4,
        'Duel': 5, '50/50': 5, 'Dribbled Past': 5, 
        'Dribble': 6, 'Foul Committed': 7, 'Foul Won': 7, 'Offside': 7,
        'Dispossessed': 8, 'Miscontrol': 8, 'Error': 8, 'Goal Keeper': 9 
    }

    status_360 = matches_df.loc[matches_df['match_id'] == match_id, 'match_status_360'].values
    match_has_360 = 1.0 if len(status_360) > 0 and status_360[0] == 'available' else 0.0
    
    last_t_stamp = 0.0
    evento = []
    
    for row in events.to_dict('records'):
        # locación en formato x,y
        loc = row.get('location')
        if not isinstance(loc, (list, np.ndarray)) or pd.isna(loc).any():
            loc = [60.0, 40.0]
            
        # locación final en formato x,y  
        end_loc = loc 
        for col in ['pass_end_location', 'shot_end_location', 'carry_end_location']:
            end = row.get(col)
            if isinstance(end, (list, np.ndarray)) and not pd.isna(end).any():
                end_loc = end
                break

        val_team = row.get('team_name') if pd.notna(row.get('team_name')) else row.get('team')
        val_player = row.get('player_name') if pd.notna(row.get('player_name')) else row.get('player')
        
        v_team = str(val_team)
        v_player = str(val_player)
        
        # location y end location normalizados 
        v1_loc_x = float(loc[0]) / 120.0
        v2_loc_y = float(loc[1]) / 80.0
        v3_end_x = float(end_loc[0]) / 120.0
        v4_end_y = float(end_loc[1]) / 80.0
        
        # duración
        v5_duration = float(row.get('duration')) 
        if pd.isna(v5_duration):
            v5_duration = 0.0
            
        # tiempo (time_diff)
        t_stamp = str(row.get('timestamp'))
        h, m, s = t_stamp.split(':')
        seg_actual = float(h) * 3600 + float(m) * 60 + float(s) 
        v6_time = max(0.0, seg_actual - last_t_stamp) 
        last_t_stamp = seg_actual
        
        # presión y Balon Parado
        v7_under_pressure = 1.0 if row.get('under_pressure') == True else 0.0 
        pass_type = str(row.get('pass_type'))
        shot_type = str(row.get('shot_type'))
        balon_parado = (pass_type in jugadas_balon_parado or shot_type in jugadas_balon_parado)
        v8_balon_parado = 1.0 if balon_parado else 0.0
        
        # periodo y datos 360
        v9_period = float(row.get('period', 1)) / 5.0
        status_360 = matches.loc[matches['match_id'] == match_id, 'match_status_360'].values
        v10_datos_360 = match_has_360
        
        # datos de pases
        v11_pass_length = float(row.get('pass_length')) / 120.0 if pd.notna(row.get('pass_length')) else 0.0
        v12_pass_angle = float(row.get('pass_angle')) / np.pi if pd.notna(row.get('pass_angle')) else 0.0
        
        pass_height = str(row.get('pass_height', ''))
        v13_is_ground_pass = 1.0 if pass_height == 'Ground Pass' else 0.0
        v14_is_high_pass = 1.0 if pass_height in ['High Pass', 'Low Pass'] else 0.0
        v15_is_switch = 1.0 if row.get('pass_switch') == True else 0.0
        
        # patron de juego
        v16_play_pattern = [0.0] * 9 
        pattern_type = row.get('play_pattern')
        if isinstance(pattern_type, str) and pattern_type in play_pattern_id:
            v16_play_pattern[play_pattern_id[pattern_type]] = 1.0
            
        # tipo de evento
        v17_event_type = [0.0] * 10
        event_type = row.get('type')
        if isinstance(event_type, str) and event_type in event_type_id:
            v17_event_type[event_type_id[event_type]] = 1.0
        
        # Secuencia e Identidad de Posesión
        v18_possession = float(row.get('possession', 0)) 
        team = row.get('team')
        possession_team = row.get('possession_team_name') if pd.notna(row.get('possession_team_name')) else row.get('possession_team')
        v19_is_possession_team = 1.0 if team == possession_team else 0.0    
        
        # Resultado de la Acción 
        v20_is_success = 0.0
        if event_type == 'Pass':
            v20_is_success = 1.0 if pd.isna(row.get('pass_outcome')) else 0.0
        elif event_type == 'Shot':
            v20_is_success = 1.0 if row.get('shot_outcome') == 'Goal' else 0.0
        elif event_type == 'Dribble':
            v20_is_success = 1.0 if row.get('dribble_outcome') == 'Complete' else 0.0
        elif event_type in ['Carry', 'Ball Recovery', 'Interception', 'Clearance']:
            v20_is_success = 1.0
            
        # 3. Geometría Avanzada
        dist_to_goal = np.sqrt((120.0 - float(loc[0]))**2 + (40.0 - float(loc[1]))**2)
        v21_dist_to_goal = dist_to_goal / 126.5 
        
        # 4. Contexto Táctico Específico
        v22_is_cross = 1.0 if row.get('pass_cross') == True else 0.0
        v23_counterpress = 1.0 if row.get('counterpress') == True else 0.0
        
        # 5. Goles Esperados (xG), Asistencias Esperadas (xA) y Amenaza Esperada (xT)
        xg = row.get('shot_statsbomb_xg')
        v24_xg = float(xg) if pd.notna(xg) else 0.0
        
        # StatsBomb suele guardar el xA en 'pass_expected_assists' o 'pass_xa'
        xa = row.get('pass_expected_assists', row.get('pass_xa'))
        v25_xa = float(xa) if pd.notna(xa) else 0.0
        
        # StatsBomb suele guardar la amenaza en 'xt' o 'expected_threat'
        xt = row.get('xt', row.get('expected_threat'))
        v26_xt = float(xt) if pd.notna(xt) else 0.0

        v_freeze_frame = row.get('freeze_frame') if isinstance(row.get('freeze_frame'), list) else []

        vector_evento = [
            v1_loc_x, v2_loc_y, v3_end_x, v4_end_y,
            v5_duration, v6_time, v7_under_pressure, v8_balon_parado, v9_period,
            v10_datos_360, v11_pass_length, v12_pass_angle, 
            v13_is_ground_pass, v14_is_high_pass, v15_is_switch,
            v18_possession, v19_is_possession_team, v20_is_success, 
            v21_dist_to_goal, v22_is_cross, v23_counterpress, 
            v24_xg, v25_xa, v26_xt
        ] + v16_play_pattern + v17_event_type + [v_team, v_player]
        
        evento.append(vector_evento)

    cols_pp = [f'pp_{k}' for k in play_pattern_id.keys()]
    cols_ev = ['ev_pass', 'ev_carry', 'ev_shot', 'ev_pressure', 'ev_def_recovery', 
               'ev_duel', 'ev_dribble', 'ev_foul', 'ev_turnover', 'ev_goalkeeper']
    
    cols_nuevas = ['possession', 'is_possession_team', 'is_success', 'dist_to_goal', 
                   'is_cross', 'counterpress', 'xg', 'xa', 'xt']
                   
    columnas = [
        'loc_x', 'loc_y', 'end_x', 'end_y', 'duration', 'time_diff', 
        'under_pressure', 'balon_parado', 'period', 'datos_360', 
        'pass_length', 'pass_angle', 'is_ground_pass', 'is_high_pass', 'is_switch'
    ] + cols_nuevas + cols_pp + cols_ev + ['team_name', 'player_name', 'freeze_frame']

    # Crear y retornar el df
    df_features = pd.DataFrame(evento, columns=columnas)
    return df_features

#%%
# ya quiero acabar
# sino, no sale ¿que cosa? no se

# un solo partido
def procesar_un_partido(m_id):
    try:
        df_temp = extraer_variables(match_id=m_id, credenciales=credenciales, matches_df=matches)
        df_temp.insert(0, 'match_id', m_id)
        return m_id, df_temp
    except ValueError:
        print(f"    Partido {m_id} sin datos de eventos. Saltando...")
        return m_id, None
    except Exception as e:
        print(f"    Error inesperado en partido {m_id}: {e}")
        return m_id, None

# coyeye
def procesar_conjunto(lista_ids, nombre_conjunto, hilos=20):
    print(f"Iniciando descarga en paralelo para {nombre_conjunto} con {hilos} hilos...")
    lista_df = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=hilos) as executor:
        # Enviar todas las tareas al procesador de forma simultánea
        futuros = [executor.submit(procesar_un_partido, m_id) for m_id in lista_ids]
        
        # Recolectar resultados conforme van terminando
        for futuro in concurrent.futures.as_completed(futuros):
            m_id, df_temp = futuro.result()
            if df_temp is not None and not df_temp.empty:
                lista_df.append(df_temp)
            print(f"[{nombre_conjunto}] Partido ID: {m_id} extraído exitosamente.")
            
    if lista_df:
        df_final = pd.concat(lista_df, ignore_index=True)
        print(f"{nombre_conjunto} completado. Total de eventos: {df_final.shape[0]}\n")
        return df_final
    else:
        print(f"No se encontraron datos válidos para {nombre_conjunto}.\n")
        return pd.DataFrame()

#%%

archivo_train = 'america_train.csv'
archivo_val = 'america_val.csv'
archivo_test = 'america_test.csv'

# Verificar si los 3 archivos ya existen localmente y no están vacíos
if (os.path.exists(archivo_train) and os.path.getsize(archivo_train) > 0 and 
    os.path.exists(archivo_val) and os.path.getsize(archivo_val) > 0 and 
    os.path.exists(archivo_test) and os.path.getsize(archivo_test) > 0):
    
    print("Cargando datos rápidamente desde archivos CSV locales...")
    df_train = pd.read_csv(archivo_train)
    df_val = pd.read_csv(archivo_val)
    df_test = pd.read_csv(archivo_test)
else:
    print("Archivos no encontrados o vacíos. Iniciando descarga desde StatsBomb...")
    df_train = procesar_conjunto(america_train_ids, "Entrenamiento")
    df_val = procesar_conjunto(america_val_ids, "Validación")
    df_test = procesar_conjunto(america_test_ids, "Prueba")

    print("Guardando datos en CSV locales para futuras ejecuciones...")
    df_train.to_csv(archivo_train, index=False)
    df_val.to_csv(archivo_val, index=False)
    df_test.to_csv(archivo_test, index=False)

#%%

# puro pedos con las variables
for df_temp in [df_train, df_val, df_test]:
    if 'xa' not in df_temp.columns:
        df_temp['xa'] = 0.0
    if 'xt' not in df_temp.columns:
        df_temp['xt'] = 0.0

# 1. Lista de variables tácticas y espaciales definitivas
features = [
    # Geometría y Ubicación (7)
    'loc_x', 'loc_y', 'end_x', 'end_y', 'dist_to_goal', 'pass_length', 'pass_angle',
    
    # Contexto y Ejecución (10)
    'duration', 'time_diff', 'under_pressure', 'balon_parado', 'datos_360', 
    'is_ground_pass', 'is_high_pass', 'is_switch', 'is_cross', 'counterpress',
    
    # Posesión y Éxito (3)
    'is_possession_team', 'is_success', 'period',
    
    # Métricas de Amenaza / Valor (3)
    'xg', 'xa', 'xt',
    
    # Tipos de Eventos (7)
    'ev_pass', 'ev_carry', 'ev_shot', 'ev_pressure', 'ev_def_recovery', 'ev_duel', 'ev_dribble',
    
    # Orígenes / Patrones de Juego (5)
    'pp_Regular Play', 'pp_From Corner', 'pp_From Free Kick', 'pp_From Throw In', 'pp_From Counter'
]

target = 'is_success' 

X_train = df_train[features]
y_train = df_train[target]
X_val = df_val[features]
y_val = df_val[target]
X_test = df_test[features]
y_test = df_test[target]

ratio_desbalanceo = len(y_train[y_train == 0]) / len(y_train[y_train == 1]) if len(y_train[y_train == 1]) > 0 else 1

# 2. XGBoost con regularización ajustada para múltiples variables
modelo_obv = xgb.XGBClassifier(
    n_estimators=300,          
    max_depth=7,               # Incrementado para capturar la interacción entre zonas y tipos de pase
    learning_rate=0.05,        
    subsample=0.8,             
    colsample_bytree=0.7,      
    scale_pos_weight=ratio_desbalanceo, 
    random_state=42,
    eval_metric='logloss'
)

print("Entrenando el modelo OBV con el paquete táctico completo...")
modelo_obv.fit(X_train, y_train)

predicciones_val = modelo_obv.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, predicciones_val)

print(f"Entrenamiento completado. Validación ROC-AUC: {auc:.4f}")

# calcular prob exito
#df_test['prob_exito'] = modelo_obv.predict_proba(X_test)[:, 1]
#df_test['obv'] = df_test['prob_exito'] - df_test['xg']

# Calcular probabilidad de éxito y OBV real para TODAS las matrices
for df_temp, X_temp in [(df_train, X_train), (df_val, X_val), (df_test, X_test)]:
    # Calcular la probabilidad de éxito
    df_temp['prob_exito'] = modelo_obv.predict_proba(X_temp)[:, 1]
    
    # El OBV se estima como la diferencia entre el éxito predicho y la expectativa base (xG)
    df_temp['obv'] = df_temp['prob_exito'] - df_temp['xg']

#%%

# =============================================================================
# DETECTOR DE PLAYER-TEAM FIT (Scouting Inteligente)
# =============================================================================

def recomendar_fichajes_por_zona(df_partidos, equipo_actual="América", top_n=5):
    """
    Identifica a los jugadores de otros equipos que tienen el mayor On-Ball Value (OBV)
    en las zonas de finalización o progresión, sugiriéndolos como fichajes ideales.
    """
    print(f"\nBuscando jugadores compatibles (Player-Team Fit) para: {equipo_actual}...")
    
    #obv?
    if 'obv' not in df_partidos.columns and 'obv_evento' not in df_partidos.columns:
        return "Error: Primero debes calcular e inyectar el OBV en la matriz."
        
    col_obv = 'obv_evento' if 'obv_evento' in df_partidos.columns else 'obv'
    # ver rivales
    df_scouting = df_partidos[df_partidos['team_name'] != equipo_actual].copy()
    
    # vemos quien amenaza esttando cerca de la cancha
    df_ataque = df_scouting[df_scouting['loc_x'] > 80.0]  
    
    # agrupar el impacto de estos jugadores en zonas críticas
    fit_jugadores = df_ataque.groupby(['team_name', 'player_name']).agg(
        acciones_ataque=(col_obv, 'count'),
        obv_ataque_total=(col_obv, 'sum')
    ).reset_index()
    
    # calculamos eficiencia en ataque
    fit_jugadores['obv_por_accion_ataque'] = fit_jugadores['obv_ataque_total'] / fit_jugadores['acciones_ataque']
    
    # vemos quienes han participado
    candidatos = fit_jugadores[fit_jugadores['acciones_ataque'] > 15].sort_values(
        by='obv_ataque_total', ascending=False
    )
    
    print("\nTOP REFUERZOS RECOMENDADOS (MAYOR IMPACTO EN ÚLTIMO TERCIO)")
    top_candidatos = candidatos.head(top_n)
    print(top_candidatos.to_string(index=False))
    
    # sale
    top_candidatos.to_csv('reporte_player_team_fit.csv', index=False, encoding='utf-8-sig')
    print("Archivo 'reporte_player_team_fit.csv' generado exitosamente.")
    
    return top_candidatos

candidatos_ideales = recomendar_fichajes_por_zona(df_test, equipo_actual="Club América", top_n=5)

#%%
##########################################################################################################################################################################
### ESTO ES PARA GENERAR UN VIDEO DE LA SIMULACION DEL PARTIDO
##########################################################################################################################################################################

plt.rcParams['animation.embed_limit'] = 100.0
plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

def animar_partido_completo(df_partidos, match_id_objetivo, limite_eventos=None, guardar_mp4=True, nombre_archivo='partido_completo.mp4'):
    if limite_eventos is not None:
        partido = df_partidos[df_partidos['match_id'] == match_id_objetivo].iloc[:limite_eventos].reset_index(drop=True)
    else:
        partido = df_partidos[df_partidos['match_id'] == match_id_objetivo].reset_index(drop=True)
        
    if partido.empty:
        print("No se encontraron jugadas para este partido.")
        return None

    equipos = partido['team_name'].dropna().unique()
    equipo_local = equipos[0] if len(equipos) > 0 else "Local"
    equipo_visitante = equipos[1] if len(equipos) > 1 else "Visitante"

    fig, ax = plt.subplots(figsize=(10, 6))
    
    def dibujar_cancha_base():
        ax.clear()
        cesped = patches.Rectangle((-0.05, -0.05), 1.1, 1.1, color='#2E8B57', zorder=0)
        ax.add_patch(cesped)
        
        ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color="white", linewidth=2, zorder=1) 
        ax.plot([0.5, 0.5], [0, 1], color="white", linewidth=2, zorder=1) 
        
        circulo = plt.Circle((0.5, 0.5), 0.1, color='white', fill=False, linewidth=2, zorder=1)
        ax.add_patch(circulo)
        
        ax.plot([0, 0.17, 0.17, 0], [0.21, 0.21, 0.79, 0.79], color="white", linewidth=2, zorder=1)
        ax.plot([1, 0.83, 0.83, 1], [0.21, 0.21, 0.79, 0.79], color="white", linewidth=2, zorder=1)
        
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.invert_yaxis()
        ax.axis('off')

    def actualizar_cuadro(frame):
        dibujar_cancha_base()
        
        # Historial general para calcular el marcador y el tiempo acumulado
        historial = partido.iloc[:frame+1]
        goles_local = len(historial[(historial['team_name'] == equipo_local) & (historial['ev_shot'] == 1.0) & (historial['is_success'] == 1.0)])
        goles_visitante = len(historial[(historial['team_name'] == equipo_visitante) & (historial['ev_shot'] == 1.0) & (historial['is_success'] == 1.0)])
        
        # Definir la ventana de las últimas 5 jugadas (o menos si el partido va empezando)
        inicio_ventana = max(0, frame - 4)
        ventana_jugadas = partido.iloc[inicio_ventana:frame+1]
        
        # Iterar para dibujar la secuencia completa con efecto de rastro
        for i, (idx, jugada) in enumerate(ventana_jugadas.iterrows()):
            es_actual = (idx == frame)
            
            # Las jugadas pasadas son más transparentes, la actual es totalmente sólida
            if es_actual:
                alpha_val = 1.0
                # NUEVO: Leer y proyectar a los demás jugadores
                freeze = jugada.get('freeze_frame', [])
                if isinstance(freeze, str): # Parsear si viene de un CSV guardado
                    try:
                        freeze = ast.literal_eval(freeze)
                    except (ValueError, SyntaxError):
                        freeze = []

                if isinstance(freeze, list) and len(freeze) > 0:
                    for player in freeze:
                        loc = player.get('location')
                        if loc and len(loc) >= 2:
                            px = float(loc[0]) / 120.0
                            py = float(loc[1]) / 80.0
                            es_companero = player.get('teammate', False)
                            es_actor = player.get('actor', False)
                            
                            # Filtrar al jugador con el balón para no encimar los gráficos
                            if not es_actor:
                                color_j = 'blue' if es_companero else 'black'
                                ax.scatter(px, py, color=color_j, s=60, alpha=0.4, zorder=2, edgecolors='white')
            else:
                alpha_val = 0.25 + 0.6 * (i / max(1, len(ventana_jugadas) - 1))
                
            x_ini, y_ini = jugada['loc_x'], jugada['loc_y']
            exito = jugada.get('is_success', 1.0)
            color_accion = 'cyan' if exito == 1.0 else 'red'
            
            if jugada.get('ev_pass') == 1.0:
                if 'end_x' in jugada and not pd.isna(jugada['end_x']):
                    x_fin, y_fin = jugada['end_x'], jugada['end_y']
                    ax.arrow(x_ini, y_ini, x_fin - x_ini, y_fin - y_ini, 
                             head_width=0.02, color=color_accion, alpha=alpha_val, length_includes_head=True, zorder=2)
            elif jugada.get('ev_carry') == 1.0:
                if 'end_x' in jugada and not pd.isna(jugada['end_x']):
                    x_fin, y_fin = jugada['end_x'], jugada['end_y']
                    ax.plot([x_ini, x_fin], [y_ini, y_fin], color='yellow', linestyle='dashed', linewidth=2, alpha=alpha_val, zorder=2)
            elif jugada.get('ev_shot') == 1.0:
                marcador_color = 'gold' if exito == 1.0 else 'darkred'
                ax.scatter(x_ini, y_ini, color=marcador_color, marker='*', s=300, edgecolor='black', alpha=alpha_val, zorder=5)
            else:
                ax.scatter(x_ini, y_ini, color='white', s=80, edgecolor='black', alpha=alpha_val, zorder=3)
                
            if jugada.get('under_pressure') == 1.0:
                ax.scatter(x_ini, y_ini, facecolors='none', edgecolors='orange', s=400, linewidth=2, alpha=alpha_val, zorder=4)
                
            # Solo la jugada que ocurre en el presente modifica la barra de transmisión superior
            if es_actual:
                tiempo_acumulado = historial['time_diff'].sum()
                minuto = int(tiempo_acumulado // 60)
                segundo = int(tiempo_acumulado % 60)
                equipo = jugada.get('team_name', 'Equipo')
                jugador = jugada.get('player_name', 'Desconocido')
                
                texto_marcador = f"{equipo_local} {goles_local} - {goles_visitante} {equipo_visitante}"
                titulo = f"Min {minuto:02d}:{segundo:02d} | {texto_marcador} \n {equipo}: {jugador}"
                
                if 'obv_evento' in jugada and not pd.isna(jugada['obv_evento']):
                    titulo += f" | OBV: {jugada['obv_evento']:.3f}"
                    
                ax.set_title(titulo, fontsize=11, fontweight='bold')

    anim = animation.FuncAnimation(fig, actualizar_cuadro, frames=len(partido), interval=800, repeat=False)
    plt.close() 
    
    if guardar_mp4:
        print(f"Exportando video a {nombre_archivo}... Esto tomará unos minutos.")
        anim.save(nombre_archivo, writer='ffmpeg', fps=2)
        print("Exportación exitosa. Revisa el directorio de tu entorno.")
        return None
    else:
        return HTML(anim.to_jshtml())

#%% ya dame el videooo

############################esto lo debo de apartar
#animar_partido_completo(df_test, df_test['match_id'].iloc[0], limite_eventos=None, guardar_mp4=True, nombre_archivo='partido_completo.mp4')
################################################################################################################################################################
### FIN DEL CODIGO ESTE
################################################################################################################################################################

#%%
# =============================================================================
# MÓDULOS DE ANÁLISIS TÁCTICO Y SCOUTING
# =============================================================================

def recomendar_fichajes_por_zona(df_partidos, equipo_actual="América", top_n=5):
    print(f"\nBuscando jugadores compatibles (Player-Team Fit) para: {equipo_actual}...")
    
    col_obv = 'obv_evento' if 'obv_evento' in df_partidos.columns else 'obv'
    df_scouting = df_partidos[df_partidos['team_name'] != equipo_actual].copy()
    df_ataque = df_scouting[df_scouting['loc_x'] > 80.0] 
    
    fit_jugadores = df_ataque.groupby(['team_name', 'player_name']).agg(
        acciones_ataque=(col_obv, 'count'),
        obv_ataque_total=(col_obv, 'sum')
    ).reset_index()
    
    fit_jugadores['obv_por_accion_ataque'] = fit_jugadores['obv_ataque_total'] / fit_jugadores['acciones_ataque']
    candidatos = fit_jugadores[fit_jugadores['acciones_ataque'] > 15].sort_values(by='obv_ataque_total', ascending=False)
    
    top_candidatos = candidatos.head(top_n)
    print("\nTOP REFUERZOS RECOMENDADOS (MAYOR IMPACTO EN ÚLTIMO TERCIO)")
    print(top_candidatos.to_string(index=False))
    
    top_candidatos.to_csv('reporte_player_team_fit.csv', index=False, encoding='utf-8-sig')
    return top_candidatos

def filtrar_jugadas_criticas(df_partido, top_n=3):
    metricas = ['xg', 'xa', 'obv', 'xt']
    for col in metricas:
        if col not in df_partido.columns:
            df_partido[col] = 0.0
            
    top_xg = df_partido.nlargest(top_n, 'xg')
    top_xa = df_partido.nlargest(top_n, 'xa')
    top_obv = df_partido.nlargest(top_n, 'obv')
    top_xt = df_partido.nlargest(top_n, 'xt')
    
    jugadas_clave = pd.concat([top_xg, top_xa, top_obv, top_xt]).drop_duplicates(subset=['time_diff'])
    return jugadas_clave.sort_values(by='time_diff')


# =============================================================================
# INTEGRACIÓN IA CAUSAL (Qwen 1.5B) Y EJECUCIÓN FINAL
# =============================================================================
print("\nCargando modelo causal (Qwen 1.5B)...")
generador_causal = pipeline(
    "text-generation", 
    model="Qwen/Qwen2.5-1.5B-Instruct",
    model_kwargs={"torch_dtype": torch.float16}, 
    device=0
)

def generar_reporte_completo(df_completo, match_id_objetivo, generador_ia):
    df_partido = df_completo[df_completo['match_id'] == match_id_objetivo].copy()
    if df_partido.empty:
        return "No se encontraron datos para este partido."
        
    df_partido['tiempo_acumulado'] = df_partido['time_diff'].cumsum()
    
    # 1. Corrección del nombre del equipo (StatsBomb usa 'América', no 'Club América')
    df_america = df_partido[df_partido['team_name'] == 'América']
    
    # Candado de seguridad por si el nombre varía
    if df_america.empty:
        df_america = df_partido[df_partido['team_name'].str.contains('América', na=False)]
        
    jugadas_criticas = filtrar_jugadas_criticas(df_america, top_n=3)

    # 2. Corregir el nombre oficial en el scouting
    candidatos = recomendar_fichajes_por_zona(df_completo, "América", top_n=3)
    nombres_candidatos = candidatos['player_name'].tolist()
    str_candidatos = ', '.join(nombres_candidatos)
    
    print(f"\nGenerando instrucciones del partido {match_id_objetivo}...")
    
    reporte_final = ""
    
    # === LLAMAR A LA IA JUGADA POR JUGADA ===
    for idx, jugada in jugadas_criticas.iterrows():
        tiempo_actual = jugada.get('tiempo_acumulado', 0)
        minuto = int(tiempo_actual // 60)
        segundo = int(tiempo_actual % 60)
        marca_tiempo = f"[{minuto:02d}:{segundo:02d}]"
        
        jugador = jugada.get('player_name', 'Jugador')
        
        if jugada.get('ev_shot') == 1.0: accion = "Tiro a gol"
        elif jugada.get('ev_pass') == 1.0: accion = "Pase de ruptura"
        else: accion = "Conducción ofensiva"
            
        obv = jugada.get('obv', 0.0)
        xg = jugada.get('xg', 0.0)
        xt = jugada.get('xt', 0.0)
        
        prompt_jugada = f"Jugador: {jugador} | Acción: {accion} | OBV: {obv:.2f}, xG: {xg:.2f}, xT: {xt:.2f}"

        # Estructura ChatML
        prompt_final = (
            "<|im_start|>system\n"
            "Eres el Director de Inteligencia Deportiva del Club América. Tu objetivo es proponer alternativas tácticas para un simulador.<|im_end|>\n"
            "<|im_start|>user\n"
            f"Analiza ESTA ÚNICA jugada crítica del partido:\n{prompt_jugada}\n\n"
            f"Compañeros disponibles en ataque: {str_candidatos}.\n\n"
            "Redacta UNA sola sugerencia táctica corta para esta jugada. REGLAS ESTRICTAS:\n"
            "1. Asigna a uno de los compañeros disponibles para modificar la jugada.\n"
            "2. Justifica citando textualmente el valor de OBV, xG o xT que se muestra arriba.\n"
            "3. DEBES incluir al menos una de estas palabras clave para el simulador: 'posesión', 'banda', o 'disparo'.<|im_end|>\n"
#            "4. DEBES de dar una sugerencia a todos los minutos qu fueron jugada clave. \n"
            "<|im_start|>assistant\n" 
        )

        outputs = generador_ia(
            prompt_final,
            max_new_tokens=1000, 
            do_sample=True,
            temperature=0.5, 
            return_full_text=False 
        )
        
        texto_generado = outputs[0]['generated_text'].strip()
        texto_generado = texto_generado.replace('\n', ' ')
        
        # Python inyecta el minuto manualmente para que la IA no pueda equivocarse
        reporte_final += f"- Minuto {marca_tiempo}: {texto_generado}\n"

    # para que no envie nada vacio
    if reporte_final == "":
        return "Error: No se encontraron jugadas críticas para el América en este partido."
        
    return reporte_final

#%%

# --- EJECUCIÓN ---
mi_partido_id = 4004042
sugerencias_finales = generar_reporte_completo(df_test, mi_partido_id, generador_causal)

print("\n--- SUGERENCIAS TÁCTICAS AUTOMÁTICAS ---")
print(sugerencias_finales)

#%%
nombre_archivo = 'sugerencias_tacticas.txt'

with open(nombre_archivo, "w", encoding="utf-8") as archivo:
    archivo.write("--- OBSERVACIONES AUTOMÁTICAS ---\n\n")
    archivo.write(sugerencias_finales)

print(f"El reporte táctico se ha guardado exitosamente en tu computadora como '{nombre_archivo}'.")

#%%
# =============================================================================
# 6. CONSTRUCCIÓN DEL MOTOR TÁCTICO (SIMULADOR DE MARKOV)
# =============================================================================
print("\nConstruyendo Motor Táctico de Markov...")
columnas_cancha = 6
filas_cancha = 4

def asignar_zona(x, y):
    col = int(min(x, 0.999) * columnas_cancha)
    fila = int(min(y, 0.999) * filas_cancha)
    return fila * columnas_cancha + col

# Usamos df_train como base histórica para aprender probabilidades reales
df_motor = df_train.copy() 
if df_motor.empty: df_motor = df_test.copy()

df_motor['zona_inicio'] = df_motor.apply(lambda r: asignar_zona(r['loc_x'], r['loc_y']), axis=1)
df_motor['zona_fin'] = df_motor.apply(lambda r: asignar_zona(r['end_x'], r['end_y']), axis=1)

pases = df_motor[(df_motor['ev_pass'] == 1.0) & (df_motor['is_success'] == 1.0)].copy()
transiciones = pases.groupby(['zona_inicio', 'zona_fin']).size().reset_index(name='conteo')
total_por_zona = transiciones.groupby('zona_inicio')['conteo'].transform('sum')
transiciones['prob_exito_pase'] = transiciones['conteo'] / total_por_zona

valor_zonas = df_motor.groupby('zona_inicio')['obv'].mean().reset_index()
valor_zonas.columns = ['zona_fin', 'obv_promedio_zona']

simulador = transiciones.merge(valor_zonas, on='zona_fin', how='left')
#%%

def validar_estilo_entrenador(modificadores_propuestos, matriz_pearson, nombre_dt="André Jardine"):
    print("\n--- VALIDACIÓN TÁCTICA CONTRA PATRÓN DEL ENTRENADOR ---")
    
    modificadores_aprobados = {}
    
    # 1. Validación de Posesión (Relación entre pases e impacto)
    if "asegurar_posesion" in modificadores_propuestos:
        try:
            # Revisa la correlación entre cantidad de pases y éxito/xG
            corr_posesion = matriz_pearson.loc[(nombre_dt, 'ev_pass'), 'xg']
            if corr_posesion > 0.1: # Si el DT históricamente genera peligro tocando el balón
                modificadores_aprobados["asegurar_posesion"] = True
                print(f"Táctica 'Asegurar Posesión' validada (Correlación DT: {corr_posesion:.2f})")
            else:
                print(f"Táctica 'Asegurar Posesión' rechazada: No encaja con el estilo histórico del DT ({corr_posesion:.2f})")
        except KeyError:
             modificadores_aprobados["asegurar_posesion"] = True # Aprobación por defecto si faltan datos
             
    # 2. Validación de Bandas vs Centro (Relación pases cruzados)
    if "atacar_por_bandas" in modificadores_propuestos:
        try:
            corr_bandas = matriz_pearson.loc[(nombre_dt, 'is_cross'), 'is_success']
            if corr_bandas > 0.05: 
                modificadores_aprobados["atacar_por_bandas"] = True
                print(f"Táctica 'Atacar por Bandas' validada (Correlación DT: {corr_bandas:.2f})")
            else:
                print(f"Táctica 'Atacar por Bandas' rechazada: El DT prefiere atacar por el carril central.")
        except KeyError:
             modificadores_aprobados["atacar_por_bandas"] = True
             
    # 3. Validación de Agresividad y Verticalidad
    if "agresividad_ofensiva" in modificadores_propuestos:
        # Se asume que casi todo entrenador acepta tácticas agresivas en zona final
        modificadores_aprobados["agresividad_ofensiva"] = True
        print("✅ Táctica 'Agresividad Ofensiva' validada (Universal).")

    return modificadores_aprobados

# Ejecución (Se asume que calculaste 'matriz_pearson' previamente en el código):
# tacticas_validadas = validar_estilo_entrenador(tacticas_bart, matriz_pearson, "André Jardine")
# validar_impacto_ab("América", equipo_rival, simulador, tacticas_validadas)

#%%

def evaluar_what_if_jugadas(df_completo, match_id_objetivo, motor_tactico, tacticas_ia, df_criticas):
    # 1. Asegurar cronología
    df_partido = df_completo[df_completo['match_id'] == match_id_objetivo].copy().sort_index()
    df_partido['segundo_absoluto'] = df_partido['time_diff'].cumsum()
    
    equipos = df_partido['team_name'].dropna().unique()
    equipo_local = equipos[0] if len(equipos) > 0 else "Local"
    equipo_visitante = equipos[1] if len(equipos) > 1 else "Visitante"
    
    print("\n=======================================================")
    print("     MICRO-SIMULADOR WHAT-IF (para jugadas valiosas")
    print("=======================================================")
    
    # 2. Iterar SOLAMENTE sobre las jugadas de mayor impacto
    for i, jugada in df_criticas.iterrows():
        tiempo_actual = jugada.get('segundo_absoluto', 0)
        minuto = int(tiempo_actual // 60)
        segundo = int(tiempo_actual % 60)
        equipo_accion = jugada.get('team_name', equipo_local)
        jugador_accion = jugada.get('player_name', 'Jugador')
        zona_inicio = int(jugada.get('zona_inicio', 14))
        
        # A. Rescatar la realidad (Marcador exacto hasta este segundo)
        historial = df_partido[df_partido['segundo_absoluto'] <= tiempo_actual]
        goles_loc = len(historial[(historial['team_name'] == equipo_local) & (historial['ev_shot'] == 1.0) & (historial['is_success'] == 1.0)])
        goles_vis = len(historial[(historial['team_name'] == equipo_visitante) & (historial['ev_shot'] == 1.0) & (historial['is_success'] == 1.0)])
        
        print(f"\n▶[Minuto {minuto:02d}:{segundo:02d}] Jugada Crítica de {jugador_accion} ({equipo_accion})")
        print(f"   Marcador Real: {equipo_local} {goles_loc} - {goles_vis} {equipo_visitante} | Zona de arranque: {zona_inicio}")
        print(f"   Métricas: OBV {jugada.get('obv',0):.3f} | xG {jugada.get('xg',0):.3f} | xT {jugada.get('xt',0):.3f}")
        
        # B. Simular el "What-If" para esta única posesión aplicando los consejos de la IA
        posesion_activa = True
        zona_actual = zona_inicio
        intentos_simulacion = 0
        
        while posesion_activa and intentos_simulacion < 10: # Límite de 10 pases para resolver la jugada
            intentos_simulacion += 1
            opciones = motor_tactico[motor_tactico['zona_inicio'] == zona_actual]
            
            if opciones.empty:
                posesion_activa = False
                continue
                
            mejor_opcion = opciones.sample(weights='conteo', replace=True).iloc[0]
            peligro = mejor_opcion['obv_promedio_zona']
            prob_exito = 0.82

            # Aplicar matemáticas de la IA
            multiplicador = 1.15 if "asegurar_posesion" in tacticas_ia else 1.0
            if "atacar_por_bandas" in tacticas_ia and mejor_opcion['zona_fin'] in [0, 6, 12, 18, 5, 11, 17, 23]:
                prob_exito = min(0.95, prob_exito * 1.15)
                peligro *= 1.10
                
            if random.random() < (prob_exito * multiplicador):
                zona_actual = int(mejor_opcion['zona_fin'])
                # Evaluar tiro
                if zona_actual in [4, 5, 10, 11, 16, 17, 22, 23]:
                    prob_tiro = 0.30 if "agresividad_ofensiva" in tacticas_ia else 0.15
                    prob_gol = max(0.02, peligro * 0.05) if "agresividad_ofensiva" in tacticas_ia else max(0.01, peligro * 0.025)
                    
                    if random.random() < prob_tiro:
                        if random.random() < prob_gol:
                            print("RESULTADO ALTERNO: ¡La sugerencia táctica de la IA culminó en GOL virtual!")
                        else:
                            print("RESULTADO ALTERNO: La jugada mejoró y terminó en un tiro a puerta (Atajado/Fuera).")
                        posesion_activa = False
            else:
                posesion_activa = False
                print("RESULTADO ALTERNO: La jugada fue interceptada por la defensa rival.")

def extraer_modificadores_de_texto(texto_sugerencias):
    """
    Traduce el texto de la IA a comandos matemáticos para el simulador de Markov.
    """
    texto = texto_sugerencias.lower()
    modificadores_activos = {}
    
    # Buscar patrones tácticos en la respuesta de Qwen
    if "pase" in texto or "posesión" in texto or "control" in texto:
        modificadores_activos["asegurar_posesion"] = True
    if "banda" in texto or "amplitud" in texto or "centro" in texto or "cruzado" in texto:
        modificadores_activos["atacar_por_bandas"] = True
    if "tiro" in texto or "disparo" in texto or "agresividad" in texto or "ofensiva" in texto:
        modificadores_activos["agresividad_ofensiva"] = True
        
    return modificadores_activos

# =============================================================================
# EJECUCIÓN DEL MICRO-SIMULADOR
# =============================================================================
# 1. Preparar la matriz del partido
df_partido_actual = df_test[df_test['match_id'] == mi_partido_id].copy().sort_index()
df_partido_actual['segundo_absoluto'] = df_partido_actual['time_diff'].cumsum()

# 2. Rescatar las jugadas valiosas de todo el partido (sin importar el minuto)
jugadas_criticas_reales = filtrar_jugadas_criticas(df_partido_actual, top_n=3)

# 3. Traducir sugerencias de Qwen a modificadores
tacticas_ia = extraer_modificadores_de_texto(sugerencias_finales)

# 4. Correr la evaluación jugada por jugada
evaluar_what_if_jugadas(df_test, mi_partido_id, simulador, tacticas_ia, jugadas_criticas_reales)