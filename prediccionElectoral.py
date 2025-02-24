import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from loguru import logger
import tiktoken
import time
import pandas as pd
import plotly.express as px

# Cargar variables de entorno y cliente
load_dotenv()
qclient = Groq(api_key=os.getenv('GROQ_API_KEY'))

# Configuración de la página
st.set_page_config(
    page_title="Predicción Electoral",
    page_icon="🇪🇨",
    layout="wide"
)

st.title("🗳️ Predicción Electoral Ecuador")

# Inicialización de variables de estado
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []
if "datos_procesados" not in st.session_state:
    st.session_state.datos_procesados = None
if "tiene_resultados" not in st.session_state:
    st.session_state.tiene_resultados = False

def procesar_respuesta(respuesta_chat):
    return ''.join(
        chunk.choices[0].delta.content or ''
        for chunk in respuesta_chat
    ).strip()

def dividir_texto(texto, max_tokens=4000):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(texto)
    return [tokenizer.decode(tokens[i:i + max_tokens]) 
            for i in range(0, len(tokens), max_tokens)]

def analizar_texto_fila(texto):
    try:
        chunks_texto = dividir_texto(texto)
        respuestas = []

        for chunk in chunks_texto:
            time.sleep(1)
            respuesta_stream = qclient.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Analiza el texto y determina si expresa apoyo a Noboa, Luisa González o es un voto nulo/indeciso. Responde solo con: 'Noboa', 'Luisa' o 'Nulo'."},
                    {"role": "user", "content": str(chunk)},
                ],
                model="llama3-8b-8192",
                stream=True
            )
            respuestas.append(procesar_respuesta(respuesta_stream))

        if len(respuestas) > 1:
            conteos = {
                'Noboa': respuestas.count('Noboa'),
                'Luisa': respuestas.count('Luisa'),
                'Nulo': respuestas.count('Nulo')
            }
            return max(conteos, key=conteos.get)
        return respuestas[0] if respuestas[0] in ['Noboa', 'Luisa', 'Nulo'] else 'Nulo'
            
    except Exception as e:
        logger.error(f"Error al analizar texto: {e}")
        return 'Nulo'

def procesar_archivo_excel(archivo, tamano_muestra=None):
    try:
        df = pd.read_excel(archivo)
        if tamano_muestra and tamano_muestra < len(df):
            df = df.sample(n=tamano_muestra, random_state=42)
        return df.reset_index(drop=True)
    except Exception as e:
        st.error(f"Error al procesar el archivo Excel: {str(e)}")
        return None

# Configuración de sidebar
st.sidebar.header("Configuración")
archivo_subido = st.sidebar.file_uploader("Cargar archivo Excel", type=['xlsx', 'xls'])
tamano_muestra = st.sidebar.number_input(
    "Tamaño de la muestra, recomendada 400 para 11000 registros",
    min_value=50,
    value=400
)

if archivo_subido:
    df_preview = pd.read_excel(archivo_subido)
    columna_analizar = st.sidebar.selectbox(
        "Selecciona la columna a analizar",
        options=df_preview.columns.tolist()
    )

    if st.sidebar.button("Procesar Datos"):
        with st.spinner("Procesando datos..."):
            tamano_muestra = None if tamano_muestra == 0 else tamano_muestra
            df = procesar_archivo_excel(archivo_subido, tamano_muestra)
            
            if df is not None:
                barra_progreso = st.progress(0)
                predicciones = []
                
                for i, fila in df.iterrows():
                    predicciones.append(analizar_texto_fila(str(fila[columna_analizar])))
                    barra_progreso.progress((i + 1) / len(df))
                
                df['prediccion'] = predicciones
                st.session_state.datos_procesados = df
                st.session_state.tiene_resultados = True

if st.session_state.tiene_resultados:
    df = st.session_state.datos_procesados
    resultados = df['prediccion'].value_counts()
    
    # Construccion del grafico para resultados
    fig = px.bar(
        x=resultados.index,
        y=resultados.values,
        title="Distribución de Predicciones",
        labels={'x': 'Candidato', 'y': 'Número de Votos'},
        color=resultados.index,
        color_discrete_map={
            'Noboa': '#5D9E9B',
            'Luisa': '#FF69B4',
            'Nulo': '#808080'
        }
    )
    st.plotly_chart(fig)
    
    # Métricas a mostras
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Votos Noboa", resultados.get('Noboa', 0))
    with col2: st.metric("Votos Luisa", resultados.get('Luisa', 0))
    with col3: st.metric("Votos Nulos", resultados.get('Nulo', 0))
    
    # Presentar la Conclusión de los datos obtenidos
    total_votos = resultados.sum()
    ganador = resultados.index[0]
    porcentaje_ganador = (resultados[ganador] / total_votos) * 100
    
    st.markdown(f"""
    ### Conclusión
    Según el análisis realizado:
    - El candidato con más apoyo es **{ganador}** con {porcentaje_ganador:.1f}% de los votos.
    - Se encontraron {resultados.get('Nulo', 0)} votos nulos/indecisos.
    """)
    
    st.subheader("Datos Procesados")
    st.dataframe(df)
    
    # Sección de preguntas para despues de obtener los datos y las graficas
    st.markdown("### Preguntas sobre los datos")
    pregunta_usuario = st.text_input("Haz una pregunta sobre los resultados:")
    
    if pregunta_usuario:
        with st.chat_message('user'):
            st.markdown(pregunta_usuario)
            
        with st.chat_message('assistant'):
            contexto = f"""
            Resultados de la predicción electoral:
            - Votos Noboa: {resultados.get('Noboa', 0)}
            - Votos Luisa: {resultados.get('Luisa', 0)}
            - Votos Nulos: {resultados.get('Nulo', 0)}
            - Total de votos: {total_votos}
            """
            
            chunks_contexto = dividir_texto(contexto)
            respuestas = []
            
            for chunk in chunks_contexto:
                time.sleep(3)
                respuesta_stream = qclient.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "Eres un asistente experto en análisis electoral. Responde en español latino basándote en la información proporcionada."},
                        {"role": "user", "content": f"Contexto: {chunk}\n\nPregunta: {pregunta_usuario}"},
                    ],
                    model="llama3-8b-8192",
                    stream=True,
                    temperature=0
                )
                
                respuesta_chunk = procesar_respuesta(respuesta_stream)
                if respuesta_chunk.strip():
                    respuestas.append(respuesta_chunk)
            
            if len(respuestas) > 1:
                contexto_combinado = "\n".join(respuestas)
                prompt_final = f"""
                Basándote en las siguientes respuestas parciales:
                {contexto_combinado}
                Por favor proporciona una respuesta final sintetizada y coherente a la pregunta: {pregunta_usuario}
                """
                
                respuesta_stream = qclient.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "Sintetiza las respuestas parciales en una respuesta final coherente. Responde en español latino."},
                        {"role": "user", "content": prompt_final},
                    ],
                    model="llama3-8b-8192",
                    stream=True,
                    temperature=0
                )
                
                respuesta_final = procesar_respuesta(respuesta_stream)
                st.write(respuesta_final)
                st.session_state.mensajes.extend([
                    {'role': 'user', 'content': pregunta_usuario},
                    {'role': 'assistant', 'content': respuesta_final}
                ])
            else:
                respuesta_texto = respuestas[0] if respuestas else "No pude procesar la pregunta. ¿Podrías reformularla?"
                st.write(respuesta_texto)
                st.session_state.mensajes.extend([
                    {'role': 'user', 'content': pregunta_usuario},
                    {'role': 'assistant', 'content': respuesta_texto}
                ])
