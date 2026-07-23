from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
import re
import asyncio
import httpx
from typing import Optional

app = FastAPI(title="RADAR Politico Engine", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RequestPayload(BaseModel):
    nombre: str
    fecha: Optional[str] = "julio 2026"
    forceRefresh: Optional[bool] = False

def limpiar_texto(texto: str) -> str:
    if not texto:
        return ""
    texto = re.sub(r'http\S+', '', texto)
    texto = re.sub(r'\s+', ' ', texto)
    return texto.strip()

async def extraer_datos_apify(nombre: str, fecha_ctx: str):
    apify_token = os.getenv("APIFY_API_TOKEN")
    if not apify_token:
        return "No hay token de Apify. Generando análisis por contexto."

    async with httpx.AsyncClient(timeout=45.0) as client:
        task_tweets = client.post(
            f"https://api.apify.com/v2/acts/apidojo~twitter-scraper-lite/run-sync-get-dataset-items?token={apify_token}",
            json={
                "searchTerms": [f"{nombre}", f"{nombre} oposicion", f"{nombre} gobierno"],
                "sort": "Latest",
                "maxItems": 25,
                "tweetLanguage": "es"
            }
        )
        task_noticias = client.post(
            f"https://api.apify.com/v2/acts/apify~google-search-scraper/run-sync-get-dataset-items?token={apify_token}",
            json={
                "queries": f"{nombre} noticias columna opinion {fecha_ctx}\n{nombre} oposicion denuncias edomex mexico",
                "resultsPerPage": 20,
                "maxPagesPerQuery": 1,
                "languageCode": "es",
                "countryCode": "mx"
            }
        )
        task_fb = client.post(
            f"https://api.apify.com/v2/acts/apify~facebook-posts-scraper/run-sync-get-dataset-items?token={apify_token}",
            json={"searchTerm": nombre, "maxPosts": 20}
        )

        res_tweets, res_noticias, res_fb = await asyncio.gather(
            task_tweets, task_noticias, task_fb, return_exceptions=True
        )

    tweets_clean = []
    if not isinstance(res_tweets, Exception) and res_tweets.status_code == 200:
        for t in res_tweets.json()[:20]:
            txt = limpiar_texto(t.get("text") or t.get("fullText") or "")
            if txt:
                tweets_clean.append(f"[@{t.get('author',{}).get('userName','anon')}]: {txt}")

    noticias_clean = []
    if not isinstance(res_noticias, Exception) and res_noticias.status_code == 200:
        for item in res_noticias.json():
            for r in item.get("organicResults", [])[:15]:
                title = limpiar_texto(r.get("title", ""))
                desc = limpiar_texto(r.get("description", ""))
                noticias_clean.append(f"TITULAR: {title}\nRESUMEN: {desc}")

    fb_clean = []
    if not isinstance(res_fb, Exception) and res_fb.status_code == 200:
        for f in res_fb.json()[:15]:
            txt = limpiar_texto(f.get("text") or f.get("caption") or "")
            if txt:
                fb_clean.append(f"[FB Post]: {txt}")

    return (
        f"=== CONTEXTO EXTRAÍDO Y LIMPIO PARA '{nombre}' ===\n\n"
        f"--- TWITTER/X ---\n" + "\n".join(tweets_clean) + "\n\n"
        f"--- PRENSA Y MEDIOS ---\n" + "\n---\n".join(noticias_clean) + "\n\n"
        f"--- FACEBOOK ---\n" + "\n".join(fb_clean)
    )

@app.post("/api/analizar")
async def analizar_actor(payload: RequestPayload):
    nombre = payload.nombre.strip()
    fecha_ctx = payload.fecha or "julio 2026"

    contexto = await extraer_datos_apify(nombre, fecha_ctx)
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_key:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY requerida")

    prompt = f"""Eres un Director General de Inteligencia Político-Digital. Fecha: {fecha_ctx}.

DATOS REALES EXTRAÍDOS Y PROCESADOS EN PYTHON PARA "{nombre}":
{contexto}

INSTRUCCIÓN: Devuelve UNICAMENTE un JSON estructurado con explicaciones exhaustivas (2 a 3 párrafos en campos explicativos) para no omitir ningún análisis.

Estructura JSON obligatoria:
{{
  "nombre": "{nombre}",
  "cargo": "Cargo oficial a {fecha_ctx} · Entidad / Partido",
  "fecha_analisis": "{fecha_ctx}",
  "tags": ["Tag1", "Tag2", "Tag3", "Tag4", "Tag5"],
  "kpis": [
    {{"label": "SEGUIDORES TOTALES", "valor": "X.XM", "nota": "Alcance estimado", "tipo": "acc"}},
    {{"label": "APROBACIÓN DIGITAL", "valor": "XX%", "nota": "Proporción favorable", "tipo": "suc"}},
    {{"label": "PANTALLAS DE CRISIS", "valor": "X", "nota": "Temas de tensión", "tipo": "dan"}},
    {{"label": "MECANISMO NARRATIVO", "valor": "XX/XX", "nota": "Propia vs Impuesta", "tipo": "gld"}},
    {{"label": "SENTIMIENTO POSITIVO", "valor": "XX%", "nota": "Aceptación neta", "tipo": "suc"}},
    {{"label": "TENDENCIA", "valor": "Alta / Estable", "nota": "Evolución", "tipo": "acc"}}
  ],
  "vision_general": {{
    "resumen_ejecutivo": "Escribe 2 a 3 párrafos extensos analizando a fondo la postura estratégica del personaje, principales ataques y balance general.",
    "sentimiento": [
      {{"label": "Positivo", "pct": 38}},
      {{"label": "Neutro/Informativo", "pct": 30}},
      {{"label": "Negativo", "pct": 22}},
      {{"label": "Polarizado", "pct": 10}}
    ],
    "temas": [
      {{"tema": "Tema principal 1", "pct": 35}},
      {{"tema": "Tema principal 2", "pct": 22}},
      {{"tema": "Tema principal 3", "pct": 15}},
      {{"tema": "Tema principal 4", "pct": 12}},
      {{"tema": "Tema principal 5", "pct": 9}},
      {{"tema": "Tema principal 6", "pct": 7}}
    ],
    "plataformas": [
      {{"nombre": "Facebook", "pct": 38, "tono_positivo": 45, "tono_negativo": 30}},
      {{"nombre": "X/Twitter", "pct": 28, "tono_positivo": 25, "tono_negativo": 60}},
      {{"nombre": "Noticias/Medios", "pct": 18, "tono_positivo": 30, "tono_negativo": 45}},
      {{"nombre": "Google Search", "pct": 10, "tono_positivo": 40, "tono_negativo": 35}},
      {{"nombre": "Instagram", "pct": 6, "tono_positivo": 50, "tono_negativo": 20}}
    ]
  }},
  "actores_politicos": {{
    "explicacion_ecosistema": "Análisis extenso de la relación del político con medios nacionales, locales, oposición y opinión pública.",
    "analisis_actores": [
      {{
        "categoria": "Prensa Nacional & Columnistas",
        "impacto": "Alto",
        "narrativa_dominante": "Explicación detallada del tratamiento mediático.",
        "tendencia_actitud": "Desfavorable (60%) / Neutro (40%)"
      }},
      {{
        "categoria": "Prensa Local & Portales Regionales",
        "impacto": "Medio",
        "narrativa_dominante": "Explicación detallada de la cobertura local.",
        "tendencia_actitud": "Favorable (70%)"
      }},
      {{
        "categoria": "Oposición & Voceros Críticos",
        "impacto": "Crítico",
        "narrativa_dominante": "Análisis de ataques y líneas de la oposición.",
        "tendencia_actitud": "Adverso (90%)"
      }},
      {{
        "categoria": "Ecosistema Ciudadano & Digital",
        "impacto": "Alto",
        "narrativa_dominante": "Sentimiento en comentarios de redes masivas.",
        "tendencia_actitud": "Dividido / Polarizado"
      }}
    ],
    "cruces_bivariados": [
      {{
        "eje_x": "Plataforma (X vs FB)",
        "eje_y": "Inclinación del Tono",
        "hallazgo": "Explicación analítica del comportamiento diferenciado por red."
      }},
      {{
        "eje_x": "Sentimiento",
        "eje_y": "Ejes Temáticos Clave",
        "hallazgo": "Explicación sobre qué temas generan apoyo o rechazo."
      }}
    ]
  }},
  "segmentacion_demografica": {{
    "analisis_demografico": "Análisis extenso del perfil sociodemográfico por género y grupos de edad.",
    "por_genero": [
      {{"segmento": "Hombres", "positivo": 35, "neutro": 30, "negativo": 35}},
      {{"segmento": "Mujeres", "positivo": 30, "neutro": 28, "negativo": 42}}
    ],
    "por_edad": [
      {{"segmento": "18-29 años", "positivo": 25, "neutro": 25, "negativo": 50}},
      {{"segmento": "30-44 años", "positivo": 35, "neutro": 30, "negativo": 35}},
      {{"segmento": "45-59 años", "positivo": 45, "neutro": 30, "negativo": 25}},
      {{"segmento": "60+ años", "positivo": 50, "neutro": 28, "negativo": 22}}
    ]
  }},
  "mapa_narrativas": {{
    "explicacion_narrativas": "Análisis detallado sobre el combate narrativo entre discurso propio y acusaciones.",
    "favorables": [
      {{"titulo": "Narrativa A favor 1", "descripcion": "Texto extenso explicando el impacto y fuente."}},
      {{"titulo": "Narrativa A favor 2", "descripcion": "Texto extenso explicando el impacto y fuente."}},
      {{"titulo": "Narrativa A favor 3", "descripcion": "Texto extenso explicando el impacto y fuente."}}
    ],
    "criticas": [
      {{"titulo": "Narrativa En contra 1", "descripcion": "Texto extenso del señalamiento crítico."}},
      {{"titulo": "Narrativa En contra 2", "descripcion": "Texto extenso del señalamiento crítico."}},
      {{"titulo": "Narrativa En contra 3", "descripcion": "Texto extenso del señalamiento crítico."}}
    ],
    "neutras": [
      {{"titulo": "Narrativa Neutra 1", "descripcion": "Análisis de temas informativos."}},
      {{"titulo": "Narrativa Neutra 2", "descripcion": "Análisis de temas informativos."}}
    ]
  }},
  "cronologia_eventos": {{
    "analisis_coyuntural": "Explicación extensa sobre cómo la coyuntura reciente impactó la reputación.",
    "eventos": [
      {{"fecha": "Fecha/Periodo", "badge": "EVENTO DESTACADO", "evento": "Hito 1", "lectura": "Explicación estratégica profunda."}},
      {{"fecha": "Fecha/Periodo", "badge": "PANTALLA DE CRISIS", "evento": "Hito 2", "lectura": "Explicación estratégica profunda."}},
      {{"fecha": "Fecha/Periodo", "badge": "EVENTO DESTACADO", "evento": "Hito 3", "lectura": "Explicación estratégica profunda."}},
      {{"fecha": "Fecha/Periodo", "badge": "PANTALLA DE CRISIS", "evento": "Hito 4", "lectura": "Explicación estratégica profunda."}}
    ]
  }},
  "riesgos_oportunidades": {{
    "dictamen_estrategico": "Dictamen analítico ejecutivo para la toma de decisiones.",
    "riesgos": [
      {{"nivel": "CRÍTICO", "titulo": "Riesgo 1", "descripcion": "Explicación extensa del riesgo."}},
      {{"nivel": "ALTO", "titulo": "Riesgo 2", "descripcion": "Explicación extensa del riesgo."}},
      {{"nivel": "MEDIO", "titulo": "Riesgo 3", "descripcion": "Explicación extensa del riesgo."}}
    ],
    "oportunidades": [
      {{"nivel": "ALTO", "titulo": "Oportunidad 1", "descripcion": "Análisis de la ventaja estratégica."}},
      {{"nivel": "MEDIO", "titulo": "Oportunidad 2", "descripcion": "Análisis de la ventaja estratégica."}}
    ]
  }}
}}"""

    async with httpx.AsyncClient(timeout=60.0) as client:
        res = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {openrouter_key}", "Content-Type": "application/json"},
            json={
                "model": "openai/gpt-4o",
                "max_tokens": 7500,
                "messages": [{"role": "user", "content": prompt}]
            }
        )

    if res.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Error OpenRouter: {res.text}")

    raw_text = res.json()["choices"][0]["message"]["content"]
    cleaned = re.sub(r'```json\s*', '', raw_text)
    cleaned = re.sub(r'```\s*$', '', cleaned).strip()

    match = re.search(r'\{[\s\S]*\}', cleaned)
    if not match:
        raise HTTPException(status_code=500, detail="Respuesta no válida")

    return json.loads(match.group(0))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
