"""
Evaluación del agente (Avance 3): casos de comportamiento y costo.

  python evaluate_agent.py --casos            # casos_comportamiento.json -> eval_results/casos_comportamiento.*
  python evaluate_agent.py --costo --n 8      # pipeline fijo vs agente en las mismas N preguntas
  python evaluate_agent.py --tools            # prueba las tools sin LLM (.invoke)

Las métricas de Ragas del agente se calculan con evaluate_ragas.py:
  python evaluate_ragas.py --tag agente --pipeline agente --sleep 25
  python evaluate_ragas.py --compare formatos agente

El comportamiento esperado de cada caso está escrito en casos_comportamiento.json
ANTES de ejecutarlo; el script solo compara lo obtenido con lo esperado.
"""

import argparse
import json
import os
import sys
import time

from langchain_core.callbacks import BaseCallbackHandler

RESULTS_DIR = "eval_results"


class ContadorLLM(BaseCallbackHandler):
    """Cuenta llamadas al LLM y tokens (entrada + salida) de un ChatGroq."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.llamadas = 0
        self.tokens = 0
        self.tokens_entrada = 0
        self.tokens_salida = 0

    def on_llm_end(self, response, **kwargs):
        self.llamadas += 1
        for gens in response.generations:
            for g in gens:
                u = getattr(getattr(g, "message", None), "usage_metadata", None) or {}
                self.tokens_entrada += int(u.get("input_tokens", 0) or 0)
                self.tokens_salida += int(u.get("output_tokens", 0) or 0)
        self.tokens = self.tokens_entrada + self.tokens_salida


def _usos(r) -> dict:
    out = {}
    for h in r.get("herramientas", []):
        out[h["nombre"]] = out.get(h["nombre"], 0) + 1
    return out


def verificar(caso: dict, r: dict) -> list:
    """Devuelve la lista de fallos (vacía si cumple lo esperado)."""
    import herramientas

    v, fallos = caso["verificar"], []
    usos = _usos(r)
    texto = r["respuesta"]
    if "max_tools" in v and len(r.get("herramientas", [])) > v["max_tools"]:
        fallos.append(f"usó {len(r['herramientas'])} tools (máx. {v['max_tools']})")
    for t, n in v.get("min_usos", {}).items():
        if usos.get(t, 0) < n:
            fallos.append(f"{t} se usó {usos.get(t, 0)} veces (mín. {n})")
    for t, n in v.get("max_usos", {}).items():
        if usos.get(t, 0) > n:
            fallos.append(f"{t} se usó {usos.get(t, 0)} veces (máx. {n})")
    for t, esperados in v.get("tool_args", {}).items():
        llamadas = [h["args"] for h in r.get("herramientas", []) if h["nombre"] == t]
        if not any(all(str(a.get(k)) == str(val) for k, val in esperados.items()) for a in llamadas):
            fallos.append(f"no llamó {t} con {esperados} (llamadas: {llamadas})")
    if v.get("cita_base") and "Fuente:" not in texto:
        fallos.append("no cita la base")
    if v.get("cita_web") and "web:" not in texto.lower():
        fallos.append("no marca la fuente web")
    if v.get("web_en_dominios_permitidos"):
        malos = [w["url"] for w in r.get("fuentes_web", []) if not herramientas._dominio_permitido(w["url"])]
        if malos:
            fallos.append(f"resultados web fuera de la lista: {malos}")
    if "max_chars_respuesta" in v and len(texto) > v["max_chars_respuesta"]:
        fallos.append(f"respuesta de {len(texto)} caracteres (máx. {v['max_chars_respuesta']})")
    for s in v.get("no_contiene", []):
        if s in texto:
            fallos.append(f"la respuesta contiene '{s}'")
    if "sin_respuesta" in v and bool(r["sin_respuesta"]) != v["sin_respuesta"]:
        fallos.append(f"sin_respuesta={r['sin_respuesta']} (esperado {v['sin_respuesta']})")
    return fallos


def _preparar():
    import rag_core

    store = rag_core.ensure_vector_store(progress_cb=print)
    llm = rag_core.get_llm()
    contador = ContadorLLM()
    llm.callbacks = [contador]
    return rag_core, store, llm, contador


def run_casos(args):
    rag_core, store, llm, contador = _preparar()
    with open("casos_comportamiento.json", encoding="utf-8") as f:
        casos = json.load(f)
    usar_web = rag_core.web_search_disponible()
    if not usar_web:
        print("AVISO: no hay TAVILY_API_KEY; el caso de información pública fallará.")

    filas = []
    for c in casos:
        contador.reset()
        r = rag_core.agent_pipeline(c["pregunta"], store, llm, usar_web=usar_web)
        fallos = verificar(c, r)
        fila = {
            "id": c["id"], "tipo": c["tipo"], "pregunta": c["pregunta"],
            "esperado": c["esperado"],
            "obtenido": r["respuesta"],
            "tools": [f"{h['nombre']}({', '.join(f'{k}={v!r}' for k, v in h['args'].items())})"
                      for h in r.get("herramientas", [])],
            "fuentes_web": [w["url"] for w in r.get("fuentes_web", [])],
            "llamadas_llm": r.get("llamadas_llm"),
            "tokens": contador.tokens,
            "tiempo_s": r.get("tiempo_s"),
            "cumple": not fallos,
            "fallos": fallos,
        }
        filas.append(fila)
        print(f"\n[{c['id']}] {c['tipo']} -> {'CUMPLE' if not fallos else 'NO CUMPLE'}")
        print("  Tools:", fila["tools"] or "ninguna", "| llamadas LLM:", fila["llamadas_llm"])
        print("  Respuesta:", r["respuesta"][:300].replace("\n", " "))
        for x in fallos:
            print("  -", x)
        time.sleep(args.sleep)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "casos_comportamiento.json"), "w", encoding="utf-8") as f:
        json.dump(filas, f, ensure_ascii=False, indent=1)
    lineas = ["| # | Tipo | Pregunta | Esperado | Obtenido | Tools usadas | Llam. LLM | ¿Cumple? |",
              "|---|---|---|---|---|---|---|---|"]
    for x in filas:
        obt = x["obtenido"].replace("\n", " ").replace("|", "/")
        lineas.append(
            f"| {x['id']} | {x['tipo']} | {x['pregunta']} | {x['esperado']} | "
            f"{obt[:220]}{'…' if len(obt) > 220 else ''} | {'<br>'.join(x['tools']) or 'ninguna'} | "
            f"{x['llamadas_llm']} | {'Sí' if x['cumple'] else 'No'} |")
    with open(os.path.join(RESULTS_DIR, "casos_comportamiento.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lineas) + "\n")
    print(f"\n{sum(x['cumple'] for x in filas)}/{len(filas)} casos cumplen. "
          f"Resultados en {RESULTS_DIR}/casos_comportamiento.json y .md")


def run_costo(args):
    rag_core, store, llm, contador = _preparar()
    with open("eval_dataset.json", encoding="utf-8") as f:
        preguntas = [i["pregunta"] for i in json.load(f) if i["en_corpus"]][: args.n]

    filas = []
    for i, q in enumerate(preguntas, 1):
        for pipeline in ("fijo", "agente"):
            contador.reset()
            t0 = time.perf_counter()
            if pipeline == "fijo":
                r = rag_core.rag_pipeline(q, store, llm)
            else:
                r = rag_core.agent_pipeline(q, store, llm, usar_web=False, registrar=False)
            filas.append({"pregunta": q, "pipeline": pipeline, "llamadas_llm": contador.llamadas,
                          "tokens": contador.tokens, "tiempo_s": round(time.perf_counter() - t0, 2),
                          "n_tools": len(r.get("herramientas", []))})
            print(f"[{i}/{len(preguntas)}] {pipeline:6} llamadas={contador.llamadas} "
                  f"tokens={contador.tokens} t={filas[-1]['tiempo_s']}s")
            time.sleep(args.sleep)

    resumen = {}
    for p in ("fijo", "agente"):
        fs = [f for f in filas if f["pipeline"] == p]
        resumen[p] = {c: round(sum(f[c] for f in fs) / len(fs), 2)
                      for c in ("llamadas_llm", "tokens", "tiempo_s", "n_tools")}
    resumen["n_preguntas"] = len(preguntas)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "costo_fijo_vs_agente.json"), "w", encoding="utf-8") as f:
        json.dump({"resumen": resumen, "detalle": filas}, f, ensure_ascii=False, indent=1)
    print("\n| Medida (promedio por pregunta) | Pipeline fijo | Agente |\n|---|---|---|")
    for c, n in (("llamadas_llm", "Llamadas al LLM"), ("tokens", "Tokens"), ("tiempo_s", "Tiempo (s)")):
        print(f"| {n} | {resumen['fijo'][c]} | {resumen['agente'][c]} |")
    print(f"Preguntas: {len(preguntas)}. Guardado en {RESULTS_DIR}/costo_fijo_vs_agente.json")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--casos", action="store_true")
    ap.add_argument("--costo", action="store_true")
    ap.add_argument("--tools", action="store_true", help="Prueba las tools sin LLM")
    ap.add_argument("--n", type=int, default=8, help="Preguntas para medir el costo")
    ap.add_argument("--sleep", type=float, default=8.0, help="Pausa entre preguntas (límites de Groq)")
    args = ap.parse_args()
    if args.tools:
        import runpy
        runpy.run_module("herramientas", run_name="__main__")
    if args.casos:
        run_casos(args)
    if args.costo:
        run_costo(args)
    if not (args.tools or args.casos or args.costo):
        ap.print_help()
