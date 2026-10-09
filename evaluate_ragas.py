"""
Evaluación del RAG con Ragas.

Modos:
  python evaluate_ragas.py --tag base                      # 4 métricas sobre eval_dataset.json
  python evaluate_ragas.py --rescore base                  # re-juzga solo las celdas NaN de 'base'
  python evaluate_ragas.py --tag prompt2 --reuse-context-from base
                                                           # tras cambiar el prompt: solo faithfulness
                                                           # y answer_relevancy (precision/recall se copian)
  python evaluate_ragas.py --tag k4 --k 4
  python evaluate_ragas.py --compare base prompt2          # tabla + gráfico
  python evaluate_ragas.py --from-log --last 30            # uso real (interactions.jsonl)

Cada variante de chunking usa su propia colección de Chroma (eval_c{size}_o{overlap}),
sin tocar el índice de producción. Env opcional: JUDGE_MODEL.
"""

import argparse
import ast
import json
import os
import time

import pandas as pd
from langchain_groq import ChatGroq
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
from ragas.run_config import RunConfig

import rag_core

DATASET_PATH = "eval_dataset.json"
LOG_FILE = "interactions.jsonl"
RESULTS_DIR = "eval_results"
METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
METRIC_OBJS = {
    "faithfulness": faithfulness,
    "answer_relevancy": answer_relevancy,
    "context_precision": context_precision,
    "context_recall": context_recall,
}
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "qwen/qwen3.8-27b")

ATRIBUCION = {
    "context_recall": "RECUPERACIÓN / CHUNKING: no se están trayendo los fragmentos que contienen la respuesta. "
                      "Mejoras: subir top_k, ajustar chunk_size/overlap, cambiar el modelo de embeddings, búsqueda híbrida.",
    "context_precision": "RECUPERACIÓN: llegan fragmentos irrelevantes o mal ordenados. "
                         "Mejoras: bajar top_k, chunks más pequeños, reranking.",
    "faithfulness": "GENERACIÓN: el modelo afirma cosas que no están en el contexto. "
                    "Mejoras: endurecer el prompt, más ejemplos few-shot, temperatura 0, exigir citas.",
    "answer_relevancy": "GENERACIÓN (o pregunta mal interpretada): la respuesta no atiende la pregunta o es redundante. "
                        "Mejoras: refinar el formato de salida, mejorar la reformulación de la pregunta.",
}


def diagnostico(medias: dict):
    validas = {k: v for k, v in medias.items() if v == v}  # descarta NaN
    if not validas:
        return
    peor = min(validas, key=validas.get)
    print(f"\nMétrica más baja: {peor} = {validas[peor]:.3f}")
    print(f"Componente probable -> {ATRIBUCION.get(peor, '-')}")


def load_dataset(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        items = json.load(f)
    pend = [i for i in items if i["pregunta"].startswith("<") or i["ground_truth"].startswith("<")]
    if pend:
        raise SystemExit(f"Hay {len(pend)} entradas con placeholders <...> en {path}.")
    return items


def get_store(chunk_size: int, overlap: int, rebuild: bool):
    name = f"eval_c{chunk_size}_o{overlap}"
    if not rebuild:
        store, emb = rag_core.load_existing_vector_store(collection_name=name)
        if store._collection.count() > 0:
            print(f"Usando colección existente '{name}' ({store._collection.count()} fragmentos).")
            return store, emb
    print(f"Construyendo colección '{name}'...")
    return rag_core.build_vector_store(
        progress_cb=print, chunk_size=chunk_size, chunk_overlap=overlap,
        collection_name=name, save_manifest=False,
    )


def ragas_scores(filas: list, metricas: list, emb_model) -> pd.DataFrame:
    """filas: [{user_input, retrieved_contexts, response, reference?}]"""
    extra = {"reasoning_format": "hidden"} if "qwen" in JUDGE_MODEL else {}
    judge = LangchainLLMWrapper(
        ChatGroq(model=JUDGE_MODEL, temperature=0.0,
                 api_key=rag_core.get_groq_api_key(),
                 **extra)
    )
    emb = LangchainEmbeddingsWrapper(emb_model)
    answer_relevancy.strictness = 1  # Groq no soporta n>1 generaciones por llamada
    result = evaluate(
        EvaluationDataset.from_list(filas),
        metrics=metricas,
        llm=judge,
        embeddings=emb,
        run_config=RunConfig(max_workers=2, timeout=240, max_retries=10),
    )
    return result.to_pandas()


# ───────────────────────── Modo 1: dataset completo ─────────────────────────

def run_eval(args):
    items = load_dataset(DATASET_PATH)
    store, emb_model = get_store(args.chunk_size, args.overlap, args.rebuild)
    llm = rag_core.get_llm()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    gen_path = os.path.join(RESULTS_DIR, f"{args.tag}_gen.json")

    if args.reuse_gen and os.path.isfile(gen_path):
        with open(gen_path, encoding="utf-8") as f:
            registros = json.load(f)
        print(f"Reutilizando generación guardada: {gen_path}")
    else:
        registros = []
        for i, it in enumerate(items, 1):
            r = rag_core.rag_pipeline(it["pregunta"], store, llm, k=args.k)
            registros.append({
                "pregunta": it["pregunta"], "ground_truth": it["ground_truth"],
                "en_corpus": it["en_corpus"], "respuesta": r["respuesta"],
                "contextos": [d.page_content for d in r["fragmentos"]],
                "sin_respuesta": r["sin_respuesta"],
            })
            print(f"[{i}/{len(items)}] {'OK ' if it['en_corpus'] else 'OOS'} {it['pregunta'][:70]}")
            time.sleep(args.sleep)
        with open(gen_path, "w", encoding="utf-8") as f:
            json.dump(registros, f, ensure_ascii=False, indent=1)
        print(f"Generación guardada en {gen_path}")

    dentro = [r for r in registros if r["en_corpus"]]
    fuera = [r for r in registros if not r["en_corpus"]]

    if args.reuse_context_from:
        nombres = ["faithfulness", "answer_relevancy"]
    else:
        nombres = METRICS

    df = ragas_scores(
        [{"user_input": r["pregunta"], "retrieved_contexts": r["contextos"],
          "response": r["respuesta"], "reference": r["ground_truth"]} for r in dentro],
        [METRIC_OBJS[m] for m in nombres],
        emb_model,
    )
    medias = {m: float(df[m].mean()) for m in METRICS if m in df.columns}

    if args.reuse_context_from:
        # Válido solo si k, chunk-size y overlap son iguales a los del tag de origen
        with open(os.path.join(RESULTS_DIR, f"{args.reuse_context_from}.json"), encoding="utf-8") as f:
            base = json.load(f)
        for m in ["context_precision", "context_recall"]:
            medias[m] = base["metricas"][m]

    abst = (sum(r["sin_respuesta"] for r in fuera) / len(fuera)) if fuera else None
    fallos = [r["pregunta"] for r in fuera if not r["sin_respuesta"]]
    falsos_rechazos = [r["pregunta"] for r in dentro if r["sin_respuesta"]]

    resumen = {
        "tag": args.tag,
        "config": {"k": args.k, "chunk_size": args.chunk_size, "overlap": args.overlap,
                   "embedding": rag_core.EMBEDDING_MODEL, "generador": rag_core.GROQ_MODEL,
                   "juez": JUDGE_MODEL},
        "metricas": medias,
        "nan_restantes": {m: int(df[m].isna().sum()) for m in nombres},
        "precision_recall_copiados_de": args.reuse_context_from,
        "abstencion_fuera_corpus": abst,
        "alucinaciones": fallos,
        "falsos_rechazos": falsos_rechazos,
        "n_dentro": len(dentro), "n_fuera": len(fuera),
    }
    with open(os.path.join(RESULTS_DIR, f"{args.tag}.json"), "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)
    df.to_csv(os.path.join(RESULTS_DIR, f"{args.tag}_detalle.csv"), index=False, encoding="utf-8")

    print("\n=== RESULTADOS", args.tag, "===")
    print(json.dumps(resumen, indent=2, ensure_ascii=False))
    diagnostico(medias)


# ───────────────────────── Modo 1b: completar celdas NaN ─────────────────────────

def run_rescore(tag: str):
    """Re-juzga solo las celdas NaN de <tag>_detalle.csv, métrica por métrica."""
    ruta = os.path.join(RESULTS_DIR, f"{tag}_detalle.csv")
    if not os.path.isfile(ruta):
        raise SystemExit(f"No existe {ruta}.")
    df = pd.read_csv(ruta)
    cols = [m for m in METRICS if m in df.columns]
    print("NaN antes:\n", df[cols].isna().sum(), "\n")

    emb_model = rag_core._get_embeddings()
    for m in cols:
        idxs = df.index[df[m].isna()]
        if len(idxs) == 0:
            continue
        print(f"Re-evaluando {m} en {len(idxs)} fila(s)...")
        filas = []
        for i in idxs:
            ctx = df.at[i, "retrieved_contexts"]
            if isinstance(ctx, str):
                ctx = ast.literal_eval(ctx)
            filas.append({
                "user_input": df.at[i, "user_input"],
                "retrieved_contexts": ctx,
                "response": df.at[i, "response"],
                "reference": df.at[i, "reference"],
            })
        nuevo = ragas_scores(filas, [METRIC_OBJS[m]], emb_model)
        if m in nuevo.columns:
            for i, v in zip(idxs, nuevo[m].tolist()):
                if pd.notna(v):
                    df.at[i, m] = v

    df.to_csv(ruta, index=False, encoding="utf-8")

    ruta_json = os.path.join(RESULTS_DIR, f"{tag}.json")
    with open(ruta_json, encoding="utf-8") as f:
        resumen = json.load(f)
    resumen["metricas"] = {m: float(df[m].mean()) for m in cols}
    resumen["nan_restantes"] = {m: int(df[m].isna().sum()) for m in cols}
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)

    print("\nMétricas actualizadas:", json.dumps(resumen["metricas"], indent=2))
    print("NaN restantes:", resumen["nan_restantes"])
    if any(resumen["nan_restantes"].values()):
        print("Quedan NaN: vuelve a ejecutar --rescore cuando haya cupo en Groq.")
    diagnostico(resumen["metricas"])


# ───────────────────────── Modo 2: registro de uso real ─────────────────────────

def run_log_eval(args):
    """faithfulness + answer_relevancy sobre interactions.jsonl (no requieren ground truth)."""
    if not os.path.isfile(LOG_FILE):
        raise SystemExit(f"No existe {LOG_FILE}. Usa la app local para generar interacciones primero.")
    with open(LOG_FILE, encoding="utf-8") as f:
        regs = [json.loads(l) for l in f if l.strip()]

    total = len(regs)
    rechazadas = sum(1 for r in regs if r.get("sin_respuesta"))
    evaluables = [r for r in regs if not r.get("sin_respuesta") and r.get("contextos")][-args.last:]
    if not evaluables:
        raise SystemExit("No hay interacciones con respuesta para evaluar.")
    print(f"{total} interacciones registradas, {rechazadas} sin información (abstenciones). "
          f"Evaluando {len(evaluables)}.")

    emb_model = rag_core._get_embeddings()
    df = ragas_scores(
        [{"user_input": r.get("pregunta_independiente") or r["pregunta"],
          "retrieved_contexts": r["contextos"], "response": r["respuesta"]} for r in evaluables],
        [faithfulness, answer_relevancy],
        emb_model,
    )
    df.insert(0, "fecha", [r["fecha"] for r in evaluables])
    df["k"] = [r["k"] for r in evaluables]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(os.path.join(RESULTS_DIR, "log_eval_detalle.csv"), index=False, encoding="utf-8")
    medias = {m: float(df[m].mean()) for m in ["faithfulness", "answer_relevancy"] if m in df.columns}
    resumen = {"n_interacciones": total, "n_abstenciones": rechazadas,
               "n_evaluadas": len(evaluables), "metricas": medias}
    with open(os.path.join(RESULTS_DIR, "log_eval.json"), "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)
    print(json.dumps(resumen, indent=2, ensure_ascii=False))

    bajas = df[df["faithfulness"] < 0.7] if "faithfulness" in df.columns else df.iloc[0:0]
    if len(bajas):
        print(f"\n{len(bajas)} respuesta(s) con faithfulness < 0.7 (posible alucinación). "
              f"Revísalas en {RESULTS_DIR}/log_eval_detalle.csv")


# ───────────────────────── Comparación antes/después ─────────────────────────

def compare(tag_a: str, tag_b: str):
    def cargar(t):
        with open(os.path.join(RESULTS_DIR, f"{t}.json"), encoding="utf-8") as f:
            return json.load(f)

    a, b = cargar(tag_a), cargar(tag_b)
    filas = []
    for m in METRICS:
        va, vb = a["metricas"].get(m), b["metricas"].get(m)
        filas.append([m, va, vb, (vb - va) if va is not None and vb is not None else None])
    aa, ab = a["abstencion_fuera_corpus"], b["abstencion_fuera_corpus"]
    filas.append(["abstencion_fuera_corpus", aa, ab, (ab - aa) if aa is not None and ab is not None else None])
    df = pd.DataFrame(filas, columns=["metrica", tag_a, tag_b, "delta"])

    print("\nConfig A:", a["config"])
    print("Config B:", b["config"], "\n")
    try:
        print(df.to_markdown(index=False, floatfmt=".3f"))
    except ImportError:
        print(df.round(3))
    df.to_csv(os.path.join(RESULTS_DIR, f"comparacion_{tag_a}_vs_{tag_b}.csv"), index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        d = df.dropna()
        x = range(len(d))
        w = 0.38
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.bar([i - w / 2 for i in x], d[tag_a], w, label=tag_a)
        ax.bar([i + w / 2 for i in x], d[tag_b], w, label=tag_b)
        ax.set_xticks(list(x))
        ax.set_xticklabels(d["metrica"], rotation=15, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Puntaje")
        ax.set_title(f"Métricas Ragas: {tag_a} vs {tag_b}")
        ax.legend()
        fig.tight_layout()
        out = os.path.join(RESULTS_DIR, f"comparacion_{tag_a}_vs_{tag_b}.png")
        fig.savefig(out, dpi=150)
        print(f"\nGráfico guardado en {out}")
    except ImportError:
        print("\n(instala matplotlib para generar el gráfico)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="base")
    ap.add_argument("--k", type=int, default=rag_core.DEFAULT_K)
    ap.add_argument("--chunk-size", type=int, default=rag_core.CHUNK_SIZE)
    ap.add_argument("--overlap", type=int, default=rag_core.CHUNK_OVERLAP)
    ap.add_argument("--rebuild", action="store_true", help="Reindexa aunque la colección exista")
    ap.add_argument("--sleep", type=float, default=2.0, help="Pausa entre preguntas (límites de Groq)")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument("--from-log", action="store_true", help="Evalúa interactions.jsonl")
    ap.add_argument("--last", type=int, default=30, help="Con --from-log: últimas N interacciones")
    ap.add_argument("--rescore", metavar="TAG", help="Re-juzga solo las celdas NaN de ese tag")
    ap.add_argument("--reuse-gen", action="store_true", help="Reutiliza <tag>_gen.json si existe")
    ap.add_argument("--reuse-context-from", metavar="TAG",
                    help="Solo juzga faithfulness y answer_relevancy; copia precision/recall de ese tag "
                         "(solo válido si k, chunk-size y overlap no cambian)")
    args = ap.parse_args()

    if args.compare:
        compare(*args.compare)
    elif args.rescore:
        run_rescore(args.rescore)
    elif args.from_log:
        run_log_eval(args)
    else:
        run_eval(args)