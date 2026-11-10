"""
Prueba de las tools SIN LLM (Avance 3, punto 1 y 2).

    python probar_tools.py

Llama a cada tool con .invoke(), igual que lo hace el agente, pero a mano:
no usa Groq. Imprime lo que devuelve la tool para la captura del informe y
comprueba con asserts los límites escritos en el código.
La web solo se prueba si hay TAVILY_API_KEY en el .env.
"""

import json

import rag_core as rc

SEP = "=" * 70


def main():
    store = rc.ensure_vector_store(progress_cb=print)
    registro = {"docs": [], "web": []}
    tools = {t.name: t for t in rc.crear_tools(store, k=15, usar_web=rc.web_search_disponible(), registro=registro)}
    print("\nTools disponibles:", list(tools))

    # ── buscar_normativa ──
    print(f"\n{SEP}\nbuscar_normativa.invoke (k pedido = 15, límite en código = {rc.MAX_FRAGMENTOS_POR_LLAMADA})\n{SEP}")
    args = {"consulta": "¿Qué plazo tiene la administración para responder un derecho de petición?"}
    print("args:", json.dumps(args, ensure_ascii=False))
    salida = tools["buscar_normativa"].invoke(args)
    print(salida[:2500], "\n[... salida recortada para imprimir ...]" if len(salida) > 2500 else "")
    n = len(registro["docs"])
    maximo = max(len(d.page_content) for d in registro["docs"])
    print(f"\nfragmentos devueltos: {n} (máximo {rc.MAX_FRAGMENTOS_POR_LLAMADA})")
    print(f"fragmento más largo: {maximo} caracteres (máximo {rc.MAX_CHARS_FRAGMENTO + 1})")
    assert n <= rc.MAX_FRAGMENTOS_POR_LLAMADA, "se excedió el máximo de fragmentos"
    assert maximo <= rc.MAX_CHARS_FRAGMENTO + 1, "se excedió el largo de fragmento"
    assert "[Fuente:" in salida and ("Pág." in salida or "Art." in salida), "falta documento/ubicación"
    print("Límites OK; cada fragmento trae documento y página/artículo.")

    # ── calcular_plazo ──
    print(f"\n{SEP}\ncalcular_plazo.invoke\n{SEP}")
    casos = [
        {"fecha_inicio": "2026-03-30", "cantidad_dias": 10, "tipo": "habiles"},     # cruza Semana Santa
        {"fecha_inicio": "2026-12-15", "cantidad_dias": 10, "tipo": "habiles"},     # cruza Navidad
        {"fecha_inicio": "2026-11-07", "cantidad_dias": 30, "tipo": "calendario"},
        {"fecha_inicio": "15/03/2026", "cantidad_dias": 5, "tipo": "habiles"},      # argumento inválido
    ]
    for c in casos:
        print("\nargs:", json.dumps(c, ensure_ascii=False))
        print(tools["calcular_plazo"].invoke(c))
    assert "2026-04-15" in tools["calcular_plazo"].invoke(casos[0])
    assert "No se pudo calcular" in tools["calcular_plazo"].invoke(casos[3])
    print("\nCálculos OK (Jueves y Viernes Santo omitidos; fecha inválida rechazada con mensaje).")

    # ── web_search ──
    if "web_search" in tools:
        print(f"\n{SEP}\nweb_search.invoke (sitios permitidos: {rc.WEB_DOMINIOS_PERMITIDOS})\n{SEP}")
        registro["web"].clear()
        args = {"consulta": "valor de la UVT vigente 2026"}
        print("args:", json.dumps(args, ensure_ascii=False))
        salida_web = tools["web_search"].invoke(args)
        print(salida_web)
        assert "falló" not in salida_web, "la búsqueda web falló (¿falta tavily o la API key?)"
        w = registro["web"]
        assert len(w) >= 1, "la búsqueda web no devolvió resultados"
        assert len(w) <= rc.WEB_MAX_RESULTADOS, "demasiados resultados web"
        assert all(rc._dominio_permitido(r["url"]) for r in w), "resultado fuera de la lista"
        assert all(len(r["texto"]) <= rc.WEB_MAX_CHARS_EXTRACTO + 1 for r in w), "extracto demasiado largo"
        print(f"\nresultados: {len(w)} (máximo {rc.WEB_MAX_RESULTADOS}); todos de sitios permitidos; extractos cortos. OK")
    else:
        print("\n(web_search no probada: falta TAVILY_API_KEY en el .env)")

    print("\nTodas las pruebas pasaron.")


if __name__ == "__main__":
    main()