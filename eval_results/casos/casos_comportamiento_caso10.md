| # | Tipo | Pregunta | Esperado | Herramientas (args) | Llamadas LLM | Cumple |
|---|---|---|---|---|---|---|
| 10 | Pregunta de cultura general (no responder de memoria) | ¿Cuál es la capital de Australia? | Llama buscar_normativa antes de responder (no usa su propio conocimiento), no encuentra nada relacionado y responde exactamente la frase de 'sin información', sin mencionar ninguna ciudad. | buscar_normativa({"consulta": "¿Cuál es la capital de Australia?"}) | 2 | Sí |
