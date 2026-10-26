// ──────────────────────────────────────────────────────────────
// Interfaz del asistente RAG (sin frameworks).
// Habla con el backend Flask: /api/status, /api/indexed y /api/ask.
// ──────────────────────────────────────────────────────────────

const apiStatusEl = document.getElementById("api-status");
const pdfListEl = document.getElementById("pdf-list");
const pdfCountEl = document.getElementById("pdf-count");
const vsStatusEl = document.getElementById("vs-status");
const buildLogEl = document.getElementById("build-log");
const logDetailsEl = document.getElementById("log-details");
const kSlider = document.getElementById("k-slider");
const kValue = document.getElementById("k-value");
const messagesEl = document.getElementById("messages");
const chatForm = document.getElementById("chat-form");
const preguntaInput = document.getElementById("pregunta-input");
const submitBtn = chatForm.querySelector("button[type=submit]");
const btnReset = document.getElementById("btn-reset");
const chatReadyEl = document.getElementById("chat-ready");

const panelEl = document.getElementById("panel");
const scrimEl = document.getElementById("scrim");
const btnOpenPanel = document.getElementById("btn-open-panel");
const btnClosePanel = document.getElementById("btn-close-panel");

const MAX_HISTORY = 6; // mensajes (3 turnos) enviados al backend

let vectorStoreListo = false;
let consultando = false;
let historial = []; // [{role: "user"|"assistant", content: "..."}]

// Copia del estado vacío para restaurarlo con "Nueva conversación"
const emptyStateTemplate = document.getElementById("empty-state").cloneNode(true);

// Nombres legibles de los documentos del corpus (si llega otro PDF se muestra su nombre de archivo)
const DOCUMENTOS = {
  "cpaca_ley_1437_2011.pdf": { titulo: "CPACA, Ley 1437 de 2011", corto: "CPACA" },
  "ley_1480_2011.pdf": { titulo: "Estatuto del Consumidor, Ley 1480 de 2011", corto: "Ley 1480" },
  "ley_1581_2012.pdf": { titulo: "Protección de datos, Ley 1581 de 2012", corto: "Ley 1581" },
  "codigo_civil.pdf": { titulo: "Código Civil", corto: "C. Civil" },
  "codigo_general_proceso_ley_1564_2012.pdf": { titulo: "Código General del Proceso, Ley 1564 de 2012", corto: "CGP" },
  "estatuto_tributario.pdf": { titulo: "Estatuto Tributario", corto: "E. Tributario" },
  "ley_820_2003.pdf": { titulo: "Arrendamiento de vivienda, Ley 820 de 2003", corto: "Ley 820" },
};

function infoDocumento(archivo) {
  const d = DOCUMENTOS[archivo];
  if (d) return d;
  const base = archivo.replace(/\.pdf$/i, "").replace(/_/g, " ");
  return { titulo: base, corto: base };
}

// ───────────────── Estado inicial ─────────────────

async function cargarEstado() {
  let data;
  try {
    const res = await fetch("/api/status");
    data = await res.json();
  } catch (e) {
    setTimeout(cargarEstado, 3000);
    return;
  }

  if (data.api_key_ok) {
    apiStatusEl.textContent = "Detectada";
    apiStatusEl.className = "status status--ok";
  } else {
    apiStatusEl.textContent = data.api_key_error;
    apiStatusEl.className = "status status--error";
  }

  renderPdfList(data.pdfs);

  if (data.log && data.log.length) {
    logDetailsEl.hidden = false;
    buildLogEl.textContent = data.log.join("\n");
  }

  if (data.init_status === "lista") {
    marcarVectorStoreListo("Lista");
  } else if (data.init_status === "error") {
    vsStatusEl.textContent = data.init_error || "Error al preparar la base vectorial.";
    vsStatusEl.className = "status status--error";
    setReady("error", "Sin conexión a la base");
    logDetailsEl.open = true;
  } else {
    vsStatusEl.textContent = "Preparando…";
    vsStatusEl.className = "status status--pending";
    setTimeout(cargarEstado, 3000);
  }
}

function setReady(state, texto) {
  chatReadyEl.dataset.state = state;
  chatReadyEl.textContent = texto;
}

function renderPdfList(pdfs) {
  pdfListEl.innerHTML = "";
  pdfCountEl.textContent = pdfs.length ? String(pdfs.length) : "";
  if (!pdfs.length) {
    const li = document.createElement("li");
    li.className = "pdf-list__empty";
    li.textContent = "No hay PDFs en la carpeta pdfs/";
    pdfListEl.appendChild(li);
    return;
  }
  pdfs.forEach((nombre) => {
    const li = document.createElement("li");
    li.className = "pdf-list__item";
    li.title = nombre;
    const titulo = document.createElement("span");
    titulo.className = "pdf-list__name";
    titulo.textContent = infoDocumento(nombre).titulo;
    const archivo = document.createElement("span");
    archivo.className = "pdf-list__file";
    archivo.textContent = nombre;
    archivo.title = nombre;
    li.append(titulo, archivo);
    pdfListEl.appendChild(li);
  });
}

function marcarVectorStoreListo(mensaje) {
  vectorStoreListo = true;
  vsStatusEl.textContent = mensaje;
  vsStatusEl.className = "status status--ok";
  setReady("ok", "Lista");
  habilitarEntrada(true);
  activarEstadoVacio();
}

function habilitarEntrada(activo) {
  preguntaInput.disabled = !activo;
  submitBtn.disabled = !activo;
  document.querySelectorAll(".example").forEach((b) => { b.disabled = !activo; });
}

function activarEstadoVacio() {
  const empty = document.getElementById("empty-state");
  if (!empty) return;
  const wait = empty.querySelector("#empty-wait");
  if (wait && vectorStoreListo) wait.hidden = true;
  empty.querySelectorAll(".example").forEach((b) => {
    b.disabled = !vectorStoreListo;
    b.onclick = () => enviarPregunta(b.textContent.trim());
  });
}

// ───────────────── Slider de k ─────────────────

function pintarSlider() {
  kValue.textContent = kSlider.value;
  const pct = ((kSlider.value - kSlider.min) / (kSlider.max - kSlider.min)) * 100;
  kSlider.style.setProperty("--pct", `${pct}%`);
}
kSlider.addEventListener("input", pintarSlider);
pintarSlider();

// ───────────────── Diagnóstico: qué se indexó ─────────────────

const btnIndexed = document.getElementById("btn-indexed");
const indexedPanel = document.getElementById("indexed-panel");

function indexedMensaje(texto) {
  const p = document.createElement("p");
  p.className = "indexed-item";
  p.textContent = texto;
  return p;
}

btnIndexed.addEventListener("click", async () => {
  if (indexedPanel.hidden) {
    btnIndexed.textContent = "Consultando…";
    indexedPanel.replaceChildren();
    try {
      const res = await fetch("/api/indexed");
      const data = await res.json();

      if (!res.ok) {
        indexedPanel.appendChild(indexedMensaje(data.error || "Error al consultar."));
      } else if (!data.fuentes.length) {
        indexedPanel.appendChild(indexedMensaje("La base vectorial está vacía."));
      } else {
        data.fuentes.forEach((f) => {
          const div = document.createElement("div");
          div.className = "indexed-item";
          const name = document.createElement("span");
          name.className = "indexed-item__name";
          name.textContent = f.archivo;
          const stats = document.createElement("div");
          stats.className = "indexed-item__stats";
          stats.textContent = `${f.fragmentos} fragmentos, ${f.caracteres_totales.toLocaleString("es-CO")} caracteres`;
          const sample = document.createElement("div");
          sample.className = "indexed-item__sample";
          sample.textContent = `"${f.muestra}…"`;
          div.append(name, stats, sample);
          indexedPanel.appendChild(div);
        });
      }
    } catch (e) {
      indexedPanel.appendChild(indexedMensaje("Error de red."));
    }
    indexedPanel.hidden = false;
    btnIndexed.textContent = "Ocultar";
    btnIndexed.setAttribute("aria-expanded", "true");
  } else {
    indexedPanel.hidden = true;
    btnIndexed.textContent = "Ver qué se indexó";
    btnIndexed.setAttribute("aria-expanded", "false");
  }
});

// ───────────────── Panel lateral en móvil ─────────────────

function abrirPanel() {
  panelEl.classList.add("is-open");
  scrimEl.hidden = false;
  btnOpenPanel.setAttribute("aria-expanded", "true");
  btnClosePanel.focus();
}

function cerrarPanel() {
  if (!panelEl.classList.contains("is-open")) return;
  panelEl.classList.remove("is-open");
  scrimEl.hidden = true;
  btnOpenPanel.setAttribute("aria-expanded", "false");
  btnOpenPanel.focus();
}

btnOpenPanel.addEventListener("click", abrirPanel);
btnClosePanel.addEventListener("click", cerrarPanel);
scrimEl.addEventListener("click", cerrarPanel);
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") cerrarPanel();
});

// ───────────────── Formato de respuestas ─────────────────

function escapeHtml(s) {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

// Cita con el formato pedido en el prompt: (Fuente: Documento.pdf, Pág. N)
const CITA_RE = /\(\s*Fuente:\s*([^,()]+?\.pdf)\s*[,—–-]\s*P[áa]g(?:ina)?\.?\s*(\d+)\s*\)/gi;

const ICONO_DOC = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 3h7l5 5v13H7z"/><path d="M14 3v5h5"/></svg>';

function formatoInline(texto) {
  // texto ya escapado
  return texto
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*(?!\s)([^*]+?)\*(?!\*)/g, "$1<em>$2</em>")
    .replace(CITA_RE, (_, archivo, pagina) => {
      const corto = escapeHtml(infoDocumento(archivo.trim()).corto);
      return `<button type="button" class="cite" data-fuente="${archivo.trim()}" data-pagina="${pagina}" ` +
        `title="Ver fragmento de ${archivo.trim()}, página ${pagina}">` +
        `${ICONO_DOC}${corto} <span class="cite__page">p. ${pagina}</span></button>`;
    });
}

// Markdown mínimo y seguro: párrafos, listas, negrita, cursiva y citas
function renderRespuesta(texto) {
  const bloques = escapeHtml(texto.trim()).split(/\n\s*\n/);
  const html = [];
  bloques.forEach((bloque) => {
    const lineas = bloque.split("\n").filter((l) => l.trim());
    if (!lineas.length) return;
    const esLista = lineas.every((l) => /^\s*([-*•]|\d+[.)])\s+/.test(l));
    if (esLista) {
      const ordenada = /^\s*\d+[.)]/.test(lineas[0]);
      const tag = ordenada ? "ol" : "ul";
      const items = lineas
        .map((l) => `<li>${formatoInline(l.replace(/^\s*([-*•]|\d+[.)])\s+/, ""))}</li>`)
        .join("");
      html.push(`<${tag}>${items}</${tag}>`);
    } else {
      html.push(`<p>${lineas.map(formatoInline).join("<br>")}</p>`);
    }
  });
  return html.join("");
}

// ───────────────── Chat ─────────────────

function agregarMensaje(texto, tipo) {
  const div = document.createElement("div");
  div.className = `msg msg--${tipo}`;
  const body = document.createElement("div");
  body.className = "msg__body";
  body.textContent = texto;
  div.appendChild(body);
  messagesEl.appendChild(div);
  scrollAlFinal();
  return div;
}

function scrollAlFinal() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function renderFuentes(fuentes, citadas) {
  const det = document.createElement("details");
  det.className = "sources";
  const sum = document.createElement("summary");
  sum.textContent = `Fuentes (${fuentes.length})`;
  det.appendChild(sum);

  const lista = document.createElement("div");
  lista.className = "sources__list";

  fuentes.forEach((f) => {
    const frag = document.createElement("div");
    frag.className = "fragment";
    frag.dataset.fuente = f.fuente;
    frag.dataset.pagina = String(f.pagina);
    if (citadas.has(`${f.fuente}|${f.pagina}`)) frag.classList.add("fragment--cited");
    const src = document.createElement("span");
    src.className = "fragment__source";
    src.textContent = `${f.fuente} — Pág. ${f.pagina}`;
    frag.appendChild(src);
    frag.appendChild(document.createTextNode(f.texto));
    lista.appendChild(frag);
  });
  det.appendChild(lista);
  return det;
}

function botonCopiar(texto) {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "msg__copy";
  btn.textContent = "Copiar";
  btn.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(texto);
      btn.textContent = "Copiado";
    } catch (e) {
      btn.textContent = "No se pudo copiar";
    }
    setTimeout(() => { btn.textContent = "Copiar"; }, 1800);
  });
  return btn;
}

// Clic en un sello de cita: abre las fuentes y resalta el fragmento correspondiente
messagesEl.addEventListener("click", (e) => {
  const cite = e.target.closest(".cite");
  if (!cite) return;
  const msg = cite.closest(".msg");
  const det = msg && msg.querySelector(".sources");
  if (!det) return;
  det.open = true;
  const frags = [...det.querySelectorAll(".fragment")].filter(
    (f) => f.dataset.fuente === cite.dataset.fuente && f.dataset.pagina === cite.dataset.pagina
  );
  if (!frags.length) {
    det.scrollIntoView({ behavior: "smooth", block: "nearest" });
    return;
  }
  frags[0].scrollIntoView({ behavior: "smooth", block: "center" });
  frags.forEach((f) => {
    f.classList.remove("fragment--flash");
    void f.offsetWidth;
    f.classList.add("fragment--flash");
    setTimeout(() => f.classList.remove("fragment--flash"), 1600);
  });
});

btnReset.addEventListener("click", () => {
  historial = [];
  messagesEl.replaceChildren(emptyStateTemplate.cloneNode(true));
  activarEstadoVacio();
  cerrarPanel();
  if (!preguntaInput.disabled) preguntaInput.focus();
});

// Campo de texto: crece con el contenido, Enter envía y Shift+Enter agrega una línea
function ajustarAltura() {
  preguntaInput.style.height = "auto";
  preguntaInput.style.height = `${Math.min(preguntaInput.scrollHeight, 180)}px`;
}

preguntaInput.addEventListener("input", ajustarAltura);
preguntaInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    chatForm.requestSubmit();
  }
});

chatForm.addEventListener("submit", (e) => {
  e.preventDefault();
  enviarPregunta(preguntaInput.value.trim());
});

async function enviarPregunta(pregunta) {
  if (!pregunta || !vectorStoreListo || consultando) return;
  consultando = true;

  const empty = document.getElementById("empty-state");
  if (empty) empty.remove();

  agregarMensaje(pregunta, "user");
  preguntaInput.value = "";
  ajustarAltura();
  habilitarEntrada(false);

  const pensando = agregarMensaje("", "assistant");
  pensando.classList.add("msg--loading");
  const body = pensando.querySelector(".msg__body");
  body.innerHTML = '<span class="typing" role="status" aria-label="Consultando…"><span></span><span></span><span></span></span>';

  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        pregunta,
        k: Number(kSlider.value),
        historial: historial.slice(-MAX_HISTORY),
      }),
    });
    const data = await res.json();
    pensando.classList.remove("msg--loading");

    if (!res.ok) {
      pensando.classList.add("msg--error");
      body.textContent = data.error || "Ocurrió un error.";
    } else {
      body.innerHTML = renderRespuesta(data.respuesta);
      if (data.sin_respuesta) pensando.classList.add("msg--abstain");

      // Guardar el turno en el historial (solo si salió bien)
      historial.push({ role: "user", content: pregunta });
      historial.push({ role: "assistant", content: data.respuesta });

      if (data.pregunta_independiente && data.pregunta_independiente !== pregunta) {
        const re = document.createElement("p");
        re.className = "msg__rewrite";
        re.append("Interpretada como: ");
        const em = document.createElement("em");
        em.textContent = data.pregunta_independiente;
        re.appendChild(em);
        pensando.appendChild(re);
      }

      const footer = document.createElement("div");
      footer.className = "msg__footer";

      const meta = document.createElement("span");
      meta.className = "msg__meta";
      meta.textContent = data.sin_respuesta
        ? "Sin información en la base de conocimientos"
        : `~${data.tokens_contexto_aprox} tokens de contexto · ${data.n_recuperados} fragmentos recuperados`;
      footer.appendChild(meta);
      footer.appendChild(botonCopiar(data.respuesta));

      if (data.fuentes && data.fuentes.length) {
        const citadas = new Set(
          [...body.querySelectorAll(".cite")].map((c) => `${c.dataset.fuente}|${c.dataset.pagina}`)
        );
        footer.appendChild(renderFuentes(data.fuentes, citadas));
      }
      pensando.appendChild(footer);
    }
  } catch (err) {
    pensando.classList.remove("msg--loading");
    pensando.classList.add("msg--error");
    body.textContent = "Error de red al consultar.";
  } finally {
    consultando = false;
    habilitarEntrada(true);
    preguntaInput.focus();
    pensando.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

activarEstadoVacio();
cargarEstado();
