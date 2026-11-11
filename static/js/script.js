// ──────────────────────────────────────────────────────────────
// Interfaz del asistente RAG (sin frameworks).
// Habla con el backend Flask: /api/status, /api/indexed, /api/ask, /api/transcribe,
// /api/conversaciones (lista, abrir, renombrar, borrar) y /api/auth (sesión y cuenta).
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

const webToggle = document.getElementById("web-toggle");
const btnImage = document.getElementById("btn-image");
const imageInput = document.getElementById("image-input");
const attachmentEl = document.getElementById("attachment");
const attachmentImg = document.getElementById("attachment-img");
const attachmentName = document.getElementById("attachment-name");
const btnRemoveImage = document.getElementById("btn-remove-image");
const btnMic = document.getElementById("btn-mic");

const chatTitleEl = document.getElementById("chat-title");
const convListEl = document.getElementById("conv-list");
const convEmptyEl = document.getElementById("conv-empty");
const btnLogout = document.getElementById("btn-logout");
const btnDeleteAccount = document.getElementById("btn-delete-account");
const dlgConv = document.getElementById("dlg-conv");
const dlgConvText = document.getElementById("dlg-conv-text");
const dlgConvOk = document.getElementById("dlg-conv-ok");
const dlgCuenta = document.getElementById("dlg-cuenta");
const formCuenta = document.getElementById("form-cuenta");
const dlgCuentaPass = document.getElementById("dlg-cuenta-pass");
const dlgCuentaError = document.getElementById("dlg-cuenta-error");
const dlgCuentaOk = document.getElementById("dlg-cuenta-ok");

const TITULO_POR_DEFECTO = "Asistente Legal";

const MAX_LADO_IMAGEN = 1600; // px: la imagen se reduce antes de enviarla
const MAX_SEGUNDOS_AUDIO = 60;
const PUEDE_GRABAR = !!(navigator.mediaDevices && window.MediaRecorder);

let vectorStoreListo = false;
let consultando = false;
let conversacionId = null; // conversación abierta (null = todavía no se guardó nada)
let conversaciones = []; // [{id, titulo, creado, actualizado}] del usuario
let imagenAdjunta = null; // data URL (JPEG) lista para enviar
let grabadora = null; // MediaRecorder activo
let transcribiendo = false;

// Copia del estado vacío para restaurarlo con "Nueva conversación"
const emptyStateTemplate = document.getElementById("empty-state").cloneNode(true);

// fetch que lleva al inicio de sesión cuando la sesión terminó (401)
async function api(url, opciones) {
  const res = await fetch(url, opciones);
  if (res.status === 401) {
    location.href = "/acceso";
    throw new Error("Tu sesión terminó.");
  }
  return res;
}

function apiJson(url, metodo, cuerpo) {
  return api(url, {
    method: metodo,
    headers: { "Content-Type": "application/json" },
    body: cuerpo === undefined ? undefined : JSON.stringify(cuerpo),
  });
}

// Nombres legibles de los documentos del corpus (si llega otro archivo se muestra su nombre)
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
  const base = archivo.replace(/\.(pdf|docx|tex)$/i, "").replace(/_/g, " ");
  return { titulo: base, corto: base };
}

// ───────────────── Estado inicial ─────────────────

async function cargarEstado() {
  let data;
  try {
    const res = await api("/api/status");
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

  renderPdfList(data.documentos);
  configurarWeb(data.web_ok);

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
    li.textContent = "No hay documentos en la carpeta pdfs/";
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
  btnImage.disabled = !activo;
  btnMic.disabled = !activo || !PUEDE_GRABAR || transcribiendo;
  document.querySelectorAll(".example").forEach((b) => { b.disabled = !activo; });
}

// ───────────────── Búsqueda web (tool opcional) ─────────────────

const CLAVE_WEB = "rag-web-activa";

function configurarWeb(disponible) {
  webToggle.disabled = !disponible;
  if (!disponible) {
    webToggle.checked = false;
    webToggle.title = "Configura TAVILY_API_KEY en el servidor para activarla";
    return;
  }
  webToggle.title = "";
  if (webToggle.dataset.init) return;
  webToggle.dataset.init = "1";
  try { webToggle.checked = localStorage.getItem(CLAVE_WEB) === "1"; } catch (e) { /* sin almacenamiento */ }
}

webToggle.addEventListener("change", () => {
  try { localStorage.setItem(CLAVE_WEB, webToggle.checked ? "1" : "0"); } catch (e) { /* sin almacenamiento */ }
});

// ───────────────── Imagen adjunta ─────────────────

// Reduce la imagen en el navegador y la convierte a JPEG: pesa menos y
// queda por debajo del límite de 4 MB de Groq.
function reducirImagen(archivo) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(archivo);
    const img = new Image();
    img.onload = () => {
      const escala = Math.min(1, MAX_LADO_IMAGEN / Math.max(img.width, img.height));
      const canvas = document.createElement("canvas");
      canvas.width = Math.round(img.width * escala);
      canvas.height = Math.round(img.height * escala);
      const ctx = canvas.getContext("2d");
      ctx.fillStyle = "#fff"; // fondo blanco para PNG con transparencia
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      URL.revokeObjectURL(url);
      resolve(canvas.toDataURL("image/jpeg", 0.85));
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("No se pudo leer la imagen."));
    };
    img.src = url;
  });
}

btnImage.addEventListener("click", () => imageInput.click());

imageInput.addEventListener("change", async () => {
  const archivo = imageInput.files[0];
  imageInput.value = "";
  if (!archivo) return;
  try {
    imagenAdjunta = await reducirImagen(archivo);
    attachmentImg.src = imagenAdjunta;
    attachmentName.textContent = archivo.name;
    attachmentEl.hidden = false;
    preguntaInput.placeholder = "Pregunta (opcional)";
    preguntaInput.focus();
  } catch (e) {
    quitarImagen();
    agregarMensaje(e.message, "assistant").classList.add("msg--error");
  }
});

function quitarImagen() {
  imagenAdjunta = null;
  attachmentEl.hidden = true;
  attachmentImg.removeAttribute("src");
  preguntaInput.placeholder = "Escribe tu pregunta";
}

btnRemoveImage.addEventListener("click", () => {
  quitarImagen();
  preguntaInput.focus();
});

// ───────────────── Dictado por voz (Whisper en Groq) ─────────────────

function marcarMic(estado) {
  // estado: "idle" | "grabando" | "transcribiendo"
  btnMic.dataset.state = estado;
  btnMic.setAttribute("aria-pressed", estado === "grabando" ? "true" : "false");
  btnMic.setAttribute("aria-label",
    estado === "grabando" ? "Detener grabación" : estado === "transcribiendo" ? "Transcribiendo…" : "Dictar pregunta");
  btnMic.title = btnMic.getAttribute("aria-label");
}

async function iniciarGrabacion() {
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (e) {
    agregarMensaje("No se pudo acceder al micrófono. Revisa los permisos del navegador.", "assistant")
      .classList.add("msg--error");
    return;
  }
  const trozos = [];
  grabadora = new MediaRecorder(stream);
  grabadora.addEventListener("dataavailable", (e) => { if (e.data.size) trozos.push(e.data); });
  grabadora.addEventListener("stop", () => {
    stream.getTracks().forEach((t) => t.stop());
    clearTimeout(grabadora.limite);
    const tipo = grabadora.mimeType || "audio/webm";
    grabadora = null;
    transcribir(new Blob(trozos, { type: tipo }));
  });
  grabadora.start();
  grabadora.limite = setTimeout(() => grabadora && grabadora.stop(), MAX_SEGUNDOS_AUDIO * 1000);
  marcarMic("grabando");
}

async function transcribir(blob) {
  if (!blob.size) { marcarMic("idle"); return; }
  transcribiendo = true;
  marcarMic("transcribiendo");
  btnMic.disabled = true;
  const extension = blob.type.includes("ogg") ? "ogg" : blob.type.includes("mp4") ? "mp4" : "webm";
  const datos = new FormData();
  datos.append("audio", blob, `grabacion.${extension}`);
  try {
    const res = await api("/api/transcribe", { method: "POST", body: datos });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "No se pudo transcribir el audio.");
    const actual = preguntaInput.value.trim();
    preguntaInput.value = actual ? `${actual} ${data.texto}` : data.texto;
    ajustarAltura();
  } catch (e) {
    agregarMensaje(e.message || "Error de red al transcribir.", "assistant").classList.add("msg--error");
  } finally {
    transcribiendo = false;
    marcarMic("idle");
    btnMic.disabled = preguntaInput.disabled || !PUEDE_GRABAR;
    preguntaInput.focus();
  }
}

btnMic.addEventListener("click", () => {
  if (grabadora) grabadora.stop();
  else if (!transcribiendo) iniciarGrabacion();
});
marcarMic("idle");
if (!PUEDE_GRABAR) btnMic.title = "Este navegador no permite grabar audio";

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
      const res = await api("/api/indexed");
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

// Cita con el formato pedido en el prompt: (Fuente: Documento.pdf, Pág. N) o (Fuente: Documento.docx, Art. N)
const CITA_RE = /\(\s*Fuente:\s*([^,()]+?\.(?:pdf|docx|tex))\s*[,—–-]\s*([^()]{1,60}?)\s*\)/gi;

// Ubicación normalizada para comparar la cita con el fragmento ("Pág. 3" = "pagina 3" = "p3")
function normUbic(u) {
  return String(u)
    .toLowerCase()
    .replace(/p[áa]g(?:ina)?\.?\s*/g, "p")
    .replace(/art(?:[íi]culo)?\.?\s*/g, "a")
    .replace(/\s+/g, "");
}

// Texto corto de la ubicación en la píldora: "Pág. 3" -> "p. 3"; "Art. 5" queda igual
function etiquetaUbic(u) {
  const m = String(u).match(/^p[áa]g(?:ina)?\.?\s*(\d+)$/i);
  return m ? `p. ${m[1]}` : String(u);
}

// Cita de la búsqueda web: [web: https://...] (también acepta el formato antiguo (Web: https://...))
const WEB_RE = /[\[(]\s*web:\s*(https?:\/\/[^\s()<>"\]]+?)\s*[\])]/gi;

const ICONO_WEB = '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="8.5"/><path d="M3.5 12h17M12 3.5c2.5 2.6 2.5 14.4 0 17M12 3.5c-2.5 2.6-2.5 14.4 0 17"/></svg>';

const ICONO_DOC = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 3h7l5 5v13H7z"/><path d="M14 3v5h5"/></svg>';

function formatoInline(texto) {
  // texto ya escapado
  return texto
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*(?!\s)([^*]+?)\*(?!\*)/g, "$1<em>$2</em>")
    .replace(CITA_RE, (_, archivo, ubicacion) => {
      const corto = escapeHtml(infoDocumento(archivo.trim()).corto);
      const ubic = ubicacion.trim();
      return `<button type="button" class="cite" data-fuente="${archivo.trim()}" data-ubicacion="${normUbic(ubic)}" ` +
        `title="Ver fragmento de ${archivo.trim()}, ${ubic}">` +
        `${ICONO_DOC}${corto} <span class="cite__page">${etiquetaUbic(ubic)}</span></button>`;
    })
    .replace(WEB_RE, (_, url) => {
      let host = url;
      try { host = new URL(url.replace(/&amp;/g, "&")).hostname.replace(/^www\./, ""); } catch (e) { /* URL rara: se muestra tal cual */ }
      return `<a class="cite cite--web" href="${url}" target="_blank" rel="noopener noreferrer" title="${url}">` +
        `${ICONO_WEB}${escapeHtml(host)}</a>`;
    });
}

const ITEM_RE = /^\s*([-*•]|\d+[.)])\s+/;

// Markdown mínimo y seguro: párrafos, listas, negrita, cursiva y citas
function renderRespuesta(texto) {
  const bloques = escapeHtml(texto.trim()).split(/\n\s*\n/);
  const html = [];
  bloques.forEach((bloque) => {
    const lineas = bloque.split("\n").filter((l) => l.trim());
    // Agrupa líneas consecutivas: el modelo suele poner la lista justo
    // debajo de la frase que la introduce, sin línea en blanco.
    let i = 0;
    while (i < lineas.length) {
      const esItem = ITEM_RE.test(lineas[i]);
      const grupo = [];
      while (i < lineas.length && ITEM_RE.test(lineas[i]) === esItem) grupo.push(lineas[i++]);
      if (esItem) {
        const tag = /^\s*\d+[.)]/.test(grupo[0]) ? "ol" : "ul";
        const items = grupo
          .map((l) => `<li>${formatoInline(l.replace(ITEM_RE, ""))}</li>`)
          .join("");
        html.push(`<${tag}>${items}</${tag}>`);
      } else {
        html.push(`<p>${grupo.map(formatoInline).join("<br>")}</p>`);
      }
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

function renderFuentes(fuentes, citadas, fuentesWeb = []) {
  const det = document.createElement("details");
  det.className = "sources";
  const sum = document.createElement("summary");
  sum.textContent = `Fuentes (${fuentes.length + fuentesWeb.length})`;
  det.appendChild(sum);

  const lista = document.createElement("div");
  lista.className = "sources__list";

  // Las fuentes se agrupan por origen: base de conocimientos y web
  const grupo = (texto) => {
    const h = document.createElement("p");
    h.className = "sources__group";
    h.textContent = texto;
    lista.appendChild(h);
  };
  if (fuentes.length) grupo(`Base de conocimientos · ${fuentes.length} fragmentos`);

  fuentes.forEach((f) => {
    const frag = document.createElement("div");
    frag.className = "fragment";
    frag.dataset.fuente = f.fuente;
    frag.dataset.ubicacion = normUbic(f.ubicacion);
    const citada = citadas.has(`${f.fuente}|${normUbic(f.ubicacion)}`);
    if (citada) frag.classList.add("fragment--cited");

    // Cabecera tipo libro mayor: documento a la izquierda, página a la derecha
    const head = document.createElement("div");
    head.className = "fragment__source";
    head.title = f.fuente;
    const doc = document.createElement("span");
    doc.className = "fragment__doc";
    doc.textContent = infoDocumento(f.fuente).titulo;
    head.appendChild(doc);
    if (citada) {
      const tag = document.createElement("span");
      tag.className = "fragment__tag";
      tag.textContent = "Citada";
      head.appendChild(tag);
    }
    const pag = document.createElement("span");
    pag.className = "fragment__page";
    pag.textContent = f.ubicacion;
    head.appendChild(pag);

    const texto = document.createElement("p");
    texto.className = "fragment__text";
    texto.textContent = f.texto;
    frag.append(head, texto);
    lista.appendChild(frag);
  });

  if (fuentesWeb.length) grupo(`Web (sitios oficiales permitidos) · ${fuentesWeb.length} resultados`);

  fuentesWeb.forEach((w) => {
    const frag = document.createElement("div");
    frag.className = "fragment fragment--web";
    const head = document.createElement("div");
    head.className = "fragment__source";
    const link = document.createElement("a");
    link.className = "fragment__doc";
    link.href = /^https?:\/\//i.test(w.url) ? w.url : "#";
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.textContent = w.titulo || w.url;
    const tag = document.createElement("span");
    tag.className = "fragment__page";
    try { tag.textContent = new URL(w.url).hostname.replace(/^www\./, ""); } catch (e) { tag.textContent = "Web"; }
    head.append(link, tag);
    const texto = document.createElement("p");
    texto.className = "fragment__text";
    texto.textContent = w.texto;
    frag.append(head, texto);
    lista.appendChild(frag);
  });

  det.appendChild(lista);
  return det;
}

// Qué herramientas usó el modelo para responder, con sus argumentos
const ETIQUETA_TOOL = { buscar_normativa: "Normativa", web_search: "Web", calcular_plazo: "Plazo" };

function argumentosLegibles(h) {
  const base = argumentosBase(h);
  return h.bloqueada ? `${base} (bloqueada por la política)` : base;
}

function argumentosBase(h) {
  const a = h.args || {};
  if (h.nombre === "calcular_plazo") {
    return `${a.cantidad_dias} días ${a.tipo} desde ${a.fecha_inicio}`;
  }
  return a.consulta || h.consulta || "";
}

function renderHerramientas(herramientas, uso) {
  // Sección desplegable: qué tools usó el agente y con qué argumentos
  const det = document.createElement("details");
  det.className = "tools";
  const sum = document.createElement("summary");
  const partes = [`Herramientas usadas (${herramientas.length})`];
  if (uso && uso.llamadas_llm) partes.push(`${uso.llamadas_llm} llamadas al LLM`);
  sum.textContent = partes.join(" · ");
  det.appendChild(sum);
  const p = document.createElement("div");
  p.className = "msg__tools";
  herramientas.forEach((h) => {
    const esWeb = h.nombre === "web_search";
    const chip = document.createElement("span");
    chip.className = `tool-chip tool-chip--${esWeb ? "web" : "doc"}`;
    chip.innerHTML = esWeb ? ICONO_WEB : ICONO_DOC;
    const etiqueta = document.createElement("span");
    etiqueta.textContent = `${ETIQUETA_TOOL[h.nombre] || h.nombre}: `;
    etiqueta.className = "tool-chip__label";
    const texto = argumentosLegibles(h);
    const consulta = document.createElement("span");
    consulta.textContent = texto;
    chip.append(etiqueta, consulta);
    // Al pasar el cursor se ven el nombre de la tool y todos sus argumentos
    const llamada = `${h.nombre}(${JSON.stringify(h.args || { consulta: h.consulta })})`;
    chip.title = llamada;
    const fila = document.createElement("div");
    fila.className = "tools__item";
    const codigo = document.createElement("code");
    codigo.className = "tools__call";
    codigo.textContent = llamada;
    fila.append(chip, codigo);
    p.appendChild(fila);
  });
  det.appendChild(p);
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
  if (!cite || cite.classList.contains("cite--web")) return;
  const msg = cite.closest(".msg");
  const det = msg && msg.querySelector(".sources");
  if (!det) return;
  det.open = true;
  const frags = [...det.querySelectorAll(".fragment")].filter(
    (f) => f.dataset.fuente === cite.dataset.fuente && f.dataset.ubicacion === cite.dataset.ubicacion
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

btnReset.addEventListener("click", nuevaConversacion);

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

// Completa un mensaje del asistente con su respuesta, herramientas, pie y fuentes.
// Se usa igual para una respuesta recién llegada y para una conversación guardada.
function completarRespuesta(msg, body, data, pregunta) {
  body.innerHTML = renderRespuesta(data.respuesta);
  if (data.sin_respuesta) msg.classList.add("msg--abstain");

  if (data.herramientas && data.herramientas.length) {
    msg.appendChild(renderHerramientas(data.herramientas, data.uso));
  } else if (data.pregunta_independiente && data.pregunta_independiente !== pregunta) {
    const re = document.createElement("p");
    re.className = "msg__rewrite";
    re.append("Interpretada como: ");
    const em = document.createElement("em");
    em.textContent = data.pregunta_independiente;
    re.appendChild(em);
    msg.appendChild(re);
  }

  const footer = document.createElement("div");
  footer.className = "msg__footer";

  const meta = document.createElement("span");
  meta.className = "msg__meta";
  const fuentesWeb = data.fuentes_web || [];
  const usoHerramientas = (data.herramientas || []).length > 0 || data.n_recuperados > 0;
  const partes = [`~${data.tokens_contexto_aprox} tokens de contexto`, `${data.n_recuperados} fragmentos`];
  if (fuentesWeb.length) partes.push(`${fuentesWeb.length} resultados web`);
  if (data.uso) partes.push(`${data.uso.llamadas_llm} llamadas al LLM · ${data.uso.segundos}s`);
  meta.textContent = data.sin_respuesta
    ? "Sin información en la base de conocimientos"
    : usoHerramientas ? partes.join(" · ") : "Respondida sin consultar documentos";
  footer.appendChild(meta);
  footer.appendChild(botonCopiar(data.respuesta));

  if ((data.fuentes && data.fuentes.length) || fuentesWeb.length) {
    const citadas = new Set(
      [...body.querySelectorAll(".cite:not(.cite--web)")].map((c) => `${c.dataset.fuente}|${c.dataset.ubicacion}`)
    );
    // Si la respuesta solo cita la web, los fragmentos de la base no la respaldan: no se muestran
    const soloWeb = citadas.size === 0 && fuentesWeb.length > 0 && body.querySelector(".cite--web");
    footer.appendChild(renderFuentes(soloWeb ? [] : (data.fuentes || []), citadas, fuentesWeb));
  }
  msg.appendChild(footer);

  if (data.aviso) {
    const aviso = document.createElement("p");
    aviso.className = "msg__notice";
    aviso.textContent = data.aviso;
    msg.appendChild(aviso);
  }
}

async function enviarPregunta(pregunta) {
  const imagen = imagenAdjunta;
  if ((!pregunta && !imagen) || !vectorStoreListo || consultando || grabadora || transcribiendo) return;
  consultando = true;

  const empty = document.getElementById("empty-state");
  if (empty) empty.remove();

  const msgUsuario = agregarMensaje(pregunta, "user");
  if (imagen) {
    const thumb = document.createElement("img");
    thumb.className = "msg__image";
    thumb.src = imagen;
    thumb.alt = "Imagen adjunta";
    msgUsuario.insertBefore(thumb, msgUsuario.firstChild);
    if (!pregunta) msgUsuario.querySelector(".msg__body").remove();
  }
  quitarImagen();
  preguntaInput.value = "";
  ajustarAltura();
  habilitarEntrada(false);

  const pensando = agregarMensaje("", "assistant");
  pensando.classList.add("msg--loading");
  const body = pensando.querySelector(".msg__body");
  body.innerHTML = '<span class="typing" role="status" aria-label="Consultando…"><span></span><span></span><span></span></span>';

  try {
    // El historial lo arma el servidor a partir de la conversación guardada
    const res = await apiJson("/api/ask", "POST", {
      pregunta,
      k: Number(kSlider.value),
      web: webToggle.checked,
      imagen,
      conversacion_id: conversacionId,
    });
    const data = await res.json();
    pensando.classList.remove("msg--loading");

    if (!res.ok) {
      pensando.classList.add("msg--error");
      body.textContent = data.error || "Ocurrió un error.";
      if (res.status === 404 && conversacionId) {
        // La conversación ya no existe (p. ej. se borró desde otra pestaña)
        conversacionId = null;
        sincronizarUrl();
        cargarConversaciones();
      }
    } else {
      completarRespuesta(pensando, body, data, pregunta);
      if (data.conversacion && data.conversacion.id) registrarTurnoEnLista(data.conversacion);
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

// ───────────────── Conversaciones guardadas ─────────────────

const ICONO_EDITAR = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M16.5 3.5a2.1 2.1 0 013 3L7 19l-4 1 1-4z"/></svg>';
const ICONO_BORRAR = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h16M10 11v6M14 11v6M6 7l1 12a2 2 0 002 2h6a2 2 0 002-2l1-12M9 7V4h6v3"/></svg>';

function fechaCorta(iso) {
  const d = new Date(iso);
  if (isNaN(d)) return "";
  const hoy = new Date();
  const ayer = new Date();
  ayer.setDate(hoy.getDate() - 1);
  if (d.toDateString() === hoy.toDateString()) {
    return `Hoy, ${d.toLocaleTimeString("es-CO", { hour: "numeric", minute: "2-digit" })}`;
  }
  if (d.toDateString() === ayer.toDateString()) return "Ayer";
  const opciones = d.getFullYear() === hoy.getFullYear()
    ? { day: "numeric", month: "short" }
    : { day: "numeric", month: "short", year: "numeric" };
  return d.toLocaleDateString("es-CO", opciones);
}

function sincronizarUrl() {
  const url = conversacionId ? `/chat?c=${encodeURIComponent(conversacionId)}` : "/chat";
  try { history.replaceState(null, "", url); } catch (e) { /* sin historial */ }
}

function botonConv(clase, etiqueta, icono, alClic) {
  const b = document.createElement("button");
  b.type = "button";
  b.className = `conv__btn ${clase}`;
  b.setAttribute("aria-label", etiqueta);
  b.title = etiqueta;
  b.innerHTML = icono;
  b.addEventListener("click", (e) => { e.stopPropagation(); alClic(); });
  return b;
}

function mostrarErrorLista(texto) {
  convListEl.querySelectorAll(".conv-list__error").forEach((n) => n.remove());
  const li = document.createElement("li");
  li.className = "conv-list__error";
  li.setAttribute("role", "alert");
  li.textContent = texto;
  convListEl.prepend(li);
  setTimeout(() => li.remove(), 5000);
}

function renderConversaciones() {
  convListEl.replaceChildren();
  convEmptyEl.hidden = conversaciones.length > 0;
  conversaciones.forEach((conv) => {
    const li = document.createElement("li");
    li.className = "conv";
    const activa = conv.id === conversacionId;
    if (activa) li.classList.add("is-active");

    const abrir = document.createElement("button");
    abrir.type = "button";
    abrir.className = "conv__open";
    abrir.title = conv.titulo;
    if (activa) abrir.setAttribute("aria-current", "true");
    const titulo = document.createElement("span");
    titulo.className = "conv__title";
    titulo.textContent = conv.titulo;
    const fecha = document.createElement("span");
    fecha.className = "conv__date";
    fecha.textContent = fechaCorta(conv.actualizado);
    abrir.append(titulo, fecha);
    abrir.addEventListener("click", () => abrirConversacion(conv.id));

    const acciones = document.createElement("div");
    acciones.className = "conv__actions";
    acciones.append(
      botonConv("", `Cambiar el nombre de «${conv.titulo}»`, ICONO_EDITAR, () => editarTitulo(li, conv)),
      botonConv("conv__btn--danger", `Borrar «${conv.titulo}»`, ICONO_BORRAR, () => pedirBorrado(conv))
    );

    li.append(abrir, acciones);
    convListEl.appendChild(li);
  });
}

async function cargarConversaciones() {
  try {
    const res = await api("/api/conversaciones");
    const data = await res.json();
    if (!res.ok) throw new Error(data.error);
    conversaciones = data.conversaciones;
  } catch (e) {
    conversaciones = [];
    renderConversaciones();
    mostrarErrorLista("No se pudieron cargar tus conversaciones.");
    return;
  }
  renderConversaciones();
}

// Tras una respuesta: la conversación sube al inicio de la lista y queda abierta
function registrarTurnoEnLista(conv) {
  conversacionId = conv.id;
  const ahora = new Date().toISOString();
  const existente = conversaciones.find((c) => c.id === conv.id);
  if (existente) {
    existente.actualizado = ahora;
    conversaciones = [existente, ...conversaciones.filter((c) => c.id !== conv.id)];
  } else {
    conversaciones.unshift({ id: conv.id, titulo: conv.titulo, creado: ahora, actualizado: ahora });
  }
  if (conv.titulo) chatTitleEl.textContent = conv.titulo;
  renderConversaciones();
  sincronizarUrl();
}

function pintarTurno(turno) {
  const extra = turno.extra || {};
  const usuario = agregarMensaje(turno.pregunta, "user");
  if (extra.tiene_imagen) {
    const nota = document.createElement("p");
    nota.className = "msg__imgnote";
    nota.textContent = "Incluía una imagen";
    usuario.insertBefore(nota, usuario.firstChild);
  }
  const msg = agregarMensaje("", "assistant");
  const body = msg.querySelector(".msg__body");
  completarRespuesta(msg, body, { ...extra, respuesta: turno.respuesta }, turno.pregunta);
}

async function abrirConversacion(id) {
  if (consultando) return;
  try {
    const res = await api(`/api/conversaciones/${encodeURIComponent(id)}`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "No se pudo abrir la conversación.");
    conversacionId = data.id;
    chatTitleEl.textContent = data.titulo;
    quitarImagen();
    messagesEl.replaceChildren();
    data.turnos.forEach(pintarTurno);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    renderConversaciones();
    sincronizarUrl();
    cerrarPanel();
  } catch (e) {
    mostrarErrorLista(e.message || "No se pudo abrir la conversación.");
    if (conversacionId === null) nuevaConversacion();
  }
}

function nuevaConversacion() {
  if (consultando) return;
  conversacionId = null;
  quitarImagen();
  chatTitleEl.textContent = TITULO_POR_DEFECTO;
  messagesEl.replaceChildren(emptyStateTemplate.cloneNode(true));
  activarEstadoVacio();
  renderConversaciones();
  sincronizarUrl();
  cerrarPanel();
  if (!preguntaInput.disabled) preguntaInput.focus();
}

// Renombrar: el título se convierte en un campo; Enter guarda, Esc cancela
function editarTitulo(li, conv) {
  const abrir = li.querySelector(".conv__open");
  const acciones = li.querySelector(".conv__actions");
  const input = document.createElement("input");
  input.className = "conv__rename";
  input.value = conv.titulo;
  input.maxLength = 80;
  input.setAttribute("aria-label", "Nuevo nombre de la conversación");
  abrir.hidden = true;
  acciones.hidden = true;
  li.appendChild(input);
  input.focus();
  input.select();

  let cerrado = false;
  async function terminar(guardar) {
    if (cerrado) return;
    cerrado = true;
    const nuevo = input.value.trim();
    if (guardar && nuevo && nuevo !== conv.titulo) {
      try {
        const res = await apiJson(`/api/conversaciones/${encodeURIComponent(conv.id)}`, "PATCH", { titulo: nuevo });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "No se pudo cambiar el nombre.");
        conv.titulo = data.titulo;
        if (conv.id === conversacionId) chatTitleEl.textContent = data.titulo;
      } catch (e) {
        renderConversaciones();
        mostrarErrorLista(e.message);
        return;
      }
    }
    renderConversaciones();
  }
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); terminar(true); }
    else if (e.key === "Escape") { e.preventDefault(); terminar(false); }
  });
  input.addEventListener("blur", () => terminar(true));
}

// Borrar una conversación (con confirmación)
let convPorBorrar = null;

function pedirBorrado(conv) {
  convPorBorrar = conv;
  dlgConvText.textContent = `Se borrará «${conv.titulo}» con todas sus preguntas y respuestas. No se puede deshacer.`;
  dlgConvOk.disabled = false;
  dlgConv.showModal();
}

dlgConvOk.addEventListener("click", async () => {
  const conv = convPorBorrar;
  if (!conv) return;
  dlgConvOk.disabled = true;
  try {
    const res = await apiJson(`/api/conversaciones/${encodeURIComponent(conv.id)}`, "DELETE");
    if (!res.ok && res.status !== 404) throw new Error();
    conversaciones = conversaciones.filter((c) => c.id !== conv.id);
    const eraLaAbierta = conv.id === conversacionId;
    dlgConv.close();
    if (eraLaAbierta) nuevaConversacion();
    else renderConversaciones();
  } catch (e) {
    dlgConv.close();
    mostrarErrorLista("No se pudo borrar la conversación. Inténtalo de nuevo.");
  }
  convPorBorrar = null;
});

// ───────────────── Cuenta ─────────────────

document.querySelectorAll("dialog [data-cerrar]").forEach((b) => {
  b.addEventListener("click", () => b.closest("dialog").close());
});

btnLogout.addEventListener("click", async () => {
  btnLogout.disabled = true;
  try { await apiJson("/api/auth/logout", "POST"); } catch (e) { /* la sesión ya no existía */ }
  location.href = "/";
});

btnDeleteAccount.addEventListener("click", () => {
  dlgCuentaPass.value = "";
  dlgCuentaError.hidden = true;
  dlgCuentaOk.disabled = false;
  dlgCuenta.showModal();
  dlgCuentaPass.focus();
});

async function confirmarBorrarCuenta() {
  if (!dlgCuentaPass.value) {
    dlgCuentaError.textContent = "Escribe tu contraseña para confirmar.";
    dlgCuentaError.hidden = false;
    return;
  }
  dlgCuentaOk.disabled = true;
  dlgCuentaError.hidden = true;
  try {
    const res = await apiJson("/api/auth/cuenta", "DELETE", { password: dlgCuentaPass.value });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "No se pudo eliminar la cuenta.");
    location.href = "/";
  } catch (e) {
    dlgCuentaError.textContent = e.message;
    dlgCuentaError.hidden = false;
    dlgCuentaOk.disabled = false;
  }
}

dlgCuentaOk.addEventListener("click", confirmarBorrarCuenta);
formCuenta.addEventListener("submit", (e) => {
  e.preventDefault(); // Enter en el campo de contraseña confirma; no cierra el diálogo
  confirmarBorrarCuenta();
});
dlgCuenta.addEventListener("close", () => { dlgCuentaPass.value = ""; });

// ───────────────── Arranque ─────────────────

activarEstadoVacio();
cargarEstado();
cargarConversaciones().then(() => {
  const id = new URLSearchParams(location.search).get("c");
  if (id) abrirConversacion(id);
});