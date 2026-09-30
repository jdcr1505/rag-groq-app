const apiStatusEl = document.getElementById("api-status");
const pdfListEl = document.getElementById("pdf-list");
const vsStatusEl = document.getElementById("vs-status");
const buildLogEl = document.getElementById("build-log");
const kSlider = document.getElementById("k-slider");
const kValue = document.getElementById("k-value");
const verboseToggle = document.getElementById("verbose-toggle");
const messagesEl = document.getElementById("messages");
const emptyStateEl = document.getElementById("empty-state");
const chatForm = document.getElementById("chat-form");
const preguntaInput = document.getElementById("pregunta-input");
const submitBtn = chatForm.querySelector("button[type=submit]");

let vectorStoreListo = false;

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
    apiStatusEl.textContent = "API key de Groq detectada";
    apiStatusEl.className = "status status--ok";
  } else {
    apiStatusEl.textContent = data.api_key_error;
    apiStatusEl.className = "status status--error";
  }

  renderPdfList(data.pdfs);

  if (data.log && data.log.length) {
    buildLogEl.hidden = false;
    buildLogEl.textContent = data.log.join("\n");
  }

  if (data.init_status === "lista") {
    marcarVectorStoreListo("Base vectorial lista.");
  } else if (data.init_status === "error") {
    vsStatusEl.textContent = data.init_error || "Error al preparar la base vectorial.";
    vsStatusEl.className = "status status--error";
  } else {
    vsStatusEl.textContent = "Preparando base vectorial…";
    vsStatusEl.className = "status status--pending";
    setTimeout(cargarEstado, 3000);
  }
}

function renderPdfList(pdfs) {
  pdfListEl.innerHTML = "";
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

    const span = document.createElement("span");
    span.textContent = nombre;
    span.className = "pdf-list__name";

    li.appendChild(span);
    pdfListEl.appendChild(li);
  });
}

function marcarVectorStoreListo(mensaje) {
  vectorStoreListo = true;
  vsStatusEl.textContent = mensaje;
  vsStatusEl.className = "status status--ok";
  preguntaInput.disabled = false;
  submitBtn.disabled = false;
  if (emptyStateEl) emptyStateEl.remove();
}

// ───────────────── Slider de k ─────────────────

kSlider.addEventListener("input", () => {
  kValue.textContent = kSlider.value;
});

// ───────────────── Diagnóstico: qué se indexó realmente ─────────────────

const btnIndexed = document.getElementById("btn-indexed");
const indexedPanel = document.getElementById("indexed-panel");

btnIndexed.addEventListener("click", async () => {
  if (indexedPanel.hidden) {
    btnIndexed.textContent = "Consultando…";
    try {
      const res = await fetch("/api/indexed");
      const data = await res.json();

      indexedPanel.innerHTML = "";
      if (!res.ok) {
        indexedPanel.innerHTML = `<p class="indexed-item">${data.error || "Error al consultar."}</p>`;
      } else if (!data.fuentes.length) {
        indexedPanel.innerHTML = `<p class="indexed-item">La base vectorial está vacía.</p>`;
      } else {
        data.fuentes.forEach((f) => {
          const div = document.createElement("div");
          div.className = "indexed-item";
          div.innerHTML = `
            <span class="indexed-item__name">${f.archivo}</span>
            <div class="indexed-item__stats">${f.fragmentos} fragmentos · ${f.caracteres_totales.toLocaleString()} caracteres</div>
            <div class="indexed-item__sample">"${f.muestra}…"</div>
          `;
          indexedPanel.appendChild(div);
        });
      }
      indexedPanel.hidden = false;
      btnIndexed.textContent = "Ocultar";
    } catch (e) {
      indexedPanel.innerHTML = `<p class="indexed-item">Error de red.</p>`;
      indexedPanel.hidden = false;
      btnIndexed.textContent = "Ocultar";
    }
  } else {
    indexedPanel.hidden = true;
    btnIndexed.textContent = "Ver qué se indexó";
  }
});

// ───────────────── Chat ─────────────────

function agregarMensaje(texto, tipo) {
  const div = document.createElement("div");
  div.className = `msg msg--${tipo}`;
  const p = document.createElement("p");
  p.textContent = texto;
  div.appendChild(p);
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return div;
}

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const pregunta = preguntaInput.value.trim();
  if (!pregunta || !vectorStoreListo) return;

  agregarMensaje(pregunta, "user");
  preguntaInput.value = "";
  preguntaInput.disabled = true;
  submitBtn.disabled = true;

  const pensando = agregarMensaje("Consultando…", "assistant");
  pensando.querySelector("p").innerHTML = '<span class="spinner"></span>Consultando…';

  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pregunta, k: Number(kSlider.value) }),
    });
    const data = await res.json();

    if (!res.ok) {
      pensando.classList.add("msg--error");
      pensando.querySelector("p").textContent = data.error || "Ocurrió un error.";
    } else {
      pensando.querySelector("p").textContent = data.respuesta;

      const meta = document.createElement("div");
      meta.className = "msg__meta";
      meta.textContent = `~${data.tokens_contexto_aprox} tokens de contexto · ${data.fragmentos.length} fragmentos recuperados`;
      pensando.appendChild(meta);

      if (verboseToggle.checked && data.fragmentos.length) {
        const cont = document.createElement("div");
        cont.className = "fragments";
        data.fragmentos.forEach((f) => {
          const frag = document.createElement("div");
          frag.className = "fragment";
          frag.innerHTML = `<span class="fragment__source">${f.fuente} — Pág. ${f.pagina}</span>${f.texto}`;
          cont.appendChild(frag);
        });
        pensando.appendChild(cont);
      }
    }
  } catch (err) {
    pensando.classList.add("msg--error");
    pensando.querySelector("p").textContent = "Error de red al consultar.";
  } finally {
    preguntaInput.disabled = false;
    submitBtn.disabled = false;
    preguntaInput.focus();
  }
});

cargarEstado();
