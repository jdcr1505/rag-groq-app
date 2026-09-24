const apiStatusEl = document.getElementById("api-status");
const pdfListEl = document.getElementById("pdf-list");
const pdfInputEl = document.getElementById("pdf-input");
const vsStatusEl = document.getElementById("vs-status");
const btnBuild = document.getElementById("btn-build");
const btnLoad = document.getElementById("btn-load");
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
  const res = await fetch("/api/status");
  const data = await res.json();

  if (data.api_key_ok) {
    apiStatusEl.textContent = "API key de Groq detectada";
    apiStatusEl.className = "status status--ok";
  } else {
    apiStatusEl.textContent = data.api_key_error;
    apiStatusEl.className = "status status--error";
  }

  renderPdfList(data.pdfs);

  if (data.vector_store_loaded) {
    marcarVectorStoreListo("Base vectorial cargada y lista.");
  } else if (data.vector_store_on_disk) {
    vsStatusEl.textContent = "Existe en disco — presiona 'Cargar existente'.";
    vsStatusEl.className = "status status--pending";
    btnLoad.disabled = false;
  } else {
    vsStatusEl.textContent = "Aún no construida.";
    vsStatusEl.className = "status status--pending";
    btnLoad.disabled = true;
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

    const btnDel = document.createElement("button");
    btnDel.textContent = "🗑";
    btnDel.className = "pdf-list__delete";
    btnDel.title = `Eliminar ${nombre}`;
    btnDel.addEventListener("click", () => eliminarPdf(nombre));

    li.appendChild(span);
    li.appendChild(btnDel);
    pdfListEl.appendChild(li);
  });
}

async function eliminarPdf(nombre) {
  if (!confirm(`¿Eliminar "${nombre}" de la carpeta pdfs/?\n\nRecuerda volver a presionar "Construir" después, para que la base vectorial deje de contener su información.`)) {
    return;
  }
  try {
    const res = await fetch("/api/delete_pdf", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ nombre }),
    });
    const data = await res.json();
    if (res.ok) {
      renderPdfList(data.pdfs);
    } else {
      alert(data.error || "No se pudo eliminar el archivo.");
    }
  } catch (e) {
    alert("Error de red al eliminar el archivo.");
  }
}

function marcarVectorStoreListo(mensaje) {
  vectorStoreListo = true;
  vsStatusEl.textContent = mensaje;
  vsStatusEl.className = "status status--ok";
  preguntaInput.disabled = false;
  submitBtn.disabled = false;
  if (emptyStateEl) emptyStateEl.remove();
}

// ───────────────── Subida de PDFs ─────────────────

pdfInputEl.addEventListener("change", async () => {
  if (!pdfInputEl.files.length) return;

  const formData = new FormData();
  for (const archivo of pdfInputEl.files) {
    formData.append("pdfs", archivo);
  }

  const res = await fetch("/api/upload", { method: "POST", body: formData });
  const data = await res.json();

  if (res.ok) {
    renderPdfList(data.pdfs);
  } else {
    alert(data.error || "No se pudo subir el archivo.");
  }
  pdfInputEl.value = "";
});

// ───────────────── Construcción de la base vectorial ─────────────────

btnBuild.addEventListener("click", async () => {
  btnBuild.disabled = true;
  btnLoad.disabled = true;
  buildLogEl.hidden = false;
  buildLogEl.textContent = "Iniciando…\n(esto puede tardar 1-2 minutos la primera vez)";
  vsStatusEl.textContent = "Construyendo…";
  vsStatusEl.className = "status status--pending";

  try {
    const res = await fetch("/api/build", { method: "POST" });
    const data = await res.json();

    buildLogEl.textContent = (data.log || []).join("\n");

    if (res.ok && data.ok) {
      marcarVectorStoreListo("Base vectorial lista.");
    } else {
      vsStatusEl.textContent = data.error || "Error al construir la base.";
      vsStatusEl.className = "status status--error";
    }
  } catch (e) {
    vsStatusEl.textContent = "Error de red al construir la base.";
    vsStatusEl.className = "status status--error";
  } finally {
    btnBuild.disabled = false;
    btnLoad.disabled = false;
  }
});

btnLoad.addEventListener("click", async () => {
  btnBuild.disabled = true;
  btnLoad.disabled = true;
  vsStatusEl.textContent = "Cargando…";
  vsStatusEl.className = "status status--pending";

  try {
    const res = await fetch("/api/load", { method: "POST" });
    const data = await res.json();

    if (res.ok && data.ok) {
      marcarVectorStoreListo("Base vectorial cargada.");
    } else {
      vsStatusEl.textContent = data.error || "Error al cargar la base.";
      vsStatusEl.className = "status status--error";
    }
  } catch (e) {
    vsStatusEl.textContent = "Error de red al cargar la base.";
    vsStatusEl.className = "status status--error";
  } finally {
    btnBuild.disabled = false;
    btnLoad.disabled = false;
  }
});

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
