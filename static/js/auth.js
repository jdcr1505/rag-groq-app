// ──────────────────────────────────────────────────────────────
// Acceso: alterna entre iniciar sesión y crear cuenta, valida lo
// básico en el navegador (el servidor vuelve a validar todo) y
// envía el formulario como JSON a /api/auth/login o /registro.
// ──────────────────────────────────────────────────────────────

const form = document.getElementById("form");
const tituloEl = document.getElementById("titulo");
const campoNombre = document.getElementById("campo-nombre");
const nombreEl = document.getElementById("nombre");
const correoEl = document.getElementById("correo");
const passEl = document.getElementById("password");
const ayudaPass = document.getElementById("ayuda-pass");
const errorEl = document.getElementById("error");
const enviarBtn = document.getElementById("enviar");
const cambioEl = document.getElementById("cambio");
const verPass = document.getElementById("ver-pass");
const tabs = document.querySelectorAll(".tab");

let modo = document.body.dataset.modo === "registro" ? "registro" : "login";

function mostrarError(texto, campo) {
  errorEl.textContent = texto;
  errorEl.hidden = false;
  [nombreEl, correoEl, passEl].forEach((c) => c.removeAttribute("aria-invalid"));
  if (campo) { campo.setAttribute("aria-invalid", "true"); campo.focus(); }
}

function limpiarError() {
  errorEl.hidden = true;
  [nombreEl, correoEl, passEl].forEach((c) => c.removeAttribute("aria-invalid"));
}

function aplicarModo(nuevo, { enfocar = false } = {}) {
  modo = nuevo;
  const registro = modo === "registro";
  tituloEl.textContent = registro ? "Crear cuenta" : "Iniciar sesión";
  document.title = `${registro ? "Crear cuenta" : "Iniciar sesión"} · Normativa colombiana`;
  campoNombre.hidden = !registro;
  nombreEl.required = registro;
  ayudaPass.hidden = !registro;
  passEl.autocomplete = registro ? "new-password" : "current-password";
  enviarBtn.textContent = registro ? "Crear cuenta" : "Iniciar sesión";
  tabs.forEach((t) => t.setAttribute("aria-selected", String(t.dataset.modo === modo)));
  cambioEl.replaceChildren();
  cambioEl.append(registro ? "¿Ya tienes cuenta? " : "¿Eres nuevo? ");
  const enlace = document.createElement("button");
  enlace.type = "button";
  enlace.textContent = registro ? "Inicia sesión" : "Crea tu cuenta";
  enlace.addEventListener("click", () => aplicarModo(registro ? "login" : "registro", { enfocar: true }));
  cambioEl.appendChild(enlace);
  limpiarError();
  try { history.replaceState(null, "", registro ? "/acceso?modo=registro" : "/acceso"); } catch (e) { /* sin historial */ }
  if (enfocar) (registro ? nombreEl : correoEl).focus();
}

tabs.forEach((t) => t.addEventListener("click", () => aplicarModo(t.dataset.modo, { enfocar: true })));

verPass.addEventListener("click", () => {
  const oculto = passEl.type === "password";
  passEl.type = oculto ? "text" : "password";
  verPass.textContent = oculto ? "Ocultar" : "Mostrar";
  verPass.setAttribute("aria-pressed", String(oculto));
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  limpiarError();
  const registro = modo === "registro";
  const nombre = nombreEl.value.trim();
  const correo = correoEl.value.trim();
  const password = passEl.value;

  if (registro && !nombre) return mostrarError("Escribe tu nombre.", nombreEl);
  if (!/^[^@\s]+@[^@\s]+\.[^@\s.]{2,}$/.test(correo)) return mostrarError("Escribe un correo válido, por ejemplo nombre@dominio.com.", correoEl);
  if (!password) return mostrarError("Escribe tu contraseña.", passEl);
  if (registro && password.length < 8) return mostrarError("La contraseña debe tener al menos 8 caracteres.", passEl);

  enviarBtn.disabled = true;
  enviarBtn.textContent = registro ? "Creando cuenta…" : "Entrando…";
  try {
    const res = await fetch(registro ? "/api/auth/registro" : "/api/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(registro ? { nombre, correo, password } : { correo, password }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || "No se pudo completar la operación. Inténtalo de nuevo.");
    location.href = "/chat";
  } catch (err) {
    const sinRed = err instanceof TypeError;
    mostrarError(sinRed ? "No hay conexión con el servidor. Revisa tu internet e inténtalo de nuevo." : err.message);
    enviarBtn.disabled = false;
    enviarBtn.textContent = registro ? "Crear cuenta" : "Iniciar sesión";
  }
});

aplicarModo(modo);