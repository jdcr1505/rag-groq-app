/* ==========================================================
   LANDING — animaciones progresivas, accesibles y sin dependencias
   ========================================================== */
(() => {
  "use strict";

  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const revealItems = document.querySelectorAll("[data-reveal]");

  // Revelación de contenido al entrar en pantalla.
  // Si el navegador no soporta IntersectionObserver, todo queda visible.
  revealItems.forEach((element) => {
    const delay = Number(element.dataset.delay || 0);
    element.style.setProperty("--delay", `${delay}ms`);
  });

  if (reduceMotion || !("IntersectionObserver" in window)) {
    revealItems.forEach((element) => element.classList.add("is-visible"));
  } else {
    const revealObserver = new IntersectionObserver((entries, observer) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        entry.target.classList.add("is-visible");
        observer.unobserve(entry.target);
      });
    }, {
      threshold: 0.12,
      rootMargin: "0px 0px -45px 0px"
    });

    revealItems.forEach((element) => revealObserver.observe(element));
  }

  // Parallax discreto, limitado a pantallas grandes y movimiento permitido.
  const parallaxItems = document.querySelectorAll("[data-parallax]");
  const canUseParallax = !reduceMotion &&
    window.matchMedia("(min-width: 801px)").matches &&
    "IntersectionObserver" in window;

  if (canUseParallax && parallaxItems.length) {
    let ticking = false;
    const visibleParallax = new Set();

    const visibilityObserver = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) visibleParallax.add(entry.target);
        else visibleParallax.delete(entry.target);
      });
    }, { rootMargin: "100px 0px" });

    parallaxItems.forEach((item) => visibilityObserver.observe(item));

    const updateParallax = () => {
      const viewportCenter = window.innerHeight / 2;

      visibleParallax.forEach((item) => {
        const rect = item.getBoundingClientRect();
        const distance = (rect.top + rect.height / 2 - viewportCenter) / viewportCenter;
        const offset = Math.max(-18, Math.min(18, distance * -12));
        const image = item.querySelector("img");
        if (image) image.style.transform = `scale(1.08) translate3d(0, ${offset}px, 0)`;
      });

      ticking = false;
    };

    const requestParallaxUpdate = () => {
      if (ticking) return;
      ticking = true;
      window.requestAnimationFrame(updateParallax);
    };

    window.addEventListener("scroll", requestParallaxUpdate, { passive: true });
    window.addEventListener("resize", requestParallaxUpdate, { passive: true });
    requestParallaxUpdate();
  }

})();
