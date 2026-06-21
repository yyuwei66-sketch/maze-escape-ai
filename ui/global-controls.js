(function () {
  let manualOverlay = null;

  function openManual() {
    if (!manualOverlay) {
      manualOverlay = document.createElement('div');
      manualOverlay.id = 'manual-overlay';
      manualOverlay.style.cssText = `
        display: none;
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 100%;
        background: rgba(0, 0, 0, 0.7);
        z-index: 9999;
        justify-content: center;
        align-items: center;
        overflow: hidden;
      `;
      document.body.appendChild(manualOverlay);

      window.closeManualOverlay = function () {
        manualOverlay.style.display = 'none';
        window.gamePaused = false;
      };
    }

    if (!manualOverlay.dataset.loaded) {
        fetch('/ui/manual/manual.html')
        .then(res => res.text())
        .then(html => {
          const parser = new DOMParser();
          const doc = parser.parseFromString(html, 'text/html');

          let styleText = '';
          doc.querySelectorAll('style').forEach(style => {
            styleText += style.textContent + '\n';
          });
          styleText = styleText.replace(
            /(^|\})\s*(html\s*,\s*body|body\s*,\s*html|body|html)\s*\{/g,
            '$1 #manual-content-box {'
          );

          const scopedStyle = document.createElement('style');
          scopedStyle.id = 'manual-scoped-style';
          scopedStyle.textContent = styleText;
          document.head.appendChild(scopedStyle);

          const contentDiv = document.createElement('div');
          contentDiv.id = 'manual-content-box';
          contentDiv.style.cssText = `
            width: 100%;
            height: 100%;
            transform: scale(0.7);
            transform-origin: center center;
            position: relative;
          `;
          contentDiv.innerHTML = doc.body.innerHTML;
          manualOverlay.appendChild(contentDiv);

          const scripts = doc.querySelectorAll('script');
          scripts.forEach(oldScript => {
            const newScript = document.createElement('script');
            newScript.textContent = oldScript.textContent;
            document.body.appendChild(newScript);
          });

          manualOverlay.dataset.loaded = 'true';
        });
    }

    manualOverlay.style.display = 'flex';
    window.gamePaused = true;
  }

  function isManualOpen() {
    return manualOverlay && manualOverlay.style.display === 'flex';
  }

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      if (isManualOpen()) {
        window.closeManualOverlay();
      } else {
        window.location.href = '/';
      }
    }
    if (e.key.toLowerCase() === 'm') {
      if (isManualOpen()) {
        window.closeManualOverlay();
      } else {
        openManual();
      }
    }
  });
})();