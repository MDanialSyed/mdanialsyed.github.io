// Google Analytics 4, loaded only when _config.yml sets google_analytics.
(function () {
  var script = document.currentScript;
  var id = script && script.getAttribute('data-ga-id');
  if (!id) return;
  window.dataLayer = window.dataLayer || [];
  function gtag() { window.dataLayer.push(arguments); }
  window.gtag = gtag;
  gtag('js', new Date());
  gtag('config', id);
})();
