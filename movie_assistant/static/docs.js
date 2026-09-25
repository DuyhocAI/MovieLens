window.addEventListener('load', () => {
  if (!window.SwaggerUIBundle) {
    document.querySelector('#swagger-ui').textContent =
      'Swagger UI could not be loaded (offline?). The raw spec is available at /openapi.json.';
    return;
  }
  window.SwaggerUIBundle({ url: '/openapi.json', dom_id: '#swagger-ui', deepLinking: true });
});
