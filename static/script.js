// Simple client script: refresh the dashboard every 2 minutes
setTimeout(function refresh() {
  fetch(window.location.href).then(() => {
    // reload page to get fresh server-side predictions
    window.location.reload();
  });
}, 120000);

// Placeholder for future client scripts (CSV preview is handled inline in the template)
