// Contact form: sends the message to Formspree without leaving the page.
// Without JavaScript the form still works as a normal POST to Formspree.
(function () {
  var form = document.getElementById('contactForm');
  if (!form || !window.fetch || !window.FormData) return;
  var status = document.getElementById('contactStatus');
  var button = document.getElementById('sendMessageButton');

  function show(kind, text) {
    status.className = 'contact-status alert alert-' + kind;
    status.textContent = text;
  }

  form.addEventListener('submit', function (event) {
    event.preventDefault();
    if (!form.checkValidity()) {
      form.reportValidity();
      return;
    }
    button.disabled = true;
    show('info', 'Sending…');
    fetch(form.action, {
      method: 'POST',
      body: new FormData(form),
      headers: { Accept: 'application/json' }
    }).then(function (response) {
      if (!response.ok) throw new Error('HTTP ' + response.status);
      form.reset();
      var filled = form.querySelectorAll('.floating-label-form-group-with-value');
      for (var i = 0; i < filled.length; i++) filled[i].classList.remove('floating-label-form-group-with-value');
      show('success', 'Thank you, your message has been sent. I will get back to you soon.');
    }).catch(function () {
      show('danger', 'Sorry, your message could not be sent. Please try again later, or email me directly.');
    }).then(function () {
      button.disabled = false;
    });
  });
})();
