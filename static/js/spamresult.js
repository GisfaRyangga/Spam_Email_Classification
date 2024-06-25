const alertPlaceholder = document.getElementById('OutputResult')
const appendAlert = (message, type) => {
  const wrapper = document.createElement('div')
  wrapper.innerHTML = [
    '<div class="alert alert-success alert-dismissible" role="alert">                             ',
    '<div><i class="bi bi-check-circle"></i> EMAIL TERDETEKSI HAM</div>                           ',
    '<button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button> ',
    '</div>'
  ].join('')

  // For Detected SPAM
  // '<div class="alert alert-danger alert-dismissible" role="alert">                               ',
  // '<div><i class="bi bi-check-circle"></i> EMAIL TERDETEKSI SPAM</div>                           ',
  // '<button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>  ',
  // '</div>'

  alertPlaceholder.append(wrapper)
}

const alertTrigger = document.getElementById('liveAlertBtn')
if (alertTrigger) {
  alertTrigger.addEventListener('click', () => {
    appendAlert('')
  })
}