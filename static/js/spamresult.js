const alertPlaceholder = document.getElementById('OutputResult');

function returnAlertClass(type) {
  if (type.toUpperCase().includes('HAM')) {
    return 'alert-success';
  } else if (type.toUpperCase().includes('SPAM')) {
    return 'alert-danger';
  } else {
    return '';
  }
}

const appendAlert = (message, type) => {
  const alertClass = returnAlertClass(type)
  const wrapper = document.createElement('div');
  wrapper.className = `alert ${alertClass} alert-dismissible` // Use template literals for cleaner class assignment
  wrapper.innerHTML = `
    <div><i class="bi bi-check-circle"></i> EMAIL TERDETEKSI ${message}</div>
    <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
  `;
  alertPlaceholder.append(wrapper);
};

// const alertTrigger = document.getElementById('liveAlertBtn');
// if (alertTrigger) {
//   alertTrigger.addEventListener('click', () => {
//     // You can remove this line as it doesn't seem to be doing anything specific
//     // appendAlert('');
//   });
// }

$(document).ready(function() {
  $("#submit-button").click(function(event) {
    event.preventDefault();

    var emailText = $("#email-text").val();

    $.ajax({
      url: "/detect",
      type: "POST",
      data: { emailText: emailText },
      success: function(response) {
        const message = response === 'HAM' ? 'HAM' : 'SPAM';
        appendAlert(message, response);
      },
      error: function(jqXHR, textStatus, errorThrown) {
        console.error("Error:", textStatus, errorThrown);
        $("#OutputResult").html("Error occurred. Please try again.");
      }
    });
  });
});
