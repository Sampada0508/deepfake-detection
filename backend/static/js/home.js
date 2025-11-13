

// openChat: redirect to Flask chat page
function openChat(){
  // use Jinja url_for to create correct path
  window.location.href = "{{ url_for('chat_page') }}";
}

// small friendly link behavior (open visualization/articles in new tab)
document.addEventListener('DOMContentLoaded', () => {
  const links = document.querySelectorAll('a');
  links.forEach(a=>{
    if(a.getAttribute('href') && (a.getAttribute('href').startsWith('/visualization') || a.getAttribute('href').startsWith('/articles'))){
      a.addEventListener('click', (e)=>{
        // allow routing if endpoints exist; otherwise open in new window as placeholder
        // no-op here (will try to load)
      });
    }
  });
});
