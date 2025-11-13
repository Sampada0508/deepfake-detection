// small UX helpers
document.addEventListener('DOMContentLoaded', () => {
  const imgInput = document.getElementById('image-file');
  const vidInput = document.getElementById('video-file');

  // show filename on choose
  function bindInput(input){
    if(!input) return;
    input.addEventListener('change', e=>{
      const file = input.files[0];
      if(!file) return;
      // small validation
      if(input.accept.includes('image') && !file.type.startsWith('image')) {
        alert('Please pick an image file.');
        input.value = '';
      }
      if(input.accept.includes('video') && !file.type.startsWith('video')) {
        alert('Please pick a video file.');
        input.value = '';
      }
    });
  }
  bindInput(imgInput);
  bindInput(vidInput);
});
