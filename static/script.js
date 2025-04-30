document.getElementById('fileInput').addEventListener('change', function(e) {
    const fileName = e.target.files.length > 0 ? e.target.files[0].name : 'No file chosen';
    document.querySelector('.file-upload span').textContent = fileName;
});