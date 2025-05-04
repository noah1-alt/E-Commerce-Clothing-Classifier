document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('upload-form');
    const fileInput = document.getElementById('fileInput');
    const fileNameSpan = document.querySelector('.file-upload span');
    const imagePreview = document.getElementById('imagePreview');
    const classifyButton = document.getElementById('classifyButton');
    const loading = document.getElementById('loading');

    // Update file name and show image preview
    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        fileNameSpan.textContent = file ? file.name : 'No file chosen';
        imagePreview.innerHTML = '<p class="preview-placeholder">No image selected</p>';
        classifyButton.style.display = 'none';

        if (file) {
            // Broader MIME type check
            const validTypes = ['image/png', 'image/jpeg', 'image/jpg'];
            if (validTypes.includes(file.type)) {
                console.log('Valid image selected:', file.name, file.type);
                const img = document.createElement('img');
                img.src = URL.createObjectURL(file);
                img.alt = 'Selected image';
                img.onload = () => {
                    console.log('Image loaded successfully');
                    imagePreview.innerHTML = ''; // Clear placeholder
                    imagePreview.appendChild(img);
                    classifyButton.style.display = 'block';
                };
                img.onerror = () => {
                    console.error('Failed to load image');
                    fileNameSpan.textContent = 'Error loading image';
                    imagePreview.innerHTML = '<p class="preview-placeholder">Error loading image</p>';
                };
            } else {
                console.log('Invalid file type:', file.type);
                fileNameSpan.textContent = 'Please select a PNG or JPEG image';
            }
        }
    });

    // Show loading indicator on form submit
    form.addEventListener('submit', () => {
        console.log('Form submitted');
        loading.style.display = 'block';
        fileNameSpan.textContent = 'No file chosen';
        imagePreview.innerHTML = '<p class="preview-placeholder">No image selected</p>';
        classifyButton.style.display = 'none';
    });

    // Cleanup object URLs on page unload to prevent memory leaks
    window.addEventListener('unload', () => {
        const img = imagePreview.querySelector('img');
        if (img && img.src.startsWith('blob:')) {
            URL.revokeObjectURL(img.src);
        }
    });
});