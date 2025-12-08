document.addEventListener('DOMContentLoaded', () => {
    const uploadButton = document.getElementById('upload-button');
    const fileInput = document.getElementById('file-input');

    uploadButton.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', (event) => {
        const file = event.target.files[0]; // Get the first file selected

        if (file) {
            processFile(file);
        } else {
            console.log('No file selected.');
        }
    });
});

function processFile(file) {
    const reader = new FileReader();

    reader.onload = (e) => {
        const data = JSON.parse(e.target.result);
        initGame(data);
    };

    reader.onerror = (e) => {
        console.error('Error reading file:', e);
    };

    reader.readAsText(file);
}
