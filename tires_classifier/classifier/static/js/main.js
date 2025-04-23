let video = document.getElementById('video');
let canvas = document.getElementById('canvas');
let context = canvas.getContext('2d');
let stream = null;

function resetPage() {
    // Stop camera if it's running
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        video.srcObject = null;
        stream = null;
    }

    // Clear canvas
    context.clearRect(0, 0, canvas.width, canvas.height);

    // Reset file input
    document.getElementById('fileInput').value = '';

    // Clear result
    document.getElementById('result').innerHTML = '';
}

// Start camera
document.getElementById('startCamera').addEventListener('click', async () => {
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: "environment" }
        });
        video.srcObject = stream;
    } catch (err) {
        console.error("Error accessing camera:", err);
        alert("Error accessing camera. Please make sure you've granted camera permissions.");
    }
});

// Capture image
document.getElementById('captureImage').addEventListener('click', () => {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert to base64 and send to server
    const imageData = canvas.toDataURL('image/jpeg');
    uploadImage(imageData);
});

// Handle file input
document.getElementById('fileInput').addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        const formData = new FormData();
        formData.append('file', file);

        fetch('/upload/', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => handleResponse(data))
        .catch(error => handleError(error));
    }
});

// Upload image to server
function uploadImage(imageData) {
    const formData = new FormData();
    formData.append('image', imageData);

    fetch('/upload/', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => handleResponse(data))
    .catch(error => handleError(error));
}

function handleResponse(data) {
    if (data.success) {
        document.getElementById('result').innerHTML = `
            <h3>Classification Result:</h3>
            <img src="${data.image_url}" style="max-width: 100%; margin-bottom: 20px;">
            <p>${data.result}</p>
        `;
    } else {
        handleError(data.error);
    }
}

function handleError(error) {
    console.error('Error:', error);
    document.getElementById('result').innerHTML =
        `<p style="color: red;">Error: ${error}</p>`;
}
