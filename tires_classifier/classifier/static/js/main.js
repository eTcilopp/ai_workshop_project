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
        const reader = new FileReader();
        reader.onload = (e) => {
            uploadImage(e.target.result);
        };
        reader.readAsDataURL(file);
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
    .then(data => {
        if (data.success) {
            document.getElementById('result').innerHTML =
                `<h3>Classification Result:</h3><p>${data.result}</p>`;
        } else {
            document.getElementById('result').innerHTML =
                `<p style="color: red;">Error: ${data.error}</p>`;
        }
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('result').innerHTML =
            `<p style="color: red;">Error uploading image</p>`;
    });
}
