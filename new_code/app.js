// WebSocket URL - replace with your server's address
const wsUrl = 'ws://localhost:8765';

// Access the video element
const video = document.getElementById('video');

// Setup the WebSocket connection
const socket = new WebSocket(wsUrl);

socket.onopen = () => {
    console.log('WebSocket connection established');
};

socket.onerror = (error) => {
    console.error('WebSocket error:', error);
};

socket.onclose = () => {
    console.log('WebSocket connection closed');
};

// Access the webcam and stream video
navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
        
        // Create a canvas to capture frames from the video
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');

        // Set canvas size to match the video
        video.addEventListener('loadedmetadata', () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
        });

        // Function to capture frames and send them to the WebSocket server
        function sendFrame() {
            // Draw the current frame to the canvas
            context.drawImage(video, 0, 0, canvas.width, canvas.height);
            
            // Convert the canvas to a JPEG image
            canvas.toBlob(blob => {
                // Send the image data to the WebSocket server
                if (socket.readyState === WebSocket.OPEN) {
                    socket.send(blob);
                }
            }, 'image/jpeg');
            
            // Request the next frame
            requestAnimationFrame(sendFrame);
        }

        // Start sending frames
        sendFrame();
    })
    .catch(err => {
        console.error('Error accessing webcam:', err);
    });
