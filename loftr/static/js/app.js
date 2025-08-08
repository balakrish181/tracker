document.getElementById('uploadForm').addEventListener('submit', async function (event) {
    event.preventDefault();
    const submitBtn = document.getElementById('submitBtn');
    submitBtn.disabled = true;
    submitBtn.textContent = 'Processing...';

    const image1 = document.getElementById('image1').files[0];
    const image2 = document.getElementById('image2').files[0];
    const radius = document.getElementById('radius').value;

    if (!image1 || !image2) {
        alert('Please select both images.');
        submitBtn.disabled = false;
        submitBtn.textContent = 'Match Moles';
        return;
    }

    const formData = new FormData();
    formData.append('image_path1', image1);
    formData.append('image_path2', image2);
    formData.append('radius', radius);

    try {
        const response = await fetch('/match_moles', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (response.ok) {
            // Display results
            const resultsDiv = document.getElementById('results');
            resultsDiv.classList.remove('d-none');

            // Message
            const messageDiv = document.getElementById('message');
            messageDiv.textContent = result.message;
            messageDiv.className = 'alert alert-success';

            // Annotated Image
            document.getElementById('annotatedImage').src = result.image_path + '?t=' + new Date().getTime(); // Avoid cache
            
            // Yolo Image
            document.getElementById('yoloImage').src = result.yolo_image_path + '?t=' + new Date().getTime(); // Avoid cache
            // Matches Count
            document.getElementById('matchesCount').textContent = result.matches_count;

            // Bounding Boxes
            document.getElementById('bboxes').textContent = JSON.stringify(result.bboxes, null, 2);

            // Query Points
            document.getElementById('queryPoints').textContent = JSON.stringify(result.query_points, null, 2);

            // Matched Points
            document.getElementById('matchedPoints').textContent = JSON.stringify(result.matched_points, null, 2);
        } else {
            throw new Error(result.detail || 'An error occurred');
        }
    } catch (error) {
        const messageDiv = document.getElementById('message');
        messageDiv.textContent = error.message;
        messageDiv.className = 'alert alert-danger';
        document.getElementById('results').classList.remove('d-none');
    } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = 'Match Moles';
    }
});