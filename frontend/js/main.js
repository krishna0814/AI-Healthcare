/**
 * AI Healthcare System - Main JavaScript
 * Handles API calls and UI interactions
 */

// API Configuration
const API_BASE_URL = 'http://localhost:5000';

// Check API status on page load
document.addEventListener('DOMContentLoaded', function() {
    console.log('AI Healthcare System loaded successfully');
    checkAPIStatus();
});

/**
 * Check if the backend API is running
 */
async function checkAPIStatus() {
    try {
        const response = await fetch(`${API_BASE_URL}/models/status`);
        const data = await response.json();
        console.log('API Status:', data);
    } catch (error) {
        console.error('API not available. Make sure the backend is running on port 5000.');
    }
}

/**
 * Make a prediction request to the API
 */
async function makePrediction(endpoint, data) {
    try {
        const response = await fetch(`${API_BASE_URL}/predict/${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        return await response.json();
    } catch (error) {
        console.error('Prediction error:', error);
        throw error;
    }
}

/**
 * Capitalize first letter
 */
function capitalizeFirstLetter(string) {
    return string.charAt(0).toUpperCase() + string.slice(1);
}

/**
 * Validate form input
 */
function validateForm(form) {
    const inputs = form.querySelectorAll('input[required], select[required]');
    let isValid = true;
    inputs.forEach(input => {
        if (!input.value || input.value === '') {
            isValid = false;
            input.style.borderColor = '#e74c3c';
        } else {
            input.style.borderColor = '#e0e0e0';
        }
    });
    return isValid;
}

/**
 * Set button loading state
 */
function setButtonLoading(button, loading) {
    if (loading) {
        button.disabled = true;
        button.dataset.originalText = button.innerHTML;
        button.innerHTML = '<span class="loading"></span> Processing...';
    } else {
        button.disabled = false;
        button.innerHTML = button.dataset.originalText || button.innerHTML;
    }
}

/**
 * Get form data as object
 */
function getFormData(form) {
    const formData = new FormData(form);
    const data = {};
    for (let [key, value] of formData.entries()) {
        data[key] = value;
    }
    return data;
}

window.AIHealthcare = {
    makePrediction,
    displayPredictionResult: function(result, diseaseType) {
        const resultSection = document.getElementById('resultSection');
        if (!resultSection) return;
        resultSection.style.display = 'block';
    },
    validateForm,
    setButtonLoading,
    getFormData,
    API_BASE_URL
};

