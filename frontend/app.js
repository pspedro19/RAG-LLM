// ============================================
// Configuration
// ============================================
const CONFIG = {
    apiUrl: localStorage.getItem('apiUrl') || 'http://localhost:8000',
    apiKey: localStorage.getItem('apiKey') || '',
    conversationId: null
};

// ============================================
// State Management
// ============================================
const state = {
    isTyping: false,
    currentImage: null,
    messages: []
};

// ============================================
// DOM Elements
// ============================================
const elements = {
    // Sidebar
    visionSidebar: document.getElementById('visionSidebar'),
    toggleVision: document.getElementById('toggleVision'),
    closeSidebar: document.getElementById('closeSidebar'),

    // Image Upload
    uploadArea: document.getElementById('uploadArea'),
    imageInput: document.getElementById('imageInput'),
    selectImageBtn: document.getElementById('selectImageBtn'),
    previewArea: document.getElementById('previewArea'),
    previewImage: document.getElementById('previewImage'),
    removeImage: document.getElementById('removeImage'),
    classifyBtn: document.getElementById('classifyBtn'),
    resultsArea: document.getElementById('resultsArea'),
    visionLoader: document.getElementById('visionLoader'),

    // Chat
    welcomeScreen: document.getElementById('welcomeScreen'),
    chatMessages: document.getElementById('chatMessages'),
    messageInput: document.getElementById('messageInput'),
    sendBtn: document.getElementById('sendBtn'),
    attachBtn: document.getElementById('attachBtn'),

    // Email
    enableEmail: document.getElementById('enableEmail'),
    emailInput: document.getElementById('emailInput'),

    // Settings
    toggleSettings: document.getElementById('toggleSettings'),
    settingsModal: document.getElementById('settingsModal'),
    closeSettings: document.getElementById('closeSettings'),
    saveSettings: document.getElementById('saveSettings'),
    resetSettings: document.getElementById('resetSettings'),

    // Toast
    toastContainer: document.getElementById('toastContainer')
};

// ============================================
// Initialization
// ============================================
document.addEventListener('DOMContentLoaded', () => {
    initEventListeners();
    loadSettings();

    // Generate conversation ID
    CONFIG.conversationId = generateUUID();

    console.log('✅ Frontend initialized');
    showToast('Sistema listo', 'success');
});

// ============================================
// Event Listeners
// ============================================
function initEventListeners() {
    // Sidebar
    elements.toggleVision.addEventListener('click', toggleSidebar);
    elements.closeSidebar.addEventListener('click', toggleSidebar);

    // Image Upload
    elements.selectImageBtn.addEventListener('click', () => elements.imageInput.click());
    elements.imageInput.addEventListener('change', handleImageSelect);
    elements.removeImage.addEventListener('click', resetImageUpload);
    elements.classifyBtn.addEventListener('click', classifyImage);

    // Nuevo: Enviar imagen al chat
    const sendToChatBtn = document.getElementById('sendToChatBtn');
    if (sendToChatBtn) {
        sendToChatBtn.addEventListener('click', sendImageToChat);
    }

    // Drag and drop
    elements.uploadArea.addEventListener('click', () => elements.imageInput.click());
    elements.uploadArea.addEventListener('dragover', handleDragOver);
    elements.uploadArea.addEventListener('dragleave', handleDragLeave);
    elements.uploadArea.addEventListener('drop', handleDrop);

    // Chat
    elements.sendBtn.addEventListener('click', sendMessage);
    elements.messageInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    elements.attachBtn.addEventListener('click', () => {
        toggleSidebar();
        elements.imageInput.click();
    });

    // Example queries
    document.querySelectorAll('.example-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            elements.messageInput.value = btn.dataset.query;
            sendMessage();
        });
    });

    // Email toggle
    elements.enableEmail.addEventListener('change', (e) => {
        elements.emailInput.style.display = e.target.checked ? 'block' : 'none';
    });

    // Settings
    elements.toggleSettings.addEventListener('click', () => {
        elements.settingsModal.style.display = 'flex';
    });

    elements.closeSettings.addEventListener('click', closeSettingsModal);
    elements.saveSettings.addEventListener('click', saveSettings);
    elements.resetSettings.addEventListener('click', resetSettings);

    // Click outside modal to close
    elements.settingsModal.addEventListener('click', (e) => {
        if (e.target === elements.settingsModal) {
            closeSettingsModal();
        }
    });
}

// ============================================
// Sidebar Functions
// ============================================
function toggleSidebar() {
    elements.visionSidebar.classList.toggle('active');
}

// ============================================
// Image Upload Functions
// ============================================
function handleDragOver(e) {
    e.preventDefault();
    elements.uploadArea.classList.add('drag-over');
}

function handleDragLeave(e) {
    e.preventDefault();
    elements.uploadArea.classList.remove('drag-over');
}

function handleDrop(e) {
    e.preventDefault();
    elements.uploadArea.classList.remove('drag-over');

    const files = e.dataTransfer.files;
    if (files.length > 0 && files[0].type.startsWith('image/')) {
        handleImage(files[0]);
    } else {
        showToast('Por favor sube una imagen válida', 'error');
    }
}

function handleImageSelect(e) {
    const file = e.target.files[0];
    if (file && file.type.startsWith('image/')) {
        handleImage(file);
    }
}

function handleImage(file) {
    state.currentImage = file;

    const reader = new FileReader();
    reader.onload = (e) => {
        elements.previewImage.src = e.target.result;
        elements.uploadArea.style.display = 'none';
        elements.previewArea.style.display = 'block';
        elements.resultsArea.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

function resetImageUpload() {
    state.currentImage = null;
    elements.imageInput.value = '';
    elements.uploadArea.style.display = 'block';
    elements.previewArea.style.display = 'none';
    elements.resultsArea.style.display = 'none';
}

async function classifyImage() {
    if (!state.currentImage) {
        showToast('No hay imagen para clasificar', 'error');
        return;
    }

    // Show loader
    elements.previewArea.style.display = 'none';
    elements.visionLoader.style.display = 'block';

    const formData = new FormData();
    formData.append('file', state.currentImage);

    try {
        const response = await fetch(`${CONFIG.apiUrl}/vision/classify`, {
            method: 'POST',
            headers: CONFIG.apiKey ? { 'Authorization': `Bearer ${CONFIG.apiKey}` } : {},
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const result = await response.json();
        displayVisionResults(result);
        showToast('Imagen clasificada exitosamente', 'success');

    } catch (error) {
        console.error('Error classifying image:', error);
        showToast(`Error: ${error.message}`, 'error');
        elements.visionLoader.style.display = 'none';
        elements.previewArea.style.display = 'block';
    }
}

// Nueva función: Enviar imagen al chat para análisis conversacional
function sendImageToChat() {
    if (!state.currentImage) {
        showToast('No hay imagen seleccionada', 'error');
        return;
    }

    // Cerrar sidebar
    toggleSidebar();

    // Mostrar indicador de imagen adjunta en el input
    elements.messageInput.placeholder = '📷 Imagen adjunta - Pregunta algo sobre ella...';
    elements.messageInput.focus();

    showToast('Imagen adjunta al chat. Haz tu pregunta!', 'success');
}

function displayVisionResults(result) {
    elements.visionLoader.style.display = 'none';
    elements.resultsArea.style.display = 'block';

    // Top prediction
    const topPred = result.top_prediction;
    elements.topPrediction.innerHTML = `
        <h4>${topPred.class}</h4>
        <div class="confidence">${topPred.confidence.toFixed(1)}%</div>
    `;

    // All predictions
    const predictionsHTML = result.predictions.map((pred, index) => `
        <div class="prediction-item">
            <div>
                <strong>${index + 1}. ${pred.class_name}</strong>
                <div class="prediction-bar">
                    <div class="prediction-fill" style="width: ${pred.confidence_percent}%"></div>
                </div>
            </div>
            <span>${pred.confidence_percent.toFixed(1)}%</span>
        </div>
    `).join('');

    document.getElementById('allPredictions').innerHTML = predictionsHTML;

    // Model info
    document.getElementById('modelInfo').innerHTML = `
        <strong>Modelo:</strong> ${result.model}<br>
        <strong>Dispositivo:</strong> ${result.device}<br>
        <strong>Tiempo:</strong> ${result.inference_time.toFixed(3)}s<br>
        <strong>Tamaño imagen:</strong> ${result.image_size[0]} x ${result.image_size[1]}
    `;
}

// ============================================
// Chat Functions
// ============================================
function hideWelcomeScreen() {
    elements.welcomeScreen.style.display = 'none';
    elements.chatMessages.style.display = 'block';
}

async function sendMessage() {
    const message = elements.messageInput.value.trim();

    if (!message) return;

    // Hide welcome screen on first message
    if (state.messages.length === 0) {
        hideWelcomeScreen();
    }

    // Determinar si hay imagen adjunta
    const hasImage = state.currentImage !== null;

    // Add user message (con indicador de imagen si aplica)
    const displayMessage = hasImage ? `${message} 📷` : message;
    addMessage('user', displayMessage);
    elements.messageInput.value = '';

    // Show typing indicator
    showTypingIndicator();

    // Prepare request
    const userEmail = elements.enableEmail.checked ? elements.emailInput.value : null;

    try {
        let response;

        if (hasImage) {
            // Usar endpoint de visión con FormData
            console.log('Enviando consulta con imagen...');
            const formData = new FormData();
            formData.append('query', message);
            formData.append('image', state.currentImage);
            formData.append('conversation_id', CONFIG.conversationId);
            if (userEmail) formData.append('user_email', userEmail);

            response = await fetch(`${CONFIG.apiUrl}/query/vision`, {
                method: 'POST',
                headers: CONFIG.apiKey ? { 'Authorization': `Bearer ${CONFIG.apiKey}` } : {},
                body: formData
            });

            // Limpiar imagen después de enviar
            state.currentImage = null;
            resetImageUpload();
            elements.messageInput.placeholder = 'Escribe tu pregunta sobre Curazao...';

        } else {
            // Usar endpoint normal de texto
            const payload = {
                query: message,
                conversation_id: CONFIG.conversationId,
                user_email: userEmail
            };

            response = await fetch(`${CONFIG.apiUrl}/query`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...(CONFIG.apiKey && { 'Authorization': `Bearer ${CONFIG.apiKey}` })
                },
                body: JSON.stringify(payload)
            });
        }

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const result = await response.json();

        // Remove typing indicator
        removeTypingIndicator();

        // Add bot response
        addMessage('bot', result.response, {
            queryType: result.query_type,
            time: result.total_time,
            tokens: result.tokens?.total_tokens || result.tokens?.total,
            agents: result.active_agents
        });

        // Show notification if email was sent
        if (userEmail && result.response.includes('itinerario')) {
            showToast('Email enviado a ' + userEmail, 'success');
        }

        // Show success for vision queries
        if (hasImage) {
            showToast('Imagen analizada exitosamente', 'success');
        }

    } catch (error) {
        console.error('Error sending message:', error);
        removeTypingIndicator();
        addMessage('bot', `Lo siento, ocurrió un error: ${error.message}`, { error: true });
        showToast(`Error: ${error.message}`, 'error');
    }
}

function addMessage(type, content, meta = {}) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;

    const time = new Date().toLocaleTimeString('es-ES', { hour: '2-digit', minute: '2-digit' });

    let metaHTML = '';
    if (meta.queryType) {
        metaHTML = `
            <div class="message-meta">
                <span><i class="fas fa-tag"></i> ${meta.queryType}</span>
                <span><i class="fas fa-clock"></i> ${meta.time.toFixed(2)}s</span>
                ${meta.tokens ? `<span><i class="fas fa-microchip"></i> ${meta.tokens} tokens</span>` : ''}
            </div>
        `;
    }

    messageDiv.innerHTML = `
        <div class="message-avatar">
            <i class="fas fa-${type === 'user' ? 'user' : 'robot'}"></i>
        </div>
        <div class="message-content">
            <div class="message-bubble">${formatMessage(content)}</div>
            <div class="message-time">${time}</div>
            ${metaHTML}
        </div>
    `;

    elements.chatMessages.appendChild(messageDiv);
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;

    state.messages.push({ type, content, time });
}

function formatMessage(text) {
    // Convert markdown-like formatting
    text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    text = text.replace(/\*(.*?)\*/g, '<em>$1</em>');
    text = text.replace(/\n/g, '<br>');
    return text;
}

function showTypingIndicator() {
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message bot typing-message';
    typingDiv.innerHTML = `
        <div class="message-avatar">
            <i class="fas fa-robot"></i>
        </div>
        <div class="message-content">
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        </div>
    `;

    elements.chatMessages.appendChild(typingDiv);
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
}

function removeTypingIndicator() {
    const typingMsg = document.querySelector('.typing-message');
    if (typingMsg) {
        typingMsg.remove();
    }
}

// ============================================
// Settings Functions
// ============================================
function loadSettings() {
    document.getElementById('apiUrl').value = CONFIG.apiUrl;
    document.getElementById('apiKey').value = CONFIG.apiKey;
}

function saveSettings() {
    CONFIG.apiUrl = document.getElementById('apiUrl').value;
    CONFIG.apiKey = document.getElementById('apiKey').value;

    localStorage.setItem('apiUrl', CONFIG.apiUrl);
    localStorage.setItem('apiKey', CONFIG.apiKey);

    showToast('Configuración guardada', 'success');
    closeSettingsModal();
}

function resetSettings() {
    CONFIG.apiUrl = 'http://localhost:8000';
    CONFIG.apiKey = '';

    localStorage.removeItem('apiUrl');
    localStorage.removeItem('apiKey');

    loadSettings();
    showToast('Configuración restablecida', 'info');
}

function closeSettingsModal() {
    elements.settingsModal.style.display = 'none';
}

// ============================================
// Toast Notifications
// ============================================
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;

    const icon = {
        success: 'check-circle',
        error: 'exclamation-circle',
        info: 'info-circle'
    }[type];

    toast.innerHTML = `
        <i class="fas fa-${icon}"></i>
        <span>${message}</span>
    `;

    elements.toastContainer.appendChild(toast);

    setTimeout(() => {
        toast.style.animation = 'fadeOut 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// ============================================
// Utility Functions
// ============================================
function generateUUID() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        const r = Math.random() * 16 | 0;
        const v = c === 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}

// ============================================
// Error Handling
// ============================================
window.addEventListener('error', (e) => {
    console.error('Global error:', e.error);
    showToast('Error inesperado en la aplicación', 'error');
});

window.addEventListener('unhandledrejection', (e) => {
    console.error('Unhandled promise rejection:', e.reason);
    showToast('Error de conexión', 'error');
});
