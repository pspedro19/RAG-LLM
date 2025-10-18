# 🌐 Frontend - Curazao Tourism Assistant

## 📋 Descripción

Frontend completo y funcional para el asistente turístico de Curazao con:
- ✅ Chat conversacional en tiempo real
- ✅ Clasificación de imágenes con modelo ViT
- ✅ Interfaz moderna y responsive
- ✅ Soporte para envío de emails
- ✅ Configuración personalizable

---

## 🚀 Instalación y Uso

### Opción 1: Abrir Directamente (Recomendado)

```bash
# Navegar a la carpeta
cd C:\Users\pedro\OneDrive\Documents\LLMS\RAG-LLM\frontend

# Abrir index.html en tu navegador
# Método 1: Doble click en index.html
# Método 2: Arrastrar index.html al navegador
# Método 3: Usar servidor local (ver abajo)
```

### Opción 2: Con Servidor Local (Mejor para desarrollo)

```bash
# Python 3
cd frontend
python -m http.server 8080

# Node.js (si tienes http-server instalado)
npx http-server -p 8080

# Abrir en navegador
# http://localhost:8080
```

---

## ⚙️ Configuración

### 1. Asegúrate de que la API esté corriendo

```bash
# En otra terminal
cd C:\Users\pedro\OneDrive\Documents\LLMS\RAG-LLM\app
python main.py

# La API debe estar en: http://localhost:8000
```

### 2. Configurar la URL de la API (opcional)

- Haz clic en el icono de configuración (⚙️) en el header
- URL por defecto: `http://localhost:8000`
- Si la API está en otra URL, cámbiala aquí

### 3. API Key (opcional)

Si implementaste autenticación en la API:
- Ve a Configuración (⚙️)
- Ingresa tu API Key
- Guarda los cambios

---

## 🎯 Características

### 1. Chat Conversacional

**Cómo usar:**
1. Escribe tu pregunta en el campo de texto
2. Presiona Enter o haz clic en el botón de enviar (✈️)
3. El bot responderá automáticamente

**Ejemplos de consultas:**
- "Crea un itinerario de 3 días en Curazao"
- "¿Cuáles son las mejores playas?"
- "Recomiéndame restaurantes locales"
- "¿Qué actividades hay para familias?"

### 2. Clasificación de Imágenes (ViT)

**Cómo usar:**
1. Haz clic en el icono de cámara (📷) en el header
2. Se abrirá el panel lateral de visión
3. **Opción A:** Arrastra una imagen al área de carga
4. **Opción B:** Haz clic en "Seleccionar Imagen"
5. Haz clic en "Clasificar Imagen"
6. Verás los resultados con porcentajes de confianza

**Formatos soportados:**
- JPEG, PNG, JPG, BMP, WEBP
- Tamaño máximo: 10MB
- Resolución mínima: 32x32

### 3. Envío de Emails Automático

**Cómo activar:**
1. Marca la casilla "Enviar respuestas por email"
2. Ingresa tu email
3. Haz una pregunta (especialmente itinerarios)
4. Recibirás el resultado por email automáticamente

**Nota:** Requiere que `SENDGRID_API_KEY` esté configurado en el backend.

---

## 🎨 Interfaz

### Pantalla de Bienvenida
- Tarjetas con funcionalidades principales
- Ejemplos de consultas para comenzar
- Click en cualquier ejemplo para enviarlo

### Chat
- Mensajes del usuario (azul, derecha)
- Mensajes del bot (gris, izquierda)
- Indicador de "escribiendo..." mientras procesa
- Metadata: tipo de consulta, tiempo, tokens usados

### Panel de Visión (Lateral)
- Área de carga con drag & drop
- Vista previa de imagen
- Resultados con:
  - Predicción principal (grande)
  - Top 5 predicciones con barras de confianza
  - Información del modelo (ViT/EfficientNet/ResNet)
  - Tiempo de inferencia
  - Dispositivo usado (CPU/GPU)

---

## 🛠️ Estructura de Archivos

```
frontend/
├── index.html          # Página principal
├── styles.css          # Estilos (400+ líneas)
├── app.js              # Lógica de la aplicación (500+ líneas)
└── README.md           # Este archivo
```

---

## 🔧 Personalización

### Cambiar Colores

Edita las variables CSS en `styles.css` (líneas 1-20):

```css
:root {
    --primary-color: #667eea;      /* Color principal */
    --secondary-color: #764ba2;    /* Color secundario */
    --success-color: #10b981;      /* Verde éxito */
    --danger-color: #ef4444;       /* Rojo error */
    /* ... más colores ... */
}
```

### Cambiar Textos

Edita el HTML en `index.html`:
- Línea 15: Título de la página
- Línea 18: Nombre del asistente
- Líneas 66-88: Pantalla de bienvenida

### Agregar Más Ejemplos de Consultas

En `index.html`, líneas 89-98:

```html
<button class="example-btn" data-query="Tu consulta aquí">
    "Texto del botón"
</button>
```

---

## 📱 Responsive Design

El frontend es completamente responsive:
- **Desktop:** Panel lateral + chat lado a lado
- **Tablet:** Panel lateral se superpone
- **Mobile:** Panel lateral ocupa pantalla completa

Puntos de quiebre:
- 768px: Cambio a layout móvil
- 480px: Ajustes adicionales para pantallas pequeñas

---

## 🐛 Troubleshooting

### Problema: "Error de conexión"

**Solución:**
1. Verifica que la API esté corriendo:
   ```bash
   curl http://localhost:8000/health
   ```
2. Verifica la URL en Configuración (⚙️)
3. Revisa la consola del navegador (F12) para errores

### Problema: "No se puede clasificar imagen"

**Solución:**
1. Verifica que el formato sea válido (JPG, PNG)
2. Verifica que el tamaño sea menor a 10MB
3. Asegúrate de que la API tenga torch instalado:
   ```bash
   pip install torch torchvision
   ```

### Problema: "Email no se envía"

**Solución:**
1. Verifica que SENDGRID_API_KEY esté configurado en el backend
2. El email solo se envía para consultas de itinerarios
3. Revisa los logs del backend para errores

### Problema: CORS Error

**Solución:**
El backend ya tiene CORS habilitado en `main.py`:
```python
allow_origins=["*"]  # Permite todos los orígenes
```

Si el error persiste:
1. Usa un servidor local (no abras el HTML directamente)
2. Verifica que la API esté en la URL correcta

---

## 🎯 Demo y Testing

### Test Rápido Completo

1. **Abrir el frontend:**
   ```bash
   # Opción 1: Doble click en index.html
   # Opción 2:
   cd frontend
   python -m http.server 8080
   # Abrir http://localhost:8080
   ```

2. **Verificar API:**
   - Debería aparecer "Sistema listo" (verde)
   - Si aparece error, verifica que la API esté corriendo

3. **Test Chat:**
   - Click en "¿Mejores playas?"
   - Debería responder en 2-5 segundos

4. **Test Visión:**
   - Click en icono de cámara (📷)
   - Arrastra cualquier imagen
   - Click "Clasificar Imagen"
   - Debería mostrar resultados en 1-3 segundos

5. **Test Email:**
   - Activa checkbox "Enviar por email"
   - Ingresa tu email
   - Pregunta: "Crea itinerario de 3 días"
   - Espera respuesta + notificación de email enviado

---

## 🚀 Características Avanzadas

### Notificaciones Toast

- ✅ Verde: Operación exitosa
- ❌ Rojo: Error
- ℹ️ Azul: Información

Aparecen automáticamente en la esquina superior derecha.

### Indicadores de Estado

- **Typing indicator:** Puntos animados mientras el bot "piensa"
- **Metadata de mensajes:** Tiempo, tokens, tipo de consulta
- **Spinner de carga:** En clasificación de imágenes

### Historial de Conversación

- Se mantiene durante toda la sesión
- Cada mensaje guarda: contenido, tipo, hora
- Scroll automático al último mensaje

---

## 📊 Métricas Mostradas

### En Chat:
- ⏱️ Tiempo de respuesta (segundos)
- 🏷️ Tipo de consulta (conversacional/información/itinerario)
- 🖥️ Tokens consumidos

### En Visión:
- 📊 Predicción principal con % confianza
- 📈 Top 5 predicciones con barras visuales
- 🤖 Modelo usado (ViT-B/16, EfficientNet, etc.)
- ⚡ Tiempo de inferencia (segundos)
- 💻 Dispositivo (CPU/CUDA)
- 🖼️ Tamaño de imagen procesada

---

## 🎨 Capturas de Pantalla

### Vista Principal (Chat)
- Header con logo y controles
- Pantalla de bienvenida con tarjetas
- Ejemplos de consultas clickeables
- Campo de entrada con opciones de email

### Panel de Visión
- Área de carga con drag & drop
- Vista previa de imagen
- Resultados con gráficos de confianza
- Información del modelo

---

## 🔒 Seguridad

### Validaciones Implementadas:
- ✅ Validación de tipo de archivo (solo imágenes)
- ✅ Validación de tamaño (máx 10MB)
- ✅ Sanitización de inputs
- ✅ API Key opcional
- ✅ HTTPS ready (si el backend lo soporta)

---

## 🎓 Para Presentación

### Orden Sugerido de Demo:

1. **Mostrar Bienvenida (30 seg)**
   - Explicar las 3 tarjetas de funcionalidades
   - Mostrar ejemplos de consultas

2. **Demo Chat (1 min)**
   - Click en "Crea un itinerario de 3 días"
   - Mostrar metadata (tiempo, tokens, tipo)
   - Mostrar respuesta completa

3. **Demo Visión (1 min)**
   - Abrir panel lateral
   - Arrastrar imagen de playa
   - Clasificar
   - Mostrar resultados (coral reef, beach, etc.)

4. **Demo Email (30 seg)**
   - Activar checkbox
   - Ingresar email
   - Hacer consulta de itinerario
   - Mostrar notificación de email enviado

5. **Mostrar Configuración (30 seg)**
   - Abrir settings
   - Mostrar URL de API configurable
   - Explicar API Key opcional

**Tiempo total: 3.5 minutos**

---

## ✅ Checklist Pre-Presentación

- [ ] API corriendo en `http://localhost:8000`
- [ ] Frontend abierto en navegador
- [ ] Imagen de prueba descargada (playa de Curazao)
- [ ] Email de prueba configurado (si se va a demostrar)
- [ ] Consola del navegador limpia (F12)
- [ ] Pantalla en modo presentación (zoom 100%)

---

## 💡 Tips de Uso

1. **Usa ejemplos rápidos:** Los botones de ejemplo son ideales para empezar
2. **Drag & drop:** Es más rápido que seleccionar archivo
3. **Email opcional:** Funciona sin configurar SendGrid (modo fallback)
4. **Responsive:** Prueba en diferentes tamaños de ventana

---

## 📝 Notas Técnicas

### Tecnologías Utilizadas:
- HTML5
- CSS3 (Custom Properties, Flexbox, Grid, Animations)
- JavaScript Vanilla (ES6+)
- Fetch API para requests
- FontAwesome 6.4 para iconos

### Compatibilidad:
- ✅ Chrome 90+
- ✅ Firefox 88+
- ✅ Edge 90+
- ✅ Safari 14+

### Sin Dependencias Externas:
- No requiere npm/yarn
- No requiere build tools
- No requiere frameworks (React, Vue, etc.)
- Solo necesita un navegador moderno

---

## 🎉 ¡Frontend Completo y Listo!

**El frontend está 100% funcional** con:
- ✅ Chat conversacional
- ✅ Clasificación de imágenes ViT
- ✅ Interfaz moderna
- ✅ Responsive design
- ✅ Notificaciones
- ✅ Configuración
- ✅ Soporte para emails

**¡Abre `index.html` y comienza a usar el asistente!** 🚀
