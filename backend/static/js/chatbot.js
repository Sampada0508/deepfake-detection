/* Chat front-end
 * - Tries POST /api/chat (expects JSON {message, persona}) -> {reply}
 * - If endpoint missing or fails, uses a local fallback responder
 * - No timestamps. Rename, Clear, quick replies, persona support included.
 */

(() => {
  const botNameEl = document.getElementById('botName');
  const welcomeBotEl = document.getElementById('welcomeBot');
  const panelBotName = document.getElementById('panelBotName');
  const messagesEl = document.getElementById('messages');
  const msgInput = document.getElementById('msgInput');
  const sendBtn = document.getElementById('sendBtn');
  const personaSelect = document.getElementById('personaSelect');
  const quickReplies = document.getElementById('quickReplies');
  const renameBtn = document.getElementById('renameBtn');
  const clearBtn = document.getElementById('clearBtn');

  // initial bot name
  let BOT_NAME = 'LensBot';

  // small util to add message to DOM
  function appendMessage({ sender = 'bot', text = '' }) {
    const div = document.createElement('div');
    div.className = `msg ${sender}`;
    // safe-ish HTML: we permit basic tags in fallback replies (simple)
    div.innerHTML = `<p>${escapeHtml(text).replace(/\n/g, '<br>')}</p>`;
    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  // escape HTML helper
  function escapeHtml(str) {
    if (!str) return '';
    return str
      .replaceAll('&', '&amp;')
      .replaceAll('<', '&lt;')
      .replaceAll('>', '&gt;')
      .replaceAll('"', '&quot;')
      .replaceAll("'", '&#x27;');
  }

  // try server side chat; fallback to local responder
  async function getBotReply(prompt, persona) {
    // try server endpoint first
    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: prompt, persona })
      });
      if (res.ok) {
        const j = await res.json();
        if (j && (j.reply || j.response)) {
          return j.reply || j.response;
        }
        // server responded but unexpected shape -> fallback to local
      }
      // if 404 / not ok, throw to go fallback
      throw new Error('server endpoint not available');
    } catch (e) {
      // fallback local responder
      return localResponder(prompt, persona);
    }
  }

  // small local responder (helpful placeholder until you connect a real model)
  function localResponder(prompt, persona) {
    const p = prompt.toLowerCase();
    // persona tweaking
    const tone = persona === 'technical' ? ' (technical)' : persona === 'concise' ? ' (concise)' : '';

    if (p.includes('how') && p.includes('detect')) {
      return `We detect forgeries by analyzing visual artifacts at frame and pixel levels, and combining per-frame predictions to classify videos. Steps: frame extraction → CNN analysis → temporal aggregation${tone}.`;
    }
    if (p.includes('confidence')) {
      return `Confidence is the model's probability for the chosen class (0–1). Higher means stronger model certainty. We display it as a percentage in visualizations.`;
    }
    if (p.includes('visual') || p.includes('chart')) {
      return `Open the Visualization page to see trends, confidence bins and prediction history. Use the "Visualization" link in the header.`;
    }
    if (p.includes('model') || p.includes('resnet')) {
      return `Models: frame-level ResNet18 for images and frame-wise aggregation for videos. You can improve by using temporal models (e.g., ConvLSTM or transformer-based) for sequence info.`;
    }
    if (p.includes('hello') || p.includes('hi') || p.includes('hey')) {
      return `👋 Hello! I’m ${BOT_NAME}, your TRUTHLENS assistant. How can I help today?`;
    }

    // generic helpful answer
    const generic = [
      `I can help you run detection, explain results, or point to visualizations.${tone}`,
      `Try: "How to improve video detection", "Explain confidence", or "Show visualization".${tone}`
    ];
    return generic.join(' ');
  }

  // bind send
  async function sendMessage() {
    const text = msgInput.value.trim();
    if (!text) return;
    appendMessage({ sender: 'user', text });
    msgInput.value = '';
    sendBtn.disabled = true;

    // persona value
    const persona = personaSelect ? personaSelect.value : 'friendly';
    const reply = await getBotReply(text, persona);
    appendMessage({ sender: 'bot', text: reply });
    sendBtn.disabled = false;
  }

  // attach events
  sendBtn.addEventListener('click', sendMessage);
  msgInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) sendMessage();
  });

  // quick replies
  quickReplies.querySelectorAll('.chip').forEach(btn => {
    btn.addEventListener('click', () => {
      const text = btn.textContent.trim();
      msgInput.value = text;
      msgInput.focus();
    });
  });

  // rename bot
  renameBtn.addEventListener('click', () => {
    const name = prompt('Enter new bot name', BOT_NAME) || BOT_NAME;
    BOT_NAME = name.trim() || BOT_NAME;
    botNameEl.textContent = BOT_NAME;
    welcomeBotEl.textContent = BOT_NAME;
    panelBotName.textContent = BOT_NAME;
    appendMessage({ sender: 'bot', text: `My name is now ${BOT_NAME}. How can I help?` });
  });

  // clear chat
  clearBtn.addEventListener('click', () => {
    if (!confirm('Clear the chat history?')) return;
    messagesEl.innerHTML = '';
    injectWelcome(); // re-insert welcome
  });

  // initial welcome injection (no timestamp)
  function injectWelcome() {
    messagesEl.innerHTML = '';
    const welcome = `👋 Hello! I’m ${BOT_NAME}, your TRUTHLENS AI guide. Ask me about detection, confidence levels, models, or visualizations.`;
    appendMessage({ sender: 'bot', text: welcome });
  }

  // wire up quick replies to send message quickly on double-click
  quickReplies.addEventListener('dblclick', (e) => {
    if (e.target && e.target.matches('.chip')) {
      msgInput.value = e.target.textContent.trim();
      sendMessage();
    }
  });

  // expose some keys for power users
  window.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
      msgInput.focus();
      e.preventDefault();
    }
  });

  // init
  injectWelcome();
})();