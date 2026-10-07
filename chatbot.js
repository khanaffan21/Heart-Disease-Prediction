/* ═══════════════════════════════════════════════════════
   HEART CARE AI CHATBOT — chatbot.js
   Hybrid Rule-Based + Gemini 1.5 Flash AI Engine
   ═══════════════════════════════════════════════════════ */

(function() {
  'use strict';

  /* ─── State ─── */
  let lastRiskResult = null; // set by main page after prediction
  let chatHistory = []; // multi-turn conversation memory
  let isOpen = false;
  let isProcessing = false;

  /* ─── Knowledge Base (Fast Local Responses) ─── */
  const KB = {
    greeting: {
      patterns: ['hello', 'hi', 'hey', 'good morning', 'good evening', 'namaste', 'howdy'],
      response: () => `👋 <strong>Hello! I'm HeartCare AI</strong>, your cardiac wellness assistant.<br><br>
I can help you with:
<ul>
  <li>🛡️ Heart disease <strong>prevention tips</strong></li>
  <li>💊 <strong>Treatment & medication</strong> information</li>
  <li>🧘 <strong>Yoga & breathing</strong> exercises for heart health</li>
  <li>🥗 <strong>Diet plans</strong> for cardiovascular wellness</li>
  <li>🚨 Recognizing <strong>emergency symptoms</strong></li>
  <li>📊 Understanding your <strong>risk score</strong></li>
</ul>
What would you like to know?`
    },
    symptoms: {
      patterns: ['symptom', 'signs', 'what does heart attack feel like', 'how do i know', 'warning signs', 'chest pain', 'shortness of breath', 'dizzy', 'dizziness', 'fatigue', 'tired'],
      response: () => `❤️ <strong>Common Heart Disease Symptoms:</strong><br><br>
<ul>
  <li>🔴 <strong>Chest pain or pressure</strong> (angina) — tightness, squeezing, or burning</li>
  <li>💨 <strong>Shortness of breath</strong> — even at rest or with mild activity</li>
  <li>😰 <strong>Excessive sweating</strong> — cold sweat without exertion</li>
  <li>💤 <strong>Persistent fatigue</strong> — unusual tiredness</li>
  <li>🫀 <strong>Palpitations</strong> — irregular, fluttering heartbeat</li>
  <li>🦵 <strong>Swollen legs/ankles</strong> — fluid retention</li>
  <li>🌀 <strong>Dizziness or fainting</strong></li>
</ul>
⚠️ If you experience sudden severe chest pain, <strong>call emergency services (112 / 911) immediately</strong>.`
    },
    prevention: {
      patterns: ['prevent', 'prevention', 'avoid', 'reduce risk', 'keep heart healthy', 'how to protect', 'healthy habits'],
      response: () => `🛡️ <strong>Heart Disease Prevention — Top Strategies:</strong><br><br>
<ul>
  <li>🚭 <strong>Quit smoking</strong> — #1 modifiable risk factor</li>
  <li>🏃 <strong>Exercise 150 min/week</strong> — brisk walking, swimming, cycling</li>
  <li>🥗 <strong>Heart-healthy diet</strong> — Mediterranean or DASH diet</li>
  <li>⚖️ <strong>Maintain healthy weight</strong> — BMI 18.5–24.9</li>
  <li>🩺 <strong>Control blood pressure</strong> — target below 130/80 mmHg</li>
  <li>🩸 <strong>Manage cholesterol</strong> — LDL below 100 mg/dL</li>
  <li>🍬 <strong>Control blood sugar</strong> — prevent/manage diabetes</li>
  <li>😴 <strong>Quality sleep</strong> — 7–9 hours per night</li>
  <li>🧘 <strong>Manage stress</strong> — yoga, meditation, deep breathing</li>
</ul>
Want details on any specific prevention strategy?`
    },
    suggestions: {
      patterns: ['what to do', 'what should i do', 'suggestion', 'suggestions', 'next step', 'next steps', 'what now', 'advice', 'guidance', 'recommendation', 'recommendations', 'action plan', 'help me'],
      response: () => {
        if (lastRiskResult) {
          const level = lastRiskResult.level;
          const pct = lastRiskResult.percentage;
          if (level === 'high') {
            return `🔴 <strong>Recommended Action Plan for High Risk (${pct}%):</strong><br><br>
1. 🩺 <strong>Schedule a Cardiologist Visit Urgently</strong> — Get a comprehensive check-up (ECG, Echocardiogram, Lipid Panel).<br>
2. 🥗 <strong>Switch to Heart-Healthy Diet</strong> — Reduce sodium (&lt;2,000mg/day) and avoid saturated/trans fats.<br>
3. 🏃 <strong>Light Exercise & Walking</strong> — 20-30 mins daily (consult your doctor before heavy workouts).<br>
4. 🧘 <strong>Daily Stress Reduction</strong> — Practice 4-7-8 breathing or Savasana for 10-15 mins.<br>
5. 🚭 <strong>Zero Smoking & Alcohol Limit</strong> — Eliminate tobacco completely.<br><br>
💡 Explore your detailed plan in the <strong><a href="lifestyle.html" style="color:#5eead4">Lifestyle Guide</a></strong> or ask me specific questions!`;
          }
        }
        return `🛡️ <strong>Recommended Action Steps:</strong><br><br>
1. 🩺 <strong>Consult a Doctor</strong> — Get your BP, blood sugar, and cholesterol checked.<br>
2. 🥗 <strong>Heart-Healthy Diet</strong> — Adopt the Mediterranean diet (more fish, veggies, olive oil, nuts).<br>
3. 🏃 <strong>150 Mins Exercise/Week</strong> — Moderate brisk walking, swimming, or cycling.<br>
4. 🧘 <strong>Stress Management</strong> — Yoga, 4-7-8 breathing, and adequate sleep (7-9 hrs).<br>
5. 🚭 <strong>Avoid Smoking & Excess Salt</strong>.<br><br>
Want details on <strong>diet</strong>, <strong>yoga</strong>, or <strong>prevention</strong>? Just ask!`;
      }
    },
    diet: {
      patterns: ['diet', 'food', 'eat', 'nutrition', 'meal', 'mediterranean', 'dash diet', 'cholesterol food', 'what should i eat', 'avoid food', 'fruits vegetables', 'omega'],
      response: () => `🥗 <strong>Heart-Healthy Diet Guide:</strong><br><br>
<strong>✅ Eat More:</strong>
<ul>
  <li>🐟 Fatty fish (salmon, mackerel) — Omega-3 fatty acids</li>
  <li>🥑 Avocados — healthy monounsaturated fats</li>
  <li>🫐 Berries — antioxidants, reduce arterial stiffness</li>
  <li>🥦 Leafy greens (spinach, kale) — rich in nitrates</li>
  <li>🌰 Walnuts & almonds — heart-protective fats</li>
  <li>🫒 Olive oil — polyphenols, anti-inflammatory</li>
</ul>
<strong>❌ Limit / Avoid:</strong>
<ul>
  <li>🧂 Excess sodium (salt) — max 2,300 mg/day</li>
  <li>🥩 Red & processed meats</li>
  <li>🧈 Trans fats & saturated fats</li>
  <li>🥤 Sugary drinks & refined carbs</li>
</ul>
💡 The <strong>Mediterranean Diet</strong> reduces heart disease risk by up to 30%.`
    },
    yoga: {
      patterns: ['yoga', 'exercise', 'pose', 'asana', 'pranayama', 'breathe', 'breathing', 'meditation', 'relax', 'stress', 'calm', 'mindfulness', 'savasana', 'physical activity'],
      response: () => `🧘 <strong>Yoga & Breathing Exercises for Heart Health:</strong><br><br>
<strong>Top Yoga Poses:</strong>
<ul>
  <li>🌅 <strong>Savasana</strong> (Corpse Pose) — reduces cortisol & blood pressure</li>
  <li>🦋 <strong>Baddha Konasana</strong> (Butterfly) — improves pelvic circulation</li>
  <li>🌉 <strong>Setu Bandha</strong> (Bridge Pose) — strengthens heart & back</li>
  <li>🐱 <strong>Marjaryasana</strong> (Cat-Cow) — calms nervous system</li>
  <li>🦅 <strong>Viparita Karani</strong> (Legs-Up-Wall) — lowers BP within 10 mins</li>
</ul>
<strong>Breathing (Pranayama):</strong>
<ul>
  <li>🌬️ <strong>4-7-8 Breathing</strong>: Inhale 4s → Hold 7s → Exhale 8s</li>
  <li>🔄 <strong>Anulom Vilom</strong>: Balances nervous system</li>
</ul>
⏱️ Visit the <a href="lifestyle.html" style="color:#5eead4">Lifestyle Guide</a> for interactive breathing timers!`
    },
    treatment: {
      patterns: ['treatment', 'medicine', 'medication', 'drug', 'therapy', 'cure', 'manage', 'statin', 'beta blocker', 'aspirin', 'blood thinner', 'surgery', 'stent', 'bypass'],
      response: () => `💊 <strong>Heart Disease Treatment Overview:</strong><br><br>
<strong>🔵 Lifestyle Medicine (First Line):</strong>
<ul>
  <li>Diet modification & regular physical activity</li>
  <li>Smoking cessation & stress reduction</li>
</ul>
<strong>💊 Common Medication Classes:</strong>
<ul>
  <li><strong>Statins</strong> (Atorvastatin) — lower LDL cholesterol</li>
  <li><strong>Beta-blockers</strong> (Metoprolol) — reduce heart rate & BP</li>
  <li><strong>ACE Inhibitors</strong> (Lisinopril) — relax blood vessels</li>
  <li><strong>Antiplatelet agents</strong> (Aspirin) — prevent blood clots</li>
</ul>
⚠️ <strong>Always consult a licensed cardiologist</strong> before starting or modifying medications.`
    },
    risk: {
      patterns: ['risk score', 'what does my score mean', 'risk level', 'percentage', 'probability', 'high risk', 'low risk', 'moderate risk', 'result', 'prediction'],
      response: () => {
        if (lastRiskResult) {
          const pct = lastRiskResult.percentage;
          const level = lastRiskResult.level;
          let advice = '';
          if (level === 'high') {
            advice = '🔴 Your risk is <strong>HIGH</strong>. Schedule a cardiologist consultation urgently and implement immediate lifestyle changes.';
          } else if (level === 'moderate') {
            advice = '🟠 Your risk is <strong>ELEVATED</strong>. This is the optimal time to modify diet, increase exercise, and get medical advice.';
          } else {
            advice = '🟢 Your risk is <strong>LOW</strong>. Maintain your healthy diet and regular physical activity!';
          }
          return `📊 <strong>Your Calculated Risk: ${pct}%</strong><br><br>${advice}<br><br>
<strong>Risk Ranges:</strong>
<ul>
  <li>🟢 <strong>0–35%</strong>: Low Risk</li>
  <li>🟠 <strong>35–65%</strong>: Elevated Risk</li>
  <li>🔴 <strong>65–100%</strong>: High Risk</li>
</ul>`;
        }
        return `📊 <strong>Understanding Risk Scores:</strong><br><br>
Run a prediction above to get your clinical risk assessment. The engine uses 11 diagnostic parameters including ECG, resting BP, and cholesterol level.`;
      }
    },
    emergency: {
      patterns: ['emergency', 'heart attack', 'ambulance', 'call 911', '911', '112', 'urgent', 'severe chest pain', 'collapse', 'faint'],
      response: () => `🚨 <strong>CARDIAC EMERGENCY — Act Immediately!</strong><br><br>
<strong>Heart Attack Warning Signs:</strong>
<ul>
  <li>Crushing chest pain or pressure lasting >5 minutes</li>
  <li>Pain radiating to left arm, neck, or jaw</li>
  <li>Sudden cold sweat & severe shortness of breath</li>
  <li>Dizziness or loss of consciousness</li>
</ul>
<strong>🆘 Call Emergency Immediately:</strong><br>
📞 <strong>India: 112</strong> | 📞 <strong>USA: 911</strong> | 📞 <strong>UK: 999</strong>`
    },
    thanks: {
      patterns: ['thank', 'thanks', 'thank you', 'ty', 'helpful', 'great', 'awesome', 'good bot'],
      response: () => `😊 You're very welcome! I'm always here to support your cardiac health journey.<br><br>
Is there anything else I can help you with today? 💙`
    }
  };

  /* ─── Intent Matcher ─── */
  function detectIntent(msg) {
    const lower = msg.toLowerCase().trim();
    for (const [intent, data] of Object.entries(KB)) {
      if (data.patterns.some(p => lower.includes(p))) {
        return intent;
      }
    }
    return null;
  }

  /* ─── Markdown Parser for AI Responses ─── */
  function parseMarkdown(text) {
    if (!text) return '';
    let formatted = text
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      // Bold
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      // Italics
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      // Code blocks / inline code
      .replace(/`(.*?)`/g, '<code style="background:rgba(255,255,255,0.1);padding:2px 5px;border-radius:4px;">$1</code>')
      // Headers
      .replace(/^### (.*$)/gim, '<strong style="color:#5eead4;display:block;margin-top:8px;">$1</strong>')
      .replace(/^## (.*$)/gim, '<strong style="color:#fca5a5;display:block;margin-top:10px;font-size:14px;">$1</strong>');

    // Convert bullet points to HTML list
    const lines = formatted.split('\n');
    let inList = false;
    let resultHTML = '';

    lines.forEach(line => {
      const trimmed = line.trim();
      if (trimmed.startsWith('* ') || trimmed.startsWith('- ') || trimmed.startsWith('• ')) {
        if (!inList) {
          resultHTML += '<ul style="margin:6px 0 6px 16px; display:flex; flex-direction:column; gap:4px;">';
          inList = true;
        }
        resultHTML += `<li>${trimmed.substring(2)}</li>`;
      } else {
        if (inList) {
          resultHTML += '</ul>';
          inList = false;
        }
        if (trimmed.length > 0) {
          resultHTML += `<p style="margin-bottom:6px;">${trimmed}</p>`;
        }
      }
    });

    if (inList) resultHTML += '</ul>';
    return resultHTML;
  }

  /* ─── Call Serverless Proxy /api/chat (Gemini AI Engine) ─── */
  async function callGemini(userMessage) {
    try {
      const payload = {
        message: userMessage,
        riskContext: lastRiskResult ? {
          percentage: lastRiskResult.percentage,
          level: lastRiskResult.level
        } : null,
        history: chatHistory
      };

      const resp = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!resp.ok) {
        console.warn('API route returned status:', resp.status);
        return null;
      }

      const data = await resp.json();
      return data?.reply || null;
    } catch (err) {
      console.error('Call to Gemini serverless proxy failed:', err);
      return null;
    }
  }

  /* ─── Fallback Response ─── */
  function fallbackResponse(msg) {
    if (lastRiskResult) {
      return `💡 <strong>Personalized Recommendation:</strong><br><br>
Based on your clinical evaluation (${lastRiskResult.percentage}% Risk), here are the top steps you should take:<br>
<ul>
  <li>🩺 <strong>Consult a Doctor</strong> — Review your ECG and blood pressure readings.</li>
  <li>🥗 <strong>Heart-Healthy Diet</strong> — Reduce sodium and saturated fats.</li>
  <li>🧘 <strong>Yoga & Stress Relief</strong> — Practice 4-7-8 breathing daily.</li>
</ul>
Ask me specifically about <strong>diet</strong>, <strong>yoga</strong>, <strong>prevention</strong>, or <strong>symptoms</strong>!`;
    }
    return `🤔 Here are popular health topics I can answer instantly:<br><br>
<ul>
  <li>💬 <em>"What are heart disease symptoms?"</em></li>
  <li>💬 <em>"How can I prevent heart disease?"</em></li>
  <li>💬 <em>"What yoga poses help the heart?"</em></li>
  <li>💬 <em>"What diet should I follow?"</em></li>
  <li>💬 <em>"What should I do next?"</em></li>
</ul>`;
  }

  /* ─── Time Helper ─── */
  function getTime() {
    return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }

  /* ─── DOM Helpers ─── */
  function appendMessage(content, role, isAi = false) {
    const messages = document.getElementById('chat-messages');
    const typingIndicator = document.getElementById('typing-indicator');
    
    const row = document.createElement('div');
    row.className = `msg-row ${role}`;
    
    const icon = document.createElement('div');
    icon.className = `msg-icon ${role}`;
    icon.textContent = role === 'bot' ? '🤖' : '🫀';

    const wrap = document.createElement('div');
    
    const bubble = document.createElement('div');
    bubble.className = `msg-bubble ${role}`;
    bubble.innerHTML = content;

    if (role === 'bot' && isAi) {
      const badge = document.createElement('div');
      badge.style.cssText = 'font-size:10px; color:#5eead4; margin-top:6px; font-weight:600; display:flex; align-items:center; gap:4px;';
      badge.innerHTML = '✨ <em>Powered by Gemini AI</em>';
      bubble.appendChild(badge);
    }
    
    const ts = document.createElement('div');
    ts.className = 'msg-timestamp';
    ts.textContent = getTime();
    
    wrap.appendChild(bubble);
    wrap.appendChild(ts);
    row.appendChild(icon);
    row.appendChild(wrap);
    
    messages.insertBefore(row, typingIndicator);
    messages.scrollTop = messages.scrollHeight;

    // Track in history
    chatHistory.push({ role: role === 'bot' ? 'bot' : 'user', text: content });
    if (chatHistory.length > 8) chatHistory.shift();
  }

  function showTyping() {
    const el = document.getElementById('typing-indicator');
    el.classList.add('show');
    document.getElementById('chat-messages').scrollTop = document.getElementById('chat-messages').scrollHeight;
  }

  function hideTyping() {
    document.getElementById('typing-indicator').classList.remove('show');
  }

  function setInputState(disabled) {
    isProcessing = disabled;
    const input = document.getElementById('chat-input');
    const btn = document.getElementById('chat-send-btn');
    if (input) input.disabled = disabled;
    if (btn) {
      btn.disabled = disabled;
      btn.style.opacity = disabled ? '0.5' : '1';
    }
  }

  /* ─── Send Message ─── */
  async function sendMessage(text) {
    if (!text || !text.trim() || isProcessing) return;

    const trimmed = text.trim().substring(0, 500); // 500 char max limit
    
    // Hide quick suggestions after first message
    const suggestions = document.getElementById('quick-suggestions');
    if (suggestions) suggestions.style.display = 'none';
    
    appendMessage(trimmed, 'user');
    document.getElementById('chat-input').value = '';
    document.getElementById('chat-input').style.height = 'auto';
    
    setInputState(true);
    showTyping();
    
    let responseText = null;
    let isAiResponse = false;

    // Check fast local intent first
    const intent = detectIntent(trimmed);
    
    if (intent) {
      await new Promise(r => setTimeout(r, 400 + Math.random() * 200));
      responseText = KB[intent].response();
    } else {
      // Query Gemini AI via Vercel Proxy
      const aiReply = await callGemini(trimmed);
      if (aiReply) {
        responseText = parseMarkdown(aiReply);
        isAiResponse = true;
      } else {
        responseText = fallbackResponse(trimmed);
      }
    }
    
    hideTyping();
    appendMessage(responseText, 'bot', isAiResponse);
    setInputState(false);
    document.getElementById('chat-input').focus();
  }

  /* ─── Open/Close Drawer ─── */
  function openChat() {
    isOpen = true;
    document.getElementById('chat-drawer').classList.add('open');
    document.getElementById('chat-overlay').classList.add('open');
    document.getElementById('chat-fab').style.display = 'none';
    document.getElementById('chat-input').focus();
    
    if (window._chatGreeted) return;
    window._chatGreeted = true;
    
    const fabBadge = document.querySelector('#chat-fab .fab-badge');
    if (fabBadge) fabBadge.remove();
    
    if (lastRiskResult) {
      let riskMsg = '';
      if (lastRiskResult.level === 'high') {
        riskMsg = `⚠️ I noticed your heart disease risk score is <strong>${lastRiskResult.percentage}% (High Risk)</strong>. I'm here to help you understand your results and build a personalized action plan. What would you like to ask?`;
      } else if (lastRiskResult.level === 'moderate') {
        riskMsg = `📊 Your risk score is <strong>${lastRiskResult.percentage}% (Elevated Risk)</strong>. Let's look at preventive measures, diet, and yoga routines to protect your heart.`;
      } else {
        riskMsg = `✅ Great news — your risk score is <strong>${lastRiskResult.percentage}% (Low Risk)</strong>! Ask me anything about maintaining optimal heart health.`;
      }
      setTimeout(() => appendMessage(riskMsg, 'bot'), 300);
    } else {
      setTimeout(() => appendMessage(KB.greeting.response(), 'bot'), 300);
    }
  }

  function closeChat() {
    isOpen = false;
    document.getElementById('chat-drawer').classList.remove('open');
    document.getElementById('chat-overlay').classList.remove('open');
    document.getElementById('chat-fab').style.display = 'flex';
  }

  /* ─── DOM Injection ─── */
  function injectChatHTML() {
    const html = `
      <!-- FAB -->
      <button id="chat-fab" onclick="window.HeartCareChat.open()" title="Chat with HeartCare AI">
        💬
        <div class="fab-badge">AI</div>
      </button>

      <!-- Overlay -->
      <div id="chat-overlay" onclick="window.HeartCareChat.close()"></div>

      <!-- Chat Drawer -->
      <div id="chat-drawer">
        <!-- Header -->
        <div id="chat-header">
          <div class="chat-avatar">❤️</div>
          <div class="chat-header-info">
            <div class="chat-header-name">HeartCare AI</div>
            <div class="chat-header-status">
              <div class="status-dot"></div>
              Online • Gemini 1.5 Flash + Medical KB
            </div>
          </div>
          <button id="chat-close-btn" onclick="window.HeartCareChat.close()">✕</button>
        </div>

        <!-- Quick Chips -->
        <div id="quick-suggestions">
          <div class="quick-chip" onclick="window.HeartCareChat.quickSend('What are heart disease symptoms?')">❤️ Symptoms</div>
          <div class="quick-chip" onclick="window.HeartCareChat.quickSend('How can I prevent heart disease?')">🛡️ Prevention</div>
          <div class="quick-chip" onclick="window.HeartCareChat.quickSend('Yoga poses for heart health?')">🧘 Yoga</div>
          <div class="quick-chip" onclick="window.HeartCareChat.quickSend('What diet should I follow?')">🥗 Diet</div>
          <div class="quick-chip" onclick="window.HeartCareChat.quickSend('What should I do next?')">📋 Action Plan</div>
          <div class="quick-chip" onclick="window.HeartCareChat.quickSend('When is it a heart emergency?')">🚨 Emergency</div>
        </div>

        <!-- Messages -->
        <div id="chat-messages">
          <div id="typing-indicator">
            <div class="msg-icon bot">🤖</div>
            <div class="typing-bubble">
              <div class="typing-dot"></div>
              <div class="typing-dot"></div>
              <div class="typing-dot"></div>
            </div>
          </div>
        </div>

        <!-- Input -->
        <div id="chat-input-area">
          <textarea id="chat-input" placeholder="Ask any heart health question..." rows="1"
            oninput="this.style.height='auto';this.style.height=this.scrollHeight+'px';"
            onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();window.HeartCareChat.send(this.value);}"></textarea>
          <button id="chat-send-btn" onclick="window.HeartCareChat.send(document.getElementById('chat-input').value)">➤</button>
        </div>
      </div>
    `;
    document.body.insertAdjacentHTML('beforeend', html);
  }

  /* ─── Public API ─── */
  window.HeartCareChat = {
    open: openChat,
    close: closeChat,
    send: sendMessage,
    quickSend: (text) => { sendMessage(text); },
    setRiskResult: (pct, level) => {
      lastRiskResult = { percentage: pct, level };
      const fab = document.getElementById('chat-fab');
      if (fab && !isOpen) {
        let badge = fab.querySelector('.fab-badge');
        if (!badge) {
          badge = document.createElement('div');
          badge.className = 'fab-badge';
          fab.appendChild(badge);
        }
        badge.textContent = '!';
        badge.style.background = level === 'high' ? '#ef4444' : (level === 'moderate' ? '#f59e0b' : '#10b981');
      }
    }
  };

  /* ─── Initialize ─── */
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', injectChatHTML);
  } else {
    injectChatHTML();
  }

})();
