// Vercel Serverless Function — /api/chat.js
// Secure proxy for Google Gemini 1.5 Flash API with Patient Context & History

module.exports = async function handler(req, res) {
  // CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  // Handle preflight OPTIONS request
  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  // Only allow POST
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const { message, riskContext, history } = req.body || {};

    if (!message || typeof message !== 'string' || !message.trim()) {
      return res.status(400).json({ error: 'Message is required' });
    }

    const GEMINI_KEY = process.env.GEMINI_API_KEY;

    if (!GEMINI_KEY) {
      return res.status(503).json({ 
        error: 'GEMINI_API_KEY environment variable not configured on Vercel',
        source: 'server'
      });
    }

    // Construct Medical Assistant System Prompt with Patient Context
    let systemPrompt = `You are HeartCare AI, an expert, compassionate cardiac wellness AI assistant.
You provide clear, scientifically sound advice on heart disease prevention, clinical risk management, symptoms, medication classes, Mediterranean & DASH diets, yoga poses, and breathing techniques.

Rules:
- Keep answers concise (100–180 words), structured with markdown bullet points.
- Be empathetic, encouraging, and clear.
- Do NOT prescribe exact drug dosages or diagnose medical emergencies.
- Always include a brief note encouraging consultation with a physician/cardiologist.`;

    if (riskContext && typeof riskContext === 'object') {
      systemPrompt += `\n\nPATIENT DIAGNOSTIC CONTEXT:
- Heart Disease Probability: ${riskContext.percentage || 'N/A'}%
- Risk Category: ${riskContext.level ? riskContext.level.toUpperCase() : 'N/A'}
- Primary Clinical Factors: ${riskContext.factors || 'Standard evaluation'}
Tailor your answers specifically considering this patient's risk profile!`;
    }

    // Build multi-turn Gemini payload
    const contents = [];

    if (history && Array.isArray(history) && history.length > 0) {
      const recentHistory = history.slice(-6);
      recentHistory.forEach(item => {
        if (item.text && item.role) {
          contents.push({
            role: item.role === 'user' ? 'user' : 'model',
            parts: [{ text: item.text.replace(/<[^>]*>?/gm, '') }]
          });
        }
      });
    }

    // Add current user message
    contents.push({
      role: 'user',
      parts: [{ text: message }]
    });

    if (contents.length > 0) {
      const firstUserMsgIndex = contents.findIndex(c => c.role === 'user');
      if (firstUserMsgIndex !== -1) {
        contents[firstUserMsgIndex].parts[0].text = `${systemPrompt}\n\nUser Question: ${contents[firstUserMsgIndex].parts[0].text}`;
      }
    }

    // Call Gemini REST API with 9 second timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 9000);

    const apiResponse = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${GEMINI_KEY}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ contents }),
        signal: controller.signal
      }
    );

    clearTimeout(timeoutId);

    if (!apiResponse.ok) {
      const errText = await apiResponse.text();
      console.error('Gemini API returned error:', apiResponse.status, errText);
      return res.status(apiResponse.status).json({ 
        error: 'Gemini API Error', 
        status: apiResponse.status 
      });
    }

    const data = await apiResponse.json();
    const replyText = data?.candidates?.[0]?.content?.parts?.[0]?.text;

    if (!replyText) {
      return res.status(502).json({ error: 'Empty response from Gemini' });
    }

    return res.status(200).json({ reply: replyText, isAi: true });

  } catch (err) {
    if (err.name === 'AbortError') {
      console.error('Gemini request timed out');
      return res.status(540).json({ error: 'Gemini request timed out' });
    }
    console.error('Internal proxy error:', err);
    return res.status(500).json({ error: 'Internal server error', message: err.message });
  }
};
