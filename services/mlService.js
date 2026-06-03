const ML_API_URL = process.env.ML_API_URL || 'http://127.0.0.1:8000';

// Daftar kata kasar lokal yang umum untuk pengecekan instan (Lapis 1)
const SWEAR_WORDS_BLOCKLIST = [
  'anjing', 'babi', 'bangsat', 'goblok', 'tolol', 'bodoh', 'bego', 
  'bajingan', 'brengsek', 'asu', 'kampret', 'kontol', 'memek', 
  'peler', 'pantek', 'ngentot', 'perek', 'jablay', 'lonte', 'sinting',
  // slang / abbreviations
  'ajg', 'anjg', 'anj', 'bgsd', 'bgst', 'kntl', 'mmk', 'bjg', 'bjgn',
  'ngntt', 'ngentod', 'ngewe', 'lnt', 'plr', 'pntk'
];

const leetMap = {
  '4': 'a', '@': 'a',
  '1': 'i', '!': 'i',
  '3': 'e',
  '0': 'o',
  '5': 's', '$': 's',
  '7': 't',
  '8': 'b',
  '9': 'g'
};

const translateLeet = (str) => {
  return str.toLowerCase().split('').map(char => leetMap[char] || char).join('');
};

// Pola regex dinamis untuk setiap kata kasar guna mendeteksi variasi karakter/spasi/duplikasi
const BLOCKLIST_PATTERNS = SWEAR_WORDS_BLOCKLIST.map(word => {
  const escaped = word.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const patternStr = '\\b' + escaped
    .split('')
    .map(char => `${char}+`)
    .join('[\\s\\W_]*') + '\\b';
  return new RegExp(patternStr, 'i');
});

/**
 * Checks the sentiment/politeness of a text using a multi-layer approach:
 * 1. Local Swear Word Blocklist (Instant)
 * 2. Entire Sentence ML API Check
 * 3. Word-by-word ML API Deep Verification (Leverages the model's own knowledge base)
 * 
 * @param {string} text - The input text to check.
 * @returns {Promise<{is_appropriate: boolean, prediction: string}>}
 */
const checkSentiment = async (text) => {
  if (!text || !text.trim()) {
    return { is_appropriate: true, prediction: 'sopan' };
  }

  // Lakukan penerjemahan leetspeak dasar
  const normalizedText = translateLeet(text);

  // ==========================================
  // LAPIS 1: Cek Kata Kasar Lokal (Instan & Pasti Kena)
  // ==========================================
  for (let i = 0; i < BLOCKLIST_PATTERNS.length; i++) {
    if (BLOCKLIST_PATTERNS[i].test(normalizedText)) {
      console.log(`[ML Service] Diblokir oleh Blocklist Kata Kasar Lokal: "${SWEAR_WORDS_BLOCKLIST[i]}"`);
      return { is_appropriate: false, prediction: 'kasar' };
    }
  }

  // Bersihkan teks untuk ekstraksi kata pada scanning berikutnya
  const cleanText = normalizedText.replace(/[^a-zA-Z\s]/g, ' ');
  const words = cleanText.split(/\s+/).filter(w => w.length > 0);

  // ==========================================
  // LAPIS 2: Cek Keseluruhan Kalimat via ML API
  // ==========================================
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000); // 5-second timeout

    const response = await fetch(`${ML_API_URL}/check-profanity`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text: normalizedText }),
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      console.warn(`[ML Service] ML API responded with status ${response.status}. Falling back to default 'sopan'.`);
      return { is_appropriate: true, prediction: 'sopan' };
    }

    const data = await response.json();
    
    // Jika kalimat utuh memang dianggap tidak pantas oleh ML, langsung blokir
    if (data.is_appropriate === false) {
      console.log(`[ML Service] Diblokir oleh ML API (Kalimat Utuh): "${text}"`);
      return { is_appropriate: false, prediction: data.prediction || 'kasar' };
    }

    // ==========================================
    // LAPIS 3: Deep Scan - Cek Kata per Kata via ML API
    // ==========================================
    // Ambil kata unik yang panjangnya > 2 karakter untuk menghindari stop words pendek
    const uniqueWords = [...new Set(words)].filter(w => w.length > 2);

    for (const word of uniqueWords) {
      try {
        const wordResponse = await fetch(`${ML_API_URL}/check-profanity`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ text: word }),
        });

        if (wordResponse.ok) {
          const wordData = await wordResponse.json();
          // Jika salah satu kata dalam kalimat diprediksi KASAR oleh ML, blokir seluruh kalimat!
          if (wordData.is_appropriate === false) {
            console.log(`[ML Service] Diblokir oleh ML API (Scan Kata: "${word}")`);
            return { is_appropriate: false, prediction: 'kasar' };
          }
        }
      } catch (wordErr) {
        // Abaikan error kata tunggal agar tidak menghentikan keseluruhan pengecekan
        console.error(`[ML Service] Gagal memverifikasi kata "${word}":`, wordErr.message);
      }
    }

    // Lolos semua lapis pengamanan
    return {
      is_appropriate: true,
      prediction: 'sopan',
    };
  } catch (error) {
    console.error('[ML Service] Error calling ML API:', error.message);
    console.warn('[ML Service] Falling back to default safe validation (is_appropriate: true).');
    return { is_appropriate: true, prediction: 'sopan' };
  }
};

module.exports = {
  checkSentiment,
};
