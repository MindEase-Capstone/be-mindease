const mlService = require('../services/mlService');

/**
 * Handle chatbot communication
 */
exports.handleChat = async (req, res) => {
  const { message } = req.body;

  if (!message || !message.trim()) {
    return res.status(400).json({ error: 'Message is required' });
  }

  try {
    // 1. Cek kesopanan pesan menggunakan ML API
    const validation = await mlService.checkSentiment(message);

    // 2. Jika dinilai tidak sopan
    if (!validation.is_appropriate) {
      return res.json({
        is_appropriate: false,
        reply: "Maaf, pesan Anda dideteksi mengandung kata-kata yang kurang sopan. Di MindEase, kita berkomitmen untuk menjaga ruang komunikasi yang positif, santun, dan saling menghargai. Mari gunakan bahasa yang lebih bersahabat ya! 💚"
      });
    }

    // 3. Jika pesan sopan, buat respons dinamis berdasarkan kata kunci (keyword-based)
    const lowerMessage = message.toLowerCase();
    let reply = "";

    if (lowerMessage.includes('stres') || lowerMessage.includes('stress') || lowerMessage.includes('cemas') || lowerMessage.includes('panik') || lowerMessage.includes('khawatir')) {
      reply = "Saya mengerti perasaanmu. Mengalami stres atau kecemasan memang sangat tidak nyaman dan melelahkan. Cobalah untuk duduk dengan rileks sejenak, tarik napas dalam-dalam dari hidung (4 detik), tahan (4 detik), lalu embuskan perlahan lewat mulut (4 detik). Ingatlah bahwa kamu sudah berusaha keras, dan tidak apa-apa jika belum bisa menyelesaikan semuanya hari ini. Kamu aman di sini. 🤗";
    } 
    else if (lowerMessage.includes('tidur') || lowerMessage.includes('insomnia') || lowerMessage.includes('begadang') || lowerMessage.includes('ngantuk')) {
      reply = "Kurang tidur atau sulit tidur nyenyak memang bisa sangat mempengaruhi mood dan energimu besok. Sebelum tidur malam ini, cobalah lakukan ritual sederhana: redupkan lampu kamarmu, hindari layar ponsel minimal 30 menit sebelum berbaring, dan fokuskan pikiran pada tarikan napas yang lambat. Semoga kamu bisa mendapatkan istirahat yang nyenyak dan memulihkan energi malam ini ya. 🌙";
    } 
    else if (lowerMessage.includes('sedih') || lowerMessage.includes('kecewa') || lowerMessage.includes('nangis') || lowerMessage.includes('depresi') || lowerMessage.includes('hancur')) {
      reply = "Bila kamu sedang merasa sedih atau kecewa, ketahuilah bahwa perasaan itu sepenuhnya valid. Tidak apa-apa untuk menangis atau merasa tidak berdaya sejenak. Kamu tidak perlu selalu terlihat kuat. Jika kamu merasa butuh tempat meluapkan perasaan tanpa takut dihakimi, kamu juga bisa membagikannya secara anonim di fitur 'Ruang Aman' kami. Kami ada di sini untuk mendengarmu. Peluk hangat untukmu! 🫂";
    } 
    else if (lowerMessage.includes('napas') || lowerMessage.includes('pernapasan') || lowerMessage.includes('meditasi') || lowerMessage.includes('tenang')) {
      reply = "Latihan pernapasan sangat baik untuk memicu respons relaksasi tubuh dan menenangkan pikiran yang bising. Mari kita lakukan teknik sederhana bersama: \n1. Tarik napas perlahan lewat hidung... (1... 2... 3... 4)\n2. Tahan napasmu sebentar... (1... 2... 3... 4)\n3. Hembuskan napas perlahan lewat mulut... (1... 2... 3... 4)\nUlangi siklus ini beberapa kali sampai bahu dan pikiranmu terasa lebih ringan. Bagaimana perasaanmu sekarang? 🧘";
    } 
    else if (lowerMessage.includes('halo') || lowerMessage.includes('hai') || lowerMessage.includes('hello') || lowerMessage.includes('poy')) {
      reply = "Halo! Saya AI MindEase Companion. Senang sekali bisa menyapamu hari ini. Bagaimana kabarmu saat ini? Apakah ada cerita atau beban pikiran yang ingin kamu bagikan kepada saya? Saya siap mendengarkan. 🌸";
    } 
    else if (lowerMessage.includes('terima kasih') || lowerMessage.includes('makasih') || lowerMessage.includes('thanks') || lowerMessage.includes('suwun')) {
      reply = "Sama-sama! Senang sekali saya bisa menemanimu dan membantumu sedikit meringankan beban. Ingatlah untuk selalu menyayangi dirimu sendiri dan mengambil jeda yang cukup di tengah kesibukanmu. Kamu berharga! 💚✨";
    } 
    else {
      reply = "Terima kasih banyak sudah mempercayakan cerita atau pesanmu kepada saya. Mendengar dan menuangkan apa yang sedang kamu rasakan adalah langkah awal yang sangat berani dan menyehatkan bagi mentalmu. Ingatlah bahwa beban yang berat akan terasa lebih ringan jika kita bersabar pada diri sendiri dan melangkah satu demi satu. Apakah ada hal lain yang ingin kamu diskusikan hari ini? 💫";
    }

    return res.json({
      is_appropriate: true,
      reply
    });
  } catch (err) {
    console.error('[Chat Controller] Error:', err);
    res.status(500).json({ error: 'Terjadi kesalahan pada server saat memproses pesan chat.' });
  }
};
