const { pool } = require('../config/db');

exports.getSettings = async (req, res) => {
  try {
    const result = await pool.query('SELECT * FROM settings');
    const settings = {};
    result.rows.forEach(r => { settings[r.setting_key] = r.setting_value; });
    res.json(settings);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Database error' });
  }
};

exports.getDoctors = async (req, res) => {
  try {
    const result = await pool.query('SELECT * FROM doctors ORDER BY id ASC');
    const doctors = result.rows.map(d => ({
      ...d,
      rating: parseFloat(d.rating),
      tags: d.tags ? d.tags.split(',') : []
    }));
    res.json(doctors);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Database error' });
  }
};

exports.postFeedback = async (req, res) => {
  const { user_id, name, email, type, message } = req.body;
  if (!type || !message) return res.status(400).json({ error: 'Tipe dan pesan wajib diisi' });
  
  try {
    const result = await pool.query(
      'INSERT INTO feedbacks (user_id, name, email, type, message) VALUES ($1, $2, $3, $4, $5) RETURNING id',
      [user_id || null, name, email, type, message]
    );
    res.json({ success: true, id: result.rows[0].id });
  } catch (err) {
    console.error('Error posting feedback:', err);
    res.status(500).json({ error: 'Database error' });
  }
};
