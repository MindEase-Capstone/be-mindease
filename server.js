require('dotenv').config();
const express = require('express');
const cors = require('cors');
const { initDB } = require('./config/db');
const http = require('http');
const { Server } = require('socket.io');

const authRoutes = require('./routes/authRoutes');
const postRoutes = require('./routes/postRoutes');
const moodRoutes = require('./routes/moodRoutes');
const adminRoutes = require('./routes/adminRoutes');
const publicRoutes = require('./routes/publicRoutes');
const chatRoutes = require('./routes/chatRoutes');

const app = express();
const server = http.createServer(app);
const io = new Server(server, {
  cors: {
    origin: "*", // allow all or use specific origin if needed
    methods: ["GET", "POST"]
  }
});

const PORT = process.env.PORT || 5000;

app.use(cors());
app.use(express.json());

// Initialize Database
initDB();

// Mount Routes
app.use('/api/auth', authRoutes);
app.use('/api/posts', postRoutes);
app.use('/api/moods', moodRoutes);
app.use('/api/admin', adminRoutes);
app.use('/api/public', publicRoutes);
app.use('/api/chat', chatRoutes);

// Daily Email Reminder Scheduler using node-cron (Setiap hari pukul 08:00 AM)
const cron = require('node-cron');
const { sendDailyMoodReminders } = require('./services/emailService');

cron.schedule('0 8 * * *', () => {
  console.log('⏰ [Cron Job] Memulai pengiriman otomatis email pengingat harian...');
  sendDailyMoodReminders();
});

// Socket.IO logic
const onlineUsers = new Map(); // maps socket.id -> username

io.on('connection', (socket) => {
  // Client joins safe space
  socket.on('join_safe_space', (username) => {
    onlineUsers.set(socket.id, username);
    // Broadcast updated list
    io.emit('online_users', Array.from(onlineUsers.values()));
  });

  socket.on('disconnect', () => {
    onlineUsers.delete(socket.id);
    io.emit('online_users', Array.from(onlineUsers.values()));
  });
});

server.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});
