const express = require('express');
const router = express.Router();
const publicController = require('../controllers/publicController');

router.get('/settings', publicController.getSettings);
router.get('/doctors', publicController.getDoctors);
router.post('/feedbacks', publicController.postFeedback);

module.exports = router;
