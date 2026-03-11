-- AI-Based Disease Prediction System - Database Schema
-- MySQL Database

-- Create database
CREATE DATABASE IF NOT EXISTS healthcare_db;
USE healthcare_db;

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Prediction history table
CREATE TABLE IF NOT EXISTS predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT,
    disease_type ENUM('diabetes', 'heart', 'liver') NOT NULL,
    input_data JSON NOT NULL,
    prediction_result INT NOT NULL,
    result_text VARCHAR(100) NOT NULL,
    confidence_score DECIMAL(5,2) NOT NULL,
    probability_json JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL,
    INDEX idx_user_id (user_id),
    INDEX idx_disease_type (disease_type),
    INDEX idx_created_at (created_at)
);

-- Model accuracy tracking table
CREATE TABLE IF NOT EXISTS model_metrics (
    id INT AUTO_INCREMENT PRIMARY KEY,
    disease_type ENUM('diabetes', 'heart', 'liver') NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    accuracy DECIMAL(5,4) NOT NULL,
    precision_score DECIMAL(5,4),
    recall_score DECIMAL(5,4),
    f1_score DECIMAL(5,4),
    training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_disease_type (disease_type)
);

-- Feedback table for model improvement
CREATE TABLE IF NOT EXISTS feedback (
    id INT AUTO_INCREMENT PRIMARY KEY,
    prediction_id INT NOT NULL,
    user_id INT,
    feedback_type ENUM('correct', 'incorrect') NOT NULL,
    user_comment TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (prediction_id) REFERENCES predictions(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
);

-- Sample stored procedure for getting prediction history
DELIMITER //
CREATE PROCEDURE IF NOT EXISTS get_user_predictions(
    IN p_user_id INT,
    IN p_limit INT
)
BEGIN
    SELECT 
        id,
        disease_type,
        prediction_result,
        result_text,
        confidence_score,
        created_at
    FROM predictions
    WHERE user_id = p_user_id
    ORDER BY created_at DESC
    LIMIT p_limit;
END //
DELIMITER ;

-- Sample stored procedure for disease statistics
DELIMITER //
CREATE PROCEDURE IF NOT EXISTS get_disease_statistics()
BEGIN
    SELECT 
        disease_type,
        COUNT(*) as total_predictions,
        AVG(confidence_score) as avg_confidence,
        SUM(CASE WHEN prediction_result = 1 THEN 1 ELSE 0 END) as positive_predictions,
        SUM(CASE WHEN prediction_result = 0 THEN 1 ELSE 0 END) as negative_predictions
    FROM predictions
    GROUP BY disease_type;
END //
DELIMITER ;

-- Insert sample model metrics (will be updated after training)
INSERT INTO model_metrics (disease_type, model_type, accuracy, precision_score, recall_score, f1_score) VALUES
('diabetes', 'Random Forest', 0.8500, 0.8200, 0.7800, 0.8000),
('heart', 'Random Forest', 0.8800, 0.8600, 0.8500, 0.8550),
('liver', 'Random Forest', 0.8200, 0.8000, 0.7900, 0.7950);

